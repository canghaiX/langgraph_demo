from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .config import settings


# 用空行分段的方式粗拆 passage。
# 这是“检索后重排”的切分，不是建库时的 chunk 切分。
_SPLIT_PATTERN = re.compile(r"\n\s*\n+")


@dataclass
class RankedPassage:
    """表示一个重排后的候选片段及其分数。"""

    text: str
    score: float


class RerankerService:
    """对 LightRAG 召回结果做精排。

    这里使用的是 CrossEncoder，也就是典型的 reranker 模型：
    给定 (question, passage) 对，直接输出相关性分数。
    """

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        """懒加载本地 reranker 模型。

        只有 agent 真正需要做重排时，才会加载 `bge-reranker-large`。
        """
        if self._model is None:
            settings.validate_reranker()
            self._model = CrossEncoder(
                settings.reranker_model_path,
                device=settings.reranker_device,
                trust_remote_code=True,
            )
        return self._model

    @staticmethod
    def split_passages(text: str) -> list[str]:
        """把长上下文拆成候选段落。

        这里优先按段落切。
        如果文本没有明显段落，就退化成按非空行切。
        """
        chunks = [chunk.strip() for chunk in _SPLIT_PATTERN.split(text) if chunk.strip()]
        if chunks:
            return chunks

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines

    def rerank(self, question: str, passages: list[str], top_k: int | None = None) -> list[RankedPassage]:
        """同步重排函数，适合在后台线程里运行。"""
        if not passages:
            return []

        model = self._get_model()
        pairs = [[question, passage] for passage in passages]
        scores = model.predict(pairs)
        ranked = sorted(
            (RankedPassage(text=passage, score=float(score)) for passage, score in zip(passages, scores)),
            key=lambda item: item.score,
            reverse=True,
        )
        limit = top_k or settings.reranker_top_k
        return ranked[:limit]

    async def arerank(
        self, question: str, passages: list[str], top_k: int | None = None
    ) -> list[RankedPassage]:
        """异步封装，避免阻塞主事件循环。"""
        return await asyncio.to_thread(self.rerank, question, passages, top_k)


# 暴露单例 service，供 agent tool 直接调用。
reranker_service = RerankerService()
