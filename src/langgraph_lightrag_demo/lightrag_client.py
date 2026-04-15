from __future__ import annotations

from pathlib import Path
from typing import Iterable
from collections import Counter
import io
import logging
import re

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import setup_logger, wrap_embedding_func_with_attrs

from .config import settings
from .semantic_chunker import SemanticChunk, chunk_text_by_semantics


# 打开 LightRAG 自身日志，方便你观察建库和查询过程。
setup_logger("lightrag", level="INFO")
setup_logger("langgraph_lightrag_demo", level="INFO")
logger = logging.getLogger("langgraph_lightrag_demo.pdf_ingest")


# 如果 PDF 直接抽取出来的正文太少，通常说明它更像“扫描图片 PDF”，
# 或者版面非常复杂，文本层已经不可用了。这时我们自动回退到 OCR。
#
# 这里不用过大的阈值，是为了避免把本来就很短的 PDF 误判为 OCR 场景；
# 也不用过小的阈值，是为了避免只抽出零散页码/页眉时被当成有效文本。
MIN_DIRECT_PDF_TEXT_LENGTH = 120


# OCR 结果会按页拼接，中间插入页码标记，方便后续排查召回来源。
# 这个标记会作为普通文本进入 LightRAG，后面问答时也能帮助定位答案来自哪一页。
PDF_PAGE_BREAK_TEMPLATE = "\n\n[PDF Page {page_number}]\n"


# 下面这些常量控制 PDF 清洗行为。
# 它们不是“绝对正确”的规则，而是针对常见办公文档、扫描件、报告类 PDF 做的保守启发式。
#
# 调整建议：
# - 如果你的文档正文很短、图注很多，可以适当降低阈值
# - 如果页眉页脚特别长、重复性很强，可以适当提高重复判定阈值
MIN_MEANINGFUL_LINE_LENGTH = 2
MAX_SHORT_NOISE_LINE_LENGTH = 4
HEADER_FOOTER_SCAN_LINE_COUNT = 3
REPEATED_LINE_MIN_COUNT = 2
PDF_LOG_PREVIEW_CHARS = 500


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """把项目里的聊天模型适配成 LightRAG 需要的 LLM 回调。

    LightRAG 不直接依赖某一个固定厂商，而是接收一个“给我 prompt，我返回结果”的函数。
    这里我们把 OpenAI-compatible 的聊天接口包装成它能用的形式。
    """
    return await openai_complete_if_cache(
        settings.chat_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=settings.embedding_dim,
    max_token_size=settings.embedding_max_tokens,
    model_name=settings.embedding_model,
)
async def _embedding_func(texts: list[str]):
    """把 embedding 接口包装成 LightRAG 需要的向量化函数。"""
    return await openai_embed(
        texts,
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def _normalize_extracted_text(text: str) -> str:
    """对提取出的文本做一层轻量清洗。

    这里刻意只做“保守清洗”：
    - 统一换行
    - 去掉首尾空白
    - 压缩连续空行

    不在这里做激进的段落重组或去页眉页脚，原因是：
    - OCR / PDF 抽取质量可能并不稳定
    - 过度清洗有时会破坏原始语义顺序
    - 对于 RAG，保留更多原始痕迹通常比“看起来更漂亮”更安全
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized = "\n".join(lines).strip()

    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")

    return normalized


def _preview_text_for_log(text: str, max_chars: int = PDF_LOG_PREVIEW_CHARS) -> str:
    """把长文本压缩成适合日志展示的预览。

    日志的目标是帮助你观察“清洗前后变化”，而不是把整份 PDF 全部打出来。
    所以这里会：
    - 折叠连续空白
    - 控制最大长度
    - 保留省略标记，提示预览被截断
    """
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."


def _log_pdf_cleaning_preview(
    path: Path,
    extraction_method: str,
    raw_text: str,
    cleaned_text: str,
) -> None:
    """打印 PDF 清洗前后的文本样例。

    这里按“一个文件一条日志”的方式输出，避免逐页打印造成日志过载。
    日志里会包含：
    - 使用的是 direct 还是 ocr
    - 清洗前后字符数
    - 截断后的预览文本
    """
    logger.info(
        (
            "PDF ingest preview | file=%s | method=%s | raw_chars=%d | cleaned_chars=%d\n"
            "[Before]\n%s\n"
            "[After]\n%s"
        ),
        path,
        extraction_method,
        len(raw_text),
        len(cleaned_text),
        _preview_text_for_log(raw_text),
        _preview_text_for_log(cleaned_text),
    )


def _is_page_number_line(text: str) -> bool:
    """判断一行文本是否像“页码行”。

    这里专门识别非常常见的页码形态：
    - 纯数字：`12`
    - 带斜杠：`3/10`
    - 英文页码：`Page 3`, `Page 3 of 10`
    - 中文页码：`第 3 页`, `共 10 页 第 3 页`

    之所以单独抽成函数，是为了把“删页码”这件事和其他噪声规则解耦。
    后续如果你遇到公司模板里的特殊页码格式，可以只改这里。
    """
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False

    page_number_patterns = [
        r"^\d{1,4}$",
        r"^\d{1,4}\s*/\s*\d{1,4}$",
        r"^page\s*\d{1,4}$",
        r"^page\s*\d{1,4}\s*of\s*\d{1,4}$",
        r"^第\s*\d{1,4}\s*页$",
        r"^第\s*\d{1,4}\s*页\s*/\s*共\s*\d{1,4}\s*页$",
        r"^共\s*\d{1,4}\s*页\s*第\s*\d{1,4}\s*页$",
    ]
    lowered = normalized.lower()
    return any(re.fullmatch(pattern, lowered) for pattern in page_number_patterns)


def _looks_like_caption_line(text: str) -> bool:
    """尽量保留图注、表注。

    图注和表注往往比较短，如果只按“短文本=噪声”处理，很容易误删。
    所以这里把常见的图注模式放进白名单，例如：
    - 图 1
    - 图1: 系统架构
    - Figure 2. Model overview
    - 表 3-1 评估结果
    """
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False

    caption_patterns = [
        r"^(图|表)\s*\d+([\-\.]\d+)*([:：.．、]\s*.*)?$",
        r"^(figure|fig\.?|table)\s*\d+([\-\.]\d+)*([:：.．、]\s*.*)?$",
    ]
    lowered = normalized.lower()
    return any(re.fullmatch(pattern, lowered) for pattern in caption_patterns)


def _looks_like_noise_symbol_line(text: str) -> bool:
    """判断一行是否更像图标、装饰符号、OCR 残片，而不是正文。

    典型噪声包括：
    - 只有标点或分隔符
    - 只有零散字母数字组合
    - OCR 从图标边缘识别出的碎片字符
    """
    normalized = text.strip()
    if not normalized:
        return False

    if _looks_like_caption_line(normalized):
        return False

    if re.fullmatch(r"[\W_]+", normalized, flags=re.UNICODE):
        return True

    if len(normalized) <= MAX_SHORT_NOISE_LINE_LENGTH and re.fullmatch(
        r"[A-Za-z0-9\-_/|.]+", normalized
    ):
        return True

    return False


def _clean_pdf_page_lines(lines: list[str]) -> list[str]:
    """对单页文本做逐行清洗。

    这一层不做跨页统计，只做“单页内就能判断”的规则：
    - 删空行
    - 删页码
    - 删明显符号噪声
    - 保留图注/表注

    这么分层的原因是：
    - 页码和噪声通常单页就能判断
    - 页眉页脚是否重复，则必须跨页比较后才能决定
    """
    cleaned_lines: list[str] = []

    for raw_line in lines:
        line = " ".join(raw_line.strip().split())
        if not line:
            continue

        if _is_page_number_line(line):
            continue

        if _looks_like_noise_symbol_line(line):
            continue

        # 极短文本既可能是噪声，也可能是“图 1”“附录 A”这种有效标签。
        # 所以这里先放行图注/表注；其余特别短且缺乏中文/字母信息的行才删除。
        if (
            len(line) < MIN_MEANINGFUL_LINE_LENGTH
            and not _looks_like_caption_line(line)
            and not re.search(r"[\u4e00-\u9fffA-Za-z]", line)
        ):
            continue

        cleaned_lines.append(line)

    return cleaned_lines


def _find_repeated_header_footer_lines(pages: list[list[str]]) -> set[str]:
    """统计跨页重复出现的页眉/页脚候选行。

    思路：
    - 只看每页顶部/底部若干行
    - 统计哪些行在多页重复出现
    - 这些高重复文本大概率是页眉、页脚、版权声明、公司名、模板编号等

    这里只返回“候选集合”，真正删除时仍会做额外保护，避免误删短标题。
    """
    counter: Counter[str] = Counter()

    for page_lines in pages:
        if not page_lines:
            continue

        candidates = (
            page_lines[:HEADER_FOOTER_SCAN_LINE_COUNT]
            + page_lines[-HEADER_FOOTER_SCAN_LINE_COUNT:]
        )

        # 同一页内只记一次，避免某页 OCR 把重复文本识别多次时放大权重。
        for line in set(candidates):
            counter[line] += 1

    return {
        line
        for line, count in counter.items()
        if count >= REPEATED_LINE_MIN_COUNT and len(line) > MAX_SHORT_NOISE_LINE_LENGTH
    }


def _remove_repeated_headers_and_footers(pages: list[list[str]]) -> list[list[str]]:
    """删除跨页重复页眉页脚，但尽量避免误删真实正文。"""
    repeated_lines = _find_repeated_header_footer_lines(pages)
    if not repeated_lines:
        return pages

    cleaned_pages: list[list[str]] = []
    for page_lines in pages:
        cleaned_page: list[str] = []
        for index, line in enumerate(page_lines):
            is_edge_line = (
                index < HEADER_FOOTER_SCAN_LINE_COUNT
                or index >= max(len(page_lines) - HEADER_FOOTER_SCAN_LINE_COUNT, 0)
            )
            # 只有在“页面边缘”出现的重复行才删除。
            # 这样可以减少正文恰好重复一句话时被误删的概率。
            if is_edge_line and line in repeated_lines and not _looks_like_caption_line(line):
                continue
            cleaned_page.append(line)
        cleaned_pages.append(cleaned_page)

    return cleaned_pages


def _build_pdf_text_from_pages(pages: list[list[str]]) -> str:
    """把按页清洗后的文本重新拼成最终入库文本。"""
    parts: list[str] = []
    for page_number, page_lines in enumerate(pages, start=1):
        page_text = _normalize_extracted_text("\n".join(page_lines))
        if page_text:
            parts.append(PDF_PAGE_BREAK_TEMPLATE.format(page_number=page_number))
            parts.append(page_text)
    return "".join(parts).strip()


def _extract_text_from_plain_text_file(path: Path) -> str:
    """读取普通文本文件。

    当前项目把 .txt / .md 都视为“可以直接按 UTF-8 解码的文本文件”。
    如果后续你有 GBK、GB18030 等编码需求，再在这里扩展编码探测逻辑即可。
    """
    return _normalize_extracted_text(path.read_text(encoding="utf-8"))


def _extract_text_from_pdf_direct(path: Path) -> str:
    """优先尝试直接从 PDF 文本层提取文字。

    这一步适合“电子版 PDF”：
    - 论文
    - 导出的合同
    - office 软件生成的 PDF

    如果 PDF 本身是扫描图片，或者文本层损坏，这里往往只能抽到很少的字，
    后续就会自动回退到 OCR。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'pypdf'. Please install it before ingesting PDF files."
        ) from exc

    reader = PdfReader(str(path))
    pages: list[list[str]] = []
    raw_page_texts: list[str] = []

    for page in reader.pages:
        extracted = page.extract_text() or ""
        raw_page_texts.append(_normalize_extracted_text(extracted))
        normalized = _normalize_extracted_text(extracted)
        raw_lines = normalized.splitlines() if normalized else []
        cleaned_lines = _clean_pdf_page_lines(raw_lines)
        if cleaned_lines:
            pages.append(cleaned_lines)
        else:
            pages.append([])

    pages = _remove_repeated_headers_and_footers(pages)
    cleaned_text = _build_pdf_text_from_pages(pages)
    raw_text = "\n\n".join(text for text in raw_page_texts if text).strip()
    _log_pdf_cleaning_preview(
        path=path,
        extraction_method="direct",
        raw_text=raw_text,
        cleaned_text=cleaned_text,
    )
    return cleaned_text


def _extract_text_from_pdf_ocr(path: Path) -> str:
    """对 PDF 每一页渲染图片后执行 OCR。

    这里选择“先渲染、再 OCR”的通用思路，是因为扫描版 PDF 本质上是图片集合。
    只要能把每页渲染成位图，就可以交给 OCR 引擎识别。

    当前实现使用：
    - PyMuPDF：负责把 PDF 页渲染为图片
    - RapidOCR：负责识别图片中的文字

    这样做的好处是：
    - 对中文场景比较友好
    - 不依赖系统级 Tesseract 可执行文件
    - 纯 Python 安装体验通常比传统 OCR 方案更平滑
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'pymupdf'. Please install it before OCR-ing PDF files."
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'Pillow'. Please install it before OCR-ing PDF files."
        ) from exc

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'rapidocr-onnxruntime'. Please install it before OCR-ing PDF files."
        ) from exc

    ocr_engine = RapidOCR()
    doc = fitz.open(path)
    pages: list[list[str]] = []
    raw_page_texts: list[str] = []

    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)

            # 适当放大渲染倍率，能明显改善小字和模糊扫描件的 OCR 成功率。
            # 这里选 2 倍是效果和速度之间的折中。
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))

            # RapidOCR 返回若干检测框，每个框包含识别文本。
            # 我们不保留坐标，只按检测顺序拼接为普通文本，因为 LightRAG 最终接收的是纯字符串。
            ocr_result, _ = ocr_engine(image)
            page_lines: list[str] = []
            for item in ocr_result or []:
                if len(item) >= 2:
                    text = str(item[1]).strip()
                    if text:
                        page_lines.append(text)

            raw_page_texts.append(_normalize_extracted_text("\n".join(page_lines)))
            cleaned_lines = _clean_pdf_page_lines(page_lines)
            pages.append(cleaned_lines)
    finally:
        doc.close()

    pages = _remove_repeated_headers_and_footers(pages)
    cleaned_text = _build_pdf_text_from_pages(pages)
    raw_text = "\n\n".join(text for text in raw_page_texts if text).strip()
    _log_pdf_cleaning_preview(
        path=path,
        extraction_method="ocr",
        raw_text=raw_text,
        cleaned_text=cleaned_text,
    )
    return cleaned_text


def _extract_text_from_path(path: Path) -> str:
    """根据文件后缀选择对应的文本提取策略。

    这个函数是“文件格式适配层”，它的职责是把不同来源的文件统一转换成纯文本。
    一旦文本提取成功，后续就可以无差别地交给 LightRAG。
    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return _extract_text_from_plain_text_file(path)

    if suffix == ".pdf":
        direct_text = _extract_text_from_pdf_direct(path)
        if len(direct_text) >= MIN_DIRECT_PDF_TEXT_LENGTH:
            return direct_text
        return _extract_text_from_pdf_ocr(path)

    raise ValueError(f"Unsupported file type for ingestion: {path.suffix}")


def _format_semantic_chunk_for_ingest(
    path: Path,
    chunk: SemanticChunk,
) -> str:
    """把语义 chunk 渲染成带来源元数据的入库文本。"""
    parts = ["[Source File]", path.name]
    if chunk.section_path:
        section_path = " > ".join(part for part in chunk.section_path if part)
        if section_path:
            parts.extend(["", "[Section Path]", section_path])
    if chunk.page_marker:
        parts.extend(["", chunk.page_marker])
    parts.extend(["", chunk.text.strip()])
    return "\n".join(part for part in parts if part is not None).strip()


def _build_ingest_documents(path: Path, content: str) -> list[str]:
    """把单个文件内容切成语义 chunk，并渲染成入库文档。"""
    chunks = chunk_text_by_semantics(
        content,
        max_tokens=settings.chunk_token_size,
        min_chunk_tokens=settings.semantic_chunk_min_tokens,
        overlap_sentences=settings.semantic_chunk_overlap_sentences,
    )
    if not chunks:
        return []
    return [_format_semantic_chunk_for_ingest(path, chunk) for chunk in chunks]


class LightRAGService:
    """负责管理项目里唯一的一份 LightRAG 实例。

    为什么单独包一层 service：
    - 统一初始化逻辑
    - 做懒加载，避免程序一启动就建库
    - 集中提供 ingest / query / close 这些高层操作
    """

    def __init__(self) -> None:
        self._rag: LightRAG | None = None

    async def get_rag(self) -> LightRAG:
        """懒加载初始化 LightRAG。

        第一次真正需要查库或入库时才创建实例。
        这样启动 CLI 时更轻，也更适合 agent 按需调用。
        """
        if self._rag is None:
            settings.validate()
            Path(settings.lightrag_workdir).mkdir(parents=True, exist_ok=True)
            rag = LightRAG(
                working_dir=settings.lightrag_workdir,
                llm_model_func=_llm_model_func,
                embedding_func=_embedding_func,
                chunk_token_size=settings.chunk_token_size,
                chunk_overlap_token_size=settings.chunk_overlap_token_size,
                addon_params={"language": settings.lightrag_language},
            )
            await rag.initialize_storages()
            self._rag = rag
        return self._rag

    async def ingest_text(self, text: str) -> None:
        """向知识库插入单段文本。适合后续扩展成 API 或动态入库场景。"""
        rag = await self.get_rag()
        await rag.ainsert(text)

    async def ingest_files(self, paths: Iterable[Path]) -> list[Path]:
        """批量读取文件内容并交给 LightRAG 建库。

        注意职责边界：
        - 文件解析、PDF 直抽、OCR 回退：在这里完成
        - 语义递归切分：在入库前先完成一轮预切分
        - 实体/关系抽取、索引构建：交给 LightRAG 内部完成

        这样拆分的好处是：
        - 业务层可以自由扩展更多文件格式
        - 入库前先尽量保留章节、段落、句子这些语义边界
        - LightRAG 始终只接收它最擅长处理的“纯文本”
        """
        rag = await self.get_rag()
        inserted: list[Path] = []
        for path in paths:
            content = _extract_text_from_path(path)
            if not content:
                raise ValueError(
                    f"No readable text could be extracted from file: {path}"
                )

            documents = _build_ingest_documents(path, content)
            if not documents:
                raise ValueError(f"No semantic chunks were produced for file: {path}")

            for document in documents:
                await rag.ainsert(document)
            inserted.append(path)
        return inserted

    async def query_context(self, question: str) -> str:
        """只取回 LightRAG 召回的上下文，不直接让它输出最终答案。

        这样做的原因是：
        - 回答权交给上层 agent
        - agent 可以继续决定是否 rerank、是否结合其他工具
        """
        rag = await self.get_rag()
        result = await rag.aquery(
            question,
            param=QueryParam(
                mode=settings.lightrag_query_mode,
                response_type=settings.lightrag_response_type,
                only_need_context=True,
            ),
        )
        return str(result)

    async def close(self) -> None:
        """优雅关闭存储连接，避免程序退出时留下未关闭资源。"""
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None


# 对外暴露单例 service，项目里统一复用它。
lightrag_service = LightRAGService()
