from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# 启动时加载 `.env`，这样项目里的所有模块都能通过 `settings` 读取统一配置。
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """集中管理项目的所有运行参数。

    这个类的作用是把：
    - 聊天模型配置
    - embedding 配置
    - LightRAG 配置
    - reranker 配置
    - MCP 配置
    统一收口到一个地方，避免这些值散落在多个文件里。
    """

    # 兼容保留的通用 OpenAI 风格配置。
    # 后面 chat / embedding 会优先使用各自独立配置。
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")

    # 聊天模型使用的 OpenAI-compatible 配置。
    chat_api_key: str = os.getenv("CHAT_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    chat_base_url: str = os.getenv("CHAT_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))

    # 向量模型使用的 OpenAI-compatible 配置。
    embedding_api_key: str = os.getenv(
        "EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")
    )
    embedding_base_url: str = os.getenv(
        "EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "")
    )
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # embedding 相关的模型能力参数，LightRAG 初始化时会用到。
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "3072"))
    embedding_max_tokens: int = int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))

    # LightRAG 的存储目录、查询模式和文本切块参数。
    lightrag_workdir: str = os.getenv("LIGHTRAG_WORKDIR", ".lightrag_cache")
    lightrag_query_mode: str = os.getenv("LIGHTRAG_QUERY_MODE", "hybrid")
    lightrag_response_type: str = os.getenv(
        "LIGHTRAG_RESPONSE_TYPE", "Multiple Paragraphs"
    )
    lightrag_language: str = os.getenv("LIGHTRAG_LANGUAGE", "Simplified Chinese")
    chunk_token_size: int = int(os.getenv("CHUNK_TOKEN_SIZE", "1200"))
    chunk_overlap_token_size: int = int(
        os.getenv("CHUNK_OVERLAP_TOKEN_SIZE", "100")
    )
    semantic_chunk_min_tokens: int = int(
        os.getenv("SEMANTIC_CHUNK_MIN_TOKENS", "80")
    )
    semantic_chunk_overlap_sentences: int = int(
        os.getenv("SEMANTIC_CHUNK_OVERLAP_SENTENCES", "1")
    )
    preference_store_path: str = os.getenv(
        "PREFERENCE_STORE_PATH", ".memory/preferences.json"
    )
    session_summary_store_path: str = os.getenv(
        "SESSION_SUMMARY_STORE_PATH", ".memory/session_summaries.json"
    )
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "default_user")
    session_memory_max_messages: int = int(
        os.getenv("SESSION_MEMORY_MAX_MESSAGES", "10")
    )
    session_memory_recent_messages: int = int(
        os.getenv("SESSION_MEMORY_RECENT_MESSAGES", "6")
    )

    # MCP 的开关和运行范围控制。
    enable_mcp_web_search: bool = (
        os.getenv("ENABLE_MCP_WEB_SEARCH", "true").lower() == "true"
    )
    enable_mcp_filesystem: bool = (
        os.getenv("ENABLE_MCP_FILESYSTEM", "true").lower() == "true"
    )
    mcp_web_search_max_results: int = int(
        os.getenv("MCP_WEB_SEARCH_MAX_RESULTS", "5")
    )
    mcp_filesystem_root: str = os.getenv("MCP_FILESYSTEM_ROOT", ".")

    # reranker 本地模型路径和推理参数。
    reranker_model_path: str = os.getenv("RERANKER_MODEL_PATH", "")
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cuda")
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "5"))

    # 整个 agent 的总系统提示词。
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "你是一个严谨的中文知识库助手。请优先依据检索到的上下文回答；如果上下文不足，请明确说明，不要编造。",
    )

    def validate(self) -> None:
        """校验项目主流程运行所需的关键配置。"""
        missing = []
        if not self.chat_base_url:
            missing.append("CHAT_BASE_URL or OPENAI_BASE_URL")
        if not self.chat_model:
            missing.append("CHAT_MODEL")
        if not self.embedding_base_url:
            missing.append("EMBEDDING_BASE_URL or OPENAI_BASE_URL")
        if not self.embedding_model:
            missing.append("EMBEDDING_MODEL")

        if missing:
            names = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variables: {names}. "
                "Please update your .env file before running the project."
            )

    def validate_reranker(self) -> None:
        """只有在真正使用 reranker 时才校验它的本地模型路径。"""
        if not self.reranker_model_path:
            raise ValueError(
                "Missing RERANKER_MODEL_PATH. Please point it to your local "
                "bge-reranker-large directory before using the reranker tool."
            )


# 整个项目共享这一份只读配置对象。
settings = Settings()
