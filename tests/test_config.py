from __future__ import annotations

import pytest

from langgraph_lightrag_demo.config import Settings


def test_validate_requires_chat_and_embedding_configuration() -> None:
    settings = Settings(
        chat_base_url="",
        chat_model="",
        embedding_base_url="",
        embedding_model="",
    )

    with pytest.raises(ValueError) as exc_info:
        settings.validate()

    message = str(exc_info.value)
    assert "CHAT_BASE_URL or OPENAI_BASE_URL" in message
    assert "CHAT_MODEL" in message
    assert "EMBEDDING_BASE_URL or OPENAI_BASE_URL" in message
    assert "EMBEDDING_MODEL" in message


def test_validate_passes_with_required_configuration() -> None:
    settings = Settings(
        chat_base_url="http://localhost:8000/v1",
        chat_model="demo-chat-model",
        embedding_base_url="http://localhost:8001/v1",
        embedding_model="demo-embedding-model",
    )

    settings.validate()


def test_validate_reranker_requires_local_model_path() -> None:
    settings = Settings(reranker_model_path="")

    with pytest.raises(ValueError) as exc_info:
        settings.validate_reranker()

    assert "RERANKER_MODEL_PATH" in str(exc_info.value)
