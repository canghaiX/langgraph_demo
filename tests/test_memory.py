from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_lightrag_demo.memory import (
    MemoryManager,
    SessionSummaryStore,
    UserPreferenceStore,
    build_runtime_context_text,
    infer_preference_updates,
)


def test_infer_preference_updates_extracts_explicit_preferences() -> None:
    updates = infer_preference_updates("以后都用中文回答，详细一点，多给代码，我在找实习。")

    assert updates["language"] == "zh"
    assert updates["response_style"] == "detailed"
    assert updates["code_preference"] == "prefer_examples"
    assert updates["career_focus"] == "internship"


def test_build_runtime_context_text_contains_preferences_and_summary(tmp_path: Path) -> None:
    manager = MemoryManager(
        preference_store=UserPreferenceStore(tmp_path / "prefs.json"),
        summary_store=SessionSummaryStore(tmp_path / "summaries.json"),
    )
    preferences = manager.update_preferences(
        "demo_user",
        language="zh",
        response_style="detailed",
        code_preference="prefer_examples",
        career_focus="internship",
    )

    text = build_runtime_context_text(preferences, "用户正在准备项目面试。")

    assert "默认使用中文回答" in text
    assert "回答风格偏详细" in text
    assert "实习求职视角" in text
    assert "用户正在准备项目面试。" in text


def test_compact_history_moves_old_messages_into_summary(monkeypatch, tmp_path: Path) -> None:
    manager = MemoryManager(
        preference_store=UserPreferenceStore(tmp_path / "prefs.json"),
        summary_store=SessionSummaryStore(tmp_path / "summaries.json"),
    )

    async def fake_summarize(previous_summary: str, messages: list) -> str:
        return f"{previous_summary} | summarized={len(messages)}".strip()

    monkeypatch.setattr(
        "langgraph_lightrag_demo.memory.summarize_session_memory",
        fake_summarize,
    )

    history = []
    for index in range(1, 7):
        history.append(HumanMessage(content=f"q{index}"))
        history.append(AIMessage(content=f"a{index}"))

    compacted = asyncio.run(manager.compact_history("demo_user", history))
    summary = manager.get_summary("demo_user")

    assert len(compacted) == 6
    assert summary.summarized_message_count == 6
    assert "summarized=6" in summary.summary
