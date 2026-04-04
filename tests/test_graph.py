from __future__ import annotations

import asyncio

from langgraph_lightrag_demo.graph import (
    _agent_result_to_text,
    _append_trace_event,
    _build_router_system_prompt,
    _message_content_to_text,
    clear_metrics_session,
    clear_trace_session,
    get_metrics_snapshot,
    get_trace_events,
    record_request_metric,
    rerank_retrieved_context,
    start_metrics_session,
    start_trace_session,
)


def test_message_content_to_text_handles_mixed_list_content() -> None:
    content = [
        "plain text",
        {"text": "structured text"},
        {"type": "other", "value": "kept as str"},
    ]

    text = _message_content_to_text(content)

    assert "plain text" in text
    assert "structured text" in text


def test_agent_result_to_text_extracts_last_message_content() -> None:
    class DummyMessage:
        def __init__(self, content: object) -> None:
            self.content = content

    result = {"messages": [DummyMessage("first"), DummyMessage("final answer")]}

    assert _agent_result_to_text(result) == "final answer"


def test_trace_session_collects_events() -> None:
    start_trace_session()
    _append_trace_event("router_dispatch", "Router Agent", "dispatch_to=knowledge")

    events = get_trace_events()
    clear_trace_session()

    assert events == [
        {
            "stage": "router_dispatch",
            "actor": "Router Agent",
            "detail": "dispatch_to=knowledge",
        }
    ]


def test_router_prompt_mentions_source_citation_rules() -> None:
    prompt = _build_router_system_prompt()

    assert "来源" in prompt
    assert "consult_knowledge_specialist" in prompt
    assert "consult_filesystem_specialist" in prompt
    assert "consult_web_specialist" in prompt


def test_metrics_session_records_request_metric() -> None:
    start_metrics_session()
    record_request_metric(123.456)

    metrics = get_metrics_snapshot()
    clear_metrics_session()

    assert metrics["request_count"] == 1
    assert metrics["request_latency_ms"] == [123.456]


def test_rerank_tool_updates_metrics(monkeypatch) -> None:
    class DummyPassage:
        def __init__(self, text: str, score: float) -> None:
            self.text = text
            self.score = score

    async def fake_arerank(question: str, passages: list[str]):
        return [DummyPassage(text=passages[0], score=0.9)]

    monkeypatch.setattr(
        "langgraph_lightrag_demo.graph.reranker_service.split_passages",
        lambda text: [text],
    )
    monkeypatch.setattr(
        "langgraph_lightrag_demo.graph.reranker_service.arerank",
        fake_arerank,
    )

    start_metrics_session()
    result = asyncio.run(rerank_retrieved_context.ainvoke({"question": "q", "retrieved_context": "ctx"}))
    metrics = get_metrics_snapshot()
    clear_metrics_session()

    assert "score=0.9000" in result
    assert metrics["tool_call_count"] == 1
    assert metrics["tool_error_count"] == 0
    assert "rerank_retrieved_context" in metrics["tool_latency_ms"]
