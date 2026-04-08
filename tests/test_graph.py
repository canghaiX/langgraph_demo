from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from langgraph_lightrag_demo.graph import (
    _agent_result_to_text,
    _aggregate_specialist_results,
    _append_trace_event,
    _build_execution_plan,
    _build_router_system_prompt,
    _message_content_to_text,
    _response_indicates_insufficient_evidence,
    answer_with_agentic_rag,
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
    assert "执行计划" in prompt
    assert "反思信息缺口" in prompt


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


def test_build_execution_plan_for_mixed_question_prefers_files_then_web() -> None:
    plan = _build_execution_plan("请结合项目代码和最新官网资料解释这个功能")

    assert plan.needs_tools is True
    assert plan.primary_specialist == "filesystem_specialist"
    assert "web_specialist" in plan.fallback_specialists
    assert plan.requires_aggregation is True


def test_response_indicates_insufficient_evidence_detects_common_phrases() -> None:
    assert _response_indicates_insufficient_evidence("当前证据不足，未找到相关内容。")
    assert not _response_indicates_insufficient_evidence("这是基于文件证据的明确结论。")


def test_aggregate_specialist_results_falls_back_without_model(monkeypatch) -> None:
    async def fake_ainvoke(messages):
        raise RuntimeError("model unavailable")

    class DummyModel:
        async def ainvoke(self, messages):
            return await fake_ainvoke(messages)

    monkeypatch.setattr("langgraph_lightrag_demo.graph._build_chat_model", lambda: DummyModel())

    result = asyncio.run(
        _aggregate_specialist_results(
            "问题",
            {
                "knowledge_specialist": "知识库结果",
                "web_specialist": "网页结果",
            },
        )
    )

    assert "综合多个专家结果如下" in result
    assert "知识库结果" in result
    assert "网页结果" in result


def test_answer_with_agentic_rag_retries_with_fallback(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_invoke(name: str, question: str) -> str:
        calls.append(name)
        if name == "knowledge_specialist":
            return "证据不足，未找到相关内容。"
        return "Filesystem specialist found concrete evidence."

    async def fake_aggregate(question: str, results: dict[str, str]) -> str:
        return "\n".join(results.values())

    monkeypatch.setattr("langgraph_lightrag_demo.graph._invoke_specialist_tool", fake_invoke)
    monkeypatch.setattr("langgraph_lightrag_demo.graph._aggregate_specialist_results", fake_aggregate)

    result = asyncio.run(answer_with_agentic_rag([HumanMessage(content="请解释这个项目的文档实现")]))

    assert calls[:2] == ["knowledge_specialist", "filesystem_specialist"]
    assert "Filesystem specialist found concrete evidence." in result
