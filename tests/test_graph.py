from __future__ import annotations

from langgraph_lightrag_demo.graph import (
    _agent_result_to_text,
    _append_trace_event,
    _build_router_system_prompt,
    _message_content_to_text,
    clear_trace_session,
    get_trace_events,
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
