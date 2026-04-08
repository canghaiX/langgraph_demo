from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage

from .config import settings
from .graph import (
    answer_with_agentic_rag,
    clear_metrics_session,
    clear_trace_session,
    get_metrics_snapshot,
    get_trace_events,
    record_request_metric,
    start_metrics_session,
    start_trace_session,
)
from .lightrag_client import lightrag_service
from .memory import memory_manager


# 这里定义“CLI 允许挑出来做入库”的文件类型。
# 之所以把后缀过滤放在这一层，而不是放进 LightRAG 里：
# - 用户在命令行导入目录时，可以尽早知道哪些文件会被处理
# - 不支持的后缀会在扫描目录阶段被直接跳过
# - 真正的文本提取细节仍由 lightrag_client.py 负责
#
# 当前支持：
# - .txt / .md：直接按 UTF-8 文本读取
# - .pdf：优先尝试直接抽文本；如果是扫描版或抽取结果太少，则自动回退到 OCR
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _collect_files(target: Path) -> list[Path]:
    """从文件或目录里收集需要入库的文件。"""
    if target.is_file():
        return [target]
    files = [
        path
        for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def _print_trace(result: dict, trace_events: list[dict] | None = None) -> None:
    """打印一次 agent 调用中的工具轨迹。

    multi-agent 版本下，单纯看 LangChain 最外层 message 里的 tool_calls 已经不够了，
    因为：
    - Router 会调用专家 agent
    - 专家 agent 内部还会继续调底层工具

    所以这里优先打印 graph.py 里收集的“跨层级 trace 事件”。
    如果当前没有开启 graph trace，再退回到旧的 message-level tool trace。
    """
    if trace_events:
        print("[trace] Multi-agent 调用轨迹:")
        for item in trace_events:
            stage = item.get("stage", "unknown_stage")
            actor = item.get("actor", "unknown_actor")
            detail = item.get("detail", "")
            print(f"- {stage}: {actor} {detail}".rstrip())
        return

    messages = result.get("messages", [])
    tool_events = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name", "unknown_tool")
            args = call.get("args", {})
            tool_events.append(("call", name, args))

        if isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None) or getattr(
                message, "tool_call_id", "tool_result"
            )
            tool_events.append(("result", tool_name, str(message.content)))

    if not tool_events:
        print("[trace] 未捕获到工具调用。")
        return

    print("[trace] 工具调用轨迹:")
    for event_type, name, payload in tool_events:
        if event_type == "call":
            print(f"- call: {name} args={payload}")
        else:
            text = payload.strip().replace("\r\n", "\n")
            preview = text[:300] + ("..." if len(text) > 300 else "")
            print(f"- result: {name}\n{preview}")


def _format_average(values: list[float]) -> str:
    """格式化一组耗时数据的平均值。"""
    if not values:
        return "0.000"
    return f"{sum(values) / len(values):.3f}"


def _print_metrics(metrics: dict) -> None:
    """打印一次请求级性能指标摘要。"""
    if not metrics:
        print("[metrics] 当前没有采集到指标。")
        return

    request_latencies = metrics.get("request_latency_ms", [])
    print("[metrics] 性能指标摘要:")
    print(f"- request_count: {metrics.get('request_count', 0)}")
    print(f"- avg_request_latency_ms: {_format_average(request_latencies)}")
    print(f"- router_dispatch_count: {metrics.get('router_dispatch_count', 0)}")
    print(f"- specialist_call_count: {metrics.get('specialist_call_count', 0)}")
    print(f"- specialist_error_count: {metrics.get('specialist_error_count', 0)}")
    print(f"- tool_call_count: {metrics.get('tool_call_count', 0)}")
    print(f"- tool_error_count: {metrics.get('tool_error_count', 0)}")

    dispatches = metrics.get("router_dispatch_by_specialist", {})
    if dispatches:
        print("- router_dispatch_by_specialist:")
        for name, count in sorted(dispatches.items()):
            print(f"  - {name}: {count}")

    specialist_latencies = metrics.get("specialist_latency_ms", {})
    if specialist_latencies:
        print("- specialist_latency_ms:")
        for name, elapsed in sorted(specialist_latencies.items()):
            print(f"  - {name}: {elapsed:.3f}")

    tool_latencies = metrics.get("tool_latency_ms", {})
    if tool_latencies:
        print("- tool_latency_ms:")
        for name, elapsed in sorted(tool_latencies.items()):
            print(f"  - {name}: {elapsed:.3f}")


def _build_agent_messages(user_id: str, history: list) -> list:
    """把偏好记忆和会话摘要注入当前请求。"""
    context_message = memory_manager.build_runtime_context_message(user_id)
    if context_message is None:
        return list(history)
    return [context_message, *history]


async def _run_ingest(path_str: str) -> None:
    """执行知识库导入。"""
    target = Path(path_str)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    files = _collect_files(target)
    if not files:
        raise FileNotFoundError(
            f"No supported files found under {target}. Supported types: {sorted(SUPPORTED_SUFFIXES)}"
        )

    inserted = await lightrag_service.ingest_files(files)
    print(f"Indexed {len(inserted)} file(s) into LightRAG:")
    for item in inserted:
        print(f"- {item}")


async def _run_ask(
    question: str,
    trace: bool = False,
    metrics: bool = False,
    user_id: str = settings.default_user_id,
) -> None:
    """执行单轮问答。

    适合做快速测试，也适合配合 `--trace` 看 agent 的工具选择。
    """
    memory_manager.infer_and_update_preferences(user_id, question)
    if metrics:
        start_metrics_session()
    if trace:
        start_trace_session()
    started_at = time.perf_counter()
    request_messages = _build_agent_messages(
        user_id,
        [HumanMessage(content=question)],
    )
    reply_text = await answer_with_agentic_rag(request_messages)
    if metrics:
        record_request_metric((time.perf_counter() - started_at) * 1000)
    if trace:
        _print_trace({}, get_trace_events())
        clear_trace_session()
    print(reply_text)
    await memory_manager.remember_exchange(user_id, question, reply_text)
    if metrics:
        _print_metrics(get_metrics_snapshot())
        clear_metrics_session()


async def _run_chat(
    trace: bool = False,
    metrics: bool = False,
    user_id: str = settings.default_user_id,
) -> None:
    """执行多轮对话。"""
    history = []
    print("Interactive chat started. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        memory_manager.infer_and_update_preferences(user_id, user_input)
        history.append(HumanMessage(content=user_input))
        if metrics:
            start_metrics_session()
        if trace:
            start_trace_session()
        started_at = time.perf_counter()
        request_messages = _build_agent_messages(user_id, history)
        reply_text = await answer_with_agentic_rag(request_messages)
        if metrics:
            record_request_metric((time.perf_counter() - started_at) * 1000)
        history.append(AIMessage(content=reply_text))
        history = await memory_manager.compact_history(user_id, history)
        if trace:
            _print_trace({}, get_trace_events())
            clear_trace_session()
        print(f"\nAI> {reply_text}")
        if metrics:
            _print_metrics(get_metrics_snapshot())
            clear_metrics_session()


def _run_show_preferences(user_id: str) -> None:
    """打印指定用户的偏好记忆。"""
    preferences = memory_manager.get_preferences(user_id)
    print(f"User ID: {user_id}")
    print(f"- language: {preferences.language}")
    print(f"- response_style: {preferences.response_style}")
    print(f"- code_preference: {preferences.code_preference}")
    print(f"- career_focus: {preferences.career_focus}")


def _run_set_preferences(
    user_id: str,
    language: str | None,
    response_style: str | None,
    code_preference: str | None,
    career_focus: str | None,
) -> None:
    """更新用户偏好记忆。"""
    updated = memory_manager.update_preferences(
        user_id,
        language=language,
        response_style=response_style,
        code_preference=code_preference,
        career_focus=career_focus,
    )
    print(f"Updated preferences for {user_id}:")
    print(f"- language: {updated.language}")
    print(f"- response_style: {updated.response_style}")
    print(f"- code_preference: {updated.code_preference}")
    print(f"- career_focus: {updated.career_focus}")


def _run_show_summary(user_id: str) -> None:
    """打印指定用户的会话摘要记忆。"""
    summary = memory_manager.get_summary(user_id)
    print(f"User ID: {user_id}")
    print(f"- summarized_message_count: {summary.summarized_message_count}")
    print(summary.summary or "(empty)")


def _run_clear_summary(user_id: str) -> None:
    """清空指定用户的会话摘要记忆。"""
    memory_manager.clear_summary(user_id)
    print(f"Cleared session summary for {user_id}.")


def build_parser() -> argparse.ArgumentParser:
    """定义 CLI 命令结构。"""
    parser = argparse.ArgumentParser(description="LangGraph + LightRAG demo CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest files into LightRAG")
    ingest_parser.add_argument(
        "--path",
        default="data/knowledge",
        help="File or directory to ingest",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--trace",
        action="store_true",
        help="Print tool-call trace for debugging",
    )
    ask_parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print request latency and tool/specialist metrics",
    )
    ask_parser.add_argument(
        "--user-id",
        default=settings.default_user_id,
        help="User id used for preference memory and session summary memory",
    )

    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--trace",
        action="store_true",
        help="Print multi-agent trace for each turn",
    )
    chat_parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print request latency and tool/specialist metrics for each turn",
    )
    chat_parser.add_argument(
        "--user-id",
        default=settings.default_user_id,
        help="User id used for preference memory and session summary memory",
    )

    show_pref_parser = subparsers.add_parser(
        "show-pref", help="Show stored user preferences"
    )
    show_pref_parser.add_argument(
        "--user-id",
        default=settings.default_user_id,
        help="User id to inspect",
    )

    set_pref_parser = subparsers.add_parser(
        "set-pref", help="Update stored user preferences"
    )
    set_pref_parser.add_argument(
        "--user-id",
        default=settings.default_user_id,
        help="User id to update",
    )
    set_pref_parser.add_argument(
        "--language",
        choices=["zh", "en"],
        help="Preferred response language",
    )
    set_pref_parser.add_argument(
        "--response-style",
        choices=["concise", "detailed"],
        help="Preferred response style",
    )
    set_pref_parser.add_argument(
        "--code-preference",
        choices=["prefer_examples", "explain_only"],
        help="Preferred coding explanation style",
    )
    set_pref_parser.add_argument(
        "--career-focus",
        choices=["general", "internship", "research"],
        help="Preferred career framing",
    )

    show_summary_parser = subparsers.add_parser(
        "show-summary", help="Show stored session summary memory"
    )
    show_summary_parser.add_argument(
        "--user-id",
        default=settings.default_user_id,
        help="User id to inspect",
    )

    clear_summary_parser = subparsers.add_parser(
        "clear-summary", help="Clear stored session summary memory"
    )
    clear_summary_parser.add_argument(
        "--user-id",
        default=settings.default_user_id,
        help="User id to clear",
    )
    return parser


def main() -> None:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            asyncio.run(_run_ingest(args.path))
        elif args.command == "ask":
            asyncio.run(
                _run_ask(
                    args.question,
                    trace=args.trace,
                    metrics=getattr(args, "metrics", False),
                    user_id=getattr(args, "user_id", settings.default_user_id),
                )
            )
        elif args.command == "chat":
            asyncio.run(
                _run_chat(
                    trace=getattr(args, "trace", False),
                    metrics=getattr(args, "metrics", False),
                    user_id=getattr(args, "user_id", settings.default_user_id),
                )
            )
        elif args.command == "show-pref":
            _run_show_preferences(args.user_id)
        elif args.command == "set-pref":
            _run_set_preferences(
                args.user_id,
                language=getattr(args, "language", None),
                response_style=getattr(args, "response_style", None),
                code_preference=getattr(args, "code_preference", None),
                career_focus=getattr(args, "career_focus", None),
            )
        elif args.command == "show-summary":
            _run_show_summary(args.user_id)
        elif args.command == "clear-summary":
            _run_clear_summary(args.user_id)
        else:
            parser.print_help()
    finally:
        try:
            # 退出前尽量优雅关闭 LightRAG 的存储连接。
            asyncio.run(lightrag_service.close())
        except RuntimeError:
            pass
