from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.messages.tool import ToolMessage

from .graph import clear_trace_session, get_agent, get_trace_events, start_trace_session
from .lightrag_client import lightrag_service


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


async def _run_ask(question: str, trace: bool = False) -> None:
    """执行单轮问答。

    适合做快速测试，也适合配合 `--trace` 看 agent 的工具选择。
    """
    agent = await get_agent()
    if trace:
        start_trace_session()
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
    if trace:
        _print_trace(result, get_trace_events())
        clear_trace_session()
    print(result["messages"][-1].content)


async def _run_chat(trace: bool = False) -> None:
    """执行多轮对话。"""
    agent = await get_agent()
    history = []
    print("Interactive chat started. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        if trace:
            start_trace_session()
        result = await agent.ainvoke({"messages": history})
        reply = result["messages"][-1]
        history.append(reply)
        if trace:
            _print_trace(result, get_trace_events())
            clear_trace_session()
        print(f"\nAI> {reply.content}")


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

    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--trace",
        action="store_true",
        help="Print multi-agent trace for each turn",
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
            asyncio.run(_run_ask(args.question, trace=args.trace))
        elif args.command == "chat":
            asyncio.run(_run_chat(trace=getattr(args, "trace", False)))
        else:
            parser.print_help()
    finally:
        try:
            # 退出前尽量优雅关闭 LightRAG 的存储连接。
            asyncio.run(lightrag_service.close())
        except RuntimeError:
            pass
