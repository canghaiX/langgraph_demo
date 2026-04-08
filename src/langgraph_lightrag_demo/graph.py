from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import sys

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from .config import settings
from .lightrag_client import lightrag_service
from .reranker import reranker_service


def _load_skill_text(skill_name: str) -> str:
    """读取项目内 skill，并去掉 YAML frontmatter。

    这里的目的不是做通用 skill 框架，而是把 skill 正文当成“可维护的策略说明”
    动态拼进 agent 的 system prompt。
    """
    skill_path = (
        Path(__file__).resolve().parents[2] / "skills" / skill_name / "SKILL.md"
    )
    text = skill_path.read_text(encoding="utf-8").strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text


def _build_skill_guidance_for(*skill_names: str) -> str:
    """按需拼接 skill 规则，供不同 agent 复用。

    single-agent 架构里，一个总 prompt 往往就够了；
    multi-agent 架构里，我们更希望不同角色只拿自己关心的规则。

    这样做的好处是：
    - Router Agent 更专注于“路由”
    - Knowledge Agent 更专注于知识库检索和 rerank
    - File / Web Agent 更专注于各自的信息源
    """
    sections = []
    for skill_name in skill_names:
        body = _load_skill_text(skill_name)
        sections.append(f"[Skill: {skill_name}]\n{body}")
    return "\n\n".join(sections)


def _build_chat_model() -> ChatOpenAI:
    """统一创建聊天模型实例。

    multi-agent 版本里虽然会创建多个 agent，但它们默认复用同一套模型配置。
    以后如果你想让 Router / Specialist 使用不同模型，这里会是最自然的扩展点。
    """
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        temperature=0,
    )


def _extract_latest_user_text(messages: list[BaseMessage]) -> str:
    """提取消息列表中最后一个用户问题。"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_content_to_text(message.content)
    return ""


def _message_content_to_text(content: Any) -> str:
    """把 LangChain message content 尽量转换为字符串。

    在多数情况下，`message.content` 是普通字符串；
    但当 agent 嵌套调用 agent 时，也可能出现列表或结构化片段。

    这里做统一收敛，是为了让“专家 agent 作为 Router 的 tool”时，
    返回值始终保持简单、稳定、可读。
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def _agent_result_to_text(result: dict) -> str:
    """从一次 agent 调用结果中提取最终文本回复。"""
    messages = result.get("messages", [])
    if not messages:
        return ""
    return _message_content_to_text(messages[-1].content)


def _question_mentions_web(question: str) -> bool:
    keywords = (
        "最新",
        "官网",
        "联网",
        "网页",
        "搜索",
        "新闻",
        "today",
        "latest",
        "official",
        "web",
    )
    lowered = question.lower()
    return any(keyword in lowered for keyword in keywords)


def _question_mentions_filesystem(question: str) -> bool:
    keywords = (
        "代码",
        "源码",
        "文件",
        "配置",
        "readme",
        "项目里",
        "仓库",
        ".py",
        "在哪定义",
        "哪个文件",
    )
    lowered = question.lower()
    return any(keyword in lowered for keyword in keywords)


def _question_needs_knowledge(question: str) -> bool:
    keywords = (
        "知识库",
        "资料",
        "文档",
        "总结",
        "概念",
        "项目",
        "rag",
        "lightrag",
    )
    lowered = question.lower()
    return any(keyword in lowered for keyword in keywords)


def _question_is_simple_chat(question: str) -> bool:
    lowered = question.strip().lower()
    trivial = (
        "你好",
        "hello",
        "hi",
        "谢谢",
        "thank you",
        "你是谁",
    )
    return any(lowered == item or lowered.startswith(item) for item in trivial)


def _build_execution_plan(question: str) -> ExecutionPlan:
    """按问题特征构建显式执行计划。"""
    if _question_is_simple_chat(question):
        return ExecutionPlan(
            needs_tools=False,
            primary_specialist=None,
            fallback_specialists=(),
            requires_aggregation=False,
            reasoning="问题更像闲聊或无需检索的直接问答，先直接回答。",
        )

    mentions_web = _question_mentions_web(question)
    mentions_files = _question_mentions_filesystem(question)
    needs_knowledge = _question_needs_knowledge(question)
    asks_compare = any(
        token in question for token in ("同时", "结合", "综合", "对比", "比较")
    )

    if mentions_web and mentions_files:
        return ExecutionPlan(
            needs_tools=True,
            primary_specialist="filesystem_specialist",
            fallback_specialists=("web_specialist", "knowledge_specialist"),
            requires_aggregation=True,
            reasoning="问题同时涉及本地项目与最新网页信息，先查工作区，再补网页与知识库证据。",
        )

    if mentions_web:
        return ExecutionPlan(
            needs_tools=True,
            primary_specialist="web_specialist",
            fallback_specialists=("knowledge_specialist",),
            requires_aggregation=asks_compare,
            reasoning="问题明显依赖公开互联网或最新信息，优先使用 Web Specialist。",
        )

    if mentions_files:
        return ExecutionPlan(
            needs_tools=True,
            primary_specialist="filesystem_specialist",
            fallback_specialists=(
                ("knowledge_specialist", "web_specialist")
                if asks_compare
                else ("knowledge_specialist",)
            ),
            requires_aggregation=asks_compare,
            reasoning="问题聚焦当前工作区代码与文档，优先使用 Filesystem Specialist。",
        )

    if needs_knowledge:
        return ExecutionPlan(
            needs_tools=True,
            primary_specialist="knowledge_specialist",
            fallback_specialists=("filesystem_specialist", "web_specialist"),
            requires_aggregation=asks_compare,
            reasoning="问题更像知识库/项目资料问答，优先使用 Knowledge Specialist，并在证据不足时补其他来源。",
        )

    return ExecutionPlan(
        needs_tools=False,
        primary_specialist=None,
        fallback_specialists=(),
        requires_aggregation=False,
        reasoning="问题没有明显工具依赖，先尝试直接回答。",
    )


def _response_indicates_insufficient_evidence(text: str) -> bool:
    """判断专家结果是否显示证据不足。"""
    lowered = text.lower()
    return any(pattern in lowered for pattern in _INSUFFICIENT_EVIDENCE_PATTERNS)


# 下面这些对象都是“懒加载缓存”。
#
# 对 multi-agent 系统来说，这样做尤其重要：
# - Router Agent 会频繁复用
# - Specialist Agent 的 prompt 固定，没必要每次重新构建
# - MCP server/client 启动成本不低，适合按需启动后缓存
_router_agent = None
_knowledge_agent = None
_file_agent = None
_web_agent = None
_filesystem_mcp_client = None
_web_mcp_client = None
_trace_events_var: ContextVar[list[dict[str, str]] | None] = ContextVar(
    "graph_trace_events",
    default=None,
)
_metrics_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "graph_metrics",
    default=None,
)
_SPECIALIST_LABELS = {
    "knowledge_specialist": "Knowledge Specialist",
    "filesystem_specialist": "Filesystem Specialist",
    "web_specialist": "Web Specialist",
}
_INSUFFICIENT_EVIDENCE_PATTERNS = (
    "未找到",
    "没有足够",
    "证据不足",
    "未提供",
    "无法确认",
    "未检索到",
    "未找到相关",
    "insufficient",
    "not enough",
    "not found",
    "unavailable",
)


@dataclass(frozen=True)
class ExecutionPlan:
    """描述一次问题处理的显式执行计划。"""

    needs_tools: bool
    primary_specialist: str | None
    fallback_specialists: tuple[str, ...]
    requires_aggregation: bool
    reasoning: str


def _new_metrics_state() -> dict[str, Any]:
    """创建一次请求级性能指标容器。"""
    return {
        "request_count": 0,
        "request_latency_ms": [],
        "router_dispatch_count": 0,
        "router_dispatch_by_specialist": {},
        "specialist_call_count": 0,
        "specialist_error_count": 0,
        "specialist_latency_ms": {},
        "tool_call_count": 0,
        "tool_error_count": 0,
        "tool_latency_ms": {},
    }


def start_metrics_session() -> None:
    """开始一次新的性能指标采集会话。"""
    _metrics_var.set(_new_metrics_state())


def get_metrics_snapshot() -> dict[str, Any]:
    """获取当前性能指标快照。"""
    state = _metrics_var.get()
    if state is None:
        return {}
    return deepcopy(state)


def clear_metrics_session() -> None:
    """清空当前性能指标采集会话。"""
    _metrics_var.set(None)


def _increment_metric(name: str, amount: int = 1) -> None:
    """递增整型计数指标。"""
    state = _metrics_var.get()
    if state is None:
        return
    state[name] = int(state.get(name, 0)) + amount


def _increment_named_metric(bucket: str, name: str, amount: int = 1) -> None:
    """递增按名称分组的计数指标。"""
    state = _metrics_var.get()
    if state is None:
        return
    values = state.setdefault(bucket, {})
    values[name] = int(values.get(name, 0)) + amount


def _record_duration_metric(bucket: str, name: str, elapsed_ms: float) -> None:
    """记录按名称分组的耗时指标。"""
    state = _metrics_var.get()
    if state is None:
        return
    durations = state.setdefault(bucket, {})
    durations[name] = round(float(durations.get(name, 0.0)) + elapsed_ms, 3)


def record_request_metric(elapsed_ms: float) -> None:
    """记录一次请求的总耗时。"""
    state = _metrics_var.get()
    if state is None:
        return
    state["request_count"] = int(state.get("request_count", 0)) + 1
    state.setdefault("request_latency_ms", []).append(round(float(elapsed_ms), 3))


def _build_source_citation_rules() -> str:
    """返回统一的“来源标注”规则。

    这条规则会同时下发给 Router 和各个专家 agent。
    我们希望项目回答不只是“像是对的”，还要尽量说清楚依据来自哪里。
    这对简历展示、面试讲解和实际调试都很有帮助。
    """
    return (
        "回答规则补充：\n"
        "1. 如果回答依赖工具证据，请在答案末尾添加“来源”小节。\n"
        "2. 知识库回答优先引用 `[Source File]` 和 `[PDF Page N]` 标记。\n"
        "3. 本地文件回答优先引用文件路径。\n"
        "4. 网页回答优先引用 URL 或页面标题。\n"
        "5. 如果答案没有使用任何工具证据，不要伪造来源。"
    )


def start_trace_session() -> None:
    """开始一次新的 trace 会话。

    CLI 在 `ask --trace` 或 `chat --trace` 时会先调用这里。
    后续 Router、专家 agent 和底层工具会把关键事件写进当前上下文。
    """
    _trace_events_var.set([])


def get_trace_events() -> list[dict[str, str]]:
    """获取当前 trace 会话中的事件列表。"""
    return list(_trace_events_var.get() or [])


def clear_trace_session() -> None:
    """清空当前 trace 会话。"""
    _trace_events_var.set(None)


def _preview_trace_text(text: str, max_chars: int = 240) -> str:
    """把 trace 中的长文本压成短预览，避免日志过长。"""
    compact = " ".join(str(text).replace("\r\n", "\n").split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."


def _append_trace_event(stage: str, actor: str, detail: str) -> None:
    """向当前 trace 会话追加一条事件。

    这里使用 `ContextVar` 而不是普通全局列表，是为了让一次请求内的事件
    尽量跟随当前调用上下文，而不是和其他请求混在一起。
    """
    events = _trace_events_var.get()
    if events is None:
        return
    events.append({"stage": stage, "actor": actor, "detail": detail})


async def _direct_answer(messages: list[BaseMessage]) -> str:
    """直接使用聊天模型回答无需工具的问题。"""
    _append_trace_event("router_direct_answer", "Router Agent", "plan=direct_answer")
    result = await _build_chat_model().ainvoke(messages)
    return _message_content_to_text(result.content)


async def _load_web_mcp_tools():
    """只加载网页专家所需的 MCP tools。

    进入 multi-agent 后，我们不再让一个 agent 同时看到所有工具。
    Web Agent 应该只看到网页相关工具，避免工具集过大导致误调用。
    """
    global _web_mcp_client

    if not settings.enable_mcp_web_search:
        return []

    if _web_mcp_client is None:
        _web_mcp_client = MultiServerMCPClient(
            {
                "web-search": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "langgraph_lightrag_demo.mcp_servers.web_search_server",
                    ],
                    "env": {
                        "PYTHONIOENCODING": "utf-8",
                        "MCP_WEB_SEARCH_MAX_RESULTS": str(
                            settings.mcp_web_search_max_results
                        ),
                    },
                }
            }
        )

    return await _web_mcp_client.get_tools()


async def _load_filesystem_mcp_tools():
    """只加载文件专家所需的 MCP tools。"""
    global _filesystem_mcp_client

    if not settings.enable_mcp_filesystem:
        return []

    if _filesystem_mcp_client is None:
        _filesystem_mcp_client = MultiServerMCPClient(
            {
                "filesystem": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "langgraph_lightrag_demo.mcp_servers.filesystem_server",
                    ],
                    "env": {
                        "PYTHONIOENCODING": "utf-8",
                        "MCP_FILESYSTEM_ROOT": str(
                            Path(settings.mcp_filesystem_root).resolve()
                        ),
                    },
                }
            }
        )

    return await _filesystem_mcp_client.get_tools()


def _build_router_system_prompt() -> str:
    """构建 Router Agent 的提示词。

    Router 的职责是：
    - 先判断问题是否需要工具
    - 如果需要，再决定交给哪一个专家 agent
    - 尽量避免“保险式全都查一遍”

    所以 Router 更像调度中心，而不是直接干活的专家。
    """
    return (
        f"{settings.system_prompt}\n\n"
        "你是整个 multi-agent 系统的 Router Agent。\n"
        "你的主要职责是先判断问题属于哪一类，再把问题转交给最合适的专家 agent。\n\n"
        "你当前可以调用的专家只有三个：\n"
        "- `consult_knowledge_specialist`：处理已入库知识库、私有资料、项目资料问答\n"
        "- `consult_filesystem_specialist`：处理当前工作区源码、配置、README、本地文件检索\n"
        "- `consult_web_specialist`：处理公开网页、官网、联网搜索、最新网页内容\n\n"
        "Router 决策规则：\n"
        "1. 如果问题不需要工具，直接回答。\n"
        "2. 先显式形成一个简短执行计划：要不要用工具、先问谁、是否需要备选专家。\n"
        "3. 如果问题明显依赖知识库或已入库资料，优先交给知识库专家。\n"
        "4. 如果问题明显依赖当前工作区文件，优先交给文件专家。\n"
        "5. 如果问题依赖公开网页或最新互联网信息，优先交给网页专家。\n"
        "6. 不要为了保险一次性调用所有专家。先选最可能正确的那个；只有证据不足时再考虑第二个专家。\n"
        "7. 如果首个专家证据不足，要反思信息缺口，再决定是否重试或切换专家。\n"
        "8. 如果综合了多个专家的结果，请明确区分不同来源并给出统一结论。\n\n"
        "请遵循以下项目内路由规则：\n\n"
        f"{_build_skill_guidance_for('knowledge-base-rag', 'mcp-routing')}\n\n"
        f"{_build_source_citation_rules()}"
    )


def _build_knowledge_agent_system_prompt() -> str:
    """构建 Knowledge Specialist Agent 的提示词。"""
    return (
        f"{settings.system_prompt}\n\n"
        "你是 Knowledge Specialist Agent，只负责知识库问答。\n"
        "你当前能使用的底层工具有：\n"
        "- `search_knowledge_base`：检索 LightRAG 知识库\n"
        "- `rerank_retrieved_context`：对召回上下文进行相关性重排\n\n"
        "你的职责边界：\n"
        "1. 优先处理依赖已入库知识、私有资料、项目文档的问题。\n"
        "2. 如果问题不依赖知识库，也可以直接做简洁回答，但不要伪装成“来自知识库”。\n"
        "3. 如果检索结果太长、太杂、主题混乱，再使用 rerank。\n"
        "4. 如果知识库证据不足，要明确说明。\n\n"
        "请遵循以下项目内规则：\n\n"
        f"{_build_skill_guidance_for('knowledge-base-rag', 'retrieval-rerank')}\n\n"
        f"{_build_source_citation_rules()}"
    )


def _build_file_agent_system_prompt() -> str:
    """构建 Filesystem Specialist Agent 的提示词。"""
    return (
        f"{settings.system_prompt}\n\n"
        "你是 Filesystem Specialist Agent，只负责分析当前工作区内的源码、配置、文档和本地文本文件。\n"
        "当用户在问“某个配置在哪”“项目里哪里定义了什么”“README 怎么写的”这类问题时，应优先使用本地文件工具。\n"
        "请尽量基于实际文件证据回答，而不是凭印象猜测。\n"
        "如果当前工作区里没有足够证据，要明确说明未找到。\n\n"
        "请遵循以下项目内规则：\n\n"
        f"{_build_skill_guidance_for('mcp-routing')}\n\n"
        f"{_build_source_citation_rules()}"
    )


def _build_web_agent_system_prompt() -> str:
    """构建 Web Specialist Agent 的提示词。"""
    return (
        f"{settings.system_prompt}\n\n"
        "你是 Web Specialist Agent，只负责公开网页、官网、联网搜索和页面抓取。\n"
        "当问题依赖公开互联网内容、在线文档或最新网页信息时，应优先使用网页工具。\n"
        "如果网页结果不足以支撑结论，要明确说明。\n\n"
        "请遵循以下项目内规则：\n\n"
        f"{_build_skill_guidance_for('mcp-routing')}\n\n"
        f"{_build_source_citation_rules()}"
    )


async def _get_knowledge_agent():
    """懒加载创建 Knowledge Specialist Agent。"""
    global _knowledge_agent
    if _knowledge_agent is not None:
        return _knowledge_agent

    _knowledge_agent = create_agent(
        model=_build_chat_model(),
        tools=[search_knowledge_base, rerank_retrieved_context],
        system_prompt=_build_knowledge_agent_system_prompt(),
    )
    return _knowledge_agent


async def _get_file_agent():
    """懒加载创建 Filesystem Specialist Agent。"""
    global _file_agent
    if _file_agent is not None:
        return _file_agent

    _file_agent = create_agent(
        model=_build_chat_model(),
        tools=await _load_filesystem_mcp_tools(),
        system_prompt=_build_file_agent_system_prompt(),
    )
    return _file_agent


async def _get_web_agent():
    """懒加载创建 Web Specialist Agent。"""
    global _web_agent
    if _web_agent is not None:
        return _web_agent

    _web_agent = create_agent(
        model=_build_chat_model(),
        tools=await _load_web_mcp_tools(),
        system_prompt=_build_web_agent_system_prompt(),
    )
    return _web_agent


async def _ask_specialist(agent_name: str, agent, question: str) -> str:
    """统一调用某个专家 agent，并抽取最终文本。

    这个小函数的价值在于：
    - Router 调 Knowledge/File/Web 三个专家时写法统一
    - 专家 agent 返回的复杂 message 结构被统一收敛成字符串
    - 以后如果你要在专家调用前后加日志，这里也是最好的切入点
    """
    _append_trace_event(
        "specialist_call",
        agent_name,
        f"question={_preview_trace_text(question)}",
    )
    _increment_metric("specialist_call_count")
    started_at = time.perf_counter()
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
        reply = _agent_result_to_text(result)
        _append_trace_event(
            "specialist_result",
            agent_name,
            _preview_trace_text(reply),
        )
        _record_duration_metric(
            "specialist_latency_ms",
            agent_name,
            (time.perf_counter() - started_at) * 1000,
        )
        return reply
    except Exception as exc:
        message = f"{agent_name} failed: {exc}"
        _increment_metric("specialist_error_count")
        _record_duration_metric(
            "specialist_latency_ms",
            agent_name,
            (time.perf_counter() - started_at) * 1000,
        )
        _append_trace_event("specialist_error", agent_name, message)
        return message


@tool
async def search_knowledge_base(question: str) -> str:
    """检索 LightRAG 知识库。

    这是“项目内私有知识”的主要入口。
    在 multi-agent 版本里，它属于 Knowledge Specialist 的底层能力，而不直接暴露给 Router。
    """
    _append_trace_event(
        "tool_call",
        "search_knowledge_base",
        f"question={_preview_trace_text(question)}",
    )
    _increment_metric("tool_call_count")
    started_at = time.perf_counter()
    try:
        context = await lightrag_service.query_context(question)
        _append_trace_event(
            "tool_result",
            "search_knowledge_base",
            f"context_chars={len(context)} preview={_preview_trace_text(context)}",
        )
        return context
    except Exception:
        _increment_metric("tool_error_count")
        raise
    finally:
        _record_duration_metric(
            "tool_latency_ms",
            "search_knowledge_base",
            (time.perf_counter() - started_at) * 1000,
        )


@tool
async def rerank_retrieved_context(question: str, retrieved_context: str) -> str:
    """对已召回上下文做重排，返回更相关的片段。

    这个工具不负责召回，只负责“把已经召回的内容再筛一遍”。
    在 multi-agent 版本里，它也是 Knowledge Specialist 的内部工具。
    """
    _append_trace_event(
        "tool_call",
        "rerank_retrieved_context",
        (
            f"question={_preview_trace_text(question)} "
            f"retrieved_context_chars={len(retrieved_context)}"
        ),
    )
    _increment_metric("tool_call_count")
    started_at = time.perf_counter()
    try:
        passages = reranker_service.split_passages(retrieved_context)
        ranked = await reranker_service.arerank(question, passages)
        if not ranked:
            _append_trace_event(
                "tool_result",
                "rerank_retrieved_context",
                "no_passages_after_rerank",
            )
            return "No passages were available to rerank."

        lines = [
            f"[{index}] score={item.score:.4f}\n{item.text}"
            for index, item in enumerate(ranked, start=1)
        ]
        result = "\n\n".join(lines)
        _append_trace_event(
            "tool_result",
            "rerank_retrieved_context",
            f"ranked_passages={len(ranked)} preview={_preview_trace_text(result)}",
        )
        return result
    except Exception:
        _increment_metric("tool_error_count")
        raise
    finally:
        _record_duration_metric(
            "tool_latency_ms",
            "rerank_retrieved_context",
            (time.perf_counter() - started_at) * 1000,
        )


@tool
async def consult_knowledge_specialist(question: str) -> str:
    """把问题转交给知识库专家 Agent。

    对 Router 来说，这个工具相当于“去问知识库专家”，而不是自己直接接触 LightRAG。
    这就是 multi-agent 的关键差别：上层只管委托，下层才管具体执行。
    """
    _append_trace_event(
        "router_dispatch",
        "Router Agent",
        "dispatch_to=knowledge_specialist",
    )
    _increment_metric("router_dispatch_count")
    _increment_named_metric(
        "router_dispatch_by_specialist", "knowledge_specialist"
    )
    agent = await _get_knowledge_agent()
    return await _ask_specialist("knowledge_specialist", agent, question)


@tool
async def consult_filesystem_specialist(question: str) -> str:
    """把问题转交给文件专家 Agent。"""
    if not settings.enable_mcp_filesystem:
        message = (
            "Filesystem specialist is unavailable because ENABLE_MCP_FILESYSTEM=false."
        )
        _append_trace_event("specialist_unavailable", "filesystem_specialist", message)
        return message

    _append_trace_event(
        "router_dispatch",
        "Router Agent",
        "dispatch_to=filesystem_specialist",
    )
    _increment_metric("router_dispatch_count")
    _increment_named_metric(
        "router_dispatch_by_specialist", "filesystem_specialist"
    )
    agent = await _get_file_agent()
    return await _ask_specialist("filesystem_specialist", agent, question)


@tool
async def consult_web_specialist(question: str) -> str:
    """把问题转交给网页专家 Agent。"""
    if not settings.enable_mcp_web_search:
        message = "Web specialist is unavailable because ENABLE_MCP_WEB_SEARCH=false."
        _append_trace_event("specialist_unavailable", "web_specialist", message)
        return message

    _append_trace_event(
        "router_dispatch",
        "Router Agent",
        "dispatch_to=web_specialist",
    )
    _increment_metric("router_dispatch_count")
    _increment_named_metric("router_dispatch_by_specialist", "web_specialist")
    agent = await _get_web_agent()
    return await _ask_specialist("web_specialist", agent, question)


async def _invoke_specialist_tool(specialist_name: str, question: str) -> str:
    """通过专家 tool 调用对应 specialist。"""
    if specialist_name == "knowledge_specialist":
        return await consult_knowledge_specialist.ainvoke({"question": question})
    if specialist_name == "filesystem_specialist":
        return await consult_filesystem_specialist.ainvoke({"question": question})
    if specialist_name == "web_specialist":
        return await consult_web_specialist.ainvoke({"question": question})
    raise ValueError(f"Unknown specialist: {specialist_name}")


async def _aggregate_specialist_results(
    question: str,
    results: dict[str, str],
) -> str:
    """汇总多专家结果，生成最终回答。"""
    sections = []
    for name, content in results.items():
        label = _SPECIALIST_LABELS.get(name, name)
        sections.append(f"[{label}]\n{content}")

    prompt = (
        "你是多专家结果汇总器。请基于不同专家返回的证据生成最终回答。\n"
        "要求：\n"
        "1. 优先保留被多个来源共同支持的结论。\n"
        "2. 如果不同来源结论不一致，要明确指出分歧。\n"
        "3. 如果某个专家结果明显表示证据不足，不要把它包装成确定事实。\n"
        "4. 在答案末尾添加“来源”小节，区分不同专家对应的来源线索。"
    )
    joined_sections = "\n\n".join(sections)
    human_content = (
        f"[用户问题]\n{question}\n\n"
        f"[专家结果]\n{joined_sections}\n\n"
        "请输出最终汇总答案："
    )
    _append_trace_event(
        "aggregation",
        "Router Agent",
        f"specialists={','.join(results.keys())}",
    )
    try:
        result = await _build_chat_model().ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=human_content),
            ]
        )
        return _message_content_to_text(result.content)
    except Exception:
        merged = ["综合多个专家结果如下：", *sections]
        return "\n\n".join(merged)


async def answer_with_agentic_rag(messages: list[BaseMessage]) -> str:
    """显式执行计划、失败反思重试和多专家聚合的主入口。"""
    question = _extract_latest_user_text(messages)
    if not question:
        return await _direct_answer(messages)

    plan = _build_execution_plan(question)
    _append_trace_event(
        "router_plan",
        "Router Agent",
        (
            f"needs_tools={plan.needs_tools} "
            f"primary={plan.primary_specialist or 'none'} "
            f"fallbacks={','.join(plan.fallback_specialists) or 'none'} "
            f"aggregate={plan.requires_aggregation} "
            f"reasoning={plan.reasoning}"
        ),
    )

    if not plan.needs_tools or plan.primary_specialist is None:
        return await _direct_answer(messages)

    specialist_results: dict[str, str] = {}
    primary_result = await _invoke_specialist_tool(plan.primary_specialist, question)
    specialist_results[plan.primary_specialist] = primary_result

    should_retry = _response_indicates_insufficient_evidence(primary_result)
    if should_retry:
        _append_trace_event(
            "reflection_retry",
            "Router Agent",
            (
                f"primary={plan.primary_specialist} "
                f"reason=insufficient_evidence "
                f"fallbacks={','.join(plan.fallback_specialists) or 'none'}"
            ),
        )

    for specialist_name in plan.fallback_specialists:
        if not should_retry and not plan.requires_aggregation:
            break

        result = await _invoke_specialist_tool(specialist_name, question)
        specialist_results[specialist_name] = result

        if should_retry and not _response_indicates_insufficient_evidence(result):
            should_retry = False
            if not plan.requires_aggregation:
                break

    if len(specialist_results) == 1 and not plan.requires_aggregation:
        return next(iter(specialist_results.values()))

    return await _aggregate_specialist_results(question, specialist_results)


async def get_agent():
    """懒加载创建 Router Agent。

    这里返回的不再是 single-agent 版本中的“全能代理”，
    而是整个 multi-agent 系统的总入口。

    当前架构分成三层：
    1. 底层工具层：LightRAG / reranker / MCP tools
    2. 专家 agent 层：Knowledge / File / Web Specialists
    3. Router 层：直接服务用户，负责路由和汇总

    对 CLI 而言，调用方式完全不变；
    对系统内部而言，已经从“一个 agent 调所有工具”升级成“一个 Router 调多个专家 agent”。
    """
    global _router_agent
    if _router_agent is not None:
        return _router_agent

    settings.validate()
    _router_agent = create_agent(
        model=_build_chat_model(),
        tools=[
            consult_knowledge_specialist,
            consult_filesystem_specialist,
            consult_web_specialist,
        ],
        system_prompt=_build_router_system_prompt(),
    )
    return _router_agent
