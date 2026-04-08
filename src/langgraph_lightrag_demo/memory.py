from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import settings


@dataclass
class UserPreferences:
    """描述用户长期偏好的轻量结构。"""

    language: str = "zh"
    response_style: str = "concise"
    code_preference: str = "prefer_examples"
    career_focus: str = "general"


@dataclass
class SessionSummary:
    """记录会话摘要记忆。"""

    summary: str = ""
    summarized_message_count: int = 0


class UserPreferenceStore:
    """基于 JSON 的用户偏好存储。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, user_id: str) -> UserPreferences:
        payload = self._read_all().get(user_id, {})
        return UserPreferences(**payload)

    def update(self, user_id: str, **updates: str) -> UserPreferences:
        current = self.get(user_id)
        merged = asdict(current)
        for key, value in updates.items():
            if value is not None and key in merged:
                merged[key] = value

        all_data = self._read_all()
        all_data[user_id] = merged
        self._write_all(all_data)
        return UserPreferences(**merged)

    def _read_all(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_all(self, data: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class SessionSummaryStore:
    """基于 JSON 的会话摘要存储。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, user_id: str) -> SessionSummary:
        payload = self._read_all().get(user_id, {})
        return SessionSummary(**payload)

    def set(self, user_id: str, summary: SessionSummary) -> None:
        all_data = self._read_all()
        all_data[user_id] = asdict(summary)
        self._write_all(all_data)

    def clear(self, user_id: str) -> None:
        all_data = self._read_all()
        if user_id in all_data:
            del all_data[user_id]
            self._write_all(all_data)

    def _read_all(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class MemoryManager:
    """管理用户偏好记忆与会话摘要记忆。"""

    def __init__(
        self,
        preference_store: UserPreferenceStore | None = None,
        summary_store: SessionSummaryStore | None = None,
    ) -> None:
        self.preference_store = preference_store or UserPreferenceStore(
            Path(settings.preference_store_path)
        )
        self.summary_store = summary_store or SessionSummaryStore(
            Path(settings.session_summary_store_path)
        )

    def get_preferences(self, user_id: str) -> UserPreferences:
        return self.preference_store.get(user_id)

    def update_preferences(self, user_id: str, **updates: str) -> UserPreferences:
        return self.preference_store.update(user_id, **updates)

    def infer_and_update_preferences(
        self,
        user_id: str,
        user_text: str,
    ) -> dict[str, str]:
        updates = infer_preference_updates(user_text)
        if updates:
            self.update_preferences(user_id, **updates)
        return updates

    def get_summary(self, user_id: str) -> SessionSummary:
        return self.summary_store.get(user_id)

    def clear_summary(self, user_id: str) -> None:
        self.summary_store.clear(user_id)

    def build_runtime_context_message(self, user_id: str) -> SystemMessage | None:
        text = build_runtime_context_text(
            self.get_preferences(user_id),
            self.get_summary(user_id).summary,
        )
        if not text:
            return None
        return SystemMessage(content=text)

    async def compact_history(
        self,
        user_id: str,
        history: list[BaseMessage],
    ) -> list[BaseMessage]:
        """把过长的历史压缩进会话摘要，只保留最近若干消息。"""
        max_messages = settings.session_memory_max_messages
        recent_messages = settings.session_memory_recent_messages
        if len(history) <= max_messages:
            return history

        cutoff = max(len(history) - recent_messages, 0)
        older_messages = history[:cutoff]
        remaining_messages = history[cutoff:]
        if not older_messages:
            return history

        await self.append_to_summary(user_id, older_messages)
        return remaining_messages

    async def remember_exchange(
        self,
        user_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        messages: list[BaseMessage] = [
            HumanMessage(content=user_text),
            AIMessage(content=assistant_text),
        ]
        await self.append_to_summary(user_id, messages)

    async def append_to_summary(
        self,
        user_id: str,
        messages: list[BaseMessage],
    ) -> SessionSummary:
        """把新增消息压缩进会话摘要。"""
        existing = self.get_summary(user_id)
        new_summary_text = await summarize_session_memory(existing.summary, messages)
        updated = SessionSummary(
            summary=new_summary_text,
            summarized_message_count=existing.summarized_message_count + len(messages),
        )
        self.summary_store.set(user_id, updated)
        return updated


def infer_preference_updates(user_text: str) -> dict[str, str]:
    """从用户显式表达中提取偏好更新。"""
    normalized = user_text.strip().lower()
    if not normalized:
        return {}

    updates: dict[str, str] = {}

    if any(phrase in normalized for phrase in ("以后都用中文", "默认用中文", "请用中文")):
        updates["language"] = "zh"
    if any(phrase in normalized for phrase in ("以后都用英文", "默认用英文", "please answer in english")):
        updates["language"] = "en"

    if any(phrase in normalized for phrase in ("简短一点", "简洁一点", "简单一点", "别太长")):
        updates["response_style"] = "concise"
    if any(phrase in normalized for phrase in ("详细一点", "讲细一点", "展开讲", "多讲一点")):
        updates["response_style"] = "detailed"

    if any(phrase in normalized for phrase in ("多给代码", "给代码示例", "多写代码")):
        updates["code_preference"] = "prefer_examples"
    if any(phrase in normalized for phrase in ("不用代码", "少点代码", "先别给代码")):
        updates["code_preference"] = "explain_only"

    if any(phrase in normalized for phrase in ("找实习", "实习简历", "实习面试")):
        updates["career_focus"] = "internship"
    if any(phrase in normalized for phrase in ("论文", "科研", "研究方向")):
        updates["career_focus"] = "research"

    return updates


def build_runtime_context_text(
    preferences: UserPreferences,
    session_summary: str,
) -> str:
    """把偏好与会话摘要渲染成运行时提示。"""
    preference_lines = [
        "用户偏好：",
        f"1. 默认使用{'中文' if preferences.language == 'zh' else '英文'}回答。",
        (
            "2. 回答风格偏简洁，优先先给结论再补充细节。"
            if preferences.response_style == "concise"
            else "2. 回答风格偏详细，适合分步展开解释。"
        ),
        (
            "3. 涉及代码或工程实现时，优先给出示例。"
            if preferences.code_preference == "prefer_examples"
            else "3. 涉及代码时优先解释思路，非必要不堆太多示例。"
        ),
    ]
    if preferences.career_focus == "internship":
        preference_lines.append("4. 如果问题与项目价值相关，优先从实习求职视角回答。")
    elif preferences.career_focus == "research":
        preference_lines.append("4. 如果问题与方案选择相关，优先补充研究和论文视角。")

    parts = ["\n".join(preference_lines)]
    if session_summary.strip():
        parts.append("会话摘要记忆：\n" + session_summary.strip())
    return "\n\n".join(part for part in parts if part.strip()).strip()


async def summarize_session_memory(
    previous_summary: str,
    messages: list[BaseMessage],
) -> str:
    """用聊天模型把旧摘要和新增消息压缩成新的会话摘要。"""
    transcript = render_messages_for_summary(messages)
    if not transcript:
        return previous_summary

    prompt = (
        "请根据已有会话摘要和新增对话，生成一段新的会话摘要记忆。\n"
        "要求：\n"
        "1. 只保留对后续对话有帮助的信息。\n"
        "2. 重点保留用户目标、偏好、未完成事项、关键结论和约束条件。\n"
        "3. 忽略寒暄和无关重复内容。\n"
        "4. 输出简洁中文，不超过 220 字。\n"
        "5. 如果信息不足，就忠实保留已有摘要并补充新增关键点。"
    )
    human_content = (
        f"[已有摘要]\n{previous_summary or '无'}\n\n"
        f"[新增对话]\n{transcript}\n\n"
        "请输出更新后的会话摘要："
    )

    try:
        model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
            temperature=0,
        )
        result = await model.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=human_content),
            ]
        )
        summary_text = message_content_to_text(result.content).strip()
        return summary_text or fallback_session_summary(previous_summary, transcript)
    except Exception:
        return fallback_session_summary(previous_summary, transcript)


def render_messages_for_summary(messages: list[BaseMessage]) -> str:
    """把消息列表渲染成摘要模型可读的转录文本。"""
    lines: list[str] = []
    for message in messages:
        role = "Assistant"
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        content = message_content_to_text(message.content).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def fallback_session_summary(previous_summary: str, transcript: str) -> str:
    """在摘要模型不可用时，退化成简单的拼接压缩。"""
    parts = [part.strip() for part in (previous_summary, transcript) if part.strip()]
    merged = "\n".join(parts).strip()
    if len(merged) <= 400:
        return merged
    return merged[-400:]


def message_content_to_text(content: Any) -> str:
    """把模型返回内容尽量转换成纯文本。"""
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


memory_manager = MemoryManager()
