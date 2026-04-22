from __future__ import annotations

import re
import uuid
from collections import OrderedDict
from typing import Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent.schemas import QuestionTask, SessionState, UnderstandingResult, UserInput


class DialogueManager:
    """Maintains multi-turn context using LangChain's InMemoryChatMessageHistory."""

    def __init__(self) -> None:
        self._histories: dict[str, InMemoryChatMessageHistory] = {}
        self._sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories[session_id]

    def append_user_message(self, session_id: str, content: str) -> None:
        history = self.get_chat_history(session_id)
        history.add_message(HumanMessage(content=content))

    def append_assistant_message(self, session_id: str, content: str) -> None:
        history = self.get_chat_history(session_id)
        history.add_message(AIMessage(content=content))

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        history = self.get_chat_history(session_id)
        return history.messages

    def get_history_string(self, session_id: str, max_turns: int = 10) -> str:
        messages = self.get_messages(session_id)[-max_turns * 2:]
        if not messages:
            return ""
        lines = []
        for msg in messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def rewrite_query_with_context(self, session_id: str, text: str) -> str:
        cleaned = text.strip()
        messages = self.get_messages(session_id)
        if not messages:
            return cleaned
        previous_user_messages = [
            msg.content for msg in messages if isinstance(msg, HumanMessage)
        ]
        if len(cleaned) < 12 and previous_user_messages:
            return f"结合上一轮上下文回答：上一轮用户问题是\"{previous_user_messages[-1]}\"，本轮补充问题是\"{cleaned}\""
        return cleaned

    def plan_questions(self, analysis: UnderstandingResult, rewritten_query: str) -> list[QuestionTask]:
        parts = self._split_questions(rewritten_query)
        if not parts and analysis.plan.tasks:
            return analysis.plan.tasks
        if not parts:
            parts = [rewritten_query]

        ordered: OrderedDict[str, QuestionTask] = OrderedDict()
        for part in parts:
            question = part.strip("，。；; ")
            if not question:
                continue
            ordered[question] = QuestionTask(
                id=str(uuid.uuid4())[:8],
                question=question,
                category="general",
                requires_retrieval=analysis.intent.requires_retrieval,
            )
        return list(ordered.values())

    def _split_questions(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text)
        candidates = re.split(r"[？?；;]\s*|(?<=。)", normalized)
        result: list[str] = []
        for item in candidates:
            item = item.strip()
            if not item:
                continue
            pieces = re.split(r"(?:另外|还有|以及|并且|同时|顺便问下|顺便问一下)", item)
            for piece in pieces:
                piece = piece.strip("，, ")
                if piece:
                    result.append(piece)
        return result

    def clear_history(self, session_id: str) -> None:
        if session_id in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        if session_id in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)


def get_session_history(session_id: str) -> Callable[[str], InMemoryChatMessageHistory]:
    """Factory function for RunnableWithMessageHistory."""
    manager = DialogueManager()
    return manager.get_chat_history(session_id)
