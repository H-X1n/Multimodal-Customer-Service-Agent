from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional


# =========================
# Multimodal Content
# =========================

@dataclass(slots=True)
class ContentBlock:
    """统一多模态输入结构"""
    type: Literal["text", "image", "audio", "video", "file"]
    data: Any  # text=str, image=path/url/base64, audio=path 等


@dataclass(slots=True)
class UserInput:
    """用户输入（统一多模态）"""
    contents: list[ContentBlock]
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(
        cls,
        question: str,
        images: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        stream: bool = False,
    ) -> "UserInput":
        if not question or len(question) < 1:
            raise ValueError("question 不能为空")
        if images and len(images) > 3:
            raise ValueError("images 最多支持 3 张图片")
        
        contents = [ContentBlock(type="text", data=question)]
        if images:
            for img in images:
                contents.append(ContentBlock(type="image", data=img))
        
        return cls(
            contents=contents,
            session_id=session_id or str(uuid.uuid4()),
            stream=stream,
        )

    @property
    def text(self) -> str:
        texts = [c.data for c in self.contents if c.type == "text"]
        return "\n".join(texts)

    @property
    def image_paths(self) -> list[str]:
        return [c.data for c in self.contents if c.type == "image"]

    @property
    def question(self) -> str:
        return self.text

    @property
    def images(self) -> list[str]:
        return self.image_paths


# =========================
# Understanding Layer
# =========================

@dataclass(slots=True)
class Intent:
    primary: str
    labels: list[str]
    requires_retrieval: bool


@dataclass(slots=True)
class QueryRewrite:
    original: str
    rewritten: str


@dataclass(slots=True)
class VisualContext:
    observations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QuestionTask:
    """可执行任务（支持 DAG）"""
    id: str
    question: str
    category: str
    requires_retrieval: bool = True
    priority: int = 0
    depends_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Plan:
    tasks: list[QuestionTask] = field(default_factory=list)


@dataclass(slots=True)
class IntentAnalysis:
    primary_intent: str
    intents: list[str]
    requires_retrieval: bool
    needs_image_grounding: bool
    rewritten_query: str
    question_plan: list[QuestionTask] = field(default_factory=list)
    visual_observations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UnderstandingResult:
    """统一理解输出（解耦后）"""
    intent: Intent
    rewrite: QueryRewrite
    plan: Plan
    vision: VisualContext


# =========================
# Retrieval Layer
# =========================

@dataclass(slots=True)
class EvidenceChunk:
    question: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # 新增（关键）
    score: Optional[float] = None
    source: Optional[str] = None
    chunk_id: Optional[str] = None

    image_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    question: str
    chunks: list[EvidenceChunk] = field(default_factory=list)

    status: Literal["success", "empty", "error"] = "success"
    error_message: Optional[str] = None


# =========================
# Memory Layer
# =========================

@dataclass(slots=True)
class ConversationTurn:
    role: Literal["user", "assistant", "system"]
    content: str
    image_paths: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class SessionState:
    session_id: str
    short_term_memory: list[ConversationTurn] = field(default_factory=list)
    long_term_memory: dict[str, Any] = field(default_factory=dict)
    slots: dict[str, str] = field(default_factory=dict)
    vector_memory_id: Optional[str] = None


# =========================
# Agent Output
# =========================

@dataclass(slots=True)
class AgentResponse:
    answer: str
    retrievals: list[RetrievalResult] = field(default_factory=list)
    referenced_images: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None