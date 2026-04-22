from __future__ import annotations

import os
import uuid

from agent.schemas import (
    Intent,
    Plan,
    QueryRewrite,
    QuestionTask,
    SessionState,
    UnderstandingResult,
    UserInput,
    VisualContext,
)


class MultimodalUnderstandingService:
    """Infers user intent from text and attached images."""

    def analyze(self, user_input: UserInput, session: SessionState, rewritten_query: str) -> UnderstandingResult:
        text = user_input.text.strip()
        visual_observations = self._describe_images(user_input.image_paths)
        question_plan = self._build_question_plan(rewritten_query)
        primary_intent = "general"

        return UnderstandingResult(
            intent=Intent(
                primary=primary_intent,
                labels=[primary_intent],
                requires_retrieval=True,
            ),
            rewrite=QueryRewrite(
                original=text,
                rewritten=rewritten_query,
            ),
            plan=Plan(tasks=question_plan),
            vision=VisualContext(observations=visual_observations),
        )

    def _describe_images(self, image_paths: list[str]) -> list[str]:
        observations: list[str] = []
        for path in image_paths:
            filename = os.path.basename(path)
            stem = os.path.splitext(filename)[0].replace("_", " ")
            observations.append(f"用户附带图片：{stem}")
        return observations

    def _build_question_plan(self, rewritten_query: str) -> list[QuestionTask]:
        pieces = rewritten_query.replace("；", "?").replace(";", "?").split("?")
        questions = [piece.strip("，,。 ") for piece in pieces if piece.strip("，,。 ")]
        if not questions:
            questions = [rewritten_query]
        return [
            QuestionTask(
                id=str(uuid.uuid4())[:8],
                question=question,
                category="general",
                requires_retrieval=True,
            )
            for question in questions
        ]
