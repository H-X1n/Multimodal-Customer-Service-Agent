from __future__ import annotations

import ast
import os
import re
import sys
from typing import Generator

from langchain.agents import create_agent

from agent.dialogue.manager import DialogueManager
from agent.schemas import AgentResponse, UserInput
from agent.tools.agent_tools import AGENT_TOOLS
from model.factory import chat_model
from utils.config_handler import chroma_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_system_prompts
from agent.tools.middleware import MIDDLEWARE


class CustomerSupportAgent:
    """Agent entrypoint powered by langchain.agents.create_agent."""

    def __init__(self) -> None:
        self.dialogue_manager = DialogueManager()
        self.model = chat_model

        self.agent = create_agent(
            model=self.model,
            tools=AGENT_TOOLS,
            system_prompt=load_system_prompts(),
            name="multimodal_customer_support_agent",
            middleware=MIDDLEWARE,
        )

    def answer(self, user_input: UserInput) -> AgentResponse | Generator[str, None, None]:
        session_id = user_input.session_id
        user_text = user_input.text

        rewritten_query = self.dialogue_manager.rewrite_query_with_context(
            session_id, user_text
        )

        self.dialogue_manager.append_user_message(session_id, user_text)

        user_prompt = self._build_user_prompt(user_text, rewritten_query)

        if user_input.stream:
            return self._stream_agent(user_prompt, session_id, rewritten_query)
        else:
            answer = self._invoke_agent(user_prompt, print_output=False)
            answer, referenced_images = self._sanitize_image_references(answer)
            self.dialogue_manager.append_assistant_message(session_id, answer)
            return AgentResponse(
                answer=answer,
                referenced_images=referenced_images,
                debug={"rewritten_query": rewritten_query},
            )

    def _stream_agent(
        self,
        prompt: str,
        session_id: str,
        rewritten_query: str,
    ) -> Generator[str, None, None]:
        full_content = ""
        try:
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": prompt}]}
            ):
                if isinstance(chunk, dict):
                    for node_name, value in chunk.items():
                        if node_name == "model" and isinstance(value, dict):
                            messages = value.get("messages", [])
                            for message in messages:
                                if hasattr(message, "content"):
                                    content = message.content
                                    if content:
                                        if isinstance(content, list):
                                            content = "\n".join(str(item) for item in content)
                                        if content and content != full_content:
                                            new_content = content[len(full_content):]
                                            full_content = content
                                            yield new_content

            if not full_content:
                yield "当前未生成有效回复。"

            self.dialogue_manager.append_assistant_message(session_id, full_content)

        except Exception as exc:
            logger.warning("Agent 执行失败", exc_info=True)
            yield f"知识库问答暂时不可用，错误信息：{exc}"

    def _invoke_agent(self, prompt: str, print_output: bool = True) -> str:
        try:
            full_content = ""
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": prompt}]}
            ):
                if isinstance(chunk, dict):
                    for node_name, value in chunk.items():
                        if node_name == "model" and isinstance(value, dict):
                            messages = value.get("messages", [])
                            for message in messages:
                                if hasattr(message, "content"):
                                    content = message.content
                                    if content:
                                        if isinstance(content, list):
                                            content = "\n".join(str(item) for item in content)
                                        if content and content != full_content:
                                            new_content = content[len(full_content):]
                                            if print_output:
                                                safe_content = new_content.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
                                                print(safe_content, end="", flush=True)
                                            full_content = content

            if not full_content:
                return "当前未生成有效回复。"
            return full_content

        except Exception as exc:
            logger.warning("Agent 执行失败", exc_info=True)
            return f"知识库问答暂时不可用，错误信息：{exc}"

    def _build_user_prompt(self, original_text: str, rewritten_query: str) -> str:
        return (
            f"用户问题：{original_text}\n"
            f"结合上下文后的问题：{rewritten_query}\n\n"
            "请按你的系统要求完成回答。"
        )

    def _sanitize_image_references(self, answer: str) -> tuple[str, list[str]]:
        image_ids = self._extract_image_ids(answer)
        if not image_ids and "<PIC>" not in answer:
            return answer, []

        referenced_images = self._resolve_existing_images(image_ids)
        if referenced_images:
            return answer, referenced_images

        cleaned_answer = re.sub(r"\s*\[[^\[\]]*\]\s*$", "", answer).replace("<PIC>", "")
        return cleaned_answer.strip(), []

    @staticmethod
    def _extract_image_ids(answer: str) -> list[str]:
        match = re.search(r"(\[[^\[\]]*\])\s*$", answer)
        if not match:
            return []
        try:
            parsed = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    @staticmethod
    def _resolve_existing_images(image_ids: list[str]) -> list[str]:
        if not image_ids:
            return []

        image_dir = os.path.join(get_abs_path(chroma_config["data_path"]), "插图")
        referenced_images: list[str] = []
        for image_id in image_ids:
            candidates = [image_id]
            if not os.path.splitext(image_id)[1]:
                candidates.extend(f"{image_id}{ext}" for ext in (".png", ".jpg", ".jpeg"))
            for candidate in candidates:
                image_path = os.path.abspath(os.path.join(image_dir, os.path.basename(candidate)))
                if os.path.isfile(image_path):
                    referenced_images.append(image_path)
                    break
        return referenced_images


if __name__ == "__main__":
    agent = CustomerSupportAgent()

    user_input = UserInput.from_api(
        question="相机如何使用？请结合图片说明",
        session_id="test_session_007",
        stream=False,
    )

    response = agent.answer(user_input)
    print(response.answer)
