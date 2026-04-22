from abc import ABC, abstractmethod
import base64
import mimetypes
import os
import time
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from utils.config_handler import rag_config
from utils.api_keys import get_required_api_key
import dashscope
from http import HTTPStatus

MULTIMODAL_MODEL_KEYWORDS = (
    "qwen3-vl-embedding",
    "qwen2.5-vl-embedding",
    "tongyi-embedding-vision",
    "multimodal-embedding",
)
MAX_MULTIMODAL_BATCH_SIZE = 20
MULTIMODAL_RETRY_TIMES = 3
MULTIMODAL_RETRY_SLEEP_SECONDS = 2


class MultiModalEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            embedding = self._call_embeddings([{"text": text}])[0]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._call_embeddings([{"text": text}])[0]

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        contents = [{"image": self._normalize_image_uri(uri)} for uri in uris]
        return self._call_embeddings(contents)

    def embed_text_with_image(self, text: str, image: str) -> list[float]:
        image_value = self._normalize_image_uri(image)
        if self._supports_enable_fusion():
            contents = [{"text": text}, {"image": image_value}]
            return self._call_embeddings(contents, enable_fusion=True)[0]
        return self._call_embeddings([{"text": text, "image": image_value}])[0]

    def _call_embeddings(
        self,
        contents: list[dict[str, str]],
        enable_fusion: bool = False,
    ) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for start in range(0, len(contents), MAX_MULTIMODAL_BATCH_SIZE):
            batch_contents = contents[start : start + MAX_MULTIMODAL_BATCH_SIZE]
            last_error = None
            for attempt in range(MULTIMODAL_RETRY_TIMES):
                kwargs = {
                    "api_key": get_required_api_key("DASHSCOPE_API_KEY"),
                    "model": self.model,
                    "input": batch_contents,
                }
                if enable_fusion and self._supports_enable_fusion():
                    kwargs["enable_fusion"] = True
                resp = dashscope.MultiModalEmbedding.call(**kwargs)
                if resp.status_code == HTTPStatus.OK:
                    all_embeddings.extend(item["embedding"] for item in resp.output["embeddings"])
                    break
                last_error = f"Embedding failed: {resp.code} - {resp.message}"
                if attempt < MULTIMODAL_RETRY_TIMES - 1:
                    time.sleep(MULTIMODAL_RETRY_SLEEP_SECONDS)
            else:
                raise ValueError(last_error or "Embedding failed with unknown error")
        return all_embeddings

    def _supports_enable_fusion(self) -> bool:
        return self.model == "qwen3-vl-embedding"

    def _normalize_image_uri(self, image: str) -> str:
        if image.startswith(("http://", "https://", "data:image/")):
            return image
        if not os.path.exists(image):
            return image

        mime_type, _ = mimetypes.guess_type(image)
        mime_type = mime_type or "image/png"
        with open(image, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        model_name = rag_config["chat_model_name"]
        return ChatOpenAI(
            model=model_name,
            api_key=get_required_api_key("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            streaming=True,
        )


chat_model = ChatModelFactory().generator()
