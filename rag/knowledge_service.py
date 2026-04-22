from __future__ import annotations

import ast
import os
import re

from agent.schemas import EvidenceChunk
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_config
from utils.path_tool import get_abs_path


class KnowledgeService:
    """Retrieves manual snippets and resolves related illustration assets."""

    def __init__(self) -> None:
        self.vector_store = VectorStoreService()
        self.txt_retriever = self.vector_store.get_retriever_txt()
        self.img_retriever = self.vector_store.get_retriever_img()
        self.image_index = self._build_image_index()

    def retrieve(self, query: str) -> list[EvidenceChunk]:
        docs = self.txt_retriever.invoke(query)
        chunks: list[EvidenceChunk] = []
        for doc in docs:
            image_ids = self._extract_image_ids(doc.page_content, doc.metadata)
            image_paths = self._resolve_image_paths(image_ids)
            chunks.append(
                EvidenceChunk(
                    question=query,
                    content=doc.page_content,
                    metadata=dict(doc.metadata),
                    image_ids=image_ids,
                    image_paths=image_paths,
                )
            )
        return chunks

    def retrieve_images(self, query: str) -> list[EvidenceChunk]:
        docs = self.img_retriever.invoke(query)
        chunks: list[EvidenceChunk] = []
        for doc in docs:
            source = doc.metadata.get("source", "")
            image_id = os.path.splitext(os.path.basename(source))[0] if source else ""
            chunks.append(
                EvidenceChunk(
                    question=query,
                    content=doc.page_content,
                    metadata=dict(doc.metadata),
                    image_ids=[image_id] if image_id else [],
                    image_paths=[source] if source else [],
                )
            )
        return chunks

    def _extract_image_ids(self, content: str, metadata: dict) -> list[str]:
        ids = re.findall(r"<PIC:\s*([^>]+)>", content)
        raw_image_ids = metadata.get("image_ids")
        if raw_image_ids:
            try:
                parsed = ast.literal_eval(raw_image_ids) if isinstance(raw_image_ids, str) else raw_image_ids
                if isinstance(parsed, list):
                    ids.extend(str(item) for item in parsed)
            except (ValueError, SyntaxError):
                pass
        ordered: dict[str, None] = {}
        for item in ids:
            ordered[item.strip()] = None
        return list(ordered.keys())

    def _build_image_index(self) -> dict[str, str]:
        image_dir = os.path.join(get_abs_path(chroma_config["data_path"]), "插图")
        index: dict[str, str] = {}
        if not os.path.isdir(image_dir):
            return index
        for root, _, filenames in os.walk(image_dir):
            for filename in filenames:
                stem = os.path.splitext(filename)[0].lower()
                index[stem] = os.path.join(root, filename)
        return index

    def _resolve_image_paths(self, image_ids: list[str]) -> list[str]:
        result: list[str] = []
        for image_id in image_ids:
            key = image_id.lower()
            matched = self.image_index.get(key)
            if matched:
                result.append(matched)
                continue
            normalized = key.replace("-", "_")
            matched = self.image_index.get(normalized)
            if matched:
                result.append(matched)
        return result
