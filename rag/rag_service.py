"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
import base64
import mimetypes
import os
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.config_handler import agent_config


def print_prompt(prompt):
    print('-'*50)
    print(prompt.to_string())
    print('-'*50)
    return prompt

class RagSummarizeService(object):
    def __init__(self, vector_store: VectorStoreService | None = None):
        self.vector_store = vector_store or VectorStoreService()
        self.retriever_txt = self.vector_store.get_retriever_txt()
        self.retriever_img = self.vector_store.get_retriever_img()
        self.prompt_text = load_rag_prompts()
        self.image_retrieval_config = agent_config["image_retrieval"]
        self.min_image_relevance_score = float(
            self.image_retrieval_config["min_relevance_score"]
        )

    def _retrieve_txt(self, query: str) -> list[Document]:
        return self.retriever_txt.invoke(query)

    def _retrieve_img(self, query: str) -> list[Document]:
        return self.retriever_img.invoke(query)

    def txt_rag_summarize(self, query: str) -> str:
        context_docs = self._retrieve_txt(query)
        context, counter = '', 0
        for doc in context_docs:
            counter += 1
            context += f'【参考资料{counter}】: 参考资料: {doc.page_content} | 参考元数据: {doc.metadata}\n'
        return context
    def img_rag_summarize(self, query: str) -> str | list[dict[str, Any]]:
        scored_docs = self.vector_store.similarity_search_images_with_score(query)
        context_docs = [
            (doc, score)
            for doc, score in scored_docs
            if score is not None and score >= self.min_image_relevance_score
        ]
        if not context_docs:
            return self.image_retrieval_config["no_relevant_image_message"]

        content_blocks: list[dict[str, Any]] = []
        summary_lines: list[str] = [
            "已检索到以下相关手册图片。请直接观察后续 image_url 图片内容，只在图片确实能辅助回答时引用图片。"
        ]
        image_blocks: list[dict[str, Any]] = []
        for doc, score in context_docs:
            filename = doc.metadata.get("filename", "unknown")
            summary_lines.append(f"- 文件名: {filename} | 相关性: {score:.3f}")
            data_uri = self._document_to_image_data_uri(doc)
            if data_uri:
                image_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )

        if not image_blocks:
            return self.image_retrieval_config["no_relevant_image_message"]

        content_blocks.append({"type": "text", "text": "\n".join(summary_lines)})
        content_blocks.extend(image_blocks)
        return content_blocks

    @staticmethod
    def _document_to_image_data_uri(doc: Document) -> str | None:
        page_content = doc.page_content.strip()
        if page_content.startswith("data:image/"):
            return page_content

        source = doc.metadata.get("source", "")
        mime_type, _ = mimetypes.guess_type(source)
        mime_type = mime_type or "image/png"

        if page_content:
            return f"data:{mime_type};base64,{page_content}"

        if source and os.path.isfile(source):
            with open(source, "rb") as file:
                encoded = base64.b64encode(file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
        return None
if __name__ == '__main__':
    rag = RagSummarizeService()
    # print(rag.txt_rag_summarize("相机如何充电？"))
    # print('='*20)
    print(rag.img_rag_summarize("相机如何充电？"))
