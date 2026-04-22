import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from model.factory import MultiModalEmbeddings
from utils.config_handler import chroma_config, rag_config
from utils.path_tool import get_abs_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import listdir_with_allowed_type, get_file_md5_hex, parse_file
from utils.logger_handler import logger


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return get_abs_path(path)


class VectorStoreService:
    def __init__(
        self,
        config_override: dict[str, Any] | None = None,
        txt_embedding_function: Embeddings | None = None,
        img_embedding_function: Embeddings | None = None,
        embedding_function: Embeddings | None = None,
    ):
        self.config = deepcopy(chroma_config)
        if config_override:
            self.config.update(config_override)
        self.txt_embedding_function = (
            txt_embedding_function
            or embedding_function
            or DashScopeEmbeddings(model=rag_config['txt_embedding_model_name'])
        )
        self.img_embedding_function = (
            img_embedding_function
            or embedding_function
            or MultiModalEmbeddings(model=rag_config['img_embedding_model_name'])
        )
        self.vector_store_img = Chroma(
            collection_name=self.config["img_collection_name"],
            embedding_function=self.img_embedding_function,
            persist_directory=resolve_project_path(self.config["img_persist_directory"]),
        )
        self.vector_store_txt = Chroma(
            collection_name=self.config["txt_collection_name"],
            embedding_function=self.txt_embedding_function,
            persist_directory=resolve_project_path(self.config["txt_persist_directory"]),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=self.config["separators"],
            length_function=len,
        )
        self.txt_paths, self.img_paths = listdir_with_allowed_type(
            resolve_project_path(self.config["data_path"]),
            tuple(self.config["allow_txt_type"]),
            tuple(self.config["allow_img_type"]),
        )
    def get_retriever_txt(self):
        return self.vector_store_txt.as_retriever(search_kwargs={"k": self.config["txt_k"]})

    def get_retriever_img(self):
        return self.vector_store_img.as_retriever(search_kwargs={"k": self.config["img_k"]})

    def similarity_search_images_with_score(self, query: str, k: int | None = None):
        """Search image collection and return similarity scores for relevance filtering."""
        return self.vector_store_img.similarity_search_with_relevance_scores(
            query,
            k=k or self.config["img_k"],
        )
    # 检查文件是否已存在向量数据库中
    def check_md5_hex(self, md5_for_check: str | None):
        if not md5_for_check:
            return False
        md5_store_path = resolve_project_path(self.config["md5_hex_store"])
        if not os.path.exists(md5_store_path):
            open(md5_store_path, "w", encoding="utf-8").close()
            return False
        with open(md5_store_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == md5_for_check:
                    return True
            return False

    # 将未存在向量数据库中的数据存入向量数据库
    def save_md5_hex(self, md5_for_check: str | None):
        if not md5_for_check:
            return
        with open(resolve_project_path(self.config["md5_hex_store"]), "a", encoding="utf-8") as f:
            f.write(md5_for_check + '\n')


    #将图片存入向量数据库
    def load_images(self, image_paths: list[str]):
        """
        从文件夹递归加载所有图片到知识库（带 MD5 去重）
        """
        num = len(image_paths)
        for idx,path in enumerate(image_paths):
            md5_hex = get_file_md5_hex(path)
            if self.check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {idx+1}/{num} 图片已存在: {path}")
                continue
            image_uris=[path]
            metadata=[{
                "source": path,
                "filename": os.path.basename(path),
                "file_type": "image"
            }]
            try:
                self.vector_store_img.add_images(
                    uris=image_uris,
                    metadatas=metadata,
                    ids=[f"image::{md5_hex}"],
                )
                self.save_md5_hex(md5_hex)
                logger.info(f"[加载知识库] {idx+1}/{num} 图片加载成功{path}")
            except Exception as e:
                #exc_info为True会记录详细的报错堆栈,如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{idx+1}/{num} 图片加载失败:{path} {str(e)}", exc_info=True)
                continue

    #加载文本文件到向量数据库
    def load_txts(self, txt_paths: list[str]):
        num = len(txt_paths)
        for idx,path in enumerate(txt_paths):
            md5_hex = get_file_md5_hex(path)
            if self.check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {idx+1}/{num} 文本内容已存在知识库内，跳过{path}]")
                continue
            try:
                doc = parse_file(path)
                # 添加到向量库
                split_doc= self.spliter.split_documents([doc])
                if not split_doc:
                    logger.info(f"[加载知识库] {idx+1}/{num} 文本切片为空，跳过{path}")
                    continue
                self.vector_store_txt.add_documents(
                    split_doc
                )
                self.save_md5_hex(md5_hex)
                logger.info(f"[加载知识库] {idx+1}/{num} 文本内容加载成功{path}")
            except Exception as e:
                #exc_info为True会记录详细的报错堆栈,如果为False仅记录报错信息本身
                logger.error(f"[加载知识库] {idx+1}/{num} 加载失败: {path} {str(e)}", exc_info=True)
                continue



if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_images(vs.img_paths)
    vs.load_txts(vs.txt_paths)
    print(f"文本文件数量: {len(vs.txt_paths)}")
    print(f"图片文件数量: {len(vs.img_paths)}")
    for name, store in (("文本向量库", vs.vector_store_txt), ("图片向量库", vs.vector_store_img)):
        try:
            print(f"{name}记录数: {store._collection.count()}")
        except Exception as exc:
            print(f"{name}读取失败: {type(exc).__name__}: {exc}")


