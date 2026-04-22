import os
import hashlib
import json
import ast
from PIL import Image
from typing import List
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

def get_file_md5_hex(file_path: str):
    """
    获取文件的md5的十六进制字符串
    :return:
    """
    if not os.path.exists(file_path):
        logger.error(f'[md5计算]文件{file_path}不存在')
        return None
    if not os.path.isfile(file_path):
        logger.error(f'[md5计算]路径{file_path}不是文件')
        return None
    md5_obj=hashlib.md5()
    chunk_size = 4096 #4KB分片，避免爆内存
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f'计算文件{file_path}md5失败,{str(e)}')
        return None


def listdir_with_allowed_type(
        path: str,
        text_extensions: tuple[str] = (".txt", ".pdf", ".md", ".docx", ".json", ".csv", ".epub", ".mobi"),
        image_extensions: tuple[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
):
    """
    递归返回文件夹内文本文件和图片文件的路径列表
    :param path: 根目录路径
    :param text_extensions: 文本文件后缀元组（默认包含常见文本格式）
    :param image_extensions: 图片文件后缀元组（默认包含常见图片格式）
    :return: (text_files, image_files) 两个列表，分别是文本文件路径和图片文件路径
    """
    text_files = []
    image_files = []

    # 检查路径是否为有效文件夹
    if not os.path.isdir(path):
        logger.error(f'[listdir_with_allowed_type]{path}不是文件夹')
        return text_files, image_files  # 返回两个空列表

    # 递归遍历所有子目录
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            lower_filename = filename.lower()
            full_path = os.path.join(root, filename)

            # 分类判断：文本文件
            if lower_filename.endswith(text_extensions):
                text_files.append(full_path)
            # 分类判断：图片文件
            elif lower_filename.endswith(image_extensions):
                image_files.append(full_path)

    return text_files, image_files


def pdf_loader(file_path: str,passwd=None)->Document:
    return PyPDFLoader(file_path, passwd, encoding='utf-8').load()

def txt_loader(file_path)->Document:
    return TextLoader(file_path, encoding='utf-8').load()


def parse_file(file_path: str) -> Document:
    """
    解析txt文件[string: str, img_names: list[str]]格式
    仅提取文本部分，不处理<PIC>标签
    :param file_path: txt文件路径
    :return: Document对象
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        parsed_data = json.loads(content)
    except json.JSONDecodeError:
        try:
            parsed_data = ast.literal_eval(content)
        except Exception as e:
            raise ValueError(f"文件 {file_path} 解析失败: {str(e)}")

    if not isinstance(parsed_data, list) or len(parsed_data) < 1:
        raise ValueError(f"文件 {file_path} 格式错误，应为 [text, image_ids] 格式")

    text_content = parsed_data[0]
    if not isinstance(text_content, str):
        raise ValueError(f"文件 {file_path} 第一个元素必须是字符串")

    return Document(
        page_content=text_content,
        metadata={
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_type": "text",
        }
    )


def txt_loader_x(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # 解析你的 ["文本", [ids]] 结构
    try:
        text, ids = json.loads(content)
    except:
        text, ids = ast.literal_eval(content)

    return [
        Document(
            page_content=text,
            metadata={"source": file_path, "ids": ids}
        )
    ]

if __name__ == '__main__':
    doc = parse_file(r"D:\PythonProject\agent项目\data\健身单车手册.txt")
    print(doc.page_content)