"""
为整个工程提供统一的绝对路径
"""
import os
def get_project_root()->str:
    """
    获取工程所在的根目录
    :return: 字符串根目录
    """
    #当前文件的绝对路径  D:\PythonProject\agent项目\utils\path_tool.py
    current_path = os.path.abspath(__file__)
    #获取工程的根目录，获取文件所在的文件夹绝对路径  D:\PythonProject\agent项目\utils
    current_dir=os.path.dirname(current_path)
    #获取工程根目录  D:\PythonProject\agent项目
    project_path = os.path.dirname(current_dir)
    return project_path


def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径，得到绝对路径
    :param relative_path:
    :return:
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)