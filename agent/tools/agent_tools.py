from langchain_core.tools import tool
import os
import base64
import mimetypes
from rag.rag_service import RagSummarizeService
from utils.config_handler import chroma_config
from utils.path_tool import get_abs_path
rag_service = RagSummarizeService()


@tool(description="从文本向量数据库搜索相关资料")
def txt_rag_summarize(query: str) -> str:
    return rag_service.txt_rag_summarize(query)

@tool(description="仅当用户明确需要图片、示意图、按钮/接口位置、安装/连接/操作步骤等必须图示的问题时调用；返回足够相关的手册图片，包含可由多模态模型读取的base64 image_url内容。普通问答、物流、售后、价格、寒暄等不要调用。")
def img_rag_summarize(query: str) -> str | list[dict]:
    return rag_service.img_rag_summarize(query)

@tool(description="兼容工具：仅在需要按文件名补充读取图片时调用。根据图片文件名列表获取图片内容，返回base64编码的图片信息。输入格式：逗号分隔的图片文件名，如'Camera_01.png,Camera_02.png'")
def get_image_content(filenames: str) -> str:
    image_dir = os.path.join(get_abs_path(chroma_config["data_path"]), "插图")
    results = []
    
    for filename in filenames.split(","):
        filename = filename.strip()
        if not filename:
            continue
            
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            results.append(f"图片 {filename} 不存在")
            continue
        
        try:
            mime_type, _ = mimetypes.guess_type(str(image_path))
            mime_type = mime_type or "image/png"
            
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            results.append(f"图片 {filename}:\ndata:{mime_type};base64,{image_data}")
        except Exception as e:
            results.append(f"读取图片 {filename} 失败: {str(e)}")
    
    return "\n\n".join(results)

@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report() -> str:
    return "fill_context_for_report 已调用"

@tool(description="获得杭州今天的天气")
def get_weather() -> str:
    return "晴天"



AGENT_TOOLS = [txt_rag_summarize, img_rag_summarize, get_image_content, fill_context_for_report, get_weather]
