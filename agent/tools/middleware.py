from typing import Callable
from utils.prompt_loader import load_report_prompts, load_system_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
)->ToolMessage | Command:
    logger.info(f"[tool monitor]调用工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数: {request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f'[tool monitor]工具{request.tool_call["name"]}调用成功')
        if request.tool_call["name"] == "fill_context_for_report":
            context = getattr(request.runtime, "context", None)
            if isinstance(context, dict):
                context["report"] = True
            else:
                logger.warning("[tool monitor]runtime.context 不可写，跳过 report 标记")
        return result
    except Exception as e:
        logger.error(f'工具{request.tool_call["name"]}调用失败，原因: {e}')
        raise e

@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    logger.info(f"[log before model]即将调用模型，带有{len(state['messages'])}条信息")
    last_msg = state["messages"][-1]
    msg_type = type(last_msg).__name__
    content = getattr(last_msg, 'content', '')
    if isinstance(content, str):
        content_preview = content[:100] if len(content) > 100 else content
    else:
        content_preview = str(content)[:100]
    logger.debug(f'[log before model] {msg_type} | {content_preview}')
    return None

@dynamic_prompt
def report_prompts_switch(request: ModelRequest):
    context = getattr(request.runtime, 'context', None) or {}
    is_report = context.get('report', False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()

MIDDLEWARE = [monitor_tool, log_before_model, report_prompts_switch]
