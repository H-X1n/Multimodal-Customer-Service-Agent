"""
yaml
"""
import yaml
import utils.api_keys  # noqa: F401
from utils.path_tool import get_abs_path

def load_rag_config(config_path: str=get_abs_path("config/rag.yaml"), encoding: str="utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_chroma_config(config_path: str=get_abs_path("config/chroma.yaml"), encoding: str="utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(config_path: str=get_abs_path("config/prompts.yaml"), encoding: str="utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_agent_config(config_path: str=get_abs_path("config/agent.yaml"), encoding: str="utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()
