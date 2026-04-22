import os

from dotenv import dotenv_values

from utils.path_tool import get_abs_path


API_KEYS_FILE = get_abs_path("config/api_keys.env")
API_KEYS = {
    key: value
    for key, value in dotenv_values(API_KEYS_FILE).items()
    if value
}

for key, value in API_KEYS.items():
    os.environ[key] = value


def get_api_key(name: str, default: str | None = None) -> str | None:
    """Read API keys from the centralized api_keys.env file."""
    return API_KEYS.get(name, default)


def get_required_api_key(name: str) -> str:
    value = get_api_key(name)
    if not value:
        raise RuntimeError(f"Missing required API key: {name}. Please set it in config/api_keys.env.")
    return value
