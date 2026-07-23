import os
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "patent-collate" / "config.yaml"


def get_api_key(key_name: str, config_path: Path = DEFAULT_CONFIG_PATH) -> str:
    env_var = os.environ.get("OPENAI_API_KEY")
    if env_var:
        return env_var

    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config and key_name in config and config[key_name]:
            return config[key_name]

    raise ValueError(
        "No OpenAI API key found. Set OPENAI_API_KEY environment variable, "
        "pass --api-key to the CLI, or create a config file at "
        f"{DEFAULT_CONFIG_PATH} with:\n  openai: sk-..."
    )
