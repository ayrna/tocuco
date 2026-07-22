import importlib
import importlib.util
import sys
from pathlib import Path


def load_config(*, config_name: str = None, config_path: str = None):
    if not (config_name or config_path):
        raise ValueError("Either config_name or config_path must be provided.")

    if config_name and config_path:
        raise ValueError("Provide either config_name or config_path, not both.")

    if config_path:
        path = Path(config_path)
    elif config_name:
        path = Path("execution/config") / f"{config_name}.py"

    if not path.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")

    spec = importlib.util.spec_from_file_location("config", path)
    config_module = importlib.util.module_from_spec(spec)

    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)

    return config_module
