import os
import sys
import importlib

_EXPERIMENTS = {}
_current_dir = os.path.dirname(__file__)

for filename in os.listdir(_current_dir):
    if filename.endswith(".py") and not filename.startswith("_"):
        name = filename[:-3]
        module = importlib.import_module(f".{name}", package=__name__)
        _EXPERIMENTS[name] = module

_active_config_name = "default"

if "EXECUTION_CONFIG_NAME" in os.environ:
    _active_config_name = os.environ["EXECUTION_CONFIG_NAME"]

else:
    if "--config" in sys.argv:
        _idx = sys.argv.index("--config")
        _active_config_name = sys.argv[_idx + 1]
    elif "-c" in sys.argv:
        _idx = sys.argv.index("-c")
        _active_config_name = sys.argv[_idx + 1]
    
    os.environ["EXECUTION_CONFIG_NAME"] = _active_config_name

if _active_config_name not in _EXPERIMENTS:
    raise ValueError(
        f"[ERROR] Configuration '{_active_config_name}' does not exist. "
        f"Availables: {list(_EXPERIMENTS.keys())}"
    )

CONFIG_EXP = _EXPERIMENTS[_active_config_name]