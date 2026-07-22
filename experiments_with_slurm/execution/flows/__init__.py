from .dl import run_dl_flow
from .ml import run_ml_flow

REGISTRY = {
    "dl": {
        "function": run_dl_flow,
        "supported_pipelines": ["external_cv"],
    },
    "ml": {
        "function": run_ml_flow,
        "supported_pipelines": ["internal_cv"],
    },
}

__all__ = ["REGISTRY"]
