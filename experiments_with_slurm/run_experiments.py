import argparse
import json

from execution.configs import load_config


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="default",
        help="Name of the configuration module inside execution.config (without .py extension).",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=str,
        choices=["paramsearch", "final", "full", "results_collector", "all"],
        default="all",
        help=(
            "Step(s) of the pipeline to execute."
            "For the external_cv pipeline: 'paramsearch', 'final'"
            " (final training), 'results_collector', or 'all'."
            "For the internal_cv pipeline: 'full' runs the full paramsearch (if done) and training,"
            " 'results_collector' only runs the results collector, and 'all' runs both."
        ),
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help=(
            "Whether to perform a dry run (flow function receives dry_run=True)"
            " for testing purposes. Overrides the config.py setting."
        ),
    )
    parser.add_argument(
        "--clear-logs",
        action="store_true",
        help="Whether to clear old logs before submitting new jobs.",
    )
    args = parser.parse_args()

    import os
    os.environ["EXECUTION_CONFIG_NAME"] = args.config

    CONFIG = load_config(config_name=args.config)

    from execution.flows import REGISTRY
    from execution.orchestrator import (
        clear_old_logs,
        submit_external_cv_pipeline,
        submit_internal_cv_pipeline,
    )

    if args.dry_run:
        CONFIG.dry_run = True

    config_dict = {k: v for k, v in vars(CONFIG).items() if not k.startswith("_")}
    beautiful_config = json.dumps(config_dict, indent=4)
    print(beautiful_config)

    if CONFIG.flow not in REGISTRY:
        raise ValueError(
            f"Flow {CONFIG.flow} not found in REGISTRY. Available flows: {list(REGISTRY.keys())}"
        )

    if CONFIG.pipeline not in REGISTRY[CONFIG.flow]["supported_pipelines"]:
        raise ValueError(
            f"Flow {CONFIG.flow} does not support pipeline {CONFIG.pipeline}. "
            f"Supported pipelines: {REGISTRY[CONFIG.flow]['supported_pipelines']}"
        )

    # Clear old logs before submitting new jobs
    if args.clear_logs:
        clear_old_logs()

    if CONFIG.pipeline == "internal_cv":
        submit_internal_cv_pipeline(
            flow_name=CONFIG.flow, step=args.step, dry_run=args.dry_run
        )
    else:
        submit_external_cv_pipeline(
            flow_name=CONFIG.flow, step=args.step, dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()
