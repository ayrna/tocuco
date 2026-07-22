import datetime
import inspect
import os
from functools import partial
from pathlib import Path
from shutil import rmtree
import time

import config as CONFIG

config_path = os.path.abspath(CONFIG.__file__)

import submitit

from execution.flows import REGISTRY
from execution.paramsearch import get_randomizedsearch_params
from execution.workers import (
    run_aggregator,
    run_experiment,
    run_experiment_with_best_config,
    run_results_collector,
)
from experiments.utils.estimators_external import get_estimator_config

#################################################################
# HELPER FUNCTIONS
#################################################################


def get_value_for_estimator_dataset(estimator, dataset, override, value):
    if override:
        if estimator in override and dataset in override[estimator]:
            return override[estimator][dataset]
        elif "*" in override and dataset in override["*"]:
            return override["*"][dataset]
        elif estimator in override and "*" in override[estimator]:
            return override[estimator]["*"]
    return value


def get_time_in_minutes(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 60 + m + s / 60


def get_executor_params(estimator, dataset, name_suffix=""):
    params = {}
    params["slurm_job_name"] = (
        f"{CONFIG.experiment_name}{name_suffix}_{estimator}_{dataset}"
    )
    params["slurm_time"] = get_value_for_estimator_dataset(
        estimator, dataset, CONFIG.max_time_override, CONFIG.max_time
    )

    partitions = []
    time_minutes = get_time_in_minutes(params["slurm_time"])

    if CONFIG.gpus > 0:
        if CONFIG.gpu_legacy:
            partitions.append("legacy_gpu")
        if time_minutes <= 30:
            partitions.append("short_gpu")
        elif time_minutes <= 720:
            partitions.append("normal_gpu")
        else:
            partitions.append("long_gpu")
    else:
        if time_minutes <= 30:
            partitions.append("short")
        elif time_minutes <= 720:
            partitions.append("normal")
        else:
            partitions.append("long")

    params["slurm_partition"] = ",".join(partitions)
    params["cpus_per_task"] = CONFIG.cpus
    params["mem_gb"] = get_value_for_estimator_dataset(
        estimator, dataset, CONFIG.memory_override, CONFIG.memory
    )

    gpu_type = get_value_for_estimator_dataset(
        estimator, dataset, CONFIG.gpu_type_override, CONFIG.gpu_type
    )

    gres = f"gpu:{gpu_type + ':' if gpu_type else ''}{CONFIG.gpus}"
    params["slurm_additional_parameters"] = {"gres": gres, "nice": CONFIG.nice}
    params["slurm_array_parallelism"] = CONFIG.max_concurrent_jobs
    #params["slurm_setup"] = ["source /etc/profile", "module load python/3.9"]

    return params


def generate_external_cv_experiments_configs(final):
    configs = {}
    for estimator in CONFIG.estimators:
        for dataset in CONFIG.datasets:
            configs[(estimator, dataset)] = []

            for seed in range(CONFIG.seeds):
                context = {
                    **vars(CONFIG),  # Convert module to dict
                    "dataset": dataset,
                    "estimator_name": estimator,
                    "estimator_config": None,
                    "fold": None,
                    "seed": seed,
                    "batch_size": get_value_for_estimator_dataset(
                        estimator,
                        dataset,
                        CONFIG.batch_size_override,
                        CONFIG.batch_size,
                    ),
                }

                if final:
                    context["n_folds"] = None
                    context["val_size"] = None
                    context["search_n_iter"] = None

                    configs[(estimator, dataset)].append(context)
                else:
                    base_config, param_grid = get_estimator_config(estimator)
                    param_configs = get_randomizedsearch_params(
                        param_grid, CONFIG.search_n_iter, random_state=seed
                    )

                    for param_config in param_configs:
                        # Only use folds if n_folds > 1
                        if CONFIG.n_folds and CONFIG.n_folds > 1:
                            folds_to_iterate = range(CONFIG.n_folds)
                        else:
                            folds_to_iterate = range(1)
                        
                        for fold in folds_to_iterate:
                            estimator_config = base_config.copy()
                            estimator_config.update(param_config)

                            config = context.copy()
                            config["estimator_config"] = estimator_config
                            # Only set fold if we are actually using multiple folds
                            if CONFIG.n_folds and CONFIG.n_folds > 1:
                                config["fold"] = fold
                            # else: keep fold = None (already set in context)

                            configs[(estimator, dataset)].append(config)
    return configs


def generate_internal_cv_experiments_configs():
    configs = {}
    for estimator in CONFIG.estimators:
        for dataset in CONFIG.datasets:
            configs[(estimator, dataset)] = []

            for seed in range(CONFIG.seeds):
                context = {
                    **vars(CONFIG),  # Convert module to dict
                    "dataset": dataset,
                    "estimator_name": estimator,
                    "seed": seed,
                }

                configs[(estimator, dataset)].append(context)
    return configs


def clear_old_logs():
    logs_path = Path(CONFIG.logs_output_dir)
    if logs_path.exists() and logs_path.is_dir():
        rmtree(logs_path)
        print(f"Old logs in {logs_path} cleared.")


########################################################################
# ORQUESTRATION FUNCTIONS
# These functions are responsible for submitting the jobs to the cluster
########################################################################


def submit_experiments(flow_function, step: str, timestamp, additional_params={}):
    if step not in ["paramsearch", "final", "full"]:
        raise ValueError(
            f"Invalid mode: {step}. Must be one of 'paramsearch', 'final' or 'full'."
        )

    target_args = inspect.signature(flow_function).parameters.keys()

    if step == "full":
        experiment_configs_dict = generate_internal_cv_experiments_configs()
    else:
        experiment_configs_dict = generate_external_cv_experiments_configs(
            final=(step == "final")
        )

    # Remove any keys from the configs that are not in the target function args
    clean_experiment_configs_dict = {}
    for key, configs in experiment_configs_dict.items():
        clean_configs = []
        for config in configs:
            clean_config = {k: v for k, v in config.items() if k in target_args}
            clean_configs.append(clean_config)
        clean_experiment_configs_dict[key] = clean_configs

    best_configs_file = (
        Path(CONFIG.logs_output_dir)
        / CONFIG.experiment_name
        / timestamp
        / CONFIG.best_configs_file
    )

    if step == "final":
        suffix = "f"
        executor_function = partial(
            run_experiment_with_best_config, flow_function, best_configs_file
        )
    elif step == "paramsearch":
        suffix = "ps"
        executor_function = partial(run_experiment, flow_function)
    elif step == "full":
        suffix = "icv"
        executor_function = partial(run_experiment, flow_function)

    all_jobs = []

    for (
        estimator,
        dataset,
    ), experiment_configs in clean_experiment_configs_dict.items():
        logs_dir = (
            Path(CONFIG.logs_output_dir)
            / CONFIG.experiment_name
            / timestamp
            / suffix
            / estimator
            / dataset
        )
        logs_dir.mkdir(parents=True, exist_ok=True)

        executor = submitit.AutoExecutor(folder=logs_dir)
        executor_params = get_executor_params(estimator, dataset, f"_{suffix}")
        executor_params["slurm_additional_parameters"].update(additional_params)

        """
        # Old way of submitting all jobs in one array (may exceed max array size)
        executor.update_parameters(**executor_params)

        jobs = executor.map_array(executor_function, experiment_configs)
        all_jobs.extend(jobs)

        print(
            f"Submitted {len(experiment_configs)} jobs for {estimator} on {dataset} "
            f"with cluster ID {jobs[0].job_id.split('_')[0]}."
        )"""

        executor.update_parameters(**executor_params)

        #print(f"Submitting {len(experiment_configs)} jobs for {estimator} on {dataset} ")
        
        # Divide the experiment_configs into chunks of size max_concurrent_jobs and submit each chunk separately
        dataset_jobs = []
        for i in range(0, len(experiment_configs), CONFIG.max_concurrent_jobs):
            chunk = experiment_configs[i:i + CONFIG.max_concurrent_jobs]
            
            # Send the current chunk to the executor and collect the jobs
            chunk_jobs = executor.map_array(executor_function, chunk)
            dataset_jobs.extend(chunk_jobs)
            
        all_jobs.extend(dataset_jobs)

        time.sleep(5)

        # Number of arrays submitted for this dataset
        num_arrays = ((len(experiment_configs) - 1) // CONFIG.max_concurrent_jobs) + 1

        print(
            f"Submitted {len(experiment_configs)} jobs for {estimator} on {dataset} "
            f"split across {num_arrays} arrays (First ID: {dataset_jobs[0].job_id.split('_')[0]})."
        )

    print(f"All {len(all_jobs)} jobs submitted.")

    return all_jobs


def submit_aggregator(timestamp, additional_params={}):
    logs_dir = (
        Path(CONFIG.logs_output_dir) / CONFIG.experiment_name / timestamp / "aggregator"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(
        slurm_job_name=f"{CONFIG.experiment_name}_ag",
        slurm_partition="normal",
        cpus_per_task=4,
        mem_gb=4,
        slurm_time="02:00:00",
        #slurm_setup=["source /etc/profile", "module load python/3.9"],
        slurm_additional_parameters={"nice": CONFIG.nice, **additional_params},
    )

    best_configs_path = logs_dir.parent / CONFIG.best_configs_file

    job = executor.submit(
        run_aggregator, best_configs_file=best_configs_path, config_path=config_path
    )

    print(f"Aggregator job submitted with ID {job.job_id}.")

    return job


def submit_results_collector(timestamp, external_cv, additional_params={}):
    logs_dir = (
        Path(CONFIG.logs_output_dir)
        / CONFIG.experiment_name
        / timestamp
        / "results_collector"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(
        slurm_job_name=f"{CONFIG.experiment_name}_rc",
        slurm_partition="normal",
        cpus_per_task=CONFIG.collect_n_jobs,
        mem_gb=4,
        slurm_time="02:00:00",
        #slurm_setup=["source /etc/profile", "module load python/3.9"],
        slurm_additional_parameters={"nice": CONFIG.nice, **additional_params},
    )

    job = executor.submit(run_results_collector, external_cv, config_path=config_path)

    print(f"Results collector job submitted with ID {job.job_id}.")

    return job


def submit_external_cv_pipeline(
    flow_name: str, step: str = "all", dry_run: bool = False
):
    print("Submitting external_cv pipeline...")

    if flow_name not in REGISTRY:
        raise ValueError(
            f"Flow {flow_name} not found in REGISTRY. Available flows: {list(REGISTRY.keys())}"
        )

    if "external_cv" not in REGISTRY[flow_name]["supported_pipelines"]:
        raise ValueError(
            f"Flow {flow_name} does not support pipeline external_cv. "
            f"Supported pipelines: {REGISTRY[flow_name]['supported_pipelines']}"
        )

    if step not in ["paramsearch", "final", "results_collector", "all"]:
        raise ValueError(
            f"Invalid mode: {step}. Must be one of 'paramsearch', 'final',"
            " 'results_collector' or 'all'."
        )

    flow_function = REGISTRY[flow_name]["function"]

    if dry_run:
        CONFIG.dry_run = True

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    jobs = {}

    if step in ["paramsearch", "all"]:
        ps_jobs = submit_experiments(
            flow_function, step="paramsearch", timestamp=run_timestamp
        )
        jobs["paramsearch"] = ps_jobs

        dependency_ids = set([job.job_id.split("_")[0] for job in ps_jobs])
        dependency_str = ":".join(dependency_ids)
        additional_params = {"dependency": f"afterok:{dependency_str}"}
    else:
        additional_params = {}

    if step in ["final", "all"]:
        aggregator_job = submit_aggregator(
            timestamp=run_timestamp, additional_params=additional_params
        )
        jobs["aggregator"] = [aggregator_job]

        final_jobs = submit_experiments(
            flow_function,
            step="final",
            timestamp=run_timestamp,
            additional_params={"dependency": f"afterok:{aggregator_job.job_id}"},
        )
        jobs["final"] = final_jobs

    if step in ["results_collector", "all"]:
        if "final" in jobs:
            dependency_ids = set([job.job_id.split("_")[0] for job in jobs["final"]])
        else:
            dependency_ids = set()

        if dependency_ids:
            dependency_str = ":".join(dependency_ids)
            additional_params = {"dependency": f"afterok:{dependency_str}"}
        else:
            additional_params = {}

        results_collector_job = submit_results_collector(
            timestamp=run_timestamp,
            external_cv=True,
            additional_params=additional_params,
        )
        jobs["results_collector"] = [results_collector_job]

    print("external_cv pipeline submitted.")

    return jobs


def submit_internal_cv_pipeline(
    flow_name: str, step: str = "all", dry_run: bool = False
):
    print("Submitting internal_cv pipeline...")

    if flow_name not in REGISTRY:
        raise ValueError(
            f"Flow {flow_name} not found in REGISTRY. Available flows: {list(REGISTRY.keys())}"
        )

    if "internal_cv" not in REGISTRY[flow_name]["supported_pipelines"]:
        raise ValueError(
            f"Flow {flow_name} does not support pipeline internal_cv. "
            f"Supported pipelines: {REGISTRY[flow_name]['supported_pipelines']}"
        )

    if step not in ["full", "results_collector", "all"]:
        raise ValueError(
            f"Invalid mode: {step}. Must be one of 'full', 'results_collector' or 'all'."
        )

    flow_function = REGISTRY[flow_name]["function"]

    if dry_run:
        CONFIG.dry_run = True

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    jobs = {}

    if step in ["full", "all"]:
        internal_cv_jobs = submit_experiments(
            flow_function, step="full", timestamp=run_timestamp
        )
        jobs["full"] = internal_cv_jobs

        dependency_ids = set([job.job_id.split("_")[0] for job in internal_cv_jobs])
        dependency_str = ":".join(dependency_ids)
        additional_params = {"dependency": f"afterok:{dependency_str}"}
    else:
        additional_params = {}

    if step in ["results_collector", "all"]:
        results_collector_job = submit_results_collector(
            timestamp=run_timestamp,
            external_cv=False,
            additional_params=additional_params,
        )
        jobs["results_collector"] = [results_collector_job]

    print("internal_cv pipeline submitted.")

    return jobs
