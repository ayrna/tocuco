import pickle

import numpy as np

from execution.configs import load_config
from execution.results import collect_results


def find_best_estimator_configs(
    results_dir, val_metric, greater_is_better, estimators, datasets
):
    from remayn.result_set import ResultFolder

    from experiments.utils import compute_metrics

    def filter_fn(result):
        if not result.config["estimator_name"] in estimators:
            return False

        if not result.config["dataset"] in datasets:
            return False

        if (result.config["n_folds"] is None or result.config["n_folds"] <= 1) and (
            result.config["val_size"] is None or result.config["val_size"] <= 0
        ):
            return False

        return True

    results = ResultFolder(results_dir)

    df = results.create_dataframe(
        config_columns=[
            "estimator_name",
            "dataset",
            "rs",
            "estimator_config",
            "meta_config",
        ],
        best_params_columns=["max_iter"],
        metrics_fn=compute_metrics,
        filter_fn=filter_fn,
        include_train=False,
        include_val=True,
        config_columns_prefix="",
        n_jobs=4,
    )

    if greater_is_better:
        grouped_df = df.loc[
            df.groupby(["estimator_name", "dataset", "rs"])[
                f"val_{val_metric}"
            ].idxmax()
        ]
    else:
        grouped_df = df.loc[
            df.groupby(["estimator_name", "dataset", "rs"])[
                f"val_{val_metric}"
            ].idxmin()
        ]

    configs = {}
    for idx, row in grouped_df.iterrows():
        estimator_name = row["estimator_name"]
        dataset = row["dataset"]
        rs = row["rs"]
        estimator_config = row["estimator_config"]
        meta_config = row["meta_config"]

        if type(estimator_config) == dict:
            if type(meta_config) == dict:
                # Add meta config with meta__ prefix
                for key, value in meta_config.items():
                    estimator_config[f"meta__{key}"] = value
            else:
                print(
                    f"Warning: meta_config is not a dict for {estimator_name},"
                    f" {dataset} and {rs}. Skipping meta config."
                )
        else:
            print(
                f"Warning: estimator_config is not a dict for {estimator_name},"
                f" {dataset} and {rs}. Skipping this config."
            )
            estimator_config = {}

        # Replace max_iter with the number of epochs of early stopping
        if (
            "best_max_iter" in row
            and not row["best_max_iter"] is None
            and not np.isnan(row["best_max_iter"])
        ):
            estimator_config["max_iter"] = int(row["best_max_iter"])

        configs[(estimator_name, dataset, rs)] = estimator_config

    return configs


def run_experiment(flow_function, config_dict):
    return flow_function(**config_dict)


def run_aggregator(best_configs_file, config_path):
    CONFIG = load_config(config_path=config_path)
    best_configs = find_best_estimator_configs(
        CONFIG.results_dir,
        CONFIG.val_metric,
        CONFIG.greater_is_better,
        CONFIG.estimators,
        CONFIG.datasets,
    )

    with open(best_configs_file, "wb") as f:
        pickle.dump(best_configs, f)

    print(f"Best configs saved to {best_configs_file}")


def run_experiment_with_best_config(flow_function, best_configs_file, config_dict):
    with open(best_configs_file, "rb") as f:
        best_configs = pickle.load(f)

    key = (config_dict["estimator_name"], config_dict["dataset"], config_dict["seed"])

    if (
        "estimator_config" in config_dict
        and config_dict["estimator_config"] is not None
    ):
        print(
            f"[WARNING] Config for {key} already has an estimator_config. "
            "Overriding it with the best config from the aggregator."
        )

    if key in best_configs:
        clean_config = {}
        raw_config = best_configs[key]
        
        black_list = [
            "n_iter", "param_distributions", "param_grid", "estimator", "cv", 
            "n_jobs", "pre_dispatch", "random_state", "refit", "return_train_score",
            "scoring", "verbose", "error_score"
        ]

        for k, v in raw_config.items():

            if k.startswith("estimator__"):
                clean_key = k.replace("estimator__", "")
                if clean_key not in black_list:
                    clean_config[clean_key] = v
            else:
                if k not in black_list:
                    clean_config[k] = v
                else:
                    print(f"[INFO] Skipping {k} as it is in the blacklist.")
                
        config_dict["estimator_config"] = clean_config
        return run_experiment(flow_function, config_dict)

    else:
        print(
            f"[WARNING] No best config found for {config_dict['estimator_name']},"
            f" {config_dict['dataset']} and seed {config_dict['seed']}. Skipping"
            f" this run."
        )
        return None


def run_results_collector(external_cv, config_path):
    CONFIG = load_config(config_path=config_path)

    print(CONFIG.best_params_columns_to_include)
    
    collect_results(
        CONFIG.results_dir,
        output_path=CONFIG.prepared_results_dir,
        appendix=CONFIG.prepared_results_appendix,
        methods=CONFIG.collect_methods,
        datasets=CONFIG.collect_datasets,
        seeds=CONFIG.collect_seeds,
        config_columns_to_include=CONFIG.config_columns_to_include,
        best_params_columns_to_include=CONFIG.best_params_columns_to_include,
        include_train=CONFIG.collect_train,
        include_val=CONFIG.collect_val,
        skip_zip=CONFIG.skip_zip,
        n_jobs=CONFIG.collect_n_jobs,
        external_cv=external_cv,
    )
