import json
import time
from os import environ
from random import seed as random_seed

import numpy as np
from remayn.result import make_result
from remayn.result_set import ResultFolder
from torch import cuda, manual_seed, use_deterministic_algorithms

from execution.paramsearch import get_gridsearch_params
from experiments.utils import (
    compute_metrics,
    create_dataset_resample,
    get_dataset_fold,
    get_dataset_holdout,
    load_data,
)
from experiments.utils.estimators_external import get_estimator, get_estimator_config

from execution.results import export_tocuco_seed_results


def run_dl_flow(
    *,
    data_dir,
    dataset,
    n_folds=None,
    val_size=None,
    fold=None,
    results_dir="./results",
    results_tocuco_dir="./results_tocuco",
    estimator_name="resnet18classifier",
    estimator_config={},
    batch_size=128,
    seed=0,
    interactive=False,
    n_jobs=1,
    search_n_iter=None,
    use_gpu_if_available=True,
    dry_run=False,
    cv_scoring="amae",
    export_tocuco_results=False,
):
    if dry_run:
        print("Dry run - experiment configuration:")
        print(
            json.dumps(
                get_experiment_config(
                    estimator=None,
                    estimator_name=estimator_name,
                    dataset=dataset,
                    rs=seed,
                    resample_id=0,
                    n_folds=n_folds,
                    fold=fold,
                    val_size=val_size,
                    search_n_iter=search_n_iter,
                    custom_estimator_config=estimator_config,
                ),
                indent=4,
            )
        )
        return


    if n_folds is not None and n_folds > 1 and val_size is not None and val_size > 0:
        raise ValueError("Only one of n_folds and val_size can be set")

    if search_n_iter is not None and n_folds is None and (val_size is None or val_size == 0):
        if fold is not None:
            from execution.config import CONFIG_EXP

            n_folds = getattr(CONFIG_EXP, "n_folds", 3)
        else:
            raise ValueError(
                "search_n_iter requires either n_folds, val_size or an active fold to be set for validation"
            )

    if search_n_iter is not None and search_n_iter != "gridcv" and n_jobs != 1:
        raise ValueError(
            "search_n_iter should be used with n_jobs=1 to avoid issues with RandomizedSearchCV."
        )

    # Fix seeds
    np.random.seed(seed)
    manual_seed(seed)
    random_seed(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    use_deterministic_algorithms(True, warn_only=True)

    train_dataset = load_data(
        data_dir=data_dir, dataset=dataset, partition="train", seed=seed
    )
    val_dataset = None
    test_dataset = load_data(
        data_dir=data_dir, dataset=dataset, partition="test", seed=seed
    )

    if seed != 0 and not dataset.startswith("tocuco_"):
        train_dataset, test_dataset = create_dataset_resample(
            train_dataset, test_dataset, random_state=seed
        )

    print(
        f"Loaded {dataset} dataset with {len(train_dataset)} train samples and "
        f"{len(test_dataset)} test samples."
    )

    if hasattr(train_dataset, "classes"):
        num_classes = len(train_dataset.classes)
    else:
        raise ValueError("Number of classes could not be inferred from dataset")


    if n_folds is not None and n_folds > 1:
        print(f"N_folds: {n_folds}, fold: {fold}")
        train_dataset, val_dataset = get_dataset_fold(
            train_dataset, fold=fold, n_folds=n_folds, random_state=seed
        )
        print(
            f"Using {n_folds}-fold cross-validation with fold {fold}.\n"
            f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}"
        )

    if val_size is not None and val_size > 0:
        train_dataset, val_dataset = get_dataset_holdout(
            train_dataset, test_size=val_size, random_state=seed
        )
        print(
            f"Using holdout validation with val_size {val_size}.\n"
            f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}"
        )


    # if estimator_config is not set, use the first gridsearch config
    if estimator_config is None or estimator_config == {}:

        base_config, param_grid = get_estimator_config(estimator_name)
        estimator_configs = get_gridsearch_params(param_grid)
        estimator_config = {**base_config, **estimator_configs[0]}


    # Shuffled targets. Passed to get_estimator for computing class weights
    train_targets = train_dataset.targets

    # Used to send it to get_estimator
    prior_experiment_config = get_experiment_config(
        estimator=None,
        estimator_name=estimator_name,
        dataset=dataset,
        rs=seed,
        resample_id=0,
        n_folds=n_folds,
        fold=fold,
        val_size=val_size,
        search_n_iter=search_n_iter,
        custom_estimator_config=estimator_config,
    )

    print(f"Estimator config: {estimator_config}")
    estimator = get_estimator(
        estimator_name,
        estimator_config,
        num_classes=num_classes,
        train_dataset=train_dataset,
        val_dataset=val_dataset,  # slows down training process
        random_state=seed,
        n_jobs=n_jobs,
        batch_size=batch_size,
        use_gpu_if_available=use_gpu_if_available,
        interactive=interactive,
        experiment_config=prior_experiment_config,
    )

    if search_n_iter is not None:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        base_config, param_grid = get_estimator_config(estimator_name)
        cv = n_folds if (n_folds is not None and n_folds > 0)else int(val_size*10)
        print(f"Using {cv} folds for hyperparameter search with {search_n_iter} iterations.")
        from experiments.utils.scorers import get_scorer

        estimator.verbose = 0
        if search_n_iter is None or search_n_iter == 0:
            estimator = GridSearchCV(
                estimator,
                param_grid=param_grid,
                scoring=get_scorer(cv_scoring),
                cv=cv,
                verbose=3,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )
        else:
            estimator = RandomizedSearchCV(
                estimator,
                param_distributions=param_grid,
                n_iter=search_n_iter,
                scoring=get_scorer(cv_scoring),
                cv=cv,
                verbose=3,
                n_jobs=n_jobs,
                random_state=seed,
                refit=True,  # refit the best model on the whole training set
                error_score="raise",
            )


    print(f"Estimator used: {estimator.__class__.__name__}")

    experiment_config = get_experiment_config(
        estimator,
        estimator_name,
        dataset,
        seed,
        0,
        n_folds,
        fold,
        val_size,
        search_n_iter,
        estimator_config,
    )

    print("Running experiment with config (including estimator config):")
    print(json.dumps(experiment_config, indent=4))

    if not interactive:
        results = ResultFolder(results_dir)
        if experiment_config in results:
            print("Experiment already run")
            return

    if cuda.is_available():
        gpu_properties = cuda.get_device_properties()
        print("GPU available")
        print(f"GPU properties: {gpu_properties}")
    else:
        print("GPU not available. Using CPU instead.")

    start = int(round(time.time() * 1000))

    if hasattr(train_dataset, "data") and hasattr(train_dataset, "targets"):
        estimator.fit(train_dataset.data, train_dataset.targets)
    else:
        from skorch.helper import SliceDataset
        # Images dataset
        estimator.fit(SliceDataset(train_dataset, idx=0), y=np.array(train_dataset.targets))

    train_probs = estimator.predict_proba(train_dataset)
    test_probs = estimator.predict_proba(test_dataset)

    if val_dataset is not None:
        val_probs = estimator.predict_proba(val_dataset)
        val_targets = val_dataset.targets
    else:
        val_probs = None
        val_targets = None

    train_targets = train_dataset.targets
    test_targets = test_dataset.targets

    total_time = int(round(time.time() * 1000)) - start

    experiment_config = get_experiment_config(
        estimator,
        estimator_name,
        dataset,
        seed,
        0,
        n_folds,
        fold,
        val_size,
        search_n_iter,
        estimator_config,
    )

    train_history = getattr(estimator, "train_history", None)
    val_history = getattr(estimator, "valid_history", None)

    if (
        export_tocuco_results
        and dataset.startswith("tocuco_")
        and fold is None
        and (
            search_n_iter
            is not None  # training_with_paramsearch: CV handled internally by RandomizedSearchCV
            or ((n_folds is None or n_folds <= 1) and (val_size is None or val_size <= 0))
        )
    ):
        export_tocuco_seed_results(
            data_dir=data_dir,
            results_root=results_tocuco_dir,
            estimator_name=estimator_name,
            dataset=dataset,
            seed=seed,
            train_targets=train_targets,
            train_outputs=train_probs,
            test_targets=test_targets,
            test_outputs=test_probs,
            estimator_config=estimator_config,
        )

    if not interactive:
        result = make_result(
            base_path=results_dir,
            config=experiment_config,
            predictions=np.array(test_probs),
            targets=np.array(test_targets),
            train_predictions=np.array(train_probs),
            train_targets=np.array(train_targets),
            val_predictions=np.array(val_probs),
            val_targets=np.array(val_targets),
            time=total_time,
            best_params=getattr(estimator, "best_params_", None),
            best_model=None,
            train_history=np.array(train_history),
            val_history=np.array(val_history),
        )
        print("Best params:")
        print(getattr(estimator, "best_params_", None))
        print(f"Save in: {results_dir}")
        result.save()
    else:
        best_params = getattr(estimator, "best_params_", None)
        if best_params is not None:
            print("best_params")
            print(json.dumps(best_params, indent=4))

        train_metrics = compute_metrics(train_targets, train_probs)
        print("train_metrics")
        print(json.dumps(train_metrics, indent=4))

        if val_probs is not None:
            val_metrics = compute_metrics(val_targets, val_probs)
            print("val_metrics")
            print(json.dumps(val_metrics, indent=4))

        test_metrics = compute_metrics(test_targets, test_probs)
        print("test_metrics")
        print(json.dumps(test_metrics, indent=4))

        if train_history is not None:
            print("train_history")
            print(train_history)

        if val_history is not None:
            print("val_history")
            print(val_history)


def get_experiment_config(
    estimator,
    estimator_name,
    dataset,
    rs,
    resample_id,
    n_folds,
    fold,
    val_size,
    search_n_iter,
    custom_estimator_config,
):
    config = {}

    if estimator is not None:
        config["estimator_config"] = estimator.get_params().copy()
    elif custom_estimator_config is not None:
        config["estimator_config"] = custom_estimator_config.copy()
    else:
        config["estimator_config"] = {}

    config["estimator_name"] = estimator_name
    config["search_n_iter"] = search_n_iter
    config["dataset"] = dataset
    config["rs"] = rs
    config["resample_id"] = resample_id
    config["n_folds"] = n_folds
    config["fold"] = fold
    config["val_size"] = val_size

    estimator_params_to_remove = [
        "num_classes",
        "train_targets",
        "val_dataset",
        "verbose",
        "n_jobs",
        "batch_size",
        "estimator",  # estimator from paramsearch
        "cv",
        "error_score"
    ]

    for param in estimator_params_to_remove:
        if param in config["estimator_config"]:
            del config["estimator_config"][param]
        if f"estimator__{param}" in config["estimator_config"]:
            del config["estimator_config"][f"estimator__{param}"]

    if "scoring" in config["estimator_config"]:
        config["estimator_config"]["scoring"] = str(
            config["estimator_config"]["scoring"]
        )

    return config
