import json
import time
from os import environ
from pathlib import Path
from random import seed as random_seed

import numpy as np
from remayn.result import make_result
from remayn.result_set import ResultFolder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from experiments.utils.estimators_internal import get_estimator
from experiments.utils.tocuco import load_tocuco_dataset
from experiments.utils import (
    compute_metrics,
    create_dataset_resample,
    load_data,
)

from execution.results import export_tocuco_seed_results


def run_ml_flow(
    *,
    data_dir,
    dataset,
    n_folds=None,
    val_size=None,
    search_n_iter=None,
    cv_scoring="amae",
    results_dir="./results",
    results_tocuco_dir="./results_tocuco",
    estimator_name="mlp",
    seed=0,
    interactive=False,
    n_jobs=1,
    dry_run=False,
    use_gpu_if_available=True,
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
                    val_size=val_size,
                    search_n_iter=search_n_iter,
                ),
                indent=4,
            )
        )
        return

    if n_folds is not None and n_folds > 1 and val_size is not None and val_size > 0:
        raise ValueError("Only one of n_folds and val_size can be set")

    # Fix seeds
    np.random.seed(seed)
    random_seed(seed)
    environ["PYTHONHASHSEED"] = str(seed)

    #if not dataset.startswith("tocuco_"):
    #    raise ValueError(f"Only tocuco datasets are supported, but got {dataset}")

    num_classes = 0
    X_train = X_test = y_train = y_test = None

    if dataset.startswith("tocuco_"):
        tocuco_path = Path(data_dir) / "TOCUCO"
        X_train, X_test, y_train, y_test, dataset_name, seed = load_tocuco_dataset(
            tocuco_path, dataset.replace("tocuco_", ""), seed=seed
        )

        unique_classes = np.unique(y_train)
        num_classes = unique_classes.size
    else:

        train_dataset = load_data(
            data_dir=data_dir, dataset=dataset, partition="train", seed=seed
        )
        val_dataset = None
        test_dataset = load_data(
            data_dir=data_dir, dataset=dataset, partition="test", seed=seed
        )

        import torch
        from torch.utils.data import DataLoader

        def extract_X_y(dataset, batch_size=64):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            X_all, y_all = [], []
            
            for X_batch, y_batch in loader:
                X_all.append(X_batch)
                y_all.append(y_batch)
                
            return torch.cat(X_all).numpy(), torch.cat(y_all).numpy()

        if seed != 0:
            train_dataset, test_dataset = create_dataset_resample(
                train_dataset, test_dataset, random_state=seed
            )

        X_train, y_train = extract_X_y(train_dataset)
        X_test, y_test = extract_X_y(test_dataset)

        num_classes = len(train_dataset.classes)


    if num_classes < 2:
        raise ValueError(
            f"Number of classes could not be inferred. "
            f"Expected at least 2 classes, but found {num_classes}."
        )
    
    estimator, param_grid = get_estimator(estimator_name, num_classes,random_state=seed,use_gpu_if_available=use_gpu_if_available)

    experiment_config = get_experiment_config(
        estimator,
        estimator_name,
        dataset,
        seed,
        0,
        n_folds,
        val_size,
        search_n_iter,
    )

    print("Running experiment with config:")
    print(json.dumps(experiment_config, indent=4))

    if not interactive:
        results = ResultFolder(results_dir)
        if experiment_config in results:
            print("Experiment already run")
            return

    from experiments.utils.scorers import get_scorer
    int_val_size = int(val_size * 10) if val_size is not None and val_size > 0 else None
    cv = int_val_size if int_val_size is not None else n_folds
    
    if search_n_iter is not None and search_n_iter > 0:
        estimator = RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=search_n_iter,
            scoring=get_scorer(cv_scoring),
            n_jobs=n_jobs,
            verbose=3,
            random_state=seed,
            cv=cv,
        )
    else:
        estimator = GridSearchCV(
            estimator,
            param_grid,
            verbose=3,
            scoring=get_scorer(cv_scoring),
            n_jobs=n_jobs,
            cv=cv,
        )

    start = int(round(time.time() * 1000))
    estimator.fit(X_train, y_train)

    train_probs = estimator.predict_proba(X_train)
    test_probs = estimator.predict_proba(X_test)

    total_time = int(round(time.time() * 1000)) - start

    experiment_config = get_experiment_config(
        estimator,
        estimator_name,
        dataset,
        seed,
        0,
        n_folds,
        val_size,
        search_n_iter,
    )

    train_history = getattr(estimator, "train_history", None)
    val_history = getattr(estimator, "valid_history", None)

    if (
        export_tocuco_results
        and dataset.startswith("tocuco_")
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
            train_targets=np.array(y_train),
            train_outputs=train_probs,
            test_targets=np.array(y_test),
            test_outputs=test_probs,
            estimator_config=experiment_config["estimator_config"],
        )

    if not interactive:
        result = make_result(
            base_path=results_dir,
            config=experiment_config,
            predictions=np.array(test_probs),
            targets=np.array(y_test),
            train_predictions=np.array(train_probs),
            train_targets=np.array(y_train),
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

        train_metrics = compute_metrics(y_train, train_probs)
        print("train_metrics")
        print(json.dumps(train_metrics, indent=4))

        test_metrics = compute_metrics(y_test, test_probs)
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
    val_size,
    search_n_iter,
):
    config = {}

    if estimator is not None:
        config["estimator_config"] = estimator.get_params().copy()
    else:
        config["estimator_config"] = {}

    config["estimator_name"] = estimator_name
    config["dataset"] = dataset
    config["rs"] = rs
    config["resample_id"] = resample_id
    config["n_folds"] = n_folds
    config["val_size"] = val_size
    config["search_n_iter"] = search_n_iter

    estimator_params_to_remove = [
        "num_classes",
        "train_targets",
        "val_dataset",
        "verbose",
        "n_jobs",
        "batch_size",
        "estimator",  # estimator from paramsearch
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
