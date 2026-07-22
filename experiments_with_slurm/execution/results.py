import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union
from joblib import parallel_backend
from shutil import make_archive

import csv
import errno
import json
import os
import time

from remayn.result_set import ResultFolder
from remayn.report import create_excel_columns_report, create_excel_summary_report
from experiments.utils import compute_metrics


def collect_results(
    results_path: Union[Path, str],
    output_path: Union[Path, str] = "prepared_results",
    appendix: str = "",
    methods: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    config_columns_to_include: Optional[List[str]] = None,
    best_params_columns_to_include: Optional[List[str]] = None,
    include_train: bool = True,
    include_val: bool = False,
    skip_zip: bool = False,
    n_jobs: int = 1,
    external_cv: bool = True,
):

    if config_columns_to_include is None:
        config_columns_to_include = [
            "dataset",
            "estimator_name",
            "rs",
            "estimator_config.learning_rate",
            "estimator_config.alpha",
            "estimator_config.ftype",
            "estimator_config.max_iter",
            "estimator_config.link_function",
            "estimator_config.penalization_type",
            "estimator_config.min_distance",
            "estimator_config.hidden_units",
        ]

    if best_params_columns_to_include is None:
        best_params_columns_to_include = []

    if seeds is None:
        from execution.config import CONFIG_EXP
            
        valid_seeds = list(range(getattr(CONFIG_EXP, "seeds", [0])))
    else:
        valid_seeds = seeds


    def filter_fn(result):
        if methods is not None and result.config["estimator_name"] not in methods:
            return False

        if datasets is not None and result.config["dataset"] not in datasets:
            return False

        if result.config["rs"] not in valid_seeds:
            return False

        if external_cv:
            if (
                "search_n_iter" not in result.config
                or result.config["search_n_iter"] is None
            ):
                if (
                    result.config["n_folds"] is not None
                    and result.config["n_folds"] > 1
                ):
                    return False
                if (
                    result.config["val_size"] is not None
                    and result.config["val_size"] > 0
                ):
                    return False

        return True

    results_path = Path(results_path)
    results = ResultFolder(results_path)

    with parallel_backend("multiprocessing"):
        df = results.create_dataframe(
            config_columns=config_columns_to_include,
            best_params_columns=best_params_columns_to_include,
            filter_fn=filter_fn,
            metrics_fn=compute_metrics,
            include_train=include_train,
            include_val=include_val,
            config_columns_prefix="",
            n_jobs=n_jobs,
        )

    df.sort_values(by=["dataset", "estimator_name", "rs"], inplace=True)

    pivot_key_columns = ["rs", "estimator_name", "dataset"]
    duplicate_mask = df.duplicated(subset=pivot_key_columns, keep=False)
    if duplicate_mask.any():
        duplicated_rows = df[duplicate_mask].copy()
        duplicate_groups = duplicated_rows.groupby(pivot_key_columns).size().shape[0]
        print(f"Found {len(duplicated_rows)} duplicated rows across {duplicate_groups} groups.")

        if "test_AMAE" in df.columns:
            df = (
                df.sort_values(by=["test_AMAE"], ascending=True)
                .drop_duplicates(subset=pivot_key_columns, keep="first")
                .sort_values(by=["dataset", "estimator_name", "rs"])
            )
        else:
            df = df.drop_duplicates(subset=pivot_key_columns, keep="first").sort_values(
                by=["dataset", "estimator_name", "rs"]
            )
        print(f"After deduplication: {len(df)} rows remain.")

    if df.duplicated(subset=pivot_key_columns, keep=False).any():
        raise ValueError("Still found duplicated rows after deduplication for (rs, estimator_name, dataset).")

    expected_seeds = len(valid_seeds)
    counts_per_group = (
        df.groupby(["dataset", "estimator_name"])["rs"].nunique().reset_index(name="n_seeds")
    )
    invalid_counts = counts_per_group[counts_per_group["n_seeds"] != expected_seeds]
    if not invalid_counts.empty:
        print("Invalid seed counts found (showing up to 20 rows):")
        print(invalid_counts.sort_values(by=["dataset", "estimator_name"]).head(20))
        raise ValueError(
            f"Each (dataset, estimator_name) must have exactly {expected_seeds} seeds. "
            f"Found {len(invalid_counts)} invalid groups."
        )

    if len(appendix) > 0 and not appendix.startswith("_"):
        appendix = f"_{appendix}"

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_wo_ext = (
        output_path / f'{datetime.now().strftime(r"%Y%m%d_%H%M%S")}{appendix}'
    )
    full_excel_path = f"{output_file_wo_ext}.xlsx"

    metrics_example = compute_metrics([1], [[1.0, 2.0]])
    metric_columns = list(metrics_example.keys())
    group_columns = ["dataset", "estimator_name"]

    with pd.ExcelWriter(full_excel_path, mode="w") as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        
        create_excel_summary_report(
            df, full_excel_path, group_columns, excel_writer=writer
        )
        create_excel_columns_report(
            df,
            full_excel_path,
            metric_columns=metric_columns,
            pivot_index="rs",
            pivot_columns=["estimator_name", "dataset"],
            excel_writer=writer,
        )

    print(f"Excel multi-report generated successfully at: {full_excel_path}")

    if not skip_zip:
        make_archive(str(output_file_wo_ext), "zip", str(results_path))
        print(f"Source results folder compressed into: {output_file_wo_ext}.zip")


def export_tocuco_seed_results(
    *,
    data_dir,
    results_root,
    estimator_name,
    dataset,
    seed,
    train_targets,
    train_outputs,
    test_targets,
    test_outputs,
    estimator_config,
):
    """Main entry point for the TOCUCO export.

    For a specific combination of model, dataset, and seed, it generates
    the entire folder and file structure under results_root:

        results_root/
        <estimator_name>/
            <dataset_name>/
                            predictions_by_seed/
                seed_<N>/
                train_predictions.csv
                test_predictions.csv
                                    train_confusion_matrix.txt
                                    test_confusion_matrix.txt
            hyperparameter_configuration.csv

    This function is safe for parallel runs on Condor: the
    shared file across seeds (hyperparameter_configuration.csv)
    is written with file locking (flock) to prevent race conditions.

    Args:
        data_dir: Path to the root data directory (must contain TOCUCO/).
        results_root: Root path where the results_tocuco folder will be created.
        estimator_name: Name of the estimator (e.g. 'ridgeclassifier').
        dataset: Full name of the dataset with the tocuco_ prefix
                (e.g. 'tocuco_dr09_housing').
        seed: Seed number (integer).
        train_targets: Array of true training labels.
        train_outputs: Model outputs on training data. Can be a 
                    2D array of probabilities/scores or a 1D array of 
                    direct predictions.
        test_targets: Array of true test labels.
        test_outputs: Model outputs on test data (same format as 
                    train_outputs).
        estimator_config: Dictionary containing the hyperparameter configuration
                        selected for this estimator and seed.
    """
    from sklearn.metrics import confusion_matrix

    dataset_name = dataset.replace("tocuco_", "", 1)
    model_dataset_dir = Path(results_root) / estimator_name / dataset_name
    seed_dir = model_dataset_dir / "predictions_by_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_pattern_ids, test_pattern_ids = _get_tocuco_pattern_ids(
        data_dir=data_dir,
        dataset_name=dataset_name,
        seed=seed,
        expected_train_len=len(train_targets),
        expected_test_len=len(test_targets),
    )

    train_predictions, train_probabilities = _extract_predictions_and_probabilities(
        train_outputs
    )
    test_predictions, test_probabilities = _extract_predictions_and_probabilities(test_outputs)

    _save_predictions_csv(
        file_path=seed_dir / "train_predictions.csv",
        pattern_ids=train_pattern_ids,
        targets=train_targets,
        predictions=train_predictions,
        probabilities=train_probabilities,
    )
    _save_predictions_csv(
        file_path=seed_dir / "test_predictions.csv",
        pattern_ids=test_pattern_ids,
        targets=test_targets,
        predictions=test_predictions,
        probabilities=test_probabilities,
    )

    all_labels = np.unique(
        np.concatenate(
            [
                np.array(train_targets),
                np.array(test_targets),
                np.array(train_predictions),
                np.array(test_predictions),
            ]
        )
    )
    train_cm = confusion_matrix(train_targets, train_predictions, labels=all_labels)
    test_cm = confusion_matrix(test_targets, test_predictions, labels=all_labels)

    _save_single_seed_confusion_matrix(
        file_path=seed_dir / "train_confusion_matrix.txt",
        seed=seed,
        matrix=train_cm,
    )
    _save_single_seed_confusion_matrix(
        file_path=seed_dir / "test_confusion_matrix.txt",
        seed=seed,
        matrix=test_cm,
    )

    _upsert_hyperparameter_configuration(
        model_dataset_dir / "hyperparameter_configuration.csv",
        seed,
        estimator_config if estimator_config is not None else {},
    )

def _extract_predictions_and_probabilities(outputs):
    """Separa predicciones de clase y probabilidades a partir de las salidas del modelo.

    Args:
        outputs: Array de salidas del modelo. Si es 2D se interpreta como
                 scores o probabilidades por clase; si es 1D se interpreta
                 como predicciones directas de clase.

    Returns:
        Tupla (predictions, probabilities) donde:
        - predictions: array 1D de enteros con la clase predicha.
        - probabilities: array 2D normalizado si outputs era 2D, o None
          si outputs era 1D.
    """
    outputs = np.array(outputs)
    if outputs.ndim > 1:
        probabilities = _normalize_probabilities(outputs)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities

    predictions = outputs.astype(int)
    return predictions, None


def _normalize_probabilities(raw_scores):
    """ Ensures that each row of raw_scores is a valid probability distribution.

    If the rows are already in the [0, 1] range and sum to 1 (with a 
    1e-6 tolerance), returns the array unmodified. Otherwise, applies 
    numerically stable softmax (subtracting the row-wise maximum before exp).

    Args:
        raw_scores: 2D array of shape (n_samples, n_classes).

    Returns:
        2D array of the same shape where each row sums to exactly 1.
    """
    row_sums = np.sum(raw_scores, axis=1)
    in_range = np.all((raw_scores >= -1e-8) & (raw_scores <= 1 + 1e-8))
    sums_to_one = np.allclose(row_sums, 1.0, atol=1e-6)
    if in_range and sums_to_one:
        return raw_scores

    shifted = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _format_probability_row(probability_row):
    """Serializes a probability vector as a string like '[0.1234567890, ...]'.

    Format compatible with the 'Prediction probabilities' column of the CSV.
    Each value is printed with 10 decimal places.

    Args:
        probability_row: Iterable of floats with the probabilities for each class.

    Returns:
        String with the format '[p0, p1, ..., pK]'.
    """
    formatted = ", ".join(f"{float(p):.10f}" for p in probability_row)
    return f"[{formatted}]"


def _save_predictions_csv(*, file_path, pattern_ids, targets, predictions, probabilities):
    """Writes a predictions CSV for a specific seed.

    Generated columns:
    - 'Pattern ID': global index of the pattern in the full dataset
      (position in the TOCUCO mask array).
    - 'Target': true label (expected class).
    - 'Prediction probabilities': vector of probabilities in brackets,
      present only if the model returns probabilities.
    - 'Predicition': predicted class (argmax of probabilities if available,
      or the direct model output otherwise).

    The file is always created fresh (overwrites if it already existed for that
    seed, which is safe because each seed writes to its own
    predictions_by_seed/seed_N/ subfolder).

    Args:
        file_path: Full path to the output CSV file.
        pattern_ids: Array with the global indices of the patterns.
        targets: Array with the true labels.
        predictions: Array 1D with the predicted classes.
        probabilities: 2D probability array, or None if the model does not
                       provide them.
    """
    file_path = Path(file_path)
    with file_path.open("w", newline="") as f:
        if probabilities is not None:
            fieldnames = ["Pattern ID", "Target", "Prediction probabilities", "Predicition"]
        else:
            fieldnames = ["Pattern ID", "Target", "Predicition"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, pattern_id in enumerate(pattern_ids):
            row = {
                "Pattern ID": int(pattern_id),
                "Target": int(targets[idx]),
                "Predicition": int(predictions[idx]),
            }
            if probabilities is not None:
                row["Prediction probabilities"] = _format_probability_row(probabilities[idx])
            writer.writerow(row)


def _get_tocuco_pattern_ids(
    *, data_dir, dataset_name, seed, expected_train_len, expected_test_len
):
    """Gets the global indices of the training and test patterns for a seed.

    Reads the boolean mask stored in TOCUCO/train_masks.pkl (or
    train_masks.json as a fallback) under the key '<dataset_name>_seed_<seed>'.
    True indices correspond to training patterns and False
    to test patterns.

    If the mask is not found, it emits a WARNING and returns sequential
    indices (0, 1, 2, ...) as a safe fallback.

    Args:
        data_dir: Path to the root data directory (must contain TOCUCO/).
        dataset_name: Name of the dataset without the 'tocuco_' prefix
                      (e.g. 'dr09_housing').
        seed: Seed number (integer).
        expected_train_len: Expected length of the training set,
                            used to validate that the mask matches.
        expected_test_len: Expected length of the test set.

    Returns:
        Tuple (train_ids, test_ids) with integer index arrays.

    Raises:
        ValueError: If the mask is found but its number of True/False values
                    does not match expected_train_len / expected_test_len.
    """
    from pathlib import Path

    tocuco_dir = Path(data_dir) / "TOCUCO"
    mask_key = f"{dataset_name}_seed_{seed}"
    train_mask = None

    pkl_path = tocuco_dir / "train_masks.pkl"
    if pkl_path.exists():
        import joblib

        with pkl_path.open("rb") as f:
            masks = joblib.load(f)
        if mask_key in masks:
            train_mask = np.array(masks[mask_key], dtype=bool)

    if train_mask is None:
        json_path = tocuco_dir / "train_masks.json"
        if json_path.exists():
            with json_path.open("r") as f:
                masks = json.load(f)
            if mask_key in masks:
                train_mask = np.array(masks[mask_key], dtype=bool)

    if train_mask is None:
        print(
            f"[WARNING] Could not find mask '{mask_key}' in {tocuco_dir}. "
            "Pattern IDs will be sequential in each partition."
        )
        return np.arange(expected_train_len), np.arange(expected_test_len)

    train_ids = np.where(train_mask)[0]
    test_ids = np.where(~train_mask)[0]

    if len(train_ids) != expected_train_len or len(test_ids) != expected_test_len:
        raise ValueError(
            "Mismatch between TOCUCO mask lengths and loaded dataset lengths for "
            f"{dataset_name} seed {seed}: "
            f"mask train/test=({len(train_ids)}, {len(test_ids)}), "
            f"dataset train/test=({expected_train_len}, {expected_test_len})"
        )

    return train_ids, test_ids


def _save_single_seed_confusion_matrix(*, file_path, seed, matrix):
    """Saves the confusion matrix of a seed into an individual file.

    Format:
        Seed <N>
        =====================
        [[...confusion matrix...]]
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"Seed {seed}\n" f"=====================\n" f"{np.array2string(matrix)}\n"
    file_path.write_text(content)


def _with_file_lock(target_path):
    """Acquires a lock via exclusive creation of a .lock file.

    This approach avoids ``flock`` issues on network file systems
    (e.g., NFS), where ``OSError: [Errno 116] Stale file handle`` can occur.

    The lock is acquired by creating ``<target_path>.lock`` with ``O_EXCL``. If it
    already exists, it waits and retries until a timeout is reached. It also attempts
    to recover old orphaned locks.

    Args:
        target_path: Path of the data file to protect.

    Returns:
        Path of the created lock file.
    """
    lock_path = Path(f"{target_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    timeout_seconds = 120
    poll_seconds = 0.1
    stale_lock_seconds = 1800
    deadline = time.time() + timeout_seconds

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"pid={os.getpid()} time={time.time()}\n")
            return lock_path
        except FileExistsError:
            # If a lock is stale (e.g. writer crashed), recover and continue.
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_lock_seconds:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass

            if time.time() >= deadline:
                raise TimeoutError(f"Timeout waiting for lock: {lock_path}")
            time.sleep(poll_seconds)
        except OSError as e:
            # NFS can transiently raise ESTALE when touching lock files.
            if e.errno == errno.ESTALE:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if time.time() >= deadline:
                    raise TimeoutError(f"Timeout waiting for lock after ESTALE: {lock_path}")
                time.sleep(poll_seconds)
                continue
            raise


def _serialize_param_value(value):
    """Converts a hyperparameter value to a CSV-serializable type.

    Primitive types (int, float, bool, str, None) are returned as is.
    Any other type (lists, dicts, objects) is serialized as a JSON string.

    Args:
        value: Hyperparameter value.

    Returns:
        The original value if it is primitive, or its JSON string representation.
    """
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _rename_hyperparameter(param_name):
    """Normalizes an internal hyperparameter name to the paper's readable name.

    Removes the 'estimator__' prefix added by RandomizedSearchCV and applies
    an alias dictionary to unify internal code names with the names used in 
    the paper's hyperparameter table
    (e.g., n_hidden -> hidden_units, C -> regularisation_strength).

    Args:
        param_name: Internal hyperparameter name.

    Returns:
        Normalized hyperparameter name.
    """
    param_name = param_name.replace("estimator__", "")
    aliases = {
        "n_hidden": "hidden_units",
        "learning_rate_init": "learning_rate",
        "eta": "learning_rate",
        "subsample": "pattern_subsample",
        "colsample_bytree": "feature_subsample",
        "C": "regularisation_strength",
    }
    return aliases.get(param_name, param_name)


def _upsert_hyperparameter_configuration(file_path, seed, estimator_config):
    """Inserts or updates a seed's hyperparameter row in the accumulated CSV.

    The CSV has one row per seed with a 'Seed' column plus one column
    per hyperparameter. Columns are dynamically added if a new hyperparameter
    appears (e.g., different models with different parameters).
    Rows are kept sorted by seed number.

    If the seed already has a row, its values are updated (upsert).
    File access is protected with a flock for parallel executions.

    Args:
        file_path: Path to the hyperparameter configuration CSV file.
        seed: Seed number (integer).
        estimator_config: Dictionary {param_name: value} with the
                          estimator configuration for that seed.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {"Seed": int(seed)}
    for key, value in estimator_config.items():
        new_row[_rename_hyperparameter(key)] = _serialize_param_value(value)

    lock_path = _with_file_lock(file_path)
    try:
        existing_rows = []
        fieldnames = ["Seed"]

        if file_path.exists() and file_path.stat().st_size > 0:
            with file_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is not None:
                    fieldnames = reader.fieldnames
                existing_rows = list(reader)

        for key in new_row:
            if key not in fieldnames:
                fieldnames.append(key)

        updated = False
        for row in existing_rows:
            if int(row["Seed"]) == int(seed):
                for key in fieldnames:
                    if key in new_row:
                        row[key] = new_row[key]
                    elif key not in row:
                        row[key] = ""
                updated = True
                break

        if not updated:
            row = {}
            for key in fieldnames:
                row[key] = new_row.get(key, "")
            existing_rows.append(row)

        existing_rows = sorted(existing_rows, key=lambda r: int(r["Seed"]))

        with file_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            # Best effort cleanup for lock file.
            pass