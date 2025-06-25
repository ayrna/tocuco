import os
import requests
import tempfile
import zipfile
import numpy as np
import pandas as pd

from mord import LogisticAT
from xgboost import XGBClassifier
from _model_mlp_softmax import MLPClassifier
from _model_mlp_clm import MLPCLMClassifier
from _model_mlp_triangular import MLPTriangularClassifier
from _model_ebano import EBANO
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from dlordinal.metrics import amae, ranked_probability_score
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    mean_absolute_error,
    cohen_kappa_score,
)

from baseline_experiments_utils import stream_tocuco_datasets


"""
|!| DISCLAIMER:

This script will take a long time to run, as it will train a model for each dataset in the TOCUCO dataset, using
a different seed for each dataset.
    
→ The utility of this script is to show how the baseline experiments published in [1] can be reproduced. 

[1] TOCUCO: Tabular Ordinal Classification repository of the University of Cordoba. Rafael Ayllon-Gavilan,
 David Guijo-Rubio, Antonio Manuel Gomez-Orellana, Victor Manuel Vargas-Yun and Pedro A. Gutierrez.
"""

# Load TOCUCO into a temporary directory
url = "https://www.uco.es/grupos/ayrna/datasets/TOC-UCO.zip"
response = requests.get(url)
temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.write(response.content)
tmp_tocuco_path = tempfile.mkdtemp()
zip_ref = zipfile.ZipFile(temp_file.name, "r")
zip_ref.extractall(tmp_tocuco_path)
extracted_files = zip_ref.namelist()
tmp_tocuco_path = os.path.join(tmp_tocuco_path, "TOCUCO")
print("TOCUCO loaded into temporary directory:", tmp_tocuco_path)

SCORER = make_scorer(amae, greater_is_better=False)
RANDOM_SEARCH_N_ITERS = 20
CV_METHOD = StratifiedKFold(n_splits=3)
SEEDS = 30

amae_scorer = make_scorer(amae, greater_is_better=False)

cv_N_ITERS = 20
cv_hidden_units = [5, 8, 10, 15, 20, 50, 100]
cv_max_iter = [1000, 1500, 3000, 5000]
cv_learning_rate = [0.00001, 0.0001, 0.001]

MODELS = {
    "Ridge": GridSearchCV(
        RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 7),
            class_weight="balanced",
        ),
        scoring=amae_scorer,
        param_grid={"fit_intercept": [True, False], "max_iter": cv_max_iter},
        cv=CV_METHOD,
        n_jobs=1,
        verbose=3,
        error_score="raise",
    ),
    "RandomForest": RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced"),
        scoring=amae_scorer,
        param_distributions={
            "max_depth": [3, 5, 8],
            "n_estimators": [100, 250, 500, 1000],
            "ccp_alpha": [0.0, 0.05, 0.1],
            "max_features": [None, "sqrt"],
            "bootstrap": [True, False],
        },
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=CV_METHOD,
        n_jobs=1,
        verbose=3,
        error_score=("raise"),
    ),
    "XGBoost": RandomizedSearchCV(
        XGBClassifier(),
        scoring=amae_scorer,
        param_distributions={
            "max_depth": [3, 5, 8],
            "n_estimators": [100, 250, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.75, 0.95, 1.0],
            "colsample_bytree": [0.75, 0.95, 1.0],
        },
        n_iter=RANDOM_SEARCH_N_ITERS,
        cv=CV_METHOD,
        n_jobs=1,
        verbose=3,
        error_score=("raise"),
    ),
    "LogAT": GridSearchCV(
        LogisticAT(),
        param_grid={"alpha": np.logspace(-3, 3, 7), "max_iter": cv_max_iter},
        scoring=amae_scorer,
        n_jobs=1,
        cv=CV_METHOD,
        error_score="raise",
    ),
    "MLP": RandomizedSearchCV(
        MLPClassifier(class_weights="balanced"),
        param_distributions={
            "n_hidden_units": cv_hidden_units,
            "learning_rate": cv_learning_rate,
            "max_iter": cv_max_iter,
        },
        n_iter=cv_N_ITERS,
        scoring=amae_scorer,
        n_jobs=1,
        cv=CV_METHOD,
        error_score="raise",
    ),
    "MLP-CLM": RandomizedSearchCV(
        MLPCLMClassifier(class_weights="balanced"),
        param_distributions={
            "n_hidden_units": cv_hidden_units,
            "learning_rate": cv_learning_rate,
            "max_iter": cv_max_iter,
            "min_distance": [0.0, 0.1, 0.2],
        },
        n_iter=cv_N_ITERS,
        scoring=amae_scorer,
        n_jobs=1,
        cv=CV_METHOD,
        error_score="raise",
    ),
    "MLP-T": RandomizedSearchCV(
        MLPTriangularClassifier(class_weights="balanced"),
        param_distributions={
            "n_hidden_units": cv_hidden_units,
            "learning_rate": cv_learning_rate,
            "max_iter": cv_max_iter,
            "t_alpha": [0.05, 0.10],
        },
        n_iter=cv_N_ITERS,
        scoring=amae_scorer,
        n_jobs=1,
        cv=CV_METHOD,
        error_score="raise",
    ),
    ## Instructions on how to run EBANO are provided in _model_ebano.py, *you need to
    ## first collect the results of the models you want to use in the ensemble*.
    # "EBANO": EBANO(
    #     weights_cv_n_iters=1000,
    #     models_saved_results_paths=[
    # {path to results}/LogisticAT/classWeightsBalanced_cv",
    # {path to results}/MLP/triangular_cv",
    # {path to results}/MLP/CLM_cv",
    # ],
    # ),
}

results = pd.DataFrame()

for X_train, X_test, y_train, y_test, dataset_name, seed in stream_tocuco_datasets(
    tmp_tocuco_path=tmp_tocuco_path, seeds=SEEDS
):
    for model_name, model in MODELS.items():
        # Set the random state for reproducibility
        if hasattr(model, "random_state"):
            model.random_state = seed
        if hasattr(model, "estimator"):
            if hasattr(model.estimator, "random_state"):
                model.estimator.random_state = seed

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        amae_score = amae(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        if hasattr(model, "predict_proba"):
            rps = ranked_probability_score(y_test, model.predict_proba(X_test))
        else:
            preds = model.predict(X_test)
            # convert preds to one hot
            pred_proba = np.zeros((len(preds), len(np.unique(y_test))))
            pred_proba[np.arange(len(preds)), preds] = 1
            rps = ranked_probability_score(y_test, pred_proba)

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "accuracy": acc,
                        "amae": amae_score,
                        "mae": mae,
                        "qwk": qwk,
                        "rps": rps,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

        print(f"Finished: {model_name} on dataset {dataset_name} using seed {seed}")

results.to_csv("baseline_results.csv", index=False)
