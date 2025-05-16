import os
import requests
import tempfile
import zipfile
import numpy as np
import pandas as pd

from mord import LogisticAT
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dlordinal.metrics import amae, ranked_probability_score
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error, cohen_kappa_score

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
url = "https://www.uco.es/grupos/ayrna/datasets/TOCUCO.zip"
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
ADABOOST_CV_N_ITERS = 20
CV_METHOD = StratifiedKFold(n_splits=3)
SEEDS = 30

amae_scorer = make_scorer(amae, greater_is_better=False)

MODELS = {
    "XGBoost": XGBClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
    ),
    "RandomForest": RandomForestClassifier(max_depth=3, n_estimators=100),
    "RidgeClassifier": RidgeClassifierCV(alphas=np.logspace(-3, 3, 7), cv=StratifiedKFold(n_splits=3)),
    "LogisticAT": GridSearchCV(
        LogisticAT(),
        param_grid={"alpha": np.logspace(-3, 3, 7)},
        scoring=amae_scorer,
        n_jobs=1,
        cv=StratifiedKFold(n_splits=3),
        error_score="raise",
    ),
}

results = pd.DataFrame()

for X_train, X_test, y_train, y_test, dataset_name, seed in stream_tocuco_datasets(
    tmp_tocuco_path=tmp_tocuco_path, seeds=SEEDS
):
    for model_name, model in MODELS.items():
        if hasattr(model, "random_state"):
            model.random_state = seed

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
