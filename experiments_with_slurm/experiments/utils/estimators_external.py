from typing import Tuple

from torch.utils.data import Dataset


REGRESSORS = []
CLASSIFIERS = [
    "resnet18classifier",
    "resnet18clmclassifier",
    "resnet18betaclassifier",
    "resnet18clmwkclassifier",
    "resnet18clmbetaclassifier",
    "resnet18clmwkbetaclassifier",
    "mlpclassifier",
    "mlpmanualclassifier",
    "randomforestclassifier",
    "ridgeclassifier",
    "xgboostclassifier",
]
ORDINAL_CLASSIFIERS = [
    ####  Ordinal classifiers  ####

    "resnet18mceloss",
    "resnet18mcewkloss",
    "resnet18cdwce",
    "resnet18sord",
    "resnet18slace",
    "mlpsordclassifier",
    "mlpslaceclassifier",
    "oeabclassifier",
    "logatcwclassifier",
    "xgboostclassifierslace",
    "logitcwclassifier",


    ####   Ordinal classifiers (soft labeling)   ####

    "resnet18triangular",
    "resnet18beta_sl",
    "resnet18exponential",
    "mlptriangularclassifier",


    ####   Ordinal classifiers (thresholds)   ####

    "resnet18clmclassifier_oc",
    "resnet18clmwkclassifier_oc",
    "resnet18clmbetaclassifier_oc",
    "resnet18clmexponentialclassifier",
    "resnet18clmtriangularclassifier",
    "mlpclmclassifier",
    "mlpclmsordclassifier",
    "mlpclmslaceclassifier",
]

def get_estimator_config(estimator_name):

    if estimator_name in REGRESSORS:
        raise ValueError(f"Estimator {estimator_name} not found")
    elif estimator_name in CLASSIFIERS:
        if estimator_name == "resnet18classifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        elif estimator_name == "resnet18clmclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "resnet18betaclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        elif estimator_name == "resnet18clmwkclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
                "penalization_type": ["quadratic", "linear"],
            }

        elif estimator_name == "resnet18clmbetaclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "resnet18clmwkbetaclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
                "penalization_type": ["quadratic", "linear"],
            }

        elif estimator_name == "mlpclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        elif estimator_name == "mlpclassifieradabord":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-3],
                "hidden_units": [4, 8, 16],
                "hidden_layers": [1, 2, 4],
                "max_iter": [1, 250, 500, 1000],
                "scheduler": ["plateau"],
            }

        elif estimator_name == "mlpmanualclassifier":
            base_config = {
                "class_weights": "balanced",
                "verbose": 0,
                "random_state": 0,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "n_hidden_units": [4, 8, 16, 32, 64],
                "n_hidden_layers": [1, 2, 4],
                "max_iter": [250, 500, 1000],
            }

        elif estimator_name == "xgboostclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbosity": 3,
            }

            param_grid = {
                "max_depth": [3, 5, 8],
                "n_estimators": [100, 250, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.75, 0.95, 1.0],
                "colsample_bytree": [0.75, 0.95, 1.0],
            }

        elif estimator_name == "ridgeclassifier":
            base_config = {
                "class_weight": "balanced",
            }

            param_grid = {
                "max_iter": [1000, 1500, 3000, 5000],
                "fit_intercept": [True, False],
            }

        elif estimator_name == "randomforestclassifier":
            base_config = {
                "class_weight": "balanced",
            }

            param_grid = {
                "max_depth": [3, 5, 8],
                "n_estimators": [100, 250, 500, 1000],
                "ccp_alpha": [0.0, 0.05, 0.1],
                "max_features": [None, "sqrt"],
                "bootstrap": [True, False],
            }

        else:
            raise ValueError(f"Estimator {estimator_name} not found")

    elif estimator_name in ORDINAL_CLASSIFIERS:

        ########   Ordinal classifiers   ########

        if estimator_name == "resnet18mceloss":
            base_config = {
                "class_weight": "balanced",
                "verbosity": 3,
            }

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
            }

        elif estimator_name == "resnet18mcewkloss":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "use_logits": True,
            }

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "penalization_type": ["quadratic", "linear"],
                "C": [0.25, 0.5, 0.75],
            }

        elif estimator_name == "resnet18cdwce":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "alpha": [0.25, 0.5, 0.75],
            }

        elif estimator_name == "resnet18sord":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "use_logits": True,
            }

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "alpha" : [0.25, 0.5, 0.75],
            }

        elif estimator_name == "resnet18slace":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "use_logits": True,
            }

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "alpha" : [0.25, 0.5, 0.75],
            }

        elif estimator_name == "mlpsordclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "use_logits": True,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "alpha": [1, 0.3, 0.5, 0.8, 2, 3, 4, 7, 10, 15, 20, 25],
                "prox": [False],
                "ftype": [
                    "max",
                    "norm_max",
                    "norm_log",
                    "log",
                    "norm_division",
                    "division",
                ],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        elif estimator_name == "mlpslaceclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "use_logits": True,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "alpha": [1, 0.3, 0.5, 0.8, 2, 3, 4, 7, 10, 15, 20, 25],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
            }
        
        elif estimator_name == "oeabclassifier":
            base_config = {
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [0.001, 0.005],
                "n_estimators": [10, 30],
                "n_hidden": [4, 16, 32, 64],
            }

        elif estimator_name == "logatcwclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "max_iter": [1000, 1500, 3000, 5000],
            }


        elif estimator_name == "xgboostclassifierslace":
            base_config = {
                "class_weight": "balanced",
                "verbosity": 3,
                "use_logits": True,
            }

            param_grid = {
                "max_depth": [3, 5, 8],
                "n_estimators": [100, 250, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.75, 0.95, 1.0],
                "colsample_bytree": [0.75, 0.95, 1.0],
                "alpha": [1, 0.3, 0.5, 0.8, 2, 3, 4, 7, 10, 15, 20, 25],
                "max_iter": [100, 150, 300, 500],
            }


        elif estimator_name == "logitcwclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        
        ####    Ordinal classifiers (soft labeling)    ####

        elif estimator_name == "resnet18triangular":
            base_config = {
                "class_weight": "balanced",
                "verbosity": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [500, 1000, 1500, 3000, 5000],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_alpha" : [0.02, 0.03, 0.04, 0.05, 0.06]
            }


        elif estimator_name == "resnet18beta_sl":
            base_config = {
                "class_weight": "balanced",
                "verbosity": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [500, 1000, 1500, 3000, 5000],
                "loss_eta": [0.8, 0.9, 1.0],
            }


        elif estimator_name == "resnet18exponential":
            base_config = {
                "class_weight": "balanced",
                "verbosity": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_p" : [0.8, 0.9, 1.0]
            }


        elif estimator_name == "mlptriangularclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
                "loss_alpha2": [0.05, 0.10],
                "loss_eta": [0.8, 1.0],
            }


        ####    Ordinal classifiers (thresholds)    ####

        elif estimator_name == "resnet18clmclassifier_oc":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "resnet18clmwkclassifier_oc":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
                "penalization_type": ["quadratic", "linear"],
            }

        elif estimator_name == "resnet18clmbetaclassifier_oc":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "loss_eta": [0.8, 0.9, 1.0],
                "link_function": ["logit"],
            }
        
        elif estimator_name == "resnet18clmexponentialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_p" : [0.8, 0.9, 1.0]
            }

        elif estimator_name == "resnet18clmtriangularclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_alpha2" : [0.01, 0.02, 0.03, 0.04, 0.05]
            }

        elif estimator_name == "mlpclmclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "hidden_units": [4, 8, 16],
                "hidden_layers": [1, 2, 4],
                "max_iter": [100, 250, 500, 1000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "mlpclmsordclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.5, 1.0],
                "alpha": [1, 0.3, 0.5, 0.8, 2, 3, 4, 7, 10, 15, 20, 25],
                "prox": [False],
                "ftype": [
                    "max",
                    "norm_max",
                    "norm_log",
                    "log",
                    "norm_division",
                    "division",
                ],
            }


        elif estimator_name == "mlpclmslaceclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.5, 1.0],
                "alpha": [1, 0.3, 0.5, 0.8, 2, 3, 4, 7, 10, 15, 20, 25],
            }


        else:
            raise ValueError(f"Estimator {estimator_name} not found")

    else:
        raise ValueError(f"Estimator {estimator_name} not found")

    return base_config, param_grid



def get_estimator(
    estimator_name,
    config,
    *,
    num_classes,
    train_dataset,
    val_dataset,
    random_state=0,
    n_jobs=1,
    batch_size=128,
    use_gpu_if_available=True,
    interactive=False,
    experiment_config={},
):

    from copy import copy

    config = copy(config)
    if "device" in config:
        del config["device"]
    if "interactive" in config:
        del config["interactive"]

    import torch

    device = "cuda" if use_gpu_if_available and torch.cuda.is_available() else "cpu"

    # Remove the last part after the dash
    estimator_name = estimator_name.split("-")[0]

    train_targets = getattr(train_dataset, "targets", None)

    if estimator_name in REGRESSORS:
        raise ValueError(f"Estimator {estimator_name} not found")
    elif estimator_name in CLASSIFIERS:
        if estimator_name == "resnet18classifier":
            from ..classification import ResNet18Classifier

            estimator = ResNet18Classifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmclassifier":
            from ..classification import ResNet18CLMClassifier

            estimator = ResNet18CLMClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18betaclassifier":
            from ..classification import ResNet18BetaClassifier
            
            estimator = ResNet18BetaClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwkclassifier":
            from ..classification import ResNet18CLMWKClassifier

            estimator = ResNet18CLMWKClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmbetaclassifier":
            from ..classification import ResNet18CLMBetaClassifier

            estimator = ResNet18CLMBetaClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwkbetaclassifier":
            from ..classification import ResNet18CLMWKBetaClassifier

            estimator = ResNet18CLMWKBetaClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "mlpclassifier":
            from ..classification import MLPClassifier

            estimator = MLPClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "mlpclassifieradabord":
            from ..classification import MLPClassifier

            estimator = MLPClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "mlpmanualclassifier":
            from ..classification import MLPManualClassifier

            estimator = MLPManualClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "xgboostclassifier":
            from ..classification import XGBoostClassifier

            estimator = XGBoostClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "ridgeclassifier":
            from ..classification import RidgeClassifier

            estimator = RidgeClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
            )

        elif estimator_name == "randomforestclassifier":
            from ..classification import RandomForestClassifier

            estimator = RandomForestClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        else:
            raise ValueError(f"Estimator {estimator_name} not found")

    elif estimator_name in ORDINAL_CLASSIFIERS:

        ########   Ordinal classifiers   ########

        if estimator_name == "resnet18mceloss":
            from ..ordinal_classification import ResNet18MCE

            estimator = ResNet18MCE(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "resnet18mcewkloss":
            from ..ordinal_classification import ResNet18MCEWK

            estimator = ResNet18MCEWK(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "resnet18cdwce":
            from ..ordinal_classification import ResNet18CDWCE

            estimator = ResNet18CDWCE(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "resnet18sord":
            from ..ordinal_classification import ResNet18SORD

            estimator = ResNet18SORD(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "resnet18slace":
            from ..ordinal_classification import ResNet18SLACE

            estimator = ResNet18SLACE(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "mlpsordclassifier":
            from ..ordinal_classification import MLPSordClassifier

            estimator = MLPSordClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "mlpslaceclassifier":
            from ..ordinal_classification import MLPSLACEClassifier

            estimator = MLPSLACEClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )
        
        elif estimator_name == "oeabclassifier":
            from ..ordinal_classification import OEABClassifier

            estimator = OEABClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
            )

        elif estimator_name == "logatcwclassifier":
            from ..ordinal_classification import LogATcw

            estimator = LogATcw(**config)

        elif estimator_name == "xgboostclassifierslace":
            from ..ordinal_classification import XGBoostClassifierSLACE

            estimator = XGBoostClassifierSLACE(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                n_jobs=n_jobs,
                device=device,
            )

        elif estimator_name == "logitcwclassifier":
            from ..ordinal_classification import LogITcw

            estimator = LogITcw(**config)


        ####    Ordinal classifiers (soft labeling)    ####


        elif estimator_name == "resnet18triangular":
            from ..ordinal_classification.soft_labeling import ResNet18Triangular

            estimator = ResNet18Triangular(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18beta_sl":
            from ..ordinal_classification.soft_labeling import ResNet18Beta

            estimator = ResNet18Beta(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18exponential":
            from ..ordinal_classification.soft_labeling import ResNet18Exponential

            estimator = ResNet18Exponential(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "mlptriangularclassifier":
            from ..ordinal_classification.soft_labeling import MLPTriangularClassifier

            estimator = MLPTriangularClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )


        ####    Ordinal classifiers (thresholds)    ####

        elif estimator_name == "resnet18clmclassifier_oc":
            from ..ordinal_classification.thresholds import ResNet18CLMClassifier

            estimator = ResNet18CLMClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwkclassifier_oc":
            from ..ordinal_classification.thresholds import ResNet18CLMWKClassifier

            estimator = ResNet18CLMWKClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmbetaclassifier_oc":
            from ..ordinal_classification.thresholds import ResNet18CLMBetaClassifier

            estimator = ResNet18CLMBetaClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )
        
        elif estimator_name == "resnet18clmexponentialclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMExponentialClassifier

            estimator = ResNet18CLMExponentialClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmtriangularclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMTriangularClassifier

            estimator = ResNet18CLMTriangularClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "mlpclmclassifier":
            from ..ordinal_classification.thresholds import MLPCLMClassifier

            estimator = MLPCLMClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "mlpclmsordclassifier":
            from ..ordinal_classification.thresholds import MLPCLMSordClassifier

            estimator = MLPCLMSordClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "mlpclmslaceclassifier":
            from ..ordinal_classification.thresholds import MLPCLMSLACEClassifier

            estimator = MLPCLMSLACEClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        else:
            raise ValueError(f"Estimator {estimator_name} not found")
    else:
        raise ValueError(f"Estimator {estimator_name} not found")
    return estimator
