import torch

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


def get_estimator(estimator_name, num_classes, random_state, use_gpu_if_available):
    estimator = None
    param_grid = None

    device = "cuda" if torch.cuda.is_available() and use_gpu_if_available else "cpu"


    if estimator_name in REGRESSORS:
        raise ValueError(f"Estimator {estimator_name} not found")
    elif estimator_name in CLASSIFIERS:
        if estimator_name == "resnet18classifier":
            from ..classification import ResNet18Classifier

            estimator = ResNet18Classifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
            }
        elif estimator_name == "resnet18clmclassifier":
            from ..classification import ResNet18CLMClassifier

            estimator = ResNet18CLMClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "resnet18betaclassifier":
            from ..classification import ResNet18BetaClassifier
            
            estimator = ResNet18BetaClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        elif estimator_name == "resnet18clmwkclassifier":
            from ..classification import ResNet18CLMWKClassifier

            estimator = ResNet18CLMWKClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
                "penalization_type": ["quadratic", "linear"],
            }

        elif estimator_name == "resnet18clmbetaclassifier":
            from ..classification import ResNet18CLMBetaClassifier

            estimator = ResNet18CLMBetaClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "resnet18clmwkbetaclassifier":
            from ..classification import ResNet18CLMWKBetaClassifier

            estimator = ResNet18CLMWKBetaClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
                "penalization_type": ["quadratic", "linear"],
            }

        elif estimator_name == "mlpclassifier":
            from ..classification import MLPClassifier

            estimator = MLPClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        elif estimator_name == "mlpclassifieradabord":
            from ..classification import MLPClassifier

            estimator = MLPClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-3],
                "hidden_units": [4, 8, 16],
                "hidden_layers": [1, 2, 4],
                "max_iter": [1, 250, 500, 1000],
                "scheduler": ["plateau"],
            }

        elif estimator_name == "mlpmanualclassifier":
            from ..classification import MLPManualClassifier

            estimator = MLPManualClassifier(
                random_state=random_state,
                num_classes=num_classes,
                n_hidden_layers = 2,
                n_hidden_units = 16,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "n_hidden_units": [4, 8, 16, 32, 64],
                "n_hidden_layers": [1, 2, 4],
                "max_iter": [250, 500, 1000],
            }

        elif estimator_name == "xgboostclassifier":
            from ..classification import XGBoostClassifier

            estimator = XGBoostClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "max_depth": [3, 5, 8],
                "n_estimators": [100, 250, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.75, 0.95, 1.0],
                "colsample_bytree": [0.75, 0.95, 1.0],
            }

        elif estimator_name == "ridgeclassifier":
            from ..classification import RidgeClassifier

            estimator = RidgeClassifier(
                num_classes=num_classes,
                class_weight="balanced",
            )

            param_grid = {
                "max_iter": [1000, 1500, 3000, 5000],
                "fit_intercept": [True, False],
            }

        elif estimator_name == "randomforestclassifier":
            from ..classification import RandomForestClassifier
            
            estimator = RandomForestClassifier(
                num_classes=num_classes,
                class_weight="balanced",
            )

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
            from ..ordinal_classification import ResNet18MCE

            estimator = ResNet18MCE(
                max_iter=500,
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
            }

        elif estimator_name == "resnet18mcewkloss":
            from ..ordinal_classification import ResNet18MCEWK

            estimator = ResNet18MCEWK(
                max_iter=500,
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "penalization_type": ["quadratic", "linear"],
                "C": [0.25, 0.5, 0.75],
            }

        elif estimator_name == "resnet18cdwce":
            from ..ordinal_classification import ResNet18CDWCE

            estimator = ResNet18CDWCE(
                max_iter=500,
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "alpha": [0.25, 0.5, 0.75],
            }

        elif estimator_name == "resnet18sord":
            from ..ordinal_classification import ResNet18SORD

            estimator = ResNet18SORD(
                max_iter=500,
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "alpha" : [0.25, 0.5, 0.75],
            }

        elif estimator_name == "resnet18slace":
            from ..ordinal_classification import ResNet18SLACE

            estimator = ResNet18SLACE(
                max_iter=500,
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "alpha" : [0.25, 0.5, 0.75],
            }

        elif estimator_name == "mlpsordclassifier":
            from ..ordinal_classification import MLPSordClassifier

            estimator = MLPSordClassifier(
                num_classes=num_classes,
                alpha=1,
                device = device,
                class_weight="balanced",
            )

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
            from ..ordinal_classification import MLPSLACEClassifier

            estimator = MLPSLACEClassifier(
                num_classes=num_classes,
                alpha=1,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "alpha": [1, 0.3, 0.5, 0.8, 2, 3, 4, 7, 10, 15, 20, 25],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
            }
        
        elif estimator_name == "oeabclassifier":
            from ..ordinal_classification import OEABClassifier

            estimator = OEABClassifier(
                num_classes=num_classes,
                device = device,
            )

            param_grid = {
                "learning_rate": [0.001, 0.005],
                "n_estimators": [10, 30],
                "n_hidden": [4, 16, 32, 64],
            }

        elif estimator_name == "logatcwclassifier":
            from ..ordinal_classification import LogATcw

            estimator = LogATcw(
                class_weight="balanced",
            )

            param_grid = {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        elif estimator_name == "xgboostclassifierslace":
            from ..ordinal_classification import XGBoostClassifierSLACE

            estimator = XGBoostClassifierSLACE(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

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
            from ..ordinal_classification import LogITcw

            estimator = LogITcw(
                class_weight="balanced",
            )

            param_grid = {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "max_iter": [1000, 1500, 3000, 5000],
            }

        ####    Ordinal classifiers (soft labeling)    ####

        elif estimator_name == "resnet18triangular":
            from ..ordinal_classification.soft_labeling import ResNet18Triangular

            estimator = ResNet18Triangular(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [500, 1000, 1500, 3000, 5000],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_alpha" : [0.02, 0.03, 0.04, 0.05, 0.06]
            }

        elif estimator_name == "resnet18beta_sl":
            from ..ordinal_classification.soft_labeling import ResNet18Beta

            estimator = ResNet18Beta(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [500, 1000, 1500, 3000, 5000],
                "loss_eta": [0.8, 0.9, 1.0],
            }

        elif estimator_name == "resnet18exponential":
            from ..ordinal_classification.soft_labeling import ResNet18Exponential

            estimator = ResNet18Exponential(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [1000, 1500, 3000, 5000],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_p" : [0.8, 0.9, 1.0]
            }

        elif estimator_name == "mlptriangularclassifier":
            from ..ordinal_classification.soft_labeling import MLPTriangularClassifier

            estimator = MLPTriangularClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "hidden_units": [5, 8, 10, 15, 20, 50, 100],
                "max_iter": [1000, 1500, 3000, 5000],
                "loss_alpha2": [0.05, 0.10],
                "loss_eta": [0.8, 1.0],
            }


        ####    Ordinal classifiers (thresholds)    ####

        elif estimator_name == "resnet18clmclassifier_oc":
            from ..ordinal_classification.thresholds import ResNet18CLMClassifier

            estimator = ResNet18CLMClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "resnet18clmwkclassifier_oc":
            from ..ordinal_classification.thresholds import ResNet18CLMWKClassifier

            estimator = ResNet18CLMWKClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
                "penalization_type": ["quadratic", "linear"],
            }

        elif estimator_name == "resnet18clmbetaclassifier_oc":
            from ..ordinal_classification.thresholds import ResNet18CLMBetaClassifier

            estimator = ResNet18CLMBetaClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "loss_eta": [0.8, 0.9, 1.0],
                "link_function": ["logit"],
            }
        
        elif estimator_name == "resnet18clmexponentialclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMExponentialClassifier

            estimator = ResNet18CLMExponentialClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_p" : [0.8, 0.9, 1.0]
            }

        elif estimator_name == "resnet18clmtriangularclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMTriangularClassifier

            estimator = ResNet18CLMTriangularClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "max_iter": [100, 150, 300, 500],
                "loss_eta": [0.8, 0.9, 1.0],
                "loss_alpha2" : [0.01, 0.02, 0.03, 0.04, 0.05]
            }

        elif estimator_name == "mlpclmclassifier":
            from ..ordinal_classification.thresholds import MLPCLMClassifier

            estimator = MLPCLMClassifier(
                num_classes=num_classes,
                device = device,
                class_weight="balanced",
            )

            param_grid = {
                "learning_rate": [1e-5, 1e-4, 1e-3],
                "hidden_units": [4, 8, 16],
                "hidden_layers": [1, 2, 4],
                "max_iter": [100, 250, 500, 1000],
                "link_function": ["logit"],
                "min_distance": [0.0, 0.1, 0.2],
            }

        elif estimator_name == "mlpclmsordclassifier":
            from ..ordinal_classification.thresholds import MLPCLMSordClassifier

            estimator = MLPCLMSordClassifier(
                num_classes=num_classes,
                alpha=1,
                device = device,
                class_weight="balanced",
            )

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
            from ..ordinal_classification.thresholds import MLPCLMSLACEClassifier

            estimator = MLPCLMSLACEClassifier(
                num_classes=num_classes,
                alpha=1,
                device = device,
                class_weight="balanced",
            )

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

    return estimator, param_grid
