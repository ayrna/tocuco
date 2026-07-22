import numpy as np
import torch
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight


def torch_loss_wrapper(num_classes, device="cpu"):
    """
    Wrapper analítico puro y escalado. Recibe las muestras y calcula las
    derivadas exactas basándose en los pesos inyectados dinámicamente.
    """

    def fobj(y_true, y_pred, sample_weight=None):
        labels = np.asarray(y_true).astype(np.int64)
        n_samples = labels.shape[0]

        if num_classes > 2:
            preds = y_pred.reshape(n_samples, num_classes)
        else:
            preds = y_pred.reshape(-1, 1)
            preds = np.concatenate([-preds, preds], axis=1)

        preds_t = torch.tensor(preds, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs_t = torch.softmax(preds_t, dim=1)
            probs_np = probs_t.cpu().numpy()

        labels_one_hot = np.zeros_like(probs_np)
        labels_one_hot[np.arange(n_samples), labels] = 1.0
        grads_np = probs_np - labels_one_hot

        hess_np = probs_np * (1.0 - probs_np)

        # Escala de la Loss (Crucial para mantener la tracción del boosting original)
        grads_np = grads_np * n_samples
        hess_np = hess_np * n_samples

        hess_np = np.maximum(hess_np, 1e-4)

        if sample_weight is not None:
            sw = np.array(sample_weight).reshape(-1, 1)
            grads_np = grads_np * sw
            hess_np = hess_np * sw

        return (grads_np.reshape(-1), hess_np.reshape(-1))

    return fobj


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        num_classes,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=1.0,
        colsample_bytree=0.5,
        n_jobs=1,
        random_state=0,
        verbosity=0,
        class_weight=None,
        train_targets=None,
        device: str = "cpu",
    ):
        # Todos los parámetros deben mapearse exactamente uno a uno en el init
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = verbosity
        self.class_weight = class_weight
        self.train_targets = train_targets
        self.device = device

    def fit(self, X, y=None, **fit_params):
        if y is None:
            y = self.train_targets
        y = np.array(y)
        X = self._ensure_array(X)

        # =========================================================================
        # 1. RE-INICIALIZACIÓN EN TIEMPO DE FIT (Soliuciona el bug del Clone)
        # =========================================================================
        # Al instanciarlo aquí, garantizamos que use los hiperparámetros actuales
        # elegidos por el RandomizedSearchCV en esta iteración concreta.
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=torch_loss_wrapper(num_classes=self.num_classes, device=self.device),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=self.verbosity,
            num_class=self.num_classes,
        )

        # =========================================================================
        # 2. BALANCEO DINÁMICO DE CLASES
        # =========================================================================
        if self.class_weight == "balanced":
            unique_classes = np.unique(y)
            cw = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y)
            class_weight_dict = dict(zip(unique_classes, cw))
            fold_sample_weights = np.array([class_weight_dict[cls] for cls in y])
        else:
            fold_sample_weights = np.ones(len(y))

        # =========================================================================
        # 3. ZERO-WEIGHT PADDING FOR MISSING CLASSES
        # =========================================================================
        classes_in_y = np.unique(y)
        expected_classes = np.arange(self.num_classes)
        missing_classes = np.setdiff1d(expected_classes, classes_in_y)

        if len(missing_classes) > 0:
            X_dummy = np.repeat(X[0:1], len(missing_classes), axis=0)
            y_dummy = missing_classes
            X = np.concatenate([X, X_dummy], axis=0)
            y = np.concatenate([y, y_dummy], axis=0)

            sw_dummy = np.zeros(len(missing_classes))
            fold_sample_weights = np.concatenate([fold_sample_weights, sw_dummy], axis=0)

        if "sample_weight" in fit_params:
            external_sw = fit_params["sample_weight"]
            if len(missing_classes) > 0:
                external_sw = np.concatenate(
                    [external_sw, np.zeros(len(missing_classes))], axis=0
                )
            fold_sample_weights = fold_sample_weights * external_sw

        fit_params["sample_weight"] = fold_sample_weights

        return self.model_.fit(X, y, **fit_params)

    def _ensure_array(self, X):
        if hasattr(X, "data"):
            X = X.data
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        return X

    def predict(self, X):
        return self.model_.predict(self._ensure_array(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._ensure_array(X))
