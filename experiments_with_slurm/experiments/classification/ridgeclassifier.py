import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV


class RidgeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        num_classes,
        alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        fit_intercept=True,
        max_iter=1000,
        cv=None,
        scoring=None,
        class_weight=None,
        train_targets=None,
    ):
        self.num_classes = num_classes
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.cv = cv
        self.scoring = scoring
        self.class_weight = class_weight
        self.train_targets = train_targets

        self.model_ = None
        self.best_params_ = {}

        self.initialize()

    def initialize(self):
        self.model_ = RidgeClassifierCV(
            alphas=self.alphas,
            class_weight=self.class_weight,
        )

    def _ensure_array(self, X):
        if hasattr(X, "data"):
            X = X.data
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        elif hasattr(X, "numpy"):
            X = X.cpu().numpy() if hasattr(X, "cpu") else X.numpy()

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        return X

    def fit(self, X, y=None, **fit_params):
        if y is None:
            y = self.train_targets
        if y is None:
            raise ValueError("y is required when train_targets is None")

        X = self._ensure_array(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n_fit_loops = 1
        if self.max_iter is not None:
            n_fit_loops = int(self.max_iter)
            if n_fit_loops < 1:
                raise ValueError("max_iter must be >= 1 when provided")

        for _ in range(n_fit_loops):
            self.model_.fit(X, y, **fit_params)

        if hasattr(self.model_, "alpha_"):
            self.best_params_ = {"alpha": float(self.model_.alpha_), "max_iter": n_fit_loops}
        else:
            self.best_params_ = {"max_iter": n_fit_loops}

        return self

    def decision_function(self, X):
        X = self._ensure_array(X)
        return self.model_.decision_function(X)

    def predict(self, X):
        X = self._ensure_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X):
        scores = self.decision_function(X)

        # RidgeClassifierCV does not expose calibrated probabilities.
        # Convert decision scores to pseudo-probabilities for downstream metrics.
        if np.ndim(scores) == 1:
            scores = np.clip(scores, -50.0, 50.0)
            p1 = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack((1.0 - p1, p1))

        scores = np.asarray(scores, dtype=np.float64)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        denom = np.sum(exp_scores, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return exp_scores / denom

    def score(self, X, y=None, sample_weight=None):
        if y is None:
            y = self.train_targets
        if y is None:
            raise ValueError("y is required when train_targets is None")

        X = self._ensure_array(X)
        y = np.asarray(y)
        return self.model_.score(X, y, sample_weight=sample_weight)
