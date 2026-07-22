import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn.ensemble as sk_ensemble


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        max_depth=None,
        n_estimators=100,
        ccp_alpha=0.0,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        n_jobs=1,
        random_state=0,
        train_targets=None,
        num_classes=None,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.train_targets = train_targets
        self.num_classes = num_classes

        self.model_ = None
        self.best_params_ = {}
        self.initialize()

    def initialize(self):
        self.model_ = sk_ensemble.RandomForestClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            ccp_alpha=self.ccp_alpha,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
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

        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        X = self._ensure_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X):
        X = self._ensure_array(X)
        return self.model_.predict_proba(X)

    def score(self, X, y=None, sample_weight=None):
        if y is None:
            y = self.train_targets
        if y is None:
            raise ValueError("y is required when train_targets is None")

        X = self._ensure_array(X)
        y = np.asarray(y)
        return self.model_.score(X, y, sample_weight=sample_weight)
