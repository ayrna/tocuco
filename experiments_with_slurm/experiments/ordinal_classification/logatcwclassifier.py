import numpy as np
from mord import LogisticAT
from mord.threshold_based import threshold_fit
from sklearn.utils import compute_sample_weight


class LogATcw(LogisticAT):
    def __init__(self, class_weight=None, alpha=1.0, verbose=0, max_iter=1000):
        self.class_weight = class_weight
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y):
        # Ensure y is a numpy array (some callers pass lists)
        y = np.asarray(y)

        _y = y.astype(int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError("y must only contain integer values")

        self.classes_ = np.unique(_y)
        min_class = self.classes_.min()
        max_class = self.classes_.max()
        self.n_class_ = max_class - min_class + 1

        # Check if all classes [min_class, max_class] are present
        expected_classes = set(range(min_class, max_class + 1))
        present_classes = set(self.classes_)
        missing_classes = expected_classes - present_classes

        X_array = X
        if hasattr(X, "data"):
            X_array = X.data
        if hasattr(X_array, "numpy"):
            X_array = X_array.cpu().numpy() if hasattr(X_array, "cpu") else X_array.numpy()
        if not isinstance(X_array, np.ndarray):
            X_array = np.asarray(X_array)

        # If there are missing classes, add synthetic samples to avoid mord error
        if len(missing_classes) > 0:
            print(
                f"Warning: Missing classes {sorted(missing_classes)} in training data. Adding synthetic samples."
            )
            X_append = []
            y_append = []

            # Get the mean of existing samples to create synthetic ones
            mean_sample = np.mean(X_array, axis=0, keepdims=True)

            for missing_class in sorted(missing_classes):
                # Add one synthetic sample per missing class
                X_append.append(mean_sample)
                y_append.append(missing_class)

            X_append = np.vstack(X_append)
            y_append = np.array(y_append, dtype=int)

            # Concatenate original and synthetic data
            X_array = np.vstack([X_array, X_append])
            _y = np.concatenate([_y, y_append])

        y_tmp = _y - _y.min()  # we need classes that start at zero

        sample_weight = compute_sample_weight(class_weight=self.class_weight, y=_y)

        self.coef_, self.theta_ = threshold_fit(
            X_array,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="AE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            sample_weight=sample_weight,
        )
        return self

    def _ensure_array(self, X):
        # Accept LabeledTensorDataset, torch tensors, lists, numpy arrays
        if hasattr(X, "data"):
            X = X.data

        if hasattr(X, "numpy"):
            X = X.cpu().numpy() if hasattr(X, "cpu") else X.numpy()

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        return X

    def predict_proba(self, X):
        X = self._ensure_array(X)
        # Delegate to mord.LogisticAT.predict_proba
        return super().predict_proba(X)

    def predict(self, X):
        X = self._ensure_array(X)
        return super().predict(X)
