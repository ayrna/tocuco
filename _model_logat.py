import numpy as np
from mord import LogisticAT
from mord.threshold_based import threshold_fit
from sklearn.utils import compute_sample_weight


class LogAT(LogisticAT):
    def __init__(self, class_weight, alpha=1.0, verbose=0, max_iter=1000):
        self.class_weight = class_weight
        self.alpha = alpha
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y):

        sample_weight = compute_sample_weight(class_weight=self.class_weight, y=y)
        print("LogAT sample_weight:", sample_weight)

        _y = np.array(y).astype(int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError("y must only contain integer values")
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit(
            X,
            y_tmp,
            self.alpha,
            self.n_class_,
            mode="AE",
            verbose=self.verbose,
            max_iter=self.max_iter,
            sample_weight=sample_weight,
        )
        return self
