import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin
from dlordinal.metrics import amae


class EBANO(BaseEstimator, ClassifierMixin):
    """
    EBANO receives the paths to the saved results of the models
    and computes the optimal weights for each model using cross-validation.

    The path to the model must match the format:
    `{model_saved_results_path}/{dataset_name}/seed_{random_state}_{split}.csv`,
    where `split` can be either "train" or "test". The CSV file should contain a column
    named "y_proba" with the predicted probabilities for each class.
    """

    def __init__(
        self,
        models_saved_results_paths,
        weights_cv_n_iters=1000,
        random_state=None,
    ):
        self.weights_cv_n_iters = weights_cv_n_iters
        self.models_saved_results_paths = models_saved_results_paths
        self.random_state = random_state
        self.estimator_weights = None
        self.dataset_name = None

    def _crossvalidate_ensemble_weights(self, targets, models_pred_probas):
        if self.weights_cv_n_iters <= 0:
            raise ValueError("num_iter must be positive")

        random_state = check_random_state(self.random_state)

        num_estimators = len(models_pred_probas)
        best_weights = random_state.uniform(size=num_estimators)
        best_amae_score = float("inf")

        for _ in range(self.weights_cv_n_iters):
            weights = random_state.uniform(size=num_estimators)
            weights /= np.sum(weights)

            probas = self._compute_probas(models_pred_probas, weights)
            preds = np.argmax(probas, axis=1)

            amae_score = amae(
                y_true=targets,
                y_pred=preds,
            )
            if amae_score < best_amae_score:
                best_amae_score = amae_score
                best_weights = weights

        return best_weights

    def _compute_probas(self, models_pred_probas, weights):
        weighted_probas = np.zeros((models_pred_probas[0].shape[0], self.n_classes_))
        for i, estimator_pred_probas in enumerate(models_pred_probas):
            weighted_probas += estimator_pred_probas * weights[i]
        weighted_probas = softmax(weighted_probas, axis=1)
        return weighted_probas

    def _load_model_pred_probas(self, model_saved_results_path, split):
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        seed_results_path = os.path.join(
            model_saved_results_path,
            self.dataset_name,
            f"seed_{self.random_state}_{split}.csv",
        )
        if not os.path.exists(seed_results_path):
            raise FileNotFoundError(f"Pretrained model not found at {seed_results_path}")
        model_pred_probas = pd.read_csv(seed_results_path)["y_proba"].to_numpy()
        float_probas = [
            np.fromstring(row.strip("[]"), sep=" ") for row in model_pred_probas
        ]
        float_probas = np.vstack(float_probas)
        return float_probas

    def fit(self, X, y):
        self._train_shape = X.shape
        self.n_classes_ = len(np.unique(y))
        if self.dataset_name is None:
            raise ValueError("dataset_name must be set before fitting")

        models_pred_probas = []

        for model_saved_results_path in self.models_saved_results_paths:
            model_pred_probas = self._load_model_pred_probas(
                model_saved_results_path, split="train"
            )
            if model_pred_probas.shape[1] != self.n_classes_:
                raise ValueError(
                    f"Model predictions must have shape (n_samples, {self.n_classes_}), "
                    f"but got {model_pred_probas.shape}"
                )
            if model_pred_probas.shape[0] != len(y):
                raise ValueError(
                    f"Model predictions must have the same number of samples as y, "
                    f"but got {model_pred_probas.shape[0]} and {len(y)}"
                )
            models_pred_probas.append(model_pred_probas)

        self.estimator_weights = self._crossvalidate_ensemble_weights(
            y,
            models_pred_probas=models_pred_probas,
        )

        return self

    def predict_proba(self, X):
        split = "test"
        if self._train_shape == X.shape:
            split = "train"
        models_pred_probas = []
        for model_saved_results_path in self.models_saved_results_paths:
            model_pred_probas = self._load_model_pred_probas(
                model_saved_results_path, split=split
            )
            models_pred_probas.append(model_pred_probas)

        return self._compute_probas(models_pred_probas, self.estimator_weights)

    def predict(self, X):
        pred_probas = self.predict_proba(X)
        return np.argmax(pred_probas, axis=1)
