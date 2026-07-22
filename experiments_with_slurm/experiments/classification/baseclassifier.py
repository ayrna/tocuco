import inspect
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.helper import predefined_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """
    Replaces a submodule in `model` given its dotted path `name` with `new_module`.
    Example: name="classifier.1" or name="fc"
    """
    parts = name.split(".")
    submodule = model
    for p in parts[:-1]:
        submodule = getattr(submodule, p)
    setattr(submodule, parts[-1], new_module)


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        num_classes,
        device="cpu",
        max_iter=1000,
        class_weight=None,
        train_targets=None,
        learning_rate=1e-3,
        verbose=0,
        batch_size=128,
        val_dataset=None,
        n_jobs=1,
        interactive=False,
    ):

        self.num_classes = num_classes
        self.device = device
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.train_targets = train_targets
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.batch_size = batch_size
        self.val_dataset = val_dataset
        self.n_jobs = n_jobs
        self.interactive = interactive
        self.best_params_ = {}
        self.net_ = None

    def initialize(self, y):
        if self.class_weight is not None and self.train_targets is None:
            raise ValueError("class_weight requires train_targets")

        if y is None:
            raise ValueError("Training targets 'y' are required to initialize the model.")
            
        y_arr = np.asarray(y)

        self.computed_weights_ = torch.tensor(
            compute_class_weight(
                self.class_weight,
                classes=np.arange(self.num_classes),
                y=y_arr,
            ),
            dtype=torch.float32,
        ).to(self.device)

        classifier_class = self.get_classifier_class()
        self.net_ = classifier_class(
            module=self.build_model().to(self.device),
            criterion=self.build_loss(self.computed_weights_).to(self.device),  # type: ignore
            optimizer=AdamW,
            lr=self.learning_rate,
            max_epochs=self.max_iter,
            train_split=predefined_split(self.val_dataset),  # type: ignore
            callbacks=self.build_callbacks(),
            device=self.device,
            verbose=self.verbose,
            iterator_train__batch_size=self.batch_size,
            iterator_train__shuffle=True,
            iterator_train__num_workers=self.n_jobs - 1,
            iterator_train__pin_memory=(self.device == "cuda"),
            iterator_valid__batch_size=self.batch_size,
            iterator_valid__shuffle=False,
            iterator_valid__num_workers=self.n_jobs - 1,
            iterator_valid__pin_memory=(self.device == "cuda"),
        )

    def get_classifier_class(self):
        return NeuralNetClassifier

    def build_loss(self, computed_class_weights) -> torch.nn.Module:
        return CrossEntropyLoss(weight=computed_class_weights)

    def build_final_layer(self, in_features):
        return torch.nn.Linear(in_features, self.num_classes).to(self.device)

    def build_base_model(self):
        raise NotImplementedError(
            "build_base_model() must be implemented in the subclass"
        )

    def build_model(self):
        model = self.build_base_model()

        # Replace the final layer with a new one that matches the number of classes
        current_final_layer = list(model.named_modules())[-1][1]
        final_layer_name = list(model.named_modules())[-1][0]
        set_module_by_name(
            model,
            final_layer_name,
            self.build_final_layer(current_final_layer.in_features),
        )

        return model

    def build_callbacks(self):
        callbacks = []
        if self.val_dataset is not None:

            # def amae_scorer(net, X, y):
            #     from dlordinal.metrics import amae

            #     y_proba = net.predict_proba(X)
            #     return amae(y, y_proba)

            # def get_lr(net, X, y):
            #     return net.optimizer_.param_groups[0]["lr"]

            # callbacks.append(
            #     (
            #         "amae_scoring",
            #         EpochScoring(amae_scorer, lower_is_better=True, name="valid_amae"),
            #     )
            # )
            # callbacks.append(
            #     (
            #         "lr_scoring",
            #         EpochScoring(get_lr, lower_is_better=False, name="lr"),
            #     )
            # )
            callbacks.append(
                (
                    "early_stopping",
                    EarlyStopping(monitor="valid_loss", patience=40, load_best=True),
                )
            )
            # callbacks.append(
            #     (
            #         "lr_scheduler",
            #         LRScheduler(
            #             ReduceLROnPlateau,
            #             monitor="valid_amae",
            #             factor=0.75,
            #             patience=10,
            #         ),
            #     )
            # )
        return callbacks

    def fit(
        self, X: Union[ArrayLike, Dataset], y: Optional[ArrayLike] = None, **fit_params
    ):
        if y is None:
            y = self.train_targets
        y = np.array(y)

        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        elif hasattr(X, "astype"):
            X = X.astype(np.float32)
        else:
            X = np.array(X, dtype=np.float32)

        self.initialize(y=y)

        r = self.net_.fit(X, y, **fit_params)

        # Save best epoch from early stopping callback
        for name, callback in self.net_.callbacks_:
            if name == "early_stopping":
                self.best_params_["max_iter"] = callback.best_epoch_
                break

        return r

    def predict(self, X: Union[ArrayLike, Dataset]):
        if not isinstance(X, Dataset):
            if hasattr(X, "astype"):
                X = X.astype(np.float32)
            else:
                X = np.array(X, dtype=np.float32)
                
        check_is_fitted(self, "net_")
        return self.net_.predict(X)

    def predict_proba(self, X: Union[ArrayLike, Dataset]):
        if not isinstance(X, Dataset):
            if hasattr(X, "astype"):
                X = X.astype(np.float32)
            else:
                X = np.array(X, dtype=np.float32)
                
        check_is_fitted(self, "net_")
        return self.net_.predict_proba(X)

    def score(
        self,
        X: Union[ArrayLike, Dataset],
        y: Optional[ArrayLike] = None,
        sample_weight=None,
    ):
        check_is_fitted(self, "net_")
        if y is None:
            y = self.train_targets
        y = np.array(y)
        if hasattr(X, "astype"):
            X = X.astype(np.float32)
        else:
            X = np.array(X, dtype=np.float32)
        return self.net_.score(X, y, sample_weight)

    @property
    def train_history(self):
        check_is_fitted(self, "net_")
        return (
            [float(l) for l in self.net_.history[:, "train_loss"]]
            if self.net_.history
            else []
        )

    @property
    def valid_history(self):
        check_is_fitted(self, "net_")
        return (
            [float(l) for l in self.net_.history[:, "valid_loss"]]
            if self.net_.history and self.val_dataset
            else []
        )

    @property
    def classes_(self):
        return np.arange(self.num_classes)

    def _get_param_names(self):
        """Override sklearn's _get_param_names to include all mixin
        and base class parameters.
        """
        param_names = set()

        for cls in self.__class__.__mro__:
            # Avoid duplicate init from BaseEstimator, object, etc.
            if "__init__" in cls.__dict__:
                sig = inspect.signature(cls.__init__)
                for name, param in sig.parameters.items():
                    if name == "self":
                        continue
                    if param.kind in (
                        param.POSITIONAL_OR_KEYWORD,
                        param.KEYWORD_ONLY,
                    ):
                        param_names.add(name)

        return sorted(param_names)
