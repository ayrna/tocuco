import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split
from torch.nn import CrossEntropyLoss, Flatten, Linear, Module, Sequential, Sigmoid
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLPModule(Module):
    def __init__(self, input_dim, hidden_units, hidden_layers, num_classes):
        super().__init__()

        layers = []
        layers.append(Flatten())

        # Primera capa oculta
        layers.append(Linear(input_dim, hidden_units))
        layers.append(Sigmoid())

        # Capas ocultas adicionales
        for _ in range(hidden_layers - 1):
            layers.append(Linear(hidden_units, hidden_units))
            layers.append(Sigmoid())

        # Capa de salida
        layers.append(Linear(hidden_units, num_classes))

        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        num_classes,
        hidden_units=64,
        hidden_layers=2,
        device="cpu",
        max_iter=1000,
        class_weight=None,
        train_targets=None,
        learning_rate=1e-3,
        verbose=0,
        batch_size=128,
        val_dataset=None,
        n_jobs=1,
        scheduler=None,
    ):
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.device = device
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.train_targets = train_targets
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.batch_size = batch_size
        self.val_dataset = val_dataset
        self.n_jobs = n_jobs
        self.scheduler = scheduler
        self.best_params_ = {}
        self.net_ = None
        self.input_dim_ = None

    def initialize(self, input_dim, y):
        if self.class_weight is not None and self.train_targets is None:
            raise ValueError("class_weight requires train_targets")

        if y is None:
            raise ValueError("Training targets 'y' are required to initialize the model.")
            
        y_arr = np.asarray(y)

        self.input_dim_ = input_dim

        self.computed_weights_ = torch.tensor(
            compute_class_weight(
                self.class_weight,
                classes=np.arange(self.num_classes),
                y=y_arr,
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.net_ = NeuralNetClassifier(
            module=self.get_model().to(self.device),
            criterion=self.get_loss().to(self.device),
            optimizer=AdamW,
            lr=self.learning_rate,
            max_epochs=self.max_iter,
            train_split=predefined_split(self.val_dataset),
            callbacks=self.get_callbacks(),
            device=self.device,
            verbose=self.verbose,
            iterator_train__batch_size=self.batch_size,
            iterator_train__shuffle=True,
            iterator_train__num_workers=self.n_jobs - 1,
            iterator_train__pin_memory=True,
            iterator_valid__batch_size=self.batch_size,
            iterator_valid__shuffle=False,
            iterator_valid__num_workers=self.n_jobs - 1,
            iterator_valid__pin_memory=True,
        )

    def get_loss(self) -> torch.nn.Module:
        return CrossEntropyLoss(weight=self.computed_weights_)

    def get_model(self):
        model = MLPModule(
            input_dim=self.input_dim_,
            hidden_units=self.hidden_units,
            hidden_layers=self.hidden_layers,
            num_classes=self.num_classes,
        ).to(self.device)
        return model

    def get_callbacks(self):
        callbacks = []
        if self.val_dataset is not None:
            callbacks.append(("early_stopping", EarlyStopping(patience=40, load_best=True)))

        if self.scheduler is not None:
            callbacks.append(self.get_scheduler_callback())

        return callbacks

    def get_scheduler_callback(self):
        if isinstance(self.scheduler, tuple):
            return self.scheduler

        if isinstance(self.scheduler, str):
            scheduler_name = self.scheduler.lower()
            if scheduler_name == "plateau":
                monitor = "valid_loss" if self.val_dataset is not None else "train_loss"
                return (
                    "lr_scheduler",
                    LRScheduler(
                        ReduceLROnPlateau,
                        monitor=monitor,
                        mode="min",
                        factor=0.75,
                        patience=50,
                        min_lr=1e-5,
                        threshold=1e-2,
                        # threshold_mode="abs",
                    ),
                )

            raise ValueError(
                f"Unsupported scheduler '{self.scheduler}'. Supported string values: ['plateau']."
            )

        return ("lr_scheduler", self.scheduler)

    def fit(self, X, y=None, **fit_params):
        if y is None:
            y = self.train_targets
        y = np.array(y)


        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        elif hasattr(X, "astype"):
            X = X.astype(np.float32)
        else:
            X = np.array(X, dtype=np.float32)

        # sklearn expects classifiers to expose the attribute `classes_`
        # after fit. Set it here so scorers and utilities that access
        # `estimator.classes_` do not fail (e.g. during cross-validation
        # scoring or grid/randomized search).
        try:
            # Preserve original order of classes as provided in y
            self.classes_ = np.unique(y)
        except Exception:
            # Fallback: ensure attribute exists even if y is malformed
            self.classes_ = np.array([])

        # Initialize on first fit
        if self.net_ is None:
            X_array = np.array(X) if not isinstance(X, np.ndarray) else X
            input_dim = X_array.shape[1] if len(X_array.shape) > 1 else 1
            self.initialize(input_dim, y)

        r = self.net_.fit(X, y, **fit_params)

        # Save best epoch from early stopping callback
        for name, callback in self.net_.callbacks_:
            if name == "early_stopping":
                self.best_params_["max_iter"] = callback.best_epoch_
                break

        return r

    def predict(self, X):
        if hasattr(X, "astype"):
            X = X.astype(np.float32)
        else:
            X = np.array(X, dtype=np.float32)

        return self.net_.predict(X)

    def predict_proba(self, X):
        if isinstance(X, torch.utils.data.Dataset):
            pass
        elif hasattr(X, "astype"):
            X = X.astype(np.float32)
        else:
            X = np.array(X, dtype=np.float32)

        return self.net_.predict_proba(X)

    def score(self, X, y=None, sample_weight=None):
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
        return self.net_.history[:, "train_loss"] if self.net_.history else []

    @property
    def valid_history(self):
        return (
            self.net_.history[:, "valid_loss"]
            if self.net_.history and self.val_dataset
            else []
        )
