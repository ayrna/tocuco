import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(BaseEstimator, ClassifierMixin):
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
        self.best_params_ = {}
        self.net_ = None

        self.initialize()

    def initialize(self):
        if self.class_weight is not None and self.train_targets is None:
            raise ValueError("class_weight requires train_targets")

        if not isinstance(self.train_targets, list) and not isinstance(
            self.train_targets, np.ndarray
        ):
            raise ValueError("train_targets must be a list or numpy array")

        self.computed_weights_ = torch.tensor(
            compute_class_weight(
                self.class_weight,
                classes=np.arange(self.num_classes),
                y=self.train_targets,
            ),
            dtype=torch.float32,
        ).to(self.device)

        self.net_ = NeuralNetClassifier(
            module=self.get_model().to(self.device),
            criterion=self.get_loss().to(self.device),  # type: ignore
            optimizer=AdamW,
            lr=self.learning_rate,
            max_epochs=self.max_iter,
            train_split=predefined_split(self.val_dataset),  # type: ignore
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
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(self.device)
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes).to(
            self.device
        )
        return model

    def get_callbacks(self):
        callbacks = []
        if self.val_dataset is not None:
            callbacks.append(
                ("early_stopping", EarlyStopping(patience=40, load_best=True))
            )
            # callbacks.append(
            #     (
            #         "lr_scheduler",
            #         LRScheduler(StepLR, monitor="val_loss", step_size=5, gamma=0.95),
            #     )
            # )
        return callbacks

    def fit(self, X, y=None, **fit_params):
        if y is None:
            y = self.train_targets
        y = np.array(y)

        r = self.net_.fit(X, y, **fit_params)

        # Save best epoch from early stopping callback
        for name, callback in self.net_.callbacks_:
            if name == "early_stopping":
                self.best_params_["max_iter"] = callback.best_epoch_
                break

        return r

    def predict(self, X):
        return self.net_.predict(X)

    def predict_proba(self, X):
        return self.net_.predict_proba(X)

    def score(self, X, y=None, sample_weight=None):
        if y is None:
            y = self.train_targets
        y = np.array(y)
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
