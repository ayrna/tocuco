from abc import abstractmethod

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.nn import CrossEntropyLoss


class MLPBaseClassifier(BaseEstimator, ClassifierMixin):
    @abstractmethod
    def __init__(
        self,
        n_hidden_layers,
        n_hidden_units,
        learning_rate=1e-3,
        class_weights="balanced",
        max_iter=100,
        random_state=None,
        *,
        num_classes=None,
        device="cpu",
        train_targets=None,
        verbose=0,
        val_dataset=None,
        n_jobs=1,
        batch_size=128,
    ):
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.max_iter = max_iter
        self.random_state = random_state

        self.num_classes = num_classes
        self.input_shape = None
        self.device = device
        self.train_targets = train_targets
        self.verbose = verbose
        self.val_dataset = val_dataset
        self.n_jobs = n_jobs
        self.batch_size = batch_size

        self.model = None
        self.best_params_ = {}
        self._train_history = []

    @abstractmethod
    def _setup_model(self):
        self.model = None
        return self

    def _initialize(self, X, y):
        self.classes_ = np.unique(y)
        inferred_num_classes = len(self.classes_)
        if self.num_classes is None:
            self.num_classes = inferred_num_classes

        if self.num_classes < inferred_num_classes:
            raise ValueError(
                f"num_classes={self.num_classes} is smaller than classes in y={inferred_num_classes}."
            )

        self.input_shape = X.shape[1]
        self.max_iter = int(self.max_iter)

        if self.class_weights is None:
            class_weights_arr = np.ones(self.num_classes, dtype=np.float32)
        elif self.class_weights in ("equal", "balanced"):
            class_weights_arr = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(self.num_classes),
                y=y,
            )
        else:
            raise ValueError(
                f"Invalid class_weights value: {self.class_weights}. "
                + "Use None, 'equal' or 'balanced'."
            )

        self._class_weights = torch.from_numpy(class_weights_arr).float().to(self.device)

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        self._setup_model()

        if self.model is None:
            raise ValueError(
                "No model was set up. Override the _setup_model method to define the model"
            )

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.75,
            patience=50,
            min_lr=1e-5,
            threshold=1e-2,
        )

        return self

    def _compute_loss(self, y, pred):
        if not hasattr(self, "loss"):
            self.loss = CrossEntropyLoss(weight=self._class_weights)
        return self.loss(pred, y)

    def _check_X_y(self, X, y):
        if X.shape[1] != self.input_shape:
            raise ValueError(
                f"Input shape {X.shape[1:]} does not match expected shape {self.input_shape}."
            )

        if len(np.unique(y)) > self.num_classes:
            raise ValueError(
                f"Number of classes in y exceeds expected num_classes={self.num_classes}."
            )

        return X, y

    def fit(self, X, y=None, verbose=False):
        if y is None:
            y = self.train_targets
        if y is None:
            raise ValueError("Targets are required either in fit(y=...) or train_targets.")

        if hasattr(X, "data"):
            X = X.data
        if hasattr(y, "data"):
            y = y.data

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        first_pass = self.model is None
        if first_pass:
            self._initialize(X, y)

        self._check_X_y(X, y)

        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)

        self.model.train()
        self._train_history = []

        for epoch in range(self.max_iter):
            pred = self.model(X_t)
            loss = self._compute_loss(y_t, pred)

            self.optimizer.zero_grad()
            if loss is None:
                raise ValueError(
                    "Loss is None. Override the _compute_loss method to define the loss."
                )
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss)

            loss_value = float(loss.detach().cpu().item())
            self._train_history.append(loss_value)

            if verbose or self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_iter} - loss: {loss_value:.6f}")

        self.best_params_["max_iter"] = self.max_iter
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError(
                "No model was set up. Override the _setup_model method to define the model."
            )

        if hasattr(X, "data"):
            X = X.data
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_t = torch.from_numpy(X).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_t)
            probas = torch.softmax(out, dim=1)
        return probas.cpu().numpy()

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def score(self, X, y=None, sample_weight=None):
        if y is None:
            y = self.train_targets
        if y is None:
            raise ValueError("Targets are required for score when train_targets is None.")

        y = np.array(y)
        y_pred = self.predict(X)

        if sample_weight is None:
            return float(np.mean(y_pred == y))

        sample_weight = np.asarray(sample_weight)
        return float(np.average(y_pred == y, weights=sample_weight))

    @property
    def train_history(self):
        return self._train_history

    @property
    def valid_history(self):
        return []


class MLPModel(nn.Module):
    def __init__(
        self,
        n_hidden_layers,
        n_hidden_units,
        input_shape,
        num_classes,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()

        self.hidden_first = nn.Linear(input_shape, n_hidden_units)

        self.hidden_layers = None
        if n_hidden_layers - 1 > 0:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(n_hidden_units, n_hidden_units) for _ in range(n_hidden_layers - 1)]
            )

        self.classification = nn.Linear(n_hidden_units, num_classes)
        self.output = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        h = self.activation(self.hidden_first(x))
        if self.hidden_layers is not None:
            for hidden_layer in self.hidden_layers:
                h = self.activation(hidden_layer(h))
        classification = self.classification(h)
        out = self.output(classification)
        return out


class MLPManualClassifier(MLPBaseClassifier):
    def __init__(
        self,
        *,
        num_classes,
        n_hidden_layers=1,
        n_hidden_units=4,
        hidden_layers=None,
        hidden_units=None,
        learning_rate=1e-3,
        class_weights=None,
        class_weight=None,
        max_iter=100,
        random_state=None,
        device="cpu",
        train_targets=None,
        verbose=0,
        val_dataset=None,
        n_jobs=1,
        batch_size=128,
    ):
        # Keep original constructor args as attributes for sklearn get_params/clone.
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.class_weight = class_weight

        if hidden_layers is not None:
            n_hidden_layers = hidden_layers
        if hidden_units is not None:
            n_hidden_units = hidden_units

        if class_weights is None:
            class_weights = class_weight if class_weight is not None else "balanced"

        super().__init__(
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            learning_rate=learning_rate,
            class_weights=class_weights,
            max_iter=max_iter,
            random_state=random_state,
            num_classes=num_classes,
            device=device,
            train_targets=train_targets,
            verbose=verbose,
            val_dataset=val_dataset,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )

    def _setup_model(self):
        self.model = MLPModel(
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )
        return self


class MLPClassifier(MLPManualClassifier):
    pass
