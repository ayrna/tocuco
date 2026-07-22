import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin


class ordinal_Network:
    """
    A lightweight Multi-Layer Perceptron architecture optimized using a custom
    loss function. The objective enforces monotonic constraints on adjacent output
    nodes to preserve ordinal consistency across classes.

    Attributes:
        n_input (int): Number of input features.
        n_hidden (int): Number of units in the hidden layer.
        n_output (int): Number of output nodes (equivalent to number of classes).
        net (torch.nn.Sequential): PyTorch sequential model container.
    """

    def __init__(self, n_input, n_hidden, n_output):
        """Initializes the base network layer parameters and architecture.

        Args:
            n_input (int): Input feature dimensions.
            n_hidden (int): Hidden layer feature dimensions.
            n_output (int): Total target classes mapped into the ordinal array.
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.net = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_output),
            nn.Tanh(),
        )

    def fit(self, X, y, sample_weight, learning_rate=0.001, max_iter=1000):
        """Fits the single neural network base using Adam optimization.

        The total cost penalizes standard mean squared deviation weighted by AdaBoost
        sample importance, combined with an upper-bounded restriction preventing
        higher-order threshold inversions.

        Args:
            X (torch.Tensor): Feature training tensor of shape (n_samples, n_features).
            y (torch.Tensor): Target encoded ordinal tensor of shape (n_samples, n_classes).
            sample_weight (np.ndarray): Adaptive weight distribution from the outer loop.
            learning_rate (float, optional): Optimization learning rate. Defaults to 0.001.
            max_iter (int, optional): Epoch training iterations. Defaults to 1000.
        """
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32)
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        for _ in range(max_iter):
            hypothesis = self.net(X)

            # Core weighted structural error
            cost = ((y - hypothesis) * (y - hypothesis) * sample_weight).sum()

            # Monotonic ordinal ordering penalty function
            loss = torch.max(
                torch.tensor(0.0),
                hypothesis[:, 0 : (self.n_output - 1)] - hypothesis[:, 1 : self.n_output],
            ).sum() / (len(X) * (self.n_output - 1))

            optimizer.zero_grad()
            cost = cost + loss * (0.5)
            cost.backward()
            optimizer.step()


class OEABClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal Ensemble AdaBoost (OEAB) Classifier.

    An adaptive boosting ensemble designed specifically for targets that exhibit an
    inherent rank or structural ordering. It wraps an array of neural net base learners
    and handles complex target encoding/decoding while maintaining scikit-learn compatibility.

    Attributes:
        num_classes (int): Absolute number of distinct ordinal classes.
        n_estimators (int): Quantity of stacked learners to optimize.
        n_hidden (int): Hidden units assigned per individual network estimator.
        learning_rate (float): Base step size parameters for sub-learners.
        max_iter (int): Epoch limit per internal training step.
        verbose (int): Output logging verbosity tracker (0 for quiet).
        random_state (int, optional): Pseudo-random seed control parameter.
        train_targets (array-like, optional): Fallback target container.
        estimator_list (list): Storage of initialized and optimized weak networks.
        alpha_list (np.ndarray): Computed estimator trust coefficients.
        enco_y (dict): Mapped hash-table linking actual values to binary arrays.
        class_order_ (np.ndarray): Inferred operational array of target positions.
        classes_ (np.ndarray): Scikit-learn required alias referencing class order.
        best_params_ (dict): Tracked optimal training state summary parameters.
    """

    def __init__(
        self,
        *,
        num_classes,
        n_estimators=10,
        n_hidden=6,
        learning_rate=0.001,
        max_iter=1000,
        verbose=0,
        random_state=None,
        train_targets=None,
    ):
        """Initializes the OEAB ensemble model setup configurations."""
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.train_targets = train_targets

        self.estimator_list = []
        self.alpha_list = []
        self.enco_y = {}
        self.best_params_ = {}

    def _ensure_array(self, X):
        """Safely processes various input types into standard NumPy ndarrays.

        Handles PyTorch tensors across varying devices and native structures smoothly.

        Args:
            X (array-like or torch.Tensor): Arbitrary raw feature input data.

        Returns:
            np.ndarray: Uniformly extracted feature array.
        """
        if hasattr(X, "data"):
            X = X.data
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        elif hasattr(X, "numpy"):
            X = X.cpu().numpy() if hasattr(X, "cpu") else X.numpy()

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        return X

    def _resolve_class_order(self, y):
        """Resolves target categorical listings into an ordered sequence array.

        Args:
            y (np.ndarray): Raw array containing target training labels.

        Returns:
            np.ndarray: Sorted array representing the operational ordinal classes.

        Raises:
            ValueError: If decoded class count contradicts pre-configured `num_classes`.
        """
        y = np.asarray(y)

        if self.train_targets is not None:
            classes = np.unique(np.asarray(self.train_targets))
        else:
            min_class = int(np.min(y))
            classes = np.arange(min_class, min_class + self.num_classes)

        classes = np.sort(classes)
        if len(classes) != self.num_classes:
            raise ValueError(
                f"num_classes ({self.num_classes}) does not match the number of classes inferred ({len(classes)})"
            )

        return classes

    def encoding_y(self, classes):
        """Constructs an ordered cumulative step matrix for ordinal target representations.

        Transforms simple ranking integers into complex multidimensional steps composed
        of 1 and -1 values to denote relative higher/lower group status.

        Args:
            classes (np.ndarray): Ordered reference array of classes.

        Returns:
            self: Configured object with a populated `enco_y` mapping dictionary.
        """
        self.n_class = self.num_classes
        en = np.ones((self.n_class, self.n_class), dtype=np.float32)
        for i in range(self.n_class):
            if i > 0:
                en[i, 0:i] = -1

        d = {}
        for i, j in enumerate(classes):
            d[j] = en[i, :]
        self.enco_y = d
        return self

    def fit(self, X, y=None, **fit_params):
        """Fits the complete OEAB Ensemble via an adapted AdaBoost sequence.

        Sequentially trains internal sub-networks, calculates performance margins
        against the ordinal subspace matrix, and updates example weight distributions.

        Args:
            X (array-like): Input sample feature matrix of shape (n_samples, n_features).
            y (array-like, optional): Correct ground-truth target ranking labels of shape (n_samples,).
            **fit_params: Arbitrary additional fitting parameters.

        Returns:
            self: The fully trained OEAB Classifier instance.
        """
        if y is None:
            y = self.train_targets
        if y is None:
            raise ValueError("y is required when train_targets is None")

        # Standardizing input formats
        X = self._ensure_array(X)
        y = np.asarray(y)
        self.class_order_ = self._resolve_class_order(y)
        self.classes_ = self.class_order_.copy()

        # Applying execution seed if present
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        N = len(y)
        self.n_input = X.shape[1]
        self.estimator_list, self.alpha_list = [], []

        # Target layout preprocessing
        self.encoding_y(self.class_order_)
        enco_tr_Y = list(map(lambda x: self.enco_y[x], y))
        enco_tr_Y = np.array(enco_tr_Y)

        # PyTorch network variable transformations
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tr_tensor = torch.tensor(enco_tr_Y, dtype=torch.float32)

        # Global matrix sample cell weights allocation initialization
        sample_weight = np.ones((N, self.n_class)) / (N * self.n_class)

        for m in range(self.n_estimators):
            network = ordinal_Network(self.n_input, self.n_hidden, self.n_class)
            network.fit(
                X_tensor,
                Y_tr_tensor,
                sample_weight,
                self.learning_rate,
                self.max_iter,
            )

            with torch.no_grad():
                y_predict = network.net(X_tensor).numpy()

            # Reverse decoding of target vectors via dot product matching scores
            pre_y = np.zeros(len(y_predict))
            for i in range(len(y_predict)):
                score = {}
                for j in self.enco_y.keys():
                    score[j] = np.sum(y_predict[i] * self.enco_y[j])
                pre_y[i] = max(score, key=score.get)

            g_vector = np.array(list(map(lambda x: self.enco_y[x], pre_y)))

            # Computing iteration-specific step error rate
            err = ((enco_tr_Y != g_vector) * sample_weight).sum() / (sample_weight.sum())
            if err < 0.001:
                err = 0.001

            # Estimator confidence score estimation (Alpha)
            alpha = np.log((1.0 - err) / err) / 2
            if alpha < 0:
                alpha = 0.0

            # Exponential sample weight recalculation
            sample_weight *= (enco_tr_Y == g_vector) * np.exp(-alpha) + (
                enco_tr_Y != g_vector
            ) * np.exp(alpha)
            sample_weight = sample_weight / np.sum(sample_weight)

            self.estimator_list.append(network)
            self.alpha_list.append(alpha)

            if self.verbose and (m + 1) % max(1, self.n_estimators // 5) == 0:
                print(f"[OEAB] trained estimators: {m + 1}/{self.n_estimators}")

        self.alpha_list = np.asarray(self.alpha_list)
        self.best_params_ = {
            "n_estimators": self.n_estimators,
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
        }
        return self

    def decision_function(self, X):
        """Calculates raw continuous ensemble cumulative decision margins.

        Aggregates outputs from the sequence of underlying networks weighted
        by their respective trust values (alphas).

        Args:
            X (array-like): Input feature records of shape (n_samples, n_features).

        Returns:
            np.ndarray: Raw ensemble log-margin matrix of shape (n_samples, n_classes).
        """
        X = self._ensure_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        f_vector_test = np.zeros(shape=(len(X), self.n_class))

        for m in range(self.n_estimators):
            with torch.no_grad():
                y_predict = self.estimator_list[m].net(X_tensor).numpy()

            score = np.zeros(shape=(len(X), self.n_class))
            for i, j in enumerate(self.class_order_):
                score[:, i] = np.dot(y_predict, self.enco_y[j])

            pre_idx = np.argmax(score, axis=1)
            pre_y = self.class_order_[pre_idx]
            g_vector = np.array(list(map(lambda x: self.enco_y[x], pre_y)))
            f_vector_test += g_vector * self.alpha_list[m]

        return f_vector_test

    def predict_proba(self, X):
        """Calculates class membership probability distributions.

        Converts raw margins using a cumulative sigmoid link function and maps
        adjacent threshold interval differences (p - s) into properly bound
        and normalized probabilities.

        Args:
            X (array-like): Feature array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Computed probability array of shape (n_samples, n_classes),
                where each row sums to exactly 1.0.
        """
        f_vector_test = self.decision_function(X)

        # Cumulative sigmoid transformation mapping
        p = np.exp(2 * f_vector_test) / (1 + np.exp(2 * f_vector_test))
        s = np.zeros(p.shape)
        s[:, 1:] = p[:, 0 : (p.shape[1] - 1)]

        # Class probability extraction via interval splits
        probabilities = p - s

        # Numerical stabilization, clipping bounds and normalizing steps
        probabilities = np.clip(probabilities, 0.0, 1.0)

        denom = np.sum(probabilities, axis=1, keepdims=True)
        denom = np.where(denom <= 0.0, 1.0, denom)
        probabilities = probabilities / denom

        # Safety handler preventing rows from collapsing to absolute zero
        zero_rows = np.sum(probabilities, axis=1) == 0.0
        if np.any(zero_rows):
            probabilities[zero_rows] = 1.0 / self.n_class

        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return probabilities

    def predict(self, X):
        """Predicts the optimal rank order labels for the provided records.

        Args:
            X (array-like): Test set features matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Categorical target array containing predicted class ranks.
        """
        X = self._ensure_array(X)
        probabilidades = self.predict_proba(X)
        pred_idx = np.argmax(probabilidades, axis=1)
        preds = self.class_order_[pred_idx]
        return preds

    def score(self, X, y=None, sample_weight=None):
        """Computes classification accuracy following the scikit-learn API.

        This method is used by default in many sklearn utilities (e.g. cross-validation,
        model evaluation helpers) when no custom scorer is provided.

        Args:
            X (array-like): Evaluation features matrix of shape (n_samples, n_features).
            y (array-like, optional): Associated target actual ground-truth labels.
            sample_weight (array-like, optional): Performance evaluation custom example weights.

        Returns:
            float: Mean accuracy (or weighted mean accuracy when `sample_weight` is given).
        """
        if y is None:
            y = self.train_targets
        if y is None:
            raise ValueError("y is required when train_targets is None")

        X = self._ensure_array(X)
        y = np.asarray(y)
        preds = self.predict(X)
        if sample_weight is None:
            return np.mean(preds == y)

        return np.average(preds == y, weights=sample_weight)
