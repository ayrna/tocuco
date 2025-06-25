from _model_mlp_base import MLPBaseClassifier
from _model_mlp_softmax import MLPModel
from dlordinal.losses import TriangularLoss
from torch.nn import CrossEntropyLoss


class MLPTriangularClassifier(MLPBaseClassifier):
    def __init__(
        self,
        n_hidden_layers=1,
        n_hidden_units=4,
        t_alpha=0.01,
        learning_rate=1e-3,
        class_weights=None,
        max_iter=100,
        random_state=None,
    ):
        self.t_alpha = t_alpha
        super().__init__(
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            learning_rate=learning_rate,
            class_weights=class_weights,
            max_iter=max_iter,
            random_state=random_state,
        )

    def _setup_model(self):
        self.model = MLPModel(
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_units=self.n_hidden_units,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )
        return self

    def _compute_loss(self, y, pred):
        if not hasattr(self, "loss"):
            self.loss = TriangularLoss(
                base_loss=CrossEntropyLoss(weight=self._class_weights),
                num_classes=self.num_classes,
                alpha2=self.t_alpha,
            )
        loss = self.loss(pred, y)

        return loss
