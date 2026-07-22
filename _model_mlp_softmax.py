from torch import nn
from _model_mlp_base import MLPBaseClassifier


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
                [
                    nn.Linear(n_hidden_units, n_hidden_units)
                    for _ in range(n_hidden_layers - 1)
                ]
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


class MLPClassifier(MLPBaseClassifier):
    def __init__(
        self,
        n_hidden_layers=1,
        n_hidden_units=4,
        learning_rate=1e-3,
        class_weights=None,
        max_iter=100,
        random_state=None,
    ):
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
