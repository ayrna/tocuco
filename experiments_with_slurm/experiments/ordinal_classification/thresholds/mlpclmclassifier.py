from dlordinal.output_layers import CLM
from torch.nn import Linear, Sequential

from ...classification import MLPClassifier
from ...losses import ProbCrossEntropyLoss


class MLPCLMClassifier(MLPClassifier):
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
        link_function="logit",
        min_distance=0.0,
        hidden_units=64,
        hidden_layers=1,
    ):

        self.link_function = link_function
        self.min_distance = min_distance

        super().__init__(
            num_classes=num_classes,
            device=device,
            max_iter=max_iter,
            class_weight=class_weight,
            train_targets=train_targets,
            learning_rate=learning_rate,
            verbose=verbose,
            batch_size=batch_size,
            val_dataset=val_dataset,
            n_jobs=n_jobs,
            hidden_units=hidden_units,
            hidden_layers=hidden_layers,
        )

    def get_model(self):
        model = super().get_model()
        # Replace the last layer of the network with CLM
        layers = list(model.network.children())[:-1]  # Remove last Linear layer
        layers.append(Linear(self.hidden_units, 1))
        layers.append(
            CLM(
                self.num_classes,
                link_function=self.link_function,
                min_distance=self.min_distance,
            )
        )
        model.network = Sequential(*layers)
        return model.to(self.device)

    def get_thresholds(self):
        if self.net_ is None:
            raise ValueError("Model is not fitted yet")
        clm_layer = self.net_.module_.network[-1]
        return clm_layer._convert_thresholds(
            clm_layer.thresholds_b, clm_layer.thresholds_a, clm_layer.min_distance
        )

    def get_loss(self):
        return ProbCrossEntropyLoss(
            weight=self.computed_weights_, num_classes=self.num_classes
        ).to(self.device)
