from dlordinal.output_layers import CLM
from torch.nn import Linear, Sequential

from ...classification import ResNet18Classifier
from ...losses import ProbCrossEntropyLoss

# from ...losses import ProbCrossEntropyLoss
# from .clm import CLM


class ResNet18CLMClassifier(ResNet18Classifier):
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
        interactive=False,
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
        )

    def get_model(self):
        model = super().get_model()
        model.fc = Sequential(
            Linear(model.fc.in_features, 1),
            CLM(
                self.num_classes,
                link_function=self.link_function,
                min_distance=self.min_distance,
            ),
        )
        return model.to(self.device)

    def get_thresholds(self):
        if self.net_ is None:
            raise ValueError("Model is not fitted yet")
        clm_layer = self.net_.module.fc[-1]
        return clm_layer._convert_thresholds(
            clm_layer.thresholds_b, clm_layer.thresholds_a, clm_layer.min_distance
        )

    def get_loss(self):
        return ProbCrossEntropyLoss(
            weight=self.computed_weights_, num_classes=self.num_classes
        ).to(self.device)
