from dlordinal.losses import BetaLoss
from torch.nn import CrossEntropyLoss

from ...classification import ResNet18Classifier


class ResNet18Beta(ResNet18Classifier):
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
        loss_eta=0.85,
    ):
        self.loss_eta = loss_eta

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

    def get_loss(self):
        return BetaLoss(
            base_loss=CrossEntropyLoss(
                weight=self.computed_weights_,
            ),
            num_classes=self.num_classes,
            params_set="standard",
            eta=self.loss_eta,
        ).to(self.device)
