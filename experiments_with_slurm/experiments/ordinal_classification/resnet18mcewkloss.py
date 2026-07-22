from dlordinal.losses import MCEAndWKLoss

from ..classification import ResNet18Classifier


class ResNet18MCEWK(ResNet18Classifier):
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
        C=0.5,
        penalization_type="quadratic",
        reduction="mean",
        use_logits=True,
        interactive=False,
    ):
        self.C = C
        self.penalization_type = penalization_type
        self.reduction = reduction
        self.use_logits = use_logits

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
        return MCEAndWKLoss(
            num_classes=self.num_classes,
            C=self.C,
            wk_penalization_type=self.penalization_type,
            weight=self.computed_weights_,
            reduction=self.reduction,
            use_logits=self.use_logits,
        ).to(self.device)
