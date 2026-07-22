from ..classification import ResNet18Classifier
from ..losses import SORDLoss


class ResNet18SORD(ResNet18Classifier):
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
        alpha=1.0,
        use_logits=True,
        prox=False,
        ftype="max",
        interactive=False,
    ):
        self.alpha = alpha
        self.use_logits = use_logits
        self.train_targets = train_targets
        self.prox = prox
        self.ftype = ftype

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
        return SORDLoss(
            alpha=self.alpha,
            num_classes=self.num_classes,
            train_targets=self.train_targets,
            prox=self.prox,
            ftype=self.ftype,
            weight=self.computed_weights_,
            use_logits=self.use_logits,
        ).to(self.device)
