from typing import Optional

import torch

from ..classification import MLPClassifier
from ..losses import SORDLoss


class MLPSordClassifier(MLPClassifier):
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
        hidden_units=64,
        hidden_layers=1,
        alpha: float,
        prox: bool = False,
        ftype: str = "max",
        weight: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ):

        self.alpha = alpha
        self.prox = prox
        self.ftype = ftype
        self.weight = weight
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
            hidden_units=hidden_units,
            hidden_layers=hidden_layers,
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
