from typing import Optional

import torch

from ..classification import XGBoostClassifier
from ..losses import SLACELoss


class XGBoostClassifierSLACE(XGBoostClassifier):
    def __init__(
        self,
        *,
        num_classes,
        n_estimators=None,
        max_iter=100,
        learning_rate=0.1,
        max_depth=10,
        subsample=1.0,
        colsample_bytree=0.5,
        alpha=0.1,
        use_logits=False,
        weight: Optional[torch.Tensor] = None,
        n_jobs=1,
        random_state=0,
        verbosity=0,
        class_weight=None,
        train_targets=None,
        device: str = "cpu",
    ):
        self.alpha = alpha
        self.use_logits = use_logits
        self.weight = weight
        self.max_iter = max_iter

        super().__init__(
            num_classes=num_classes,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=verbosity,
            class_weight=class_weight,
            train_targets=train_targets,
            device=device,
        )

    def get_loss(self):
        return SLACELoss(
            num_classes=self.num_classes,
            alpha=self.alpha,
            train_targets=self.train_targets,
            weight=self.computed_weights_,
            use_logits=self.use_logits,
        ).to(self.device)
