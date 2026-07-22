import torch
from dlordinal.losses import WKLoss

from ..losses import ProbCrossEntropyLoss


class WKMixin:
    def __init__(self, *, num_classes, penalization_type="quadratic", **kwargs):
        self.num_classes = num_classes
        self.penalization_type = penalization_type
        super().__init__(num_classes=num_classes, **kwargs)

    def build_loss(self, computed_class_weights) -> torch.nn.Module:
        return WKLoss(
            num_classes=self.num_classes,
            penalization_type=self.penalization_type,
            weight=computed_class_weights,
        )


class ProbCrossEntropyMixin:
    def __init__(self, *, num_classes, **kwargs):
        self.num_classes = num_classes
        super().__init__(num_classes=num_classes, **kwargs)

    def build_loss(self, computed_class_weights) -> torch.nn.Module:
        return ProbCrossEntropyLoss(
            num_classes=self.num_classes,
            weight=computed_class_weights,
        )
