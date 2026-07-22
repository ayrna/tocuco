import torch
from dlordinal.losses import BetaLoss


class BetaSoftLabelMixin:
    def __init__(self, *, num_classes, loss_eta=1.0, **kwargs):
        self.num_classes = num_classes
        self.loss_eta = loss_eta
        super().__init__(num_classes=num_classes, **kwargs)

    def build_loss(self, computed_class_weights) -> torch.nn.Module:
        return BetaLoss(
            base_loss=super().build_loss(computed_class_weights),
            num_classes=self.num_classes,
            eta=self.loss_eta,
        )
