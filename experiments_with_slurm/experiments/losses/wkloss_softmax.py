from typing import Optional

import torch
from dlordinal.losses import WKLoss


class WKLossSoftmax(WKLoss):
    def __init__(
        self,
        num_classes: int,
        penalization_type: str = "quadratic",
        weight: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-10,
    ):
        super().__init__(num_classes, penalization_type, weight, epsilon)

    def forward(self, y_pred, y_true):
        # Apply softmax to the predictions and call the parent forward method
        return super().forward(torch.nn.functional.softmax(y_pred, dim=1), y_true)
