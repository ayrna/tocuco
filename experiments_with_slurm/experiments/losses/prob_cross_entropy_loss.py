from typing import Optional

import torch
from torch import Tensor
from torch.nn import KLDivLoss


class ProbCrossEntropyLoss(torch.nn.Module):
    """CrossEntropyLoss that should receive probabilities as input."""

    def __init__(self, num_classes, weight: Optional[Tensor] = None) -> None:
        super().__init__()
        self.weight = weight
        self.kld_loss = (KLDivLoss(reduction="none"),)
        self.num_classes = num_classes

    def forward(self, input, target) -> Tensor:
        log_preds = torch.log(input + 1e-9)
        kld = torch.nn.KLDivLoss(reduction="none")

        # Convert target to one-hot
        if target.dim() == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).float()

        kld_loss = kld(log_preds, target)

        if self.weight is not None:
            sample_weight = self.weight[target.argmax(dim=1)]
            kld_loss = kld_loss * sample_weight.unsqueeze(1)

        return kld_loss.sum(dim=1).mean()
