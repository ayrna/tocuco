from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SLACELoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        num_classes: int,
        train_targets: Tensor,
        weight: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.train_targets = train_targets
        self.use_logits = use_logits
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.use_logits:
            input = F.softmax(input, dim=1)

        phi = torch.abs(
            torch.arange(self.num_classes, device=input.device).view(1, -1)
            - target.double().view(-1, 1)
        )

        softmax_targets = F.softmax(-self.alpha * phi, dim=1).to(input.device)
        one_hot_target = F.one_hot(target, num_classes=self.num_classes).to(
            input.device
        )
        one_hot_target_comp = 1 - one_hot_target
        mass_weights = (
            one_hot_target * softmax_targets + one_hot_target_comp * softmax_targets
        )

        labels = torch.arange(self.num_classes, device=input.device)
        h = labels.view(-1, 1, 1)
        i = labels.view(1, -1, 1)
        j = labels.view(1, 1, -1)

        distance_i = torch.abs(i - h)
        distance_j = torch.abs(j - h)

        prox_dom = (distance_j <= distance_i).float()

        accumulating_softmax = (
            torch.matmul(
                prox_dom[target.long()].double(),
                torch.unsqueeze(input, 2).double(),
            )
            .double()
            .squeeze(dim=2)
        )

        per_sample_loss = -torch.sum(
            mass_weights * torch.log(accumulating_softmax + 1e-9), dim=1
        )  # [Batch]

        if self.weight is not None:
            sample_weights = self.weight[target].to(input.device)  # [B]
            per_sample_loss = per_sample_loss * sample_weights  # [B]

        return per_sample_loss.mean()
