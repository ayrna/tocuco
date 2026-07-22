from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import one_hot


class CustomTargetsLoss(torch.nn.Module):
    """Base class to implement a soft labelling loss.

    Parameters
    ----------
    cls_probs : Tensor
        The class probabilities tensor.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    """

    def __init__(
        self,
        base_loss: Module,
        cls_probs: Tensor,
        eta: float = 1.0,
    ):
        super().__init__()

        self.base_loss = base_loss
        self.num_classes = cls_probs.size(0)
        self.eta = eta

        # Default class probs initialized to ones
        self.register_buffer("cls_probs", cls_probs.float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Method that is called to compute the loss.

        Parameters
        ----------
        input : Tensor
            The input tensor.
        target : Tensor
            The target tensor.

        Returns
        -------
        loss: Tensor
            The computed loss.
        """

        y_prob = self.get_buffer("cls_probs")[target]
        target_oh = one_hot(target, self.num_classes)

        y_true = (1.0 - self.eta) * target_oh + self.eta * y_prob

        return self.base_loss(input, y_true)
