from .cdw import CDWCELoss
from .custom_targets_loss import CustomTargetsLoss
from .prob_cross_entropy_loss import ProbCrossEntropyLoss
from .slace import SLACELoss
from .sord import SORDLoss
from .wkloss_softmax import WKLossSoftmax

__all__ = [
    "CDWCELoss",
    "CustomTargetsLoss",
    "WKLossSoftmax",
    "ProbCrossEntropyLoss",
    "SLACELoss",
    "SORDLoss",
]
