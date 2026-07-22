from .baseclassifier import BaseClassifier
from .labelling_mixins import BetaSoftLabelMixin
from .losses_mixins import ProbCrossEntropyMixin, WKMixin
from .models_mixins import ResNet18Mixin
from .outputs_mixins import CLMMixin


class ResNet18Classifier(ResNet18Mixin, BaseClassifier):
    pass


class ResNet18CLMClassifier(
    CLMMixin, ProbCrossEntropyMixin, ResNet18Mixin, BaseClassifier
):
    pass


class ResNet18BetaClassifier(BetaSoftLabelMixin, ResNet18Mixin, BaseClassifier):
    pass


class ResNet18CLMWKClassifier(WKMixin, CLMMixin, ResNet18Mixin, BaseClassifier):
    pass


class ResNet18CLMBetaClassifier(
    BetaSoftLabelMixin, ProbCrossEntropyMixin, CLMMixin, ResNet18Mixin, BaseClassifier
):
    pass


class ResNet18CLMWKBetaClassifier(
    BetaSoftLabelMixin, WKMixin, CLMMixin, ResNet18Mixin, BaseClassifier
):
    pass
