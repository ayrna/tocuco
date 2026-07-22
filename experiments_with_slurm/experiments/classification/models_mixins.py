from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Mixin:
    def build_base_model(self):
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
