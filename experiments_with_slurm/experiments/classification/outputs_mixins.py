from dlordinal.output_layers import CLM
from torch.nn import Linear, Sequential


class CLMMixin:
    def __init__(
        self, *, num_classes, link_function="logit", min_distance=0.0, **kwargs
    ):

        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance

        super().__init__(num_classes=num_classes, **kwargs)

    def build_final_layer(self, in_features):
        return Sequential(
            Linear(in_features, 1),
            CLM(
                self.num_classes,
                link_function=self.link_function,
                min_distance=self.min_distance,
            ),
        )
