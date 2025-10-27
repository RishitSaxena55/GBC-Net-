from imports import *
from mssop_layer import *

class MSSOPClassifier(nn.Module):
    """16-layer MS-SoP Classifier"""
    def __init__(self, in_ch, out_ch, H, W, Ho, Wo):
        super().__init__()

        # 16 MS-SoP layers
        self.layers = nn.ModuleList([
            MSSOPLayer(in_ch, out_ch, H, W, Ho, Wo),
            *[MSSOPLayer(out_ch, out_ch, H, W, Ho, Wo) for _ in range(15)]
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return x
