from imports import *
from multi_scale_block import *
from second_order_pooling_block import *

class MSSOPLayer(nn.Module):
    """Single MS-SoP Layer = MultiScale + SOP + Residual"""
    def __init__(self, in_ch, out_ch, H, W, Ho, Wo):
        super().__init__()
        self.multi_scale_block = MultiScaleBlock(in_ch, out_ch)
        self.second_order_pooling_block = SecondOrderPoolingBlock(
            out_ch, out_ch, H, W, Ho, Wo
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.multi_scale_block(x)
        residual = x
        x = self.second_order_pooling_block(x)
        x = x + residual
        out = self.relu(x)
        return out
