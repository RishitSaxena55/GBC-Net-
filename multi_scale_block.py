from imports import *

class MultiScaleBlock(nn.Module):
    """
    Multi-Scale Block from GBCNet
    Res2Net-inspired hierarchical multi-scale feature extraction
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels // 4

        self.conv_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_reduce(self.conv_reduce(x)))

        # Split into 4 slices
        slice_ch = x.shape[1] // 4
        X1, X2, X3, X4 = torch.split(x, slice_ch, dim=1)

        # Hierarchical processing
        Y1 = X1
        Y2 = self.relu(self.bn1(self.conv1(X2)))
        Y3 = self.relu(self.bn2(self.conv2(X3 + Y2)))
        Y4 = self.relu(self.bn3(self.conv3(X4 + Y3)))

        return torch.cat([Y1, Y2, Y3, Y4], dim=1)

