from imports import *

class SecondOrderPoolingBlock(nn.Module):
    """
    Second-Order Pooling Block from GBCNet
    Computes covariance-based attention weights
    """
    def __init__(self, in_ch, out_ch, H, W, Ho, Wo):
        super().__init__()

        # Channel pathway
        self.conv1_c = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv2_c = nn.Conv2d(out_ch, 4 * out_ch, kernel_size=(1, out_ch))
        self.conv3_c = nn.Conv2d(4 * out_ch, in_ch, kernel_size=1)

        # Height pathway
        self.conv1_h = nn.Conv2d(H, Ho, kernel_size=1)
        self.conv2_h = nn.Conv2d(Ho, 4 * Ho, kernel_size=(1, Ho))
        self.conv3_h = nn.Conv2d(4 * Ho, H, kernel_size=1)

        # Width pathway
        self.conv1_w = nn.Conv2d(W, Wo, kernel_size=1)
        self.conv2_w = nn.Conv2d(Wo, 4 * Wo, kernel_size=(1, Wo))
        self.conv3_w = nn.Conv2d(4 * Wo, W, kernel_size=1)

    def covariance(self, x):
        """Compute normalized covariance matrix"""
        mean = torch.mean(x, dim=2, keepdim=True)
        x_centered = x - mean
        C = torch.matmul(x_centered, x_centered.transpose(1, 2))
        C = C / x.size(2)
        # Numerical stability
        eye = torch.eye(C.size(1)).to(C.device).unsqueeze(0)
        C = C + 1e-5 * eye
        return C

    def get_weight(self, C, layer2, layer3):
        C = torch.unsqueeze(C, dim=2)
        C = layer2(C)
        w = layer3(C)
        return w

    def forward(self, x):
        org_x = x

        # Channel pathway
        x_c = self.conv1_c(x)
        x_c = torch.reshape(x_c, (x_c.shape[0], x_c.shape[1], -1))
        C_c = self.covariance(x_c)
        w_d = self.get_weight(C_c, self.conv2_c, self.conv3_c)
        Z_d = org_x * w_d

        # Height pathway
        x_h = torch.transpose(org_x, 1, 2)
        org_x_h = x_h
        x_h = self.conv1_h(x_h)
        x_h = torch.reshape(x_h, (x_h.shape[0], x_h.shape[1], -1))
        C_h = self.covariance(x_h)
        w_h = self.get_weight(C_h, self.conv2_h, self.conv3_h)
        Z_h = org_x_h * w_h

        # Width pathway
        x_w = torch.transpose(org_x, 1, 3)
        org_x_w = x_w
        x_w = self.conv1_w(x_w)
        x_w = torch.reshape(x_w, (x_w.shape[0], x_w.shape[1], -1))
        C_w = self.covariance(x_w)
        w_w = self.get_weight(C_w, self.conv2_w, self.conv3_w)
        Z_w = org_x_w * w_w

        # Combine
        Z_h_T = torch.transpose(Z_h, 1, 2)
        Z_w_T = torch.transpose(Z_w, 1, 3)
        Z = Z_d + Z_h_T + Z_w_T

        return Z

