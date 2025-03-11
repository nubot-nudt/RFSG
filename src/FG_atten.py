import time
import torch.optim
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable

# weight decouple
class SEPAtten(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Conv2d(planes, planes,kernel_size=1,stride=1),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1,groups=planes),
            nn.Sigmoid()
        )
        self.beta = nn.Sequential(
            nn.Conv2d(planes, planes,kernel_size=1,stride=1),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1,groups=planes)
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, planes,kernel_size=1,stride=1),
            nn.ReLU(True),
            nn.Conv2d(planes, planes,kernel_size=1,stride=1)
        )
    def forward(self, x):
        gama = self.gamma(x)
        beta = self.beta(x)
        scale = self.scale(x)
        weight = scale*gama+beta
        return weight
# cross attention
class GFAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()

        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1)

        self.g_conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1)
        self.g_conv2 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)

        self.conv1 = SEPAtten(dim)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv2 = SEPAtten(dim)
        kernel_size = 3
        self.conv_1d1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x, conds):
        q, k = self.to_qk(conds).chunk(2, dim=1)
        dots = torch.matmul(q, k.transpose(-1, -2)) * ((conds.size(-1)) ** -0.5)
        attn = self.attend(dots)
        out = torch.matmul(attn, conds)

        c_w = self.g_conv1(self.avg(conds))
        c_w = self.conv_1d1(c_w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out_c = self.g_conv2(c_w)*out

        out_GF = (self.conv1(out_c)*x)+(self.conv2(conds)*conds)
        return out_GF

class SE(nn.Module):
    def __init__(self, planes, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.squeeze(x)
        x1 = x1.view(b, c)
        x1 = self.excite(x1)

        return x1.view(b, c, 1, 1)*x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1)*x




class GFMoudle(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def __init__(self, channels=[32, 64, 128, 256], res=[32, 16, 8, 4], reduction="none"):
        super().__init__()
        self.reduction = reduction
        # if self.reduction == "none":
        head_list = [32, 16, 8, 4]
        self.f = nn.ModuleList([
                GFAttention(channels[i], head_list[i])
                for i in range(len(channels))
            ])

    def forward(self, x, conds, i):
        # if self.reduction == "none":
        out = self.f[i](x, conds)

        return out
if __name__ == "__main__":
    part_model = GFMoudle()
    print(part_model)
    dummy = torch.rand([2,256,4,4])
    # dummy1 = torch.rand([2, 256, 4, 4])
    # dummy2 = torch.rand([2, 256, 4, 4])
    # dist = F.cosine_similarity(dummy, dummy1)
    # print(dist.shape, dist)
    # dist = F.cosine_similarity(dummy *dummy2, dummy1*dummy2)
    # print(dist.shape, dist)
    out = part_model(dummy, dummy, 3)
    print(out.shape)