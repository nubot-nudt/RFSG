from typing import Type
import torch
import torch.nn as nn
from src.resnet import ResNet, BasicBlock
from einops import reduce
from habitat import logger




class ConditionResNet(ResNet):
    def forward(self, x):
        x = self.stem(x)

        interm_o = []
        for l in self.layers:
            x = l(x)
            interm_o.append(x)

        return x, interm_o
    
from src.FG_atten import GFMoudle
class FiLMedResNet(ResNet):
    def __init__(self, reduction, film_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.film = GFMoudle()
        self.film_layers = film_layers
        logger.info(f"filmed resnet encoder with layers: {film_layers}, reduction: {reduction}")
    
    def forward(self, x, x_cond):
        x = self.stem(x)

        self.x_aux = []
        for i,l in enumerate(self.layers):
            x = l(x)
            if i in self.film_layers:

                self.x_aux.append(x)
                x = self.film(x, x_cond[i], i)

        return x, self.x_aux

from src.image_graph import Image_Graph_Net
from src.GCN_layer import GCN
import random
class SpaceToDepth(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(N, C * 4, H // 2, W // 2)
        return x
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(y)
class self_conditiondistance(nn.Module):
    def __init__(self, channels):
        super(self_conditiondistance, self).__init__()

        self.down = SpaceToDepth()
        self.attend = SimAM()
        self.dist = nn.MSELoss()

    def forward(self, x1, x2):
        x1 = self.down(x1)
        F1, F2 = x1.chunk(2, dim=1)
        label_F = torch.mean(self.attend(x2.detach()), dim=1)
        dist_a = self.dist(torch.mean(self.attend(F1), dim=1), label_F)
        dist_b = self.dist(torch.mean(self.attend(F2), dim=1), label_F)
        dist_g = (torch.mean(dist_a)+torch.mean(dist_b))*0.2

        return dist_g
class MidFusionResNet(nn.Module):
    def __init__(
        self,
        reduction,
        film_layers,
        *args, **kwargs
    ):
        super().__init__()
        self.stem_o = FiLMedResNet(reduction, film_layers, *args, **kwargs)
        self.stem_g = ConditionResNet(*args, **kwargs)
        self.film_layers = film_layers
        self.final_spatial_compress = self.stem_o.final_spatial_compress

        self.final_channels = self.stem_o.final_channels+32

        self.max_node = 16
        self.image_graph = Image_Graph_Net(self.max_node)
        self.gcn = GCN(256+32, 32)
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 128,kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.conv_trans = nn.Conv2d(256, 32, kernel_size=1, stride=1)
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.final_channels,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, 32),
            nn.ReLU(True),
        )
        self.constant1 = nn.Sequential(
            self_conditiondistance(64),
            self_conditiondistance(128),
            self_conditiondistance(256)
        )
        self.constant2 = nn.Sequential(
            self_conditiondistance(64),
            self_conditiondistance(128),
            self_conditiondistance(256)
        )

    def forward(self, x):

        x_o = x[:,:3,...]
        x_g = x[:,3:,...]
        x_g, x_cond = self.stem_g(x_g)
        x_o, x_list = self.stem_o(x_o, x_cond)
        index_b = random.randint(0, x.shape[0] - 1)
        feat, adj = self.image_graph(x[index_b:index_b + 1, :3, ...], x[index_b:index_b + 1, 3:, ...])
        out = self.gcn(feat, adj)

        add_weight1 = self.constant1[0](x_list[0], x_list[1])+self.constant1[1](x_list[1], x_list[2])+self.constant1[2](x_list[2], x_list[3])
        add_weight2 = self.constant2[0](x_cond[0], x_cond[1])+self.constant2[1](x_cond[1], x_cond[2])+self.constant2[2](x_cond[2], x_cond[3])

        return self.compression(torch.cat([x_o, self.global_conv(x_cond[-1]) * (out.repeat(x_o.shape[0], 1, 4, 4))+self.conv_trans(x_cond[-1])], dim=1)), add_weight1+add_weight2



def resnet9(in_channels, base_planes, ngroups, film_reduction, film_layers):
    return MidFusionResNet(film_reduction, film_layers, 3, base_planes, ngroups, BasicBlock, [1, 1, 1, 1])


if __name__ == "__main__":
    mid_fusion_resnet = MidFusionResNet(
        3, 32, 16, BasicBlock, [1, 1, 1, 1]
    )
    dummy = torch.rand([2,6,128,128])
    out = mid_fusion_resnet(dummy)
    print(out.shape)