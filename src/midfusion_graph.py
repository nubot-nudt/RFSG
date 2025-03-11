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

        for i,l in enumerate(self.layers):
            x = l(x)
            if i in self.film_layers:
                x = self.film(x, x_cond[i], i)

        return x

from src.image_graph import Image_Graph_Net
from src.GCN_layer import GCN
import random
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
    def forward(self, x):
        b,c,h,w = x.shape
        # print(x.shape)
        x_o = x[:,:3,...]
        x_g = x[:,3:,...]
        x_g, x_cond = self.stem_g(x_g)
        x_o = self.stem_o(x_o, x_cond)
        index_b = random.randint(0, x.shape[0] - 1)
        feat, adj = self.image_graph(x[index_b:index_b + 1, :3, ...], x[index_b:index_b + 1, 3:, ...])
        out = self.gcn(feat, adj)
        return self.compression(torch.cat([x_o, self.global_conv(x_cond[-1]) * (out.repeat(x_o.shape[0], 1, 4, 4))+self.conv_trans(x_cond[-1])], dim=1))



def resnet9(in_channels, base_planes, ngroups, film_reduction, film_layers):
    return MidFusionResNet(film_reduction, film_layers, 3, base_planes, ngroups, BasicBlock, [1, 1, 1, 1])


if __name__ == "__main__":
    mid_fusion_resnet = MidFusionResNet(
        3, 32, 16, BasicBlock, [1, 1, 1, 1]
    )
    dummy = torch.rand([2,6,128,128])
    out = mid_fusion_resnet(dummy)
    print(out.shape)