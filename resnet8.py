import torch
from torch import nn
from torch.nn import functional as F

######################## ADAPT ########################
from adapt.approx_layers import axx_layers as approxNN

#set flag for use of AdaPT custom layers or vanilla PyTorch
use_adapt=True

#set axx mult. default = accurate
axx_mult_global = 'mul8s_acc'
#######################################################

__all__ = [
    "resnet8",
]

class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        if use_adapt:
            self.block = nn.Sequential(
            approxNN.AdaPT_Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                stride=stride,
                axx_mult=axx_mult_global
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            approxNN.AdaPT_Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                axx_mult=axx_mult_global
            ),
            nn.BatchNorm2d(num_features=out_channels),
          )
        else:
            self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                stride=stride,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=out_channels),
          )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            if use_adapt:
              self.residual = approxNN.AdaPT_Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                axx_mult=axx_mult_global,
                bias=True,
              )
            else:
              self.residual = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=True,
              )

    def forward(self, inputs):
        x = self.block(inputs)
        y = self.residual(inputs)
        return F.relu(x + y)


class Resnet8v1EEMBC(nn.Module):
    def __init__(self):
        super().__init__()
        if use_adapt:
          self.stem = nn.Sequential(
            approxNN.AdaPT_Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True, axx_mult='mul8s_acc'
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
          )
        else:
          self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
          )

        self.first_stack = ResNetBlock(in_channels=16, out_channels=16, stride=1)
        self.second_stack = ResNetBlock(in_channels=16, out_channels=32, stride=2)
        self.third_stack = ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.first_stack(x)
        x = self.second_stack(x)
        x = self.third_stack(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def resnet8(axx_mult = 'mul8s_acc'):
    global axx_mult_global
    axx_mult_global = axx_mult
    model = Resnet8v1EEMBC() 
    return model