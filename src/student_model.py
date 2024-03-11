import math
from typing import Tuple
import torch.nn.functional as F
from torch import Tensor
import torch
from torch import nn


class LightNN(nn.Module):
    def __init__(self, num_classes: int, in_features: Tuple[int, int]):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=math.prod(in_features), out_features=256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=3, padding=1,
                 pool=False, pool_kernel_size=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.conv_bn = nn.BatchNorm2d(out_channels)
        if pool:
            self.pooling = nn.MaxPool2d(pool_kernel_size)
        else:
            self.pooling = None

    def forward(self, x):
        conv = self.conv(x)
        out = F.relu(self.conv_bn(self.conv(x)))
        if self.pooling is not None:
            out = self.pooling(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=1) -> None:
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels,
                                     kernel_size, padding)
        self.conv_block2 = ConvBlock(in_channels, out_channels,
                                     kernel_size, padding)

    def forward(self, x):
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = out + residual
        return out


class ResNet9(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(ResNet9, self).__init__()
        # 1st and 2nd convolutional layer
        self.conv_block1 = ConvBlock(in_channels, 16)
        self.conv_block2 = ConvBlock(16, 32, pool=True)
        # residual block consisting of the 3rd and 4th convolutional layer
        self.res_block1 = ResidualBlock(32, 32)
        # 5th and 6th convolutional layers
        self.conv_block3 = ConvBlock(32, 64, pool=True)
        self.conv_block4 = ConvBlock(64, 128, pool=True)
        # residual block consisting of the 7th and 8th convolutional layer
        self.res_block2 = ResidualBlock(128, 128)
        # final fully-connected layer
        self.classifier = nn.Sequential(nn.MaxPool2d(3),
                                        nn.Flatten(),
                                        nn.Linear(3200, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        # x = x.permute(0, 3, 1, 2)
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.res_block1(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.res_block2(out)
        out = self.classifier(out)
        return out
