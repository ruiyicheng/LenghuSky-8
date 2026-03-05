# models/segman_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


class LayerNorm2d(nn.Module):
    """2D 层归一化"""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class BasicBlock(nn.Module):
    """基础残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SegMANEncoder(nn.Module):
    """SegMAN 编码器核心实现"""

    def __init__(self, variant='tiny'):
        super().__init__()

        # 配置参数
        if variant == 'tiny':
            channels = [32, 64, 128, 256]
            blocks = [2, 2, 2, 2]
        elif variant == 'small':
            channels = [64, 128, 256, 512]
            blocks = [2, 2, 3, 2]
        elif variant == 'base':
            channels = [96, 192, 384, 768]
            blocks = [2, 2, 4, 2]
        elif variant == 'large':
            channels = [128, 256, 512, 1024]
            blocks = [3, 3, 5, 3]
        else:
            raise ValueError(f"未知变体: {variant}")

        self.variant = variant
        self.channels = channels  # 添加channels属性供解码器使用

        # 输入stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # 构建四个阶段
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = self._make_stage(
                channels[i] if i == 0 else channels[i - 1],
                channels[i],
                blocks[i],
                stride=2 if i > 0 else 1
            )
            self.stages.append(stage)

        # 初始化权重
        self._init_weights()

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


# 变体函数
def SegMANEncoder_t(**kwargs):
    return SegMANEncoder('tiny', **kwargs)


def SegMANEncoder_s(**kwargs):
    return SegMANEncoder('small', **kwargs)


def SegMANEncoder_b(**kwargs):
    return SegMANEncoder('base', **kwargs)


def SegMANEncoder_l(**kwargs):
    return SegMANEncoder('large', **kwargs)