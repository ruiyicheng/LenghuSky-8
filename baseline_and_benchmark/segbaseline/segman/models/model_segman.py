# models/model_segman.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math

# 修复导入 - 使用相对导入
try:
    from .segman_encoder import SegMANEncoder_t, SegMANEncoder_s, SegMANEncoder_b, SegMANEncoder_l
except ImportError:
    # 备用实现
    print("警告: 无法导入 segman_encoder，使用备用实现")


    class SegMANEncoder_t(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.channels = [32, 64, 128, 256]  # 添加channels属性
            self.layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.layers(x)


    SegMANEncoder_s = SegMANEncoder_t
    SegMANEncoder_b = SegMANEncoder_t
    SegMANEncoder_l = SegMANEncoder_t


def load_encoder_pretrained(encoder: nn.Module, ckpt_path: str) -> Dict[str, int]:
    """加载编码器预训练权重"""
    if not os.path.exists(ckpt_path):
        print(f"警告: 预训练权重文件不存在: {ckpt_path}")
        return {"loaded": 0, "total": 0, "missing": 0}

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # 处理不同的检查点格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 过滤编码器权重
        encoder_state_dict = {}
        for k, v in state_dict.items():
            # 移除可能的模块前缀
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('encoder.'):
                k = k[8:]
            if k.startswith('backbone.'):
                k = k[9:]

            encoder_state_dict[k] = v

        # 获取当前模型状态字典
        model_state_dict = encoder.state_dict()

        # 匹配权重
        matched_weights = {}
        for k, v in model_state_dict.items():
            if k in encoder_state_dict and encoder_state_dict[k].shape == v.shape:
                matched_weights[k] = encoder_state_dict[k]

        # 加载匹配的权重
        if matched_weights:
            encoder.load_state_dict(matched_weights, strict=False)

        print(f"预训练权重加载完成:")
        print(f"  - 成功加载: {len(matched_weights)}/{len(model_state_dict)} 参数")

        return {
            "loaded": len(matched_weights),
            "total": len(model_state_dict),
            "missing": len(model_state_dict) - len(matched_weights)
        }

    except Exception as e:
        print(f"加载预训练权重时出错: {e}")
        return {"loaded": 0, "total": 0, "missing": 0}


class SegMAN(nn.Module):
    """完整的 SegMAN 模型"""

    def __init__(self, num_classes: int = 3, variant: str = "tiny"):
        super().__init__()

        # 根据变体选择编码器
        if variant == "tiny":
            self.encoder = SegMANEncoder_t()
            encoder_out_channels = 256  # 根据编码器实际输出通道数设置
            decoder_channels = [128, 64, 32]  # 修正通道数顺序
        elif variant == "small":
            self.encoder = SegMANEncoder_s()
            encoder_out_channels = 512
            decoder_channels = [256, 128, 64]
        elif variant == "base":
            self.encoder = SegMANEncoder_b()
            encoder_out_channels = 768
            decoder_channels = [384, 192, 96]
        elif variant == "large":
            self.encoder = SegMANEncoder_l()
            encoder_out_channels = 1024
            decoder_channels = [512, 256, 128]
        else:
            raise ValueError(f"未知的变体: {variant}")

        # 修复：获取编码器实际输出通道数
        if hasattr(self.encoder, 'channels') and self.encoder.channels:
            encoder_out_channels = self.encoder.channels[-1]

        # 简化解码器 - 修复通道数匹配问题
        self.decoder = nn.Sequential(
            # 第一层：将编码器输出通道数映射到解码器中间通道数
            nn.Conv2d(encoder_out_channels, decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # 第二层
            nn.Conv2d(decoder_channels[0], decoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # 第三层：输出到最终类别数
            nn.Conv2d(decoder_channels[1], decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # 修复：增加额外的上采样层以匹配输入尺寸
            nn.Conv2d(decoder_channels[2], decoder_channels[2] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2] // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # 最终分类层
            nn.Conv2d(decoder_channels[2] // 2, num_classes, kernel_size=1)
        )

        print(f"模型配置: variant={variant}, 编码器输出通道={encoder_out_channels}, "
              f"解码器通道={decoder_channels}, 类别数={num_classes}")

    def forward(self, x):
        # 编码器特征提取
        features = self.encoder(x)

        # 使用最后阶段的特征
        if isinstance(features, list):
            x = features[-1]
        else:
            x = features

        # 解码器上采样
        output = self.decoder(x)

        return output