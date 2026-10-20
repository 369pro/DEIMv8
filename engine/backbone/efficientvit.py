"""
EfficientViT Backbone for 2K*2K Image Processing
Based on EfficientViMBlock for high-resolution image preprocessing

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import FrozenBatchNorm2d
from ..core import register
from ..extre_module.custom_nn.module.efficientVIM import EfficientViMBlock
import logging

__all__ = ['EfficientViT']


class DownsampleBlock(nn.Module):
    """下采样块，用于逐步降低特征图分辨率"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EfficientViTStage(nn.Module):
    """EfficientViT Stage，包含多个EfficientViMBlock"""
    def __init__(self, in_channels, out_channels, num_blocks=2, downsample=True):
        super().__init__()
        
        # 可选的下采样
        if downsample and in_channels != out_channels:
            self.downsample = DownsampleBlock(in_channels, out_channels, stride=2)
        elif in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.downsample = nn.Identity()
            
        # EfficientViM blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_in_channels = out_channels if i > 0 else out_channels
            self.blocks.append(EfficientViMBlock(block_in_channels, out_channels))
    
    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


@register()
class EfficientViT(nn.Module):
    """
    EfficientViT backbone for 2K*2K image preprocessing
    
    输出多尺度特征图，兼容DFINE-S架构
    """
    
    arch_configs = {
        'S': {
            'stem_channels': [3, 32, 64],
            'stage_channels': [64, 128, 256, 512],
            'num_blocks': [2, 3, 4, 2],
        },
        'M': {
            'stem_channels': [3, 48, 96],
            'stage_channels': [96, 192, 384, 768],
            'num_blocks': [2, 4, 6, 3],
        },
        'L': {
            'stem_channels': [3, 64, 128],
            'stage_channels': [128, 256, 512, 1024],
            'num_blocks': [3, 4, 8, 4],
        }
    }
    
    def __init__(self, 
                 name='S',
                 return_idx=[1, 2, 3],
                 freeze_at=-1,
                 freeze_norm=False,
                 pretrained=None):
        super().__init__()
        
        self.return_idx = return_idx
        config = self.arch_configs[name]
        
        stem_channels = config['stem_channels']
        stage_channels = config['stage_channels']
        num_blocks = config['num_blocks']
        
        # Stem: 处理2K输入，逐步下采样
        self.stem = nn.Sequential(
            # 2048 -> 1024
            nn.Conv2d(stem_channels[0], stem_channels[1], 
                     kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels[1]),
            nn.ReLU(inplace=True),
            
            # 1024 -> 512
            nn.Conv2d(stem_channels[1], stem_channels[2], 
                     kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels[2]),
            nn.ReLU(inplace=True),
        )
        
        # Stages: 进一步特征提取
        self.stages = nn.ModuleList()
        in_channels = stem_channels[2]
        
        for i, (out_channels, blocks) in enumerate(zip(stage_channels, num_blocks)):
            stage = EfficientViTStage(
                in_channels, out_channels, 
                num_blocks=blocks, 
                downsample=(i > 0)  # 第一个stage不下采样，后续都下采样
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # 冻结参数
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at + 1, len(self.stages))):
                self._freeze_parameters(self.stages[i])
        
        if freeze_norm:
            self._freeze_norm(self)
    
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False
    
    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m
    
    def forward(self, x):
        # Stem处理: 2048x2048 -> 512x512
        x = self.stem(x)
        
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        
        return outs


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientViT('S').to(device)
    
    # 测试2K输入
    x = torch.randn(1, 3, 2048, 2048).to(device)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
