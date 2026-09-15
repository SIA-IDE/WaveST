# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, OFDRepConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "WUF",
)

############################

# WaveC2f - 小波变换增强模块 (最推荐)
# 核心创新
# 引入**离散小波变换(DWT)**实现无损下采样和频域特征增强,突破空域限制。

class DWT2D(nn.Module):
    """2D离散小波变换 - 无损分解到频域"""
    
    def __init__(self, wavelet: str = 'haar'):
        super().__init__()
        # Haar小波系数
        self.register_buffer('ll', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2)
        self.register_buffer('lh', torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2)
        self.register_buffer('hl', torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2)
        self.register_buffer('hh', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # 构建卷积核
        filters = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0)  # (4, 2, 2)
        filters = filters.unsqueeze(1).repeat(C, 1, 1, 1)  # (4C, 1, 2, 2)
        
        # 应用小波变换 (stride=2实现下采样)
        x_dwt = F.conv2d(x, filters, stride=2, groups=C)  # (B, 4C, H/2, W/2)
        
        # 分组卷积的输出按通道交错排列：C0_LL, C0_LH, ..., C1_LL, ...。
        # 先恢复频带维度，再按频带拆分，确保每个张量都包含所有输入通道的同一子带。
        x_dwt = x_dwt.view(B, C, 4, x_dwt.shape[-2], x_dwt.shape[-1])
        ll, lh, hl, hh = x_dwt.unbind(dim=2)
        return ll, lh, hl, hh


class IDWT2D(nn.Module):
    """2D逆离散小波变换"""
    
    def __init__(self):
        super().__init__()
        # 重建滤波器
        self.register_buffer('ll', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2)
        self.register_buffer('lh', torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2)
        self.register_buffer('hl', torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2)
        self.register_buffer('hh', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2)
        
    def forward(self, ll: torch.Tensor, lh: torch.Tensor, 
                hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        B, C, H, W = ll.shape
        
        # 以与 DWT 分组卷积输出一致的通道交错顺序合并子带。
        x_concat = torch.stack((ll, lh, hl, hh), dim=2)
        x_concat = x_concat.reshape(B, C * 4, H, W)  # (B, 4C, H, W)
        
        filters = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0)
        filters = filters.unsqueeze(0).repeat(C, 1, 1, 1).view(C * 4, 1, 2, 2)
        x_recon = F.conv_transpose2d(x_concat, filters, stride=2, groups=C)
        
        return x_recon

class WUF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, list):
            c_deep = c1[0]
            c_shallow = c1[1]
        else:
            raise ValueError(f"WUF expects a list of input channels from 2 layers. Got {c1}")

        self.dwt = DWT2D(wavelet='haar')
        self.idwt = IDWT2D()
        self.fusion_conv = Conv(c_deep + c_shallow, c_shallow, 1, 1)
        
        # Output projection if c2 != c_shallow
        # self.out_conv = Conv(c_shallow, c2, 1, 1) if c2 != c_shallow else nn.Identity()
        self.out_conv = Conv(c_shallow, c2, 1, 1)

    def forward(self, x):
        # x is [deep, shallow]
        deep, shallow = x[0], x[1]
        ll, lh, hl, hh = self.dwt(shallow)
        fused = torch.cat([deep, ll], dim=1)
        fused = self.fusion_conv(fused)
        out = self.idwt(fused, lh, hl, hh)
        # Output projection
        out = self.out_conv(out)
        return out