"""VIRGenerator: Teacher network for RGB -> thermal-IR translation.

U-Net with a Transformer bottleneck. Trained on REAL LLVIP visible/infrared
pairs with an L1+L2 reconstruction objective (PANav Eq. 2). Exposes 4 encoder
features for later multi-scale distillation into the Student.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import ConvBlock, Down, TransformerBottleneck, Up


class VIRGenerator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 48,
                 bottleneck_depth: int = 2, bottleneck_heads: int = 8):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.stem = ConvBlock(in_ch, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.bottleneck = TransformerBottleneck(c4, depth=bottleneck_depth,
                                                heads=bottleneck_heads)
        self.up3 = Up(c4, c4, c3)
        self.up2 = Up(c3, c3, c2)
        self.up1 = Up(c2, c2, c1)
        self.head = nn.Conv2d(c1, out_ch, 1)
        self.feat_channels: List[int] = [c1, c2, c3, c4]

    def forward(self, rgb: torch.Tensor, return_features: bool = False):
        s0 = self.stem(rgb)
        x, s1 = self.down1(s0)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x = self.bottleneck(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        ir = torch.tanh(self.head(x))
        if return_features:
            return ir, [s0, s1, s2, s3]
        return ir


def teacher_recon_loss(pred_ir: torch.Tensor, gt_ir: torch.Tensor) -> torch.Tensor:
    """PANav Eq. (2): L1 + L2 reconstruction."""
    return (pred_ir - gt_ir).abs().mean() + ((pred_ir - gt_ir) ** 2).mean()
