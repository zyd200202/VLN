"""Student network: RGB-only pseudo-IR predictor.

Same U-Net+Transformer backbone as the Teacher but RGB-only at inference. It
outputs a pseudo-IR map plus a per-pixel log-variance head (for uncertainty-
weighted distillation), and exposes 4 encoder features aligned to the Teacher
via 1x1 projections for multi-scale distillation.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, Down, TransformerBottleneck, Up


class StudentUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 48,
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
        self.ir_head = nn.Conv2d(c1, 1, 1)
        self.sigma_head = nn.Conv2d(c1, 1, 1)
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
        pseudo_ir = torch.tanh(self.ir_head(x))
        log_sigma2 = self.sigma_head(x)
        if return_features:
            return pseudo_ir, log_sigma2, [s0, s1, s2, s3]
        return pseudo_ir, log_sigma2

    @torch.no_grad()
    def physical_feature(self, rgb: torch.Tensor, out_size: int = 64) -> torch.Tensor:
        """Flattened, L2-normalized pseudo-IR embedding for downstream scoring."""
        ir, _ = self.forward(rgb)
        ir = F.interpolate(ir, size=(out_size, out_size), mode="bilinear",
                           align_corners=False)
        return F.normalize(ir.flatten(1), dim=1)


class StudentWithProjections(nn.Module):
    """Wraps StudentUNet with 1x1 projections aligning each Student stage to the
    corresponding Teacher channel count (for multi-scale distillation)."""

    def __init__(self, student: StudentUNet, teacher_channels: List[int]):
        super().__init__()
        self.student = student
        self.projections = nn.ModuleList([
            nn.Conv2d(sc, tc, 1)
            for sc, tc in zip(student.feat_channels, teacher_channels)
        ])

    def forward(self, rgb: torch.Tensor):
        ir, log_sigma2, feats = self.student(rgb, return_features=True)
        projected = [proj(f) for proj, f in zip(self.projections, feats)]
        return ir, log_sigma2, projected
