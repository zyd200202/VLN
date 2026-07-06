"""Shared U-Net + Transformer-bottleneck building blocks.

Used by the Teacher (VIRGenerator, RGB->IR) and later the Student. Plain,
well-understood architecture (~17M params for the Teacher at base=48).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.block(x)
        return self.pool(skip), skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class TransformerBottleneck(nn.Module):
    """Flatten spatial map -> Transformer encoder -> reshape back.

    Optionally concatenates a conditioning token sequence (action / language)
    to the visual tokens (used later by the world model).
    """

    def __init__(self, dim: int, depth: int = 2, heads: int = 8,
                 mlp_ratio: float = 2.0, cond_dim: int | None = None,
                 max_tokens: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_tokens = max_tokens
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim else None

    def _get_pos(self, n: int) -> torch.Tensor:
        if n > self.max_tokens:
            raise ValueError(
                f"bottleneck has {n} tokens > max_tokens={self.max_tokens}."
            )
        return self.pos[:, :n]

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self._get_pos(tokens.size(1))
        n_vis = tokens.size(1)
        if cond is not None and self.cond_proj is not None:
            ctok = self.cond_proj(cond)
            tokens = torch.cat([tokens, ctok], dim=1)
        tokens = self.encoder(tokens)
        vis = tokens[:, :n_vis].transpose(1, 2).reshape(b, c, h, w)
        return vis
