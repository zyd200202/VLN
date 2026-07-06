"""Road 1 Stage C — Eval with feature adapter wedged into GLEE.

This is imported by objnav_benchmark.py (via DRPN_ROAD1_ADAPTER env var); it
monkey-patches the backbone of the in-process GLEE so that for every
forward(rgb), the SwinL feature dict is passed through our trained
ResidualAdapter before reaching pixel_decoder + predictor.

Idempotent: calling _activate() twice has no extra effect.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

_ADAPTER = None
_ACTIVE = False


class _ResidualAdapter(nn.Module):
    """Mirror of stage_b_train_head.ResidualAdapter for inference loading."""
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 1),
                nn.GELU(),
                nn.Conv2d(c, c, 1),
            )
            for c in channels
        ])
        self.channels = list(channels)

    def forward(self, feats):
        return [x + blk(x) for x, blk in zip(feats, self.blocks)]


def _load_adapter(ckpt_path: str, device: str = "cuda:0"):
    global _ADAPTER
    ck = torch.load(ckpt_path, map_location=device)
    chans = ck["channels"]
    keys  = ck["scale_keys"]
    adp = _ResidualAdapter(chans).to(device)
    adp.load_state_dict(ck["state_dict"])
    adp.eval()
    for p in adp.parameters():
        p.requires_grad = False
    _ADAPTER = {"adp": adp, "keys": keys}
    print(f"[road1-adapter] loaded {ckpt_path}  scales={keys}  channels={chans}",
          flush=True)
    return _ADAPTER


def _wrap_backbone(backbone):
    """Wrap a SwinL backbone so its dict output passes through the adapter."""
    if getattr(backbone, "_road1_wrapped", False):
        return  # idempotent
    orig_forward = backbone.forward

    def patched_forward(x, *a, **k):
        out = orig_forward(x, *a, **k)
        # out is an OrderedDict[str -> Tensor]
        adp = _ADAPTER["adp"]
        keys = _ADAPTER["keys"]
        feats = [out[k] for k in keys if k in out]
        if len(feats) != len(keys):
            # silently bypass if the key set doesn't match (sanity)
            return out
        adapted = adp(feats)
        for k, v in zip(keys, adapted):
            out[k] = v
        return out

    backbone.forward = patched_forward
    backbone._road1_wrapped = True
    print("[road1-adapter] patched GLEE backbone forward()", flush=True)


def activate(glee_model, ckpt_path: str):
    """Public entry: load adapter from ckpt and wrap glee.backbone."""
    global _ACTIVE
    if _ACTIVE:
        return
    _load_adapter(ckpt_path, device=str(next(glee_model.parameters()).device))
    _wrap_backbone(glee_model.backbone)
    _ACTIVE = True


def maybe_activate_from_env(glee_model):
    """If DRPN_ROAD1_ADAPTER is set, activate; else no-op."""
    p = os.environ.get("DRPN_ROAD1_ADAPTER", "")
    if not p:
        return
    if not os.path.isfile(p):
        print(f"[road1-adapter][warn] DRPN_ROAD1_ADAPTER={p} not found, skipping",
              flush=True)
        return
    activate(glee_model, p)
