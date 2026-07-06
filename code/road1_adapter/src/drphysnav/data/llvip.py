"""LLVIP visible-infrared paired dataset (REAL data).

LLVIP (Jia et al., ICCV 2021 workshop) provides 15 488 strictly aligned
visible/infrared image pairs of low-light street scenes. We use it to train a
real RGB -> IR translation Teacher: in low light the visible channel degrades
while the thermal/IR channel stays informative, which is exactly the
degradation-robustness phenomenon this project studies. No synthetic data.

On-disk layout (as extracted from the official zip):
    <root>/LLVIP/visible/{train,test}/<id>.jpg
    <root>/LLVIP/infrared/{train,test}/<id>.jpg
Pairs share the same filename across the visible/ and infrared/ trees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..degradation import apply as apply_degrade
from ..degradation import random_degrade


def _to_chw_m11(img_u8_hwc: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float CHW in [-1,1]."""
    t = torch.from_numpy(img_u8_hwc.astype(np.float32) / 255.0)
    if t.ndim == 2:
        t = t[None]
    else:
        t = t.permute(2, 0, 1)
    return t * 2 - 1


class LLVIPPairs(Dataset):
    """Returns {'rgb': (3,H,W) in [-1,1], 'ir': (1,H,W) in [-1,1], 'id': str}.

    If `degrade=True`, also returns 'rgb_deg' (a randomly corrupted copy of the
    RGB) for degradation-robust Student training. If `degrade_spec=(type,sev)`
    is given, applies that fixed corruption instead of a random one (used for
    deterministic per-corruption evaluation).

    `manifest` is a list of (visible_path, infrared_path) absolute-path pairs;
    build it with `scripts/prepare_llvip.py` so the split is fixed and auditable.
    """

    def __init__(self, manifest: List[Tuple[str, str]], size: int = 256,
                 degrade: bool = False, degrade_spec=None,
                 severity_range=(1, 4)):
        self.pairs = manifest
        self.size = size
        self.degrade = degrade
        self.degrade_spec = degrade_spec
        self.severity_range = severity_range

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        vis_p, ir_p = self.pairs[idx]
        rgb_u8 = cv2.cvtColor(cv2.imread(vis_p), cv2.COLOR_BGR2RGB)
        ir_u8 = cv2.imread(ir_p, cv2.IMREAD_GRAYSCALE)
        rgb_u8 = cv2.resize(rgb_u8, (self.size, self.size), interpolation=cv2.INTER_AREA)
        ir_u8 = cv2.resize(ir_u8, (self.size, self.size), interpolation=cv2.INTER_AREA)
        out = {"rgb": _to_chw_m11(rgb_u8), "ir": _to_chw_m11(ir_u8),
               "id": Path(vis_p).stem}
        if self.degrade or self.degrade_spec is not None:
            if self.degrade_spec is not None:
                dtype, sev = self.degrade_spec
                deg_u8 = apply_degrade(rgb_u8, dtype, sev)
            else:
                deg_u8, _, _ = random_degrade(rgb_u8, self.severity_range)
            out["rgb_deg"] = _to_chw_m11(deg_u8)
        return out


def load_manifest(path: str, split: str) -> List[Tuple[str, str]]:
    with open(path) as f:
        m = json.load(f)
    return [tuple(p) for p in m[split]]
