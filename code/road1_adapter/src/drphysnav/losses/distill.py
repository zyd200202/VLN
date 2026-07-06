"""Cross-domain distillation losses (PANav Sec. 3.3).

  L_ms  : multi-scale feature alignment (MSE + cosine) over 4 stages
  L_unc : uncertainty-weighted reconstruction on the pseudo-IR output
  L_inv : clean<->degraded consistency on Student features
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


def multi_scale_alignment(student_feats: Sequence[torch.Tensor],
                          teacher_feats: Sequence[torch.Tensor],
                          weights: Sequence[float] | None = None) -> torch.Tensor:
    assert len(student_feats) == len(teacher_feats)
    n = len(student_feats)
    if weights is None:
        weights = [1.0] * n
    total = student_feats[0].new_zeros(())
    for w, s, t in zip(weights, student_feats, teacher_feats):
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear",
                              align_corners=False)
        mse = F.mse_loss(s, t)
        cos = F.cosine_similarity(s.flatten(1), t.flatten(1), dim=1).mean()
        total = total + w * (mse + (1.0 - cos))
    return total


def uncertainty_weighted_distill(student_ir: torch.Tensor,
                                 target_ir: torch.Tensor,
                                 log_sigma2: torch.Tensor) -> torch.Tensor:
    """L_unc = mean[ (S - T)^2 / sigma^2 + log sigma^2 ]. The log term keeps
    the predicted variance from diverging."""
    inv_var = torch.exp(-log_sigma2)
    sq = (student_ir - target_ir) ** 2
    return (sq * inv_var + log_sigma2).mean()


def invariance_consistency(feats_clean: Sequence[torch.Tensor],
                           feats_degraded: Sequence[torch.Tensor]) -> torch.Tensor:
    """L_inv = mean_i [ 1 - cos( S(clean)_i, S(degraded)_i ) ]."""
    total = feats_clean[0].new_zeros(())
    for c, d in zip(feats_clean, feats_degraded):
        cos = F.cosine_similarity(c.flatten(1), d.flatten(1), dim=1).mean()
        total = total + (1.0 - cos)
    return total / max(1, len(feats_clean))
