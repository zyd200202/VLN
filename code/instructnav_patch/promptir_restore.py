"""PromptIR (NeurIPS'23, 2306.13090) all-in-one blind image restoration as a drop-in restorer
for DRPhysNav component B (specialized/strong restoration). Wraps the official Restormer-style
PromptIR network + released v1.0 all-in-one checkpoint; exposes restore_bgr(bgr_uint8)->bgr_uint8
so it can replace drpn.restore_rgb. Lazy, single-instance, GPU, no-grad.

Borrowed verbatim: model arch (thirdparty/PromptIR/net/model.py) + pretrained weights
(releases/v1.0/model.ckpt). Input convention follows the official demo: RGB in [0,1], no mean/std.
"""
import os
import sys
import numpy as np
import cv2
import torch

_PIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thirdparty", "PromptIR")
if _PIR_DIR not in sys.path:
    sys.path.insert(0, _PIR_DIR)

_model = None
_device = None


def _load():
    global _model, _device
    if _model is not None:
        return _model
    from net.model import PromptIR
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ck = os.environ.get("PROMPTIR_CKPT", os.path.join(_PIR_DIR, "ckpt", "model.ckpt"))
    blob = torch.load(ck, map_location="cpu")
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    # Lightning wraps the net as self.net -> strip the 'net.' prefix to match PromptIR keys.
    sd = {(k[4:] if k.startswith("net.") else k): v for k, v in sd.items()}
    m = PromptIR(decoder=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval().to(_device)
    _model = m
    print("[PromptIR] loaded ckpt=%s on %s | missing=%d unexpected=%d"
          % (ck, _device, len(missing), len(unexpected)), flush=True)
    if len(missing) > 50:
        print("[PromptIR][WARN] many missing keys (%d) -- checkpoint/arch mismatch?" % len(missing), flush=True)
    return m


@torch.no_grad()
def restore_bgr(bgr):
    """Restore a BGR uint8 frame; returns BGR uint8 (same HxW)."""
    m = _load()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(_device)
    _, _, h, w = t.shape
    f = 8  # 3 downsamples -> input must be divisible by 8
    ph, pw = (f - h % f) % f, (f - w % f) % f
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    out = m(t)[:, :, :h, :w].clamp(0.0, 1.0)
    o = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
