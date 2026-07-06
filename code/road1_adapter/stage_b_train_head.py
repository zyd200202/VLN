"""Road 1 Stage B — feature-level adapter training.

Idea (PLAN.md Plan C, promoted to the main route):
    Train a small Conv adapter f_d : SwinL(degraded_rgb) -> SwinL(clean_rgb)
    on the (clean, degraded) pair shards from Stage A.

We freeze SwinL completely, run BOTH clean and degraded through it offline
(or on-the-fly in the train loop), and supervise the adapter with L2 on
each of the 4 SwinL feature scales.

At inference time, the adapter is wedged between SwinL.forward and the
downstream pixel_decoder / predictor, so the rest of GLEE never knows
the input was corrupted.

Saved checkpoint:
    /root/autodl-tmp/DRPhysNav/road1/ckpt/adapter_low_light_s4.pth
    keys:
        state_dict  : adapter weights
        deg_type    : "low_light"
        deg_sev     : 4
        scales      : list[int]  # SwinL stage indices the adapter covers
        channels    : list[int]  # channel dim per scale
        train_meta  : dict (losses, steps, time)
"""
from __future__ import annotations

import argparse, json, os, sys, time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/root/autodl-tmp/DRPhysNav/road1/data")
    p.add_argument("--ckpt_dir", default="/root/autodl-tmp/DRPhysNav/road1/ckpt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_split", type=int, default=100,
                   help="Hold out the last K pairs for in-loop validation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_every", type=int, default=500)
    return p.parse_args()


class ResidualAdapter(nn.Module):
    """Per-scale 1x1 conv adapter with residual connection.

    For each SwinL stage feature map of shape (B, C, H, W),
    adapter(x) = x + Conv1x1(GELU(Conv1x1(x))).  Lightweight ( ~ C*C *2 params).
    """
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
        # zero-init the residual so initial output == identity
        for blk in self.blocks:
            nn.init.zeros_(blk[-1].weight)
            nn.init.zeros_(blk[-1].bias)
        self.channels = list(channels)

    def forward(self, feats):
        """feats: list[Tensor] -- one per scale.  returns same-shape list."""
        return [x + blk(x) for x, blk in zip(feats, self.blocks)]


def _load_swin_l(device):
    # GLEE loader references config/ckpt via paths RELATIVE to InstructNav cwd,
    # so we chdir there for the load and then restore cwd.
    prev_cwd = os.getcwd()
    inav = "/root/autodl-tmp/InstructNav"
    sys.path.insert(0, inav)
    os.chdir(inav)
    try:
        from cv_utils.glee_detector import initialize_glee
        mdl = initialize_glee(device=device)
    finally:
        os.chdir(prev_cwd)
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad = False
    return mdl


def _swin_features(model, bgr_batch_uint8):
    """Run frozen SwinL on a BGR-uint8 batch and return list of 4 feature maps."""
    # GLEE uses BGR->RGB via pixel_mean/std on RGB-ish ordering. The actual
    # `glee_segmentation` does normalisation in its own way; we mirror it.
    device = next(model.parameters()).device
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std  = torch.tensor([58.395, 57.12, 57.375],   device=device).view(1, 3, 1, 1)
    # bgr_batch_uint8: (B, H, W, 3) uint8 in BGR -- convert to RGB float
    x = torch.from_numpy(bgr_batch_uint8[:, :, :, ::-1].copy()).to(device).float()
    x = x.permute(0, 3, 1, 2)
    x = (x - pixel_mean) / pixel_std
    # GLEE expects a multiple of 32 on H/W; right-pad
    B, C, H, W = x.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    feats_dict = model.backbone(x)  # dict[str -> Tensor]
    # keep deterministic ordering (Swin returns res2..res5 typically)
    keys = sorted(feats_dict.keys())
    feats = [feats_dict[k] for k in keys]
    return feats, keys


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, "train.log")
    log = open(log_path, "w")

    def L(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda:0"
    L(f"data_dir={args.data_dir} ckpt_dir={args.ckpt_dir}")

    # --- data ---
    h5 = h5py.File(os.path.join(args.data_dir, "frames.h5"), "r")
    N_total = h5["clean"].shape[0]
    val_split = min(args.val_split, max(0, N_total - args.batch))
    N_train = N_total - val_split
    L(f"frames: {N_total} (train={N_train}, val={val_split})")

    # --- frozen SwinL once, then derive channels ---
    L("loading frozen GLEE (SwinL + heads) ...")
    glee = _load_swin_l(device)
    # peek one batch to figure out feature shapes
    sample_bgr = h5["clean"][:1]  # (1, H, W, 3) uint8 BGR
    with torch.no_grad():
        sample_feats, scale_keys = _swin_features(glee, sample_bgr)
    chans = [f.shape[1] for f in sample_feats]
    shapes = [tuple(f.shape) for f in sample_feats]
    L(f"swin feature scales (keys): {scale_keys}")
    L(f"   shapes: {shapes}")
    L(f"   channels: {chans}")

    # --- adapter ---
    adapter = ResidualAdapter(chans).to(device)
    nparams = sum(p.numel() for p in adapter.parameters())
    L(f"adapter params: {nparams/1e6:.2f}M")

    opt = torch.optim.AdamW(adapter.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # --- training loop ---
    losses = []
    rng = np.random.RandomState(args.seed)
    t0 = time.time()

    def fetch_batch(train=True):
        if train:
            # h5py fancy indexing requires strictly increasing indices;
            # sample without replacement then sort.
            k = min(args.batch, N_train)
            idx = rng.choice(N_train, size=k, replace=False)
        else:
            idx = np.arange(N_train, N_total)
        idx = sorted(idx.tolist())
        clean = h5["clean"][idx]
        deg   = h5["degraded"][idx]
        return clean, deg

    for step in range(1, args.steps + 1):
        clean_bgr, deg_bgr = fetch_batch(train=True)
        with torch.no_grad():
            f_clean, _ = _swin_features(glee, clean_bgr)
            f_deg,   _ = _swin_features(glee, deg_bgr)
        f_adapted = adapter(f_deg)
        loss = sum(F.l1_loss(a, c) for a, c in zip(f_adapted, f_clean))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == 1:
            L(f"step {step:5d}/{args.steps}  loss={loss.item():.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  "
              f"elapsed={time.time()-t0:.1f}s")
        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "state_dict": adapter.state_dict(),
                "channels":   chans,
                "scale_keys": scale_keys,
                "deg_type":   "low_light",
                "deg_sev":    4,
                "step":       step,
                "loss_curve": losses,
                "args":       vars(args),
            }
            out = os.path.join(args.ckpt_dir,
                               f"adapter_low_light_s4_step{step:05d}.pth")
            torch.save(ckpt, out)
            L(f"saved {out}")
    final = os.path.join(args.ckpt_dir, "adapter_low_light_s4.pth")
    torch.save(ckpt, final)
    L(f"saved final {final}")

    # --- validation ---
    if val_split > 0:
        L("validating on hold-out...")
        clean_bgr, deg_bgr = fetch_batch(train=False)
        with torch.no_grad():
            f_clean, _ = _swin_features(glee, clean_bgr)
            f_deg,   _ = _swin_features(glee, deg_bgr)
            f_adapted = adapter(f_deg)
            l1_pre  = sum(F.l1_loss(d, c).item() for d, c in zip(f_deg, f_clean))
            l1_post = sum(F.l1_loss(a, c).item() for a, c in zip(f_adapted, f_clean))
        L(f"  hold-out L1: pre={l1_pre:.4f}  post={l1_post:.4f}  "
          f"reduction={1 - l1_post/max(l1_pre,1e-9):.1%}")

    h5.close()
    log.close()


if __name__ == "__main__":
    main()
