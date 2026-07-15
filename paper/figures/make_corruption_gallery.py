"""fig_corruption_gallery.py -- visual reference of the 4 corruption channels.

Layout: 2 scenes (rows) x 5 columns (clean + low-light + motion-blur + fog + gauss).
All corruptions applied at severity 4 (the level used in the main results).

Uses:
  - HM3D clean frames from /root/autodl-tmp/DRPhysNav/road1/data/frames.h5
    (which were themselves sampled from HM3D val scenes)
  - drphysnav.degradation.corruptions.apply(img, dtype, severity)
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/root/autodl-tmp/DRPhysNav/src")
from drphysnav.degradation.corruptions import apply  # noqa: E402

from paper_style import apply_style  # noqa: E402

apply_style()

HERE = Path(__file__).resolve().parent
H5   = "/root/autodl-tmp/DRPhysNav/road1/data/frames.h5"

# Corruption channels + severity used in the paper's main table.
CHANNELS = [
    ("clean",         None,             None),
    ("low-light",    "low_light",       4),
    ("motion-blur",  "motion_blur",     4),
    ("fog",          "fog",             4),
    ("gauss. noise", "gaussian_noise",  4),
]

# Hand-picked scene indices for visual diversity (interior / hallway / open room).
# Chosen from 1500-frame h5 by scanning for varied brightness + composition.
SCENE_IDX = [3]


def main():
    with h5py.File(H5, "r") as h:
        cleans = h["clean"][SCENE_IDX]  # (2, 480, 640, 3) uint8

    nrows, ncols = len(SCENE_IDX), len(CHANNELS)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(6.75, 1.16),
                            gridspec_kw=dict(wspace=0.03, hspace=0.10))
    if nrows == 1:
        axs = np.array([axs])

    for r, clean in enumerate(cleans):
        for c, (label, dtype, sev) in enumerate(CHANNELS):
            ax = axs[r, c]
            if dtype is None:
                img = clean
            else:
                img = apply(clean.copy(), dtype, sev)
                # Clip / cast just in case (some corruptions return float)
                if img.dtype != np.uint8:
                    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "bottom", "left", "right"):
                ax.spines[s].set_linewidth(0.4)
                ax.spines[s].set_color("#888888")
            if r == 0:
                ax.set_title(label, fontsize=8.4, pad=2)
    # severity/PSNR details live in the LaTeX caption; no in-figure footnote
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02)

    fig.savefig(HERE / "fig_corruption_gallery.pdf")
    fig.savefig(HERE / "fig_corruption_gallery.png", dpi=180)
    print("saved fig_corruption_gallery.pdf")


if __name__ == "__main__":
    main()
