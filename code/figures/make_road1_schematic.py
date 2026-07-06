"""Generate the Road-1 schematic figure used in the appendix.

Redesigned for the unified paper palette. Full-width (figure*), three stages
laid out left->right with no crossing arrows and semantic tints.

Stages:
  (A) data gen -- clean + degraded RGB pairs
  (B) train adapter -- residual conv trained via L1 feature matching
  (C) inference -- adapter wedged into frozen SwinL forward pass
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_style import (
    C_BASE, C_DEGR, C_POS, C_ORACLE, C_NEUT, C_HL,
    T_BASE, T_DEGR, T_POS, T_ORACLE, T_NEUT,
    apply_style,
)

apply_style()

HERE = Path(__file__).resolve().parent


def box(ax, x, y, w, h, txt, fc="white", ec="#666666", bold=False, fs=8.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                       fc=fc, ec=ec, lw=0.7)
    ax.add_patch(p)
    fw = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, fontweight=fw, color="#222222", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label=None, dash=False, color="#666666"):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="-|>", mutation_scale=10,
                          lw=0.9, color=color,
                          linestyle="--" if dash else "-")
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.20, label,
                ha="center", va="bottom", fontsize=7.0,
                style="italic", color="#555555")


def main():
    fig, ax = plt.subplots(figsize=(6.75, 2.60))
    ax.set_xlim(-0.5, 14.5); ax.set_ylim(-0.5, 7.6)
    ax.axis("off")

    # ---- stage A: data gen (left column) ----
    box(ax, 0.4, 4.7, 2.6, 1.0, "HM3D-val\nRGB frame", fc=T_NEUT)
    box(ax, 0.4, 1.3, 2.6, 1.0, "low-light s4\n(degrade_rgb)", fc=T_DEGR)
    arrow(ax, 1.7, 4.7, 1.7, 2.3)

    # ---- stage B: adapter (middle) ----
    box(ax, 4.4, 4.7, 3.0, 1.0, "Frozen SwinL\nbackbone", fc=T_BASE)
    box(ax, 4.4, 1.3, 3.0, 1.0, "Frozen SwinL\nbackbone (shared)", fc=T_BASE)
    arrow(ax, 3.0, 5.2, 4.4, 5.2, label="clean")
    arrow(ax, 3.0, 1.8, 4.4, 1.8, label="degraded")

    box(ax, 8.6, 1.3, 2.6, 1.0, "Residual\nAdapter",
        fc=T_POS, ec=C_POS, bold=True, fs=8.2)
    arrow(ax, 7.4, 1.8, 8.6, 1.8)

    # ---- stage C: L1 supervision (right) ----
    box(ax, 11.6, 3.0, 2.0, 1.0, r"$L_1$ feature" "\n" r"match",
        fc=T_ORACLE, ec=C_ORACLE)
    arrow(ax, 7.4, 5.2, 12.6, 4.0, label=r"$f^{\mathrm{clean}}$", dash=True)
    arrow(ax, 11.2, 1.8, 12.6, 3.0, label=r"$f^{\mathrm{adapt}}$", dash=True)

    # ---- section headers ----
    ax.text(1.7,  6.8, "(A) data gen", ha="center",
            fontsize=8.4, fontweight="bold", color="#222222")
    ax.text(8.7,  6.8, r"(B) train adapter ($L_1$, ${\sim}2$k steps)",
            ha="center", fontsize=8.4, fontweight="bold", color="#222222")
    ax.text(12.6, 6.8, "(C) supervision", ha="center",
            fontsize=8.4, fontweight="bold", color="#222222")

    # footer note (concise)
    ax.text(7.0, -0.20,
            "Inference: the same Adapter is wedged between frozen SwinL "
            "and the frozen pixel_decoder + predictor;\n"
            "the rest of the InstructNav agent is unchanged.",
            ha="center", va="center", fontsize=7.2,
            style="italic", color="#666666", linespacing=1.35)

    plt.tight_layout(pad=0.2)
    plt.savefig(HERE / "fig_road1.pdf")
    plt.savefig(HERE / "fig_road1.png", dpi=180)
    print("wrote", HERE / "fig_road1.pdf")


if __name__ == "__main__":
    main()
