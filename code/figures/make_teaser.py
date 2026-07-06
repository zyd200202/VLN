"""Page-1 teaser for DRPhysNav (v4, top-conference style).

Layout (double-column banner, two rows):
  (a) real image triplet: clean / low-light s4 / restored  -- the *input* is
      demonstrably fixed (PSNR 7.4 -> 15.9 dB, reliability r 0.39 -> 0.89)
  (b) merged slope panel: perception metrics rise, paired success does not
  (c) real Habitat top-down episode + the aggregate 26-pt reach-to-success gap
  (d) forest plot of the nine-arm intervention sweep: no deployable arm
      helps; the only significant positive is the non-deployable oracle,
      capped at +0.057
  (e) family-level coverage matrix: which evaluation dimensions each prior
      family covers vs. this paper

(a) uses the actual restoration demo frames from the paired runs; (d) is
computed from the same paired CSVs as Table 1, so every pixel and marker in
the figure is real experimental data.
"""
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from paper_style import (
    C_BASE, C_DEGR, C_POS, C_ORACLE, C_NEUT, C_HL, T_ORACLE,
    apply_style,
)

HERE = Path(__file__).resolve().parent
TD   = HERE / "fig_td_toilet.png"
DEMO = Path("/root/autodl-tmp/DRPhysNav/runs/instructnav_baseline/restore_demo.png")

apply_style()


def load_lowlight_triplet():
    """Slice the low_light row (clean | degraded | restored) out of the
    restoration demo sheet. The sheet is 5 stacked rows, each with a thin
    dark header strip; low_light is the 3rd row."""
    img = mpimg.imread(DEMO)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    h, w = img.shape[:2]
    row_h = h / 5.0
    y0 = int(2 * row_h)
    y1 = int(3 * row_h)
    band = img[y0:y1]
    # drop the header strip (dark rows at the top of the band)
    lum = band.mean(axis=(1, 2))
    body_start = 0
    for i, v in enumerate(lum[: int(row_h / 3)]):
        if v > 40:
            body_start = i
            break
    band = band[body_start:]
    third = w // 3
    tiles = [band[:, :third], band[:, third:2 * third], band[:, 2 * third:]]
    # crop sides so each tile is roughly square (taller in layout)
    out = []
    for t in tiles:
        tw = t.shape[1]
        cut = int(tw * 0.14)
        out.append(t[:, cut:tw - cut])
    return out


def panel_forest(fig, slot):
    """(d) full-width compact forest of the nine-arm sweep, vertical layout.
    Computed from the same paired CSVs as Table 1 (real data)."""
    from make_more_figures import paired_dsr, RUNS

    n300_b = RUNS / "redesign_n300/n300_B0_low_light_s4_seed0.csv"
    n150_b = RUNS / "maintable/mt_B0_low_light_s4_seed0.csv"
    uni = RUNS / "unified"
    deployable = [
        ("RES",    "300", n300_b, RUNS / "redesign_n300/n300_RES_low_light_s4_seed0.csv"),
        ("M1g",    "300", n300_b, RUNS / "redesign_n300/n300_M1g_low_light_s4_seed0.csv"),
        ("CRV",    "150", n150_b, uni / "u_CRV_low_light_s4_seed0_N150.csv"),
        ("ROUTER", "150", n150_b, RUNS / "maintable/mt_ROUTER_low_light_s4_seed0.csv"),
        ("FUSE",   "150", n150_b, uni / "u_FUSE_low_light_s4_seed0_N150.csv"),
        ("REVOKE", "150", n150_b, uni / "u_REVOKE_low_light_s4_seed0_N150.csv"),
        ("MUAP",   "150", n150_b, uni / "u_MUAP_low_light_s4_seed0_N150.csv"),
        ("DXCV",   "150", n150_b, uni / "u_DXCV_low_light_s4_seed0_N150.csv"),
    ]
    oracle = ("OracleGate", "300", n300_b,
              RUNS / "oracle/or_ORACLEGATE_low_light_s4_seed0.csv")

    rows = []
    for name, n_tag, b, c in deployable:
        s = paired_dsr(b, c)
        if s is not None:
            rows.append((name, n_tag, s))
    rows.sort(key=lambda t: -t[2]["dsr"])
    s_or = paired_dsr(oracle[2], oracle[3])
    rows.append((oracle[0], oracle[1], s_or))

    ax = fig.add_subplot(slot)
    ax.axhspan(-0.02, 0.02, color="#EDEDED", zorder=0)
    ax.axhline(0, color="#666666", lw=0.7, zorder=1)
    n_dep = len(rows) - 1
    # extra horizontal gap before the (non-deployable) oracle arm
    xpos = list(range(n_dep)) + [n_dep + 0.55]
    ax.axvline(n_dep - 0.5 + 0.28, color="#999999", lw=0.7, ls=(0, (3, 2)),
               zorder=1)

    for k, (name, n_tag, s) in enumerate(rows):
        x = xpos[k]
        d, lo, hi, p = s["dsr"], s["ci_lo"], s["ci_hi"], s["p"]
        is_sig = p < 0.05
        is_oracle = (k == n_dep)
        if is_oracle:
            col = C_ORACLE
        elif is_sig:
            col = C_DEGR
        else:
            col = C_NEUT
        ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt="o", color=col,
                    ecolor=col, capsize=2.5, ms=4.8, lw=1.2,
                    mec="white", mew=0.6, zorder=3)
        star = "*" if is_sig else ""
        ax.text(x, hi + 0.022, f"${d:+.2f}$" + star, ha="center",
                va="bottom", fontsize=6.2, color=col)

    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [f"{name}\n$N{{=}}{n}$" for name, n, _ in rows],
        fontsize=6.4, linespacing=1.15)
    ax.get_xticklabels()[-1].set_color(C_ORACLE)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylim(-0.26, 0.21)
    ax.set_yticks([-0.2, -0.1, 0, 0.1])
    ax.set_ylabel(r"paired $\Delta$SR", fontsize=6.8)
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(-0.6, xpos[-1] + 0.55)
    ax.set_title("(d) Nine-arm sweep: no deployable fix helps",
                 loc="left", pad=3, fontsize=8.5, fontweight="bold")
    ax.text(-0.45, -0.245, "deployable", ha="left", va="bottom",
            fontsize=6.2, color="#888888", style="italic")


def panel_positioning(fig, slot):
    """(e) family-level coverage matrix (compact version of the appendix
    positioning chart; prior work grouped by family)."""
    rows = [
        ("LM-guided frontier",  "SemExp / L3MVN / ESC",   [0, 0, 1, 0, .5, 0]),
        ("Value-map nav.",      "CoW / VLFM",             [0, 0, 1, 0, .5, 0]),
        ("VLM-direct nav.",     "NavGPT / InstructNav",   [0, 0, 1, 0, .5, 0]),
        ("Restoration",         "PromptIR / AirNet",      [1, 0, 0, 0, 0, 0]),
        ("Robustness bench.",   "RobustNav",              [1, 0, 1, 0, 0, .5]),
        ("This paper",          "DRPhysNav",              [1, 1, 1, 1, 1, 1]),
    ]
    cols = ["degr. RGB", "paired", "$N{\\geq}300$",
            "mech. dec.", "multi-modal", "neg. results"]

    ax = fig.add_subplot(slot)
    nr, nc = len(rows), len(cols)
    # highlight strip behind the "this paper" row (bottom row)
    ax.add_patch(Rectangle((-4.6, 0), nc + 4.6, 1,
                           fc=T_ORACLE, ec="none", zorder=0, alpha=0.65))
    for c in range(nc + 1):
        ax.plot([c, c], [0, nr], color="#D8D8D8", lw=0.4, zorder=1)
    for r in range(nr + 1):
        ax.plot([0, nc], [r, r], color="#D8D8D8", lw=0.4, zorder=1)

    for r, (name, ref, cov) in enumerate(rows):
        is_ours = (r == nr - 1)
        y = nr - r - 0.5
        ax.text(-0.15, y, name, ha="right", va="center", fontsize=6.4,
                color=C_ORACLE if is_ours else "#111111",
                fontweight="bold" if is_ours else "normal")
        for c, v in enumerate(cov):
            if v == 1:
                ax.text(c + 0.5, y, "$\\checkmark$", ha="center",
                        va="center", fontsize=8.0, fontweight="bold",
                        color=C_ORACLE if is_ours else C_BASE, zorder=3)
            elif v == 0.5:
                ax.text(c + 0.5, y, "$\\sim$", ha="center", va="center",
                        fontsize=7.5, color="#888888", zorder=3)
            else:
                ax.text(c + 0.5, y, "$-$", ha="center", va="center",
                        fontsize=7, color="#C8C8C8", zorder=3)
    for c in range(nc):
        ax.text(c + 0.30, nr + 0.18, cols[c], ha="left", va="bottom",
                fontsize=5.9, color="#222222", rotation=38,
                rotation_mode="anchor")

    ax.set_xlim(-4.6, nc + 0.15)
    ax.set_ylim(-0.15, nr + 2.8)
    ax.axis("off")
    ax.set_title("(e) Coverage map", loc="left", pad=2,
                 fontsize=8.5, fontweight="bold", y=1.00)


def main():
    fig = plt.figure(figsize=(6.75, 3.60))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.50, 1.0],
                                hspace=0.58)
    gs = gs_outer[0].subgridspec(1, 3, width_ratios=[1.55, 0.95, 1.15],
                                 wspace=0.28)

    # ============ (a) the input IS fixed: real frames ============
    gs_a = gs[0, 0].subgridspec(1, 3, wspace=0.04)
    clean, degr, rest = load_lowlight_triplet()
    panels = [
        (clean, "clean", C_NEUT,  ""),
        (degr,  "low-light s4", C_DEGR, "PSNR 7.4 dB\n$r{=}0.39$"),
        (rest,  "restored", C_POS, "PSNR 15.9 dB\n$r{=}0.89$"),
    ]
    first_ax = None
    for i, (im, lab, col, stat) in enumerate(panels):
        ax = fig.add_subplot(gs_a[0, i])
        if first_ax is None:
            first_ax = ax
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.2); s.set_color(col)
        ax.set_title(lab, fontsize=7.2, pad=2, color=col,
                     fontweight="bold")
        if stat:
            ax.text(0.5, 0.03, stat, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6.0, color="white",
                    linespacing=1.25,
                    bbox=dict(boxstyle="round,pad=0.18",
                              fc=(0, 0, 0, 0.55), ec="none"))
    # panel label above the whole strip, aligned with other panel titles
    first_ax.text(0.0, 1.42, "(a) The image is fixed ...",
                  transform=first_ax.transAxes, ha="left", va="bottom",
                  fontsize=8.5, fontweight="bold")

    # ============ (b) ...but navigation is not ============
    ax = fig.add_subplot(gs[0, 1])
    xs = [0, 1]
    # perception metrics (teal, rise)
    ax.plot(xs, [0.39, 0.89], "-o", color=C_POS, lw=1.8, ms=5.5,
            mec="white", mew=0.8, zorder=3)
    # paired success (red, flat)
    ax.plot(xs, [0.370, 0.410], "-s", color=C_DEGR, lw=1.8, ms=5.5,
            mec="white", mew=0.8, zorder=3)
    ax.text(1.08, 0.89, "reliability $r$", color=C_POS, fontsize=7.0,
            va="center")
    ax.text(1.08, 0.41, "success rate", color=C_DEGR, fontsize=7.0,
            va="center")
    ax.annotate(r"$\Delta$SR$=+0.04$" "\n(n.s., $N{=}300$)",
                xy=(0.55, 0.41), xytext=(1.15, 0.16),
                ha="left", va="bottom", fontsize=6.8, color=C_DEGR,
                arrowprops=dict(arrowstyle="-", color=C_DEGR, lw=0.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["degraded", "restored"], fontsize=7.2)
    ax.set_xlim(-0.3, 2.05)
    ax.set_ylim(0.0, 1.10)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("(b) ... navigation is not", loc="left", pad=4,
                 fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", zorder=0)

    # ============ (c) episode + aggregate gap ============
    ax = fig.add_subplot(gs[0, 2])
    if TD.exists():
        img = mpimg.imread(TD)
        h, w = img.shape[:2]
        crop = img[int(0.10*h):int(0.90*h), int(0.05*w):int(0.75*w)]
        ax.imshow(crop, extent=[0, 1, 0, 1], aspect="auto", zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color("#CCCCCC")

    traj_path = HERE / "teaser_traj_xy.json"
    if traj_path.exists():
        pts = json.load(open(traj_path))
        tx = [p[0] for p in pts]; ty = [p[1] for p in pts]
        ax.plot(tx, ty, color="white", lw=3.6, solid_capstyle="round",
                zorder=3)
        ax.plot(tx, ty, color=C_BASE, lw=1.7, solid_capstyle="round",
                zorder=4)
        ax.plot(tx[-1], ty[-1], "o", color=C_BASE, ms=5.5,
                mec="white", mew=1.0, zorder=6)

    # The goal star is already rendered into the Habitat top-down PNG;
    # we only add the 1 m success ring, centred on the *measured* pixel
    # location of that star (axes-fraction 0.706, 0.645 in this crop).
    goal_xy = (0.706, 0.645)
    ax.add_patch(Circle(goal_xy, 0.10, ec="white", fc="none", lw=2.2,
                        zorder=5))
    ax.add_patch(Circle(goal_xy, 0.10, ec=C_HL, fc="none", lw=1.1,
                        zorder=6))

    bb = dict(boxstyle="round,pad=0.22", fc=(1, 1, 1, 0.94),
              ec="#AAAAAA", lw=0.4)
    ax.text(0.03, 0.03,
            "$N{=}300$:  reach 62%  vs.  succeed 37%\n"
            "26-pt gap no test-time fix closes",
            ha="left", va="bottom", fontsize=6.6, color="#222",
            linespacing=1.35, bbox=bb)
    ax.set_title("(c) The reach-to-success gap", loc="left", pad=4,
                 fontsize=8.5, fontweight="bold")

    # ============ (d) forest + (e) coverage matrix ============
    gs_bot = gs_outer[1].subgridspec(1, 2, width_ratios=[1.72, 1.0],
                                     wspace=0.30)
    panel_forest(fig, gs_bot[0, 0])
    panel_positioning(fig, gs_bot[0, 1])

    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.075)
    fig.savefig(HERE / "fig_teaser.pdf")
    fig.savefig(HERE / "fig_teaser.png", dpi=180)
    plt.close(fig)
    print("saved fig_teaser.pdf")


if __name__ == "__main__":
    main()
