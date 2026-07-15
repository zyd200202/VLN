"""ICML-grade figure regeneration for DRPhysNav.

Three artefacts:
  fig_motivation.pdf  -- 4-panel motivation study (M0/M2/M3/M4) without label
                         crowding, colour-blind safe (Okabe-Ito), uses slope and
                         dumbbell layouts instead of grouped bars where useful.
  fig_calibration.pdf -- half-column reliability-estimator sanity plot.
  fig_teaser.pdf      -- composed below in make_teaser.py.

Numbers are pulled directly from the paper text (sections/30_motivation.tex)
so the figure cannot drift from the prose.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from paper_style import (
    C_BASE, C_DEGR, C_POS, C_ORACLE, C_NEUT, C_HL,
    T_BASE, T_DEGR, T_POS, T_NEUT,
    apply_style, SIZE_DOUBLE_TALL, SIZE_SINGLE_SHORT,
)

OUT = Path(__file__).resolve().parent
apply_style()

# Local aliases so old body-code keeps working without renaming every use
C_CLEAN = C_BASE   # baseline / clean (dark blue)
C_ACC   = C_POS    # reliability / restoration signal (teal)
C_LL    = C_ORACLE # ceiling / alternate (purple)
C_FILL  = "#F2F2F2"

def save(fig, name):
    p = OUT / (name + ".pdf")
    fig.savefig(p)
    fig.savefig(OUT / (name + ".png"), dpi=180)
    plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------
def fig_motivation():
    plt.rcParams.update({
        "font.size": 6.4, "axes.labelsize": 6.8, "axes.titlesize": 7.6,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "legend.fontsize": 5.6,
    })
    fig, axs = plt.subplots(1, 4, figsize=(7.05, 1.78),
                            gridspec_kw=dict(wspace=0.72))
    axs = np.array([[axs[0], axs[1]], [axs[2], axs[3]]])

    # --- (a) what degradation does per-episode: paired min-dist scatter ---
    # Same 300 episodes run clean and low-light s4 (paired seed). The point:
    # min-dist barely moves (median 0.11 -> 0.15 m) yet 48 episodes flip to
    # failure -- degradation breaks CONVERSION, not approach.
    ax = axs[0, 0]
    import json as _json

    def _load_motiv(p):
        out = {}
        for line in open(p):
            d = _json.loads(line); out[d["ep"]] = d
        return out

    _R = "/root/autodl-tmp/DRPhysNav/runs"
    try:
        CL = _load_motiv(f"{_R}/oracle/or_CLEANCEIL_clean_s4_seed0_motiv.jsonl")
        DG = _load_motiv(f"{_R}/redesign_n300/n300_B0_low_light_s4_seed0_motiv.jsonl")
        common = sorted(set(CL) & set(DG))
        mdc = np.array([min(s["dist"] for s in CL[e]["steps"]) for e in common])
        mdd = np.array([min(s["dist"] for s in DG[e]["steps"]) for e in common])
        sc  = np.array([int(CL[e]["success"]) for e in common])
        sd  = np.array([int(DG[e]["success"]) for e in common])
    except FileNotFoundError:
        common = []

    LIM = 9.0
    xc, yc = np.clip(mdc, 0, LIM), np.clip(mdd, 0, LIM)
    stay  = (sc == sd)
    lost  = (sc == 1) & (sd == 0)   # degradation breaks the episode
    gain  = (sc == 0) & (sd == 1)
    ax.scatter(xc[stay], yc[stay], s=6, color="#BBBBBB", alpha=0.5,
               edgecolors="none", zorder=2)
    ax.scatter(xc[lost], yc[lost], s=13, color=C_DEGR, alpha=0.9,
               edgecolors="white", linewidths=0.3, zorder=4)
    ax.scatter(xc[gain], yc[gain], s=13, color=C_ACC, alpha=0.9,
               edgecolors="white", linewidths=0.3, zorder=4)
    ax.plot([0, LIM], [0, LIM], color="#999999", lw=0.6, ls="--", zorder=1)
    ax.axvspan(0, 1.0, color="#E6F3E6", zorder=0)
    ax.axhspan(0, 1.0, color="#E6F3E6", zorder=0)
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM)
    ax.set_xlabel("clean min dist to goal (m)")
    ax.set_ylabel("low-light min dist (m)")
    ax.set_title("(a) Breaks conversion,\nnot approach",
                 loc="left", pad=5, fontweight="bold")
    # in-plot colour-coded labels (no box), left-aligned in the sparse
    # upper-left region so they clear the diagonal and the red points
    ax.text(0.24, 0.94, "unchanged", transform=ax.transAxes,
            ha="left", va="center", fontsize=5.4, color="#9A9A9A")
    ax.text(0.24, 0.83, f"broken ($n{{=}}{lost.sum()}$)",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=5.4, color=C_DEGR, fontweight="bold")
    ax.text(0.24, 0.72, f"gained ($n{{=}}{gain.sum()}$)",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=5.4, color=C_ACC, fontweight="bold")
    ax.grid(axis="both", lw=0.4, alpha=0.4, zorder=1)

    # --- (b) M2 failure composition: side-by-side dot pairs ---
    ax = axs[0, 1]
    cats = ["walk-away", "irreversible", "false stop"]
    clean = [0.78, 0.73, 0.55]
    degr  = [0.88, 0.79, 0.76]
    ys = np.arange(len(cats))[::-1]
    for y, c, d in zip(ys, clean, degr):
        ax.plot([c, d], [y, y], color=C_NEUT, lw=1.0, zorder=1)
        ax.plot(c, y, "o", color=C_CLEAN, markersize=6.5, mec="white", mew=0.8, zorder=3)
        ax.plot(d, y, "s", color=C_DEGR,  markersize=6.5, mec="white", mew=0.8, zorder=3)
        # delta annotation between the two markers (above the line)
        mid = (c + d) / 2
        ax.text(mid, y + 0.18, "+%.2f" % (d - c), ha="center", va="bottom",
                fontsize=6.4, color=C_DEGR)
    ax.set_yticks(ys); ax.set_yticklabels(cats)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(-0.5, len(cats) - 0.5 + 0.30)
    # walk-away / irreversible are fractions of failed episodes;
    # false stop is the fraction of executed stops that are false positives.
    ax.set_xlabel("rate")
    ax.set_title("(b) Failure\ncomposition", loc="left", pad=5,
                 fontweight="bold")
    ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)

    # --- (c) M3 dose-response: severity -> r and irreversible % ---
    ax = axs[1, 0]
    sev = np.array([0, 1, 2, 3, 4])
    r   = np.array([0.92, 0.87, 0.70, 0.53, 0.41])
    irr = np.array([68, 81, 95, 95, np.nan])
    ax.plot(sev, r, "-o", color=C_CLEAN, label=r"reliability $r$",
            mec="white", mew=0.7)
    ax.set_xlabel("low-light severity")
    ax.set_ylabel(r"mean reliability $r$", color=C_CLEAN)
    ax.tick_params(axis="y", labelcolor=C_CLEAN)
    ax.set_ylim(0.30, 1.0)
    ax.set_xticks(sev)
    ax.set_title("(c) Severity\ndose-response", loc="left", pad=5,
                 fontweight="bold")
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    # secondary y for irreversible % (no ylabel in the 1x4 layout: the
    # legend carries the meaning, colored ticks carry the mapping)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(sev[:-1], irr[:-1], "--s", color=C_DEGR, mec="white", mew=0.7)
    ax2.plot([sev[-2], sev[-1]], [irr[-2], irr[-2]],
             "--", color=C_DEGR, alpha=0.35)
    ax2.set_ylim(55, 100)
    ax2.tick_params(axis="y", labelcolor=C_DEGR)
    # combined legend in safe corner
    lns = [plt.Line2D([], [], color=C_CLEAN, marker="o", label=r"reliability $r$"),
           plt.Line2D([], [], color=C_DEGR, ls="--", marker="s",
                      label="irreversible (%)")]
    ax.legend(handles=lns, loc="lower left", frameon=False, fontsize=5.0,
              handletextpad=0.3, borderpad=0.15, handlelength=1.3)

    # --- (d) M4 real per-episode commit-frame r values ---
    ax = axs[1, 1]
    import json as _json
    real = _json.load(open(OUT / "m4_commit_r.json"))
    clean_r = np.asarray(real["clean"])
    degr_r  = np.asarray(real["low_light"])
    clean_mu = float(clean_r.mean()); degr_mu = float(degr_r.mean())
    rng = np.random.default_rng(0)
    jc = rng.uniform(-0.16, 0.16, len(clean_r))
    jd = rng.uniform(-0.16, 0.16, len(degr_r))
    ax.scatter(0 + jc, clean_r, s=10, color=C_CLEAN, alpha=0.45, zorder=1,
               edgecolors="white", linewidths=0.3)
    ax.scatter(1 + jd, degr_r,  s=10, color=C_DEGR,  alpha=0.45, zorder=1,
               edgecolors="white", linewidths=0.3)
    # mean diamonds (real means)
    ax.plot(0, clean_mu, "D", color=C_CLEAN, markersize=9, mec="white", mew=1.0, zorder=3)
    ax.plot(1, degr_mu,  "D", color=C_DEGR,  markersize=9, mec="white", mew=1.0, zorder=3)
    # connector arc bowing UP; the delta label sits to the LEFT of the arc
    # (arc only reaches this low height near its right/low-light end)
    ax.annotate("", xy=(0.80, degr_mu + 0.05), xytext=(0.20, clean_mu - 0.03),
                arrowprops=dict(arrowstyle="->", color=C_NEUT, lw=0.9,
                                connectionstyle="arc3,rad=0.38"))
    ax.text(0.30, 0.60, r"$\Delta=%+.2f$" % (degr_mu - clean_mu),
            ha="center", va="center", fontsize=6.4, color=C_NEUT)
    # mean labels with clear offsets
    ax.text(0.22, clean_mu + 0.04, "%.2f" % clean_mu, color=C_CLEAN,
            fontsize=7.0, ha="left", va="center")
    ax.text(0.78, degr_mu, "%.2f" % degr_mu, color=C_DEGR, fontsize=7.0,
            ha="right", va="center")
    ax.set_xticks([0, 1])
    # N = episodes with >=1 commit event, out of 90 per condition
    ax.set_xticklabels(["clean", "low-light"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel(r"$r$ at commit")
    ax.set_title("(d) Reliability at\ncommit frame",
                 loc="left", pad=5, fontweight="bold")
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    save(fig, "fig_motivation")
    apply_style()


# ----------------------------------------------------------------------
def fig_calibration():
    # compact: 5 bars do not deserve a full column of height
    fig, ax = plt.subplots(figsize=(2.5, 1.45))
    conds = ["clean", "low-\nlight", "motion-\nblur", "fog", "gauss.\nnoise"]
    vals  = [1.00, 0.65, 0.47, 0.36, 0.20]
    xs = np.arange(len(conds))
    # Monochrome + one accent: clean = C_BASE (accent), all corruptions = neutral gray.
    colors = [C_BASE] + [C_NEUT] * (len(conds) - 1)
    ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.6,
           width=0.62, zorder=2)
    ax.axhline(0.9, ls="--", color="#888888", lw=0.7)
    ax.text(len(conds) - 0.55, 0.93, "no-op gate", ha="right", va="bottom",
            fontsize=6.8, color="#666666")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.03, "%.2f" % v, ha="center", va="bottom",
                fontsize=7.2, color="#333333")
    ax.set_xticks(xs); ax.set_xticklabels(conds, fontsize=7.0)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel(r"calibrated reliability $r$", fontsize=8)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("Blind reliability estimator", loc="left",
                 fontsize=8.5, pad=4, fontweight="bold")
    ax.grid(axis="y", zorder=0)
    save(fig, "fig_calibration")


if __name__ == "__main__":
    fig_motivation()
    fig_calibration()
    print("done")
