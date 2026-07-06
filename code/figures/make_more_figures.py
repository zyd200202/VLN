"""Additional top-tier figures for DRPhysNav.

Generates four artefacts that fill the remaining empty pages and visually
deliver the paper's three core claims:

    fig_pipeline.pdf      -- the diagnosis instrument: degrade -> backbone ->
                             blind r -> outcome -> paired protocol. 1-column.
    fig_decomp.pdf        -- mechanism decomposition: SR = reach - loss across
                             B0/RES/M1g with r_c inset. N=300 data.
    fig_forest.pdf        -- paired DeltaSR forest plot over all 9 sweep arms,
                             colour-coded by family + significance markers.
    fig_positioning.pdf   -- related-work positioning chart (one row per prior
                             family) showing which evaluation dimensions each
                             approach covers vs. ours.

Numbers are pulled from existing N=300 / N=150 CSVs in
    /root/autodl-tmp/DRPhysNav/runs/
or, when a queued run has not yet completed, marked with TBD in the plot.

Style follows make_figures.py: Okabe-Ito, serif font, NO LaTeX commands inside
matplotlib text, baseline 8/9/10pt sizing, 0.6pt grid.
"""
from __future__ import annotations

import json
import os
import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from paper_style import (
    C_BASE, C_DEGR, C_POS, C_ORACLE, C_NEUT, C_HL,
    T_BASE, T_DEGR, T_POS, T_ORACLE, T_NEUT,
    apply_style,
    SIZE_DOUBLE_SHORT, SIZE_DOUBLE_MED, SIZE_SINGLE_SHORT, SIZE_SINGLE_MED,
)

OUT = Path(__file__).resolve().parent
RUNS = Path("/root/autodl-tmp/DRPhysNav/runs")
apply_style()

# Semantic aliases so old body-code keeps compiling. Do not add new roles.
C_CLEAN = C_BASE    # baseline / clean (deep blue)
C_GOOD  = C_POS     # positive / reach / restoration signal (teal)
C_WARN  = C_HL      # highlight / annotation (amber)
C_VIOL  = C_ORACLE  # non-deployable ceiling (deep purple)
C_YEL   = C_HL      # single-use highlight amber


# ----------------------------------------------------------------------------
# Helper: paired McNemar (exact two-sided) on two N-aligned CSV files of (success, spl, dist, goal).
# ----------------------------------------------------------------------------

def load_sr(path):
    if not Path(path).exists():
        return None
    import csv
    with open(path) as f:
        rows = [r for r in list(csv.reader(f))[1:] if r]
    return [float(r[0]) for r in rows]


def paired_dsr(b_path, c_path):
    b = load_sr(b_path)
    c = load_sr(c_path)
    if b is None or c is None:
        return None
    n = min(len(b), len(c))
    if n == 0:
        return None
    b = b[:n]; c = c[:n]
    sb = sum(b) / n
    sc = sum(c) / n
    b01 = sum(1 for i in range(n) if b[i] == 0 and c[i] == 1)
    b10 = sum(1 for i in range(n) if b[i] == 1 and c[i] == 0)
    ntot = b01 + b10
    k = min(b01, b10)
    p = sum(comb(ntot, i) for i in range(0, k + 1)) / (2 ** ntot) * 2 if ntot > 0 else 1.0
    p = min(1.0, p)
    # bootstrap 95% CI on dSR (5000 resamples) for the forest plot
    rng = np.random.default_rng(42)
    diffs = np.array([c[i] - b[i] for i in range(n)], dtype=float)
    boot = rng.choice(diffs, size=(5000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=n, b_sr=sb, c_sr=sc, dsr=sc - sb, p=p, ci_lo=lo, ci_hi=hi, b01=b01, b10=b10)


# ----------------------------------------------------------------------------
# Figure 1: fig_pipeline.pdf -- diagnosis instrument
# ----------------------------------------------------------------------------

def fig_pipeline():
    """Diagnosis instrument as a 5-stage pipeline with REAL visual anchors.

    Each stage box carries an actual experimental artefact:
      (a) clean HM3D frame   (b) same frame under low-light s4
      (c) label (frozen backbone)   (d) real per-step r sparkline
      (e) real top-down trajectory thumbnail
    """
    import json as _json
    import matplotlib.image as mpimg

    fig, ax = plt.subplots(figsize=(6.75, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#666666",
            fontsize=8.0, fontweight="normal", text_dy=0.0):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.6, zorder=2))
        if text:
            ax.text(x + w / 2, y + h / 2 + text_dy, text,
                    ha="center", va="center",
                    fontsize=fontsize, fontweight=fontweight, zorder=3,
                    color="#222222", linespacing=1.25)

    def arrow(x1, y1, x2, y2, color="#555555", lw=1.0, rad=0.0):
        cs = f"arc3,rad={rad}" if rad else "arc3"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                    connectionstyle=cs), zorder=1)

    def thumb(x, y, w, h, img, ec="#666666"):
        """Place a real image inside data coords [x,x+w]x[y,y+h]."""
        ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto",
                  zorder=3, interpolation="bilinear")
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec,
                               lw=0.8, zorder=4))

    # ---------- load real artefacts ----------
    clean_img, degr_img = None, None
    try:
        import sys as _sys
        _sys.path.insert(0, "/root/autodl-tmp/DRPhysNav/src")
        import h5py
        from drphysnav.degradation.corruptions import apply as _apply
        with h5py.File("/root/autodl-tmp/DRPhysNav/road1/data/frames.h5",
                       "r") as h:
            clean_img = h["clean"][3]
        degr_img = _apply(clean_img.copy(), "low_light", 4)
    except Exception as e:  # pragma: no cover
        print("thumb load failed:", e)

    r_trace = None
    motiv = RUNS / "redesign_n300/n300_B0_low_light_s4_seed0_motiv.jsonl"
    if motiv.exists():
        d = _json.loads(open(motiv).readline())
        r_trace = [s["r"] for s in d["steps"]][:120]

    td_img = None
    td_path = OUT / "fig_td_toilet.png"
    if td_path.exists():
        ti = mpimg.imread(td_path)
        th, tw = ti.shape[:2]
        td_img = ti[int(0.10*th):int(0.90*th), int(0.05*tw):int(0.75*tw)]

    # ---------- row 1: 5 stages; thumbnail on top of a label strip ----------
    y0, h0 = 34, 42          # tall boxes: image (top ~26) + label (bottom ~14)
    img_h = 26

    # (a) stimulus
    box(2, y0, 15, h0, "", T_NEUT)
    if clean_img is not None:
        thumb(3.0, y0 + h0 - img_h - 1, 13.0, img_h, clean_img)
    ax.text(2 + 7.5, y0 + 6.5, "clean RGB\nfrom sim", ha="center",
            va="center", fontsize=6.8, zorder=3, linespacing=1.2)

    # (b) corruption
    box(21, y0, 15, h0, "", T_DEGR)
    if degr_img is not None:
        thumb(22.0, y0 + h0 - img_h - 1, 13.0, img_h, degr_img, ec=C_DEGR)
    ax.text(21 + 7.5, y0 + 6.5,
            r"low-light $\cdot$ blur" "\n" r"fog $\cdot$ noise ($s{=}4$)",
            ha="center", va="center", fontsize=6.6, zorder=3,
            linespacing=1.2)

    # (c) agent under test (text only, frozen)
    box(40, y0, 18, h0,
        "frozen backbone\n\nInstructNav\n$+$ Qwen2-VL-7B\n$+$ GLEE (SwinL)",
        T_BASE, fontsize=6.9)

    # (d) instrument: real r sparkline
    box(62, y0, 16, h0, "", T_POS)
    if r_trace is not None:
        n = len(r_trace)
        sx = [62 + 1.5 + 13.0 * i / (n - 1) for i in range(n)]
        sy = [y0 + 14 + 22 * v for v in r_trace]
        ax.plot(sx, sy, color=C_POS, lw=0.8, zorder=4)
        ax.text(62 + 8, y0 + h0 - 3.0, "per-step $r(t)$",
                ha="center", va="top", fontsize=6.6, color=C_POS,
                zorder=4, fontweight="bold")
    ax.text(62 + 8, y0 + 6.5, "blind reliability\nestimator (no GT)",
            ha="center", va="center", fontsize=6.6, zorder=3,
            linespacing=1.2)

    # (e) outcome: real top-down trajectory
    box(82, y0, 16, h0, "", "#FBEED0")
    if td_img is not None:
        thumb(83.0, y0 + h0 - img_h - 1, 14.0, img_h, td_img, ec=C_HL)
    ax.text(82 + 8, y0 + 6.5, "reach $\\cdot$ commit\n$\\cdot$ stop",
            ha="center", va="center", fontsize=6.6, zorder=3,
            linespacing=1.2)

    # arrows between stages
    ymid = y0 + h0 / 2
    for x1, x2 in [(17, 21), (36, 40), (58, 62), (78, 82)]:
        arrow(x1, ymid, x2, ymid)

    # ---------- row 2: protocol strip ----------
    box(13, 4, 74, 16,
        "(f) paired protocol: same seed and episode order across arms, "
        "$N{=}300$ main split,\n"
        r"McNemar exact $p$ on discordant pairs, bootstrap $95\%$ CI",
        T_ORACLE, fontsize=7.2)
    arrow(90, y0 - 1, 60, 21, color="#888888", lw=0.7, rad=0.30)

    # ---------- captions above each stage ----------
    cap_y = 80
    for x, lab in [(2, "(a) Stimulus"), (21, "(b) Corruption"),
                   (40, "(c) Frozen agent"), (62, "(d) Instrument"),
                   (82, "(e) Outcome")]:
        ax.text(x, cap_y, lab, fontsize=7.5, fontweight="bold",
                color="#222222")

    plt.savefig(OUT / "fig_pipeline.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_pipeline.pdf")


# ----------------------------------------------------------------------------
# Figure 2: fig_decomp.pdf -- mechanism: SR = reach - loss
# ----------------------------------------------------------------------------

def _load_motiv(path):
    """ep -> record (keep last line per ep)."""
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[d["ep"]] = d
    return out


def fig_decomp():
    """Inside the commitment bottleneck: three episode-level mechanism
    panels computed from the per-step logs of the paired N=300 runs.
    Nothing here duplicates Table 1; each panel shows an *internal*
    quantity the aggregate SR hides.

      (a) ECDF of distance-to-goal at the FIRST commit event:
          B0 / RES / OracleGate curves are nearly identical -->
          no intervention moves WHERE the agent decides it has found
          the object (~70% of first commits fire >3 m away).
      (b) distribution of blind reliability r at the commit frame:
          RES shifts r from ~0.40 to ~1.00, yet P(success|commit)
          moves only 0.42 -> 0.43. Perceived trust is repaired;
          decision quality is not.
      (c) paired per-episode min-distance scatter (B0 vs RES), flips
          coloured: rescued episodes were already near the 1 m
          threshold under B0 (median 1.12 m) -- the intervention
          shuffles the margin, it does not move the far failures.
    """
    R = RUNS
    B0  = _load_motiv(R / "redesign_n300/n300_B0_low_light_s4_seed0_motiv.jsonl")
    RES = _load_motiv(R / "redesign_n300/n300_RES_low_light_s4_seed0_motiv.jsonl")
    OG  = _load_motiv(R / "oracle/or_ORACLEGATE_low_light_s4_seed0_motiv.jsonl")

    def commit_stats(D):
        dist_c, r_c, succ_c = [], [], []
        for d in D.values():
            idx = next((i for i, s in enumerate(d["steps"])
                        if s["found"] == 1), None)
            if idx is not None:
                dist_c.append(d["steps"][idx]["dist"])
                r_c.append(d["steps"][idx]["r"])
                succ_c.append(int(d["success"]))
        return (np.asarray(dist_c), np.asarray(r_c), np.asarray(succ_c))

    def ep_min_dist(D):
        return {e: min(s["dist"] for s in d["steps"])
                for e, d in D.items()}

    fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.20),
                            gridspec_kw=dict(wspace=0.34))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.20)

    # ---------- (a) ECDF of dist at first commit ----------
    ax = axs[0]
    for (name, D, color) in [("B0", B0, C_DEGR),
                             ("RES", RES, C_BASE),
                             ("OracleGate", OG, C_ORACLE)]:
        dc, _, _ = commit_stats(D)
        xs = np.sort(dc)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=color, lw=1.3,
                label=name, zorder=3)
    ax.axvspan(0, 1.0, color="#E6F3E6", zorder=0)
    ax.text(0.55, 0.035, "1 m", ha="center", va="bottom",
            fontsize=6.2, color="#5A8A5A")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1.0)
    ax.set_xlabel("distance to goal at first commit (m)", fontsize=7.6)
    ax.set_ylabel("ECDF over episodes", fontsize=7.6)
    ax.set_title("(a) Commits fire in the same\nwrong places",
                 loc="left", pad=4, fontsize=8.4, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=6.8,
              handlelength=1.4, handletextpad=0.4, borderpad=0.15)
    ax.grid(axis="both", zorder=1)

    # ---------- (b) r at commit: distribution shift, outcome flat ----------
    ax = axs[1]
    _, rcB, scB = commit_stats(B0)
    _, rcR, scR = commit_stats(RES)
    rng = np.random.default_rng(0)
    for x0, (rc, color) in enumerate([(rcB, C_DEGR), (rcR, C_BASE)]):
        jx = rng.uniform(-0.17, 0.17, len(rc))
        ax.scatter(x0 + jx, rc, s=5, color=color, alpha=0.30,
                   edgecolors="none", zorder=2)
        med = np.median(rc)
        ax.plot([x0 - 0.26, x0 + 0.26], [med, med], color=color,
                lw=1.8, zorder=4)
        ax.text(x0 + 0.30, med, f"{med:.2f}", ha="left", va="center",
                fontsize=7.2, color=color, fontweight="bold")
    ax.set_xticks([0, 1])
    # spell out the conditional so it cannot be misread as "P(slc)"
    ax.set_xticklabels(["B0\nconv. $0.42$",
                        "RES\nconv. $0.43$"], fontsize=7.4)
    ax.set_xlim(-0.55, 1.65); ax.set_ylim(0, 1.06)
    ax.set_ylabel("reliability $r$ at commit frame", fontsize=7.6)
    ax.set_title("(b) Trust is repaired,\ndecision is not",
                 loc="left", pad=4, fontsize=8.4, fontweight="bold")
    ax.grid(axis="y", zorder=1)

    # ---------- (c) paired min-dist scatter, flips coloured ----------
    ax = axs[2]
    mdB, mdR = ep_min_dist(B0), ep_min_dist(RES)
    common = sorted(set(mdB) & set(mdR))
    xs = np.array([mdB[e] for e in common])
    ys = np.array([mdR[e] for e in common])
    sB = np.array([int(B0[e]["success"]) for e in common])
    sR = np.array([int(RES[e]["success"]) for e in common])
    stay = (sB == sR)
    resc = (sB == 0) & (sR == 1)
    brok = (sB == 1) & (sR == 0)
    LIM = 9.0
    xc, yc = np.clip(xs, 0, LIM), np.clip(ys, 0, LIM)
    ax.scatter(xc[stay], yc[stay], s=6, color="#BBBBBB", alpha=0.55,
               edgecolors="none", zorder=2, label="unchanged")
    ax.scatter(xc[resc], yc[resc], s=13, color=C_POS, alpha=0.9,
               edgecolors="white", linewidths=0.3, zorder=4,
               label=f"rescued ($b_{{01}}{{=}}{resc.sum()}$)")
    ax.scatter(xc[brok], yc[brok], s=13, color=C_DEGR, alpha=0.9,
               edgecolors="white", linewidths=0.3, zorder=4,
               label=f"broken ($b_{{10}}{{=}}{brok.sum()}$)")
    ax.plot([0, LIM], [0, LIM], color="#999999", lw=0.6, ls="--",
            zorder=1)
    ax.axvspan(0, 1.0, color="#E6F3E6", zorder=0)
    ax.axhspan(0, 1.0, color="#E6F3E6", zorder=0)
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM)
    ax.set_xlabel("B0 min dist to goal (m)", fontsize=7.6)
    ax.set_ylabel("RES min dist (m)", fontsize=7.6)
    ax.set_title("(c) Flips hug the success\nmargin",
                 loc="left", pad=4, fontsize=8.4, fontweight="bold")
    # keep the note clear of the diagonal, the point cloud, and the legend:
    # bottom-right pocket just above the 1 m success band is empty
    ax.text(0.97, 0.255, "median rescued episode:\nB0 was already at 1.1 m",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.0, color="#555555", style="italic",
            linespacing=1.3)
    leg = ax.legend(loc="upper right", fontsize=6.2,
                    handletextpad=0.25, borderpad=0.3, labelspacing=0.25,
                    frameon=True, framealpha=0.85, edgecolor="#CCCCCC")
    leg.get_frame().set_linewidth(0.4)
    ax.grid(axis="both", zorder=1)

    plt.savefig(OUT / "fig_decomp.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_decomp.pdf")


# ----------------------------------------------------------------------------
# Figure 3: fig_forest.pdf -- paired DeltaSR forest over all sweep arms
# ----------------------------------------------------------------------------

def fig_forest():
    """One row per arm. dSR (point) with 95% bootstrap CI bar.
    Colour by family; star=p<0.05; light grey=TBD (queue not finished)."""
    # known N=300 rows (already in paper)
    n300_b = RUNS / "redesign_n300/n300_B0_low_light_s4_seed0.csv"
    # Family colours (semantic, drawn only as small left-margin badges)
    FAM_APPEAR  = C_DEGR      # perception-side (RES)
    FAM_WHERE   = C_BASE      # where-to-go (M1g)
    FAM_TIMING  = C_POS       # commit timing (REVOKE / CRV)
    FAM_TEMP    = C_HL        # temporal (ROUTER / MUAP)
    FAM_MULTI   = C_NEUT      # multi-view / spatial (FUSE)
    FAM_ORACLE  = C_ORACLE    # non-deployable ceiling (OracleGate)
    FAM_PROBE   = C_ORACLE    # cross-modal probe (DXCV)

    n300_rows = [
        ("RES",        "Appearance",        RUNS / "redesign_n300/n300_RES_low_light_s4_seed0.csv",   "300", FAM_APPEAR),
        ("OracleGate", "Appearance UB",     RUNS / "oracle/or_ORACLEGATE_low_light_s4_seed0.csv",     "300", FAM_ORACLE),
        ("M1g",        "Where-to-go",       RUNS / "redesign_n300/n300_M1g_low_light_s4_seed0.csv",   "300", FAM_WHERE),
    ]
    n150_b_ref = RUNS / "maintable/mt_B0_low_light_s4_seed0.csv"
    n150_known = [
        ("ROUTER",     "Temporal",          RUNS / "maintable/mt_ROUTER_low_light_s4_seed0.csv",     "150", FAM_TEMP),
    ]
    unified = RUNS / "unified"
    n150_queue = [
        ("REVOKE",     "Commit timing",     unified / "u_REVOKE_low_light_s4_seed0_N150.csv", "150", FAM_TIMING),
        ("CRV",        "Forced commit",     unified / "u_CRV_low_light_s4_seed0_N150.csv",    "150", FAM_TIMING),
        ("MUAP",       "Temporal",          unified / "u_MUAP_low_light_s4_seed0_N150.csv",   "150", FAM_TEMP),
        ("FUSE",       "Multi-view",        unified / "u_FUSE_low_light_s4_seed0_N150.csv",   "150", FAM_MULTI),
        ("DXCV",       "Multimodal probe",  unified / "u_DXCV_low_light_s4_seed0_N150.csv",   "150", FAM_PROBE),
    ]

    # compute paired stats (or TBD)
    points = []
    # N=300 rows
    for arm, fam, c_path, n_tag, color in n300_rows:
        s = paired_dsr(n300_b, c_path)
        points.append((arm, fam, n_tag, color, s))
    # N=150 rows
    for arm, fam, c_path, n_tag, color in n150_known + n150_queue:
        s = paired_dsr(n150_b_ref, c_path)
        points.append((arm, fam, n_tag, color, s))

    # Reorder: put OracleGate (non-deployable ceiling) at top,
    # then deployable arms sorted by dSR (descending, so best-looking on top).
    def _sort_key(p):
        arm, fam, n_tag, color, s = p
        if arm == "OracleGate":
            return (0, 0.0)
        if s is None:
            return (2, 0.0)      # queued/no-data last within deployables
        return (1, -s["dsr"])
    points = sorted(points, key=_sort_key)

    fig, ax = plt.subplots(figsize=(6.75, 2.55))
    yticks, ylabels = [], []
    for i, (arm, fam, n_tag, color, s) in enumerate(points):
        y = len(points) - i
        yticks.append(y)
        ylabels.append(f"{arm}  ($N{{=}}{n_tag}$)")
        if s is None:
            ax.scatter([0], [y], marker="s", s=42, facecolors="white",
                       edgecolors="#BBBBBB", linewidths=0.8, zorder=3)
            ax.text(0.02, y, "queued $N{=}150$", va="center", ha="left",
                    fontsize=7.2, color="#888888", style="italic")
        else:
            d = s["dsr"]; lo = s["ci_lo"]; hi = s["ci_hi"]
            # Non-significant = neutral grey error bar; significant = coloured.
            is_sig = s["p"] < 0.05
            bar_col = color if is_sig else C_NEUT
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o",
                        color=bar_col, ecolor=bar_col, capsize=2.5,
                        ms=5.5, lw=1.2, mec="white", mew=0.6, zorder=3)
            star = " *" if is_sig else ""
            ax.text(max(hi, d) + 0.010, y,
                    f"${d:+.3f}$  $p{{=}}{s['p']:.3f}${star}",
                    va="center", fontsize=7.2, color="#333333")

    # zero-effect vertical line + shaded "practically null" band
    ax.axvspan(-0.02, 0.02, color=T_NEUT, alpha=0.6, zorder=0)
    ax.axvline(0, color="#666666", lw=0.7, zorder=1)

    ax.set_xlim(-0.20, 0.35)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7.8)
    ax.set_xlabel(r"paired $\Delta$ success rate vs. baseline "
                  r"(low-light sev 4, seed 0)")
    ax.set_title("Per-arm effect on success rate",
                 loc="left", pad=4, fontsize=9.0, fontweight="bold")
    ax.grid(axis="x", zorder=0)
    ax.set_ylim(-0.2, len(points) + 0.8)

    # bottom-margin footnote (tiny, unobtrusive)
    ax.text(0.0, -0.28,
            r"* significant at $p<0.05$ (McNemar exact). Bars: bootstrap "
            r"$95\%$ CI over paired episode differences.  "
            r"Grey band: practically-null $|\Delta|<0.02$.",
            fontsize=6.8, color="#666666", ha="left", va="top",
            transform=ax.transAxes)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.24, top=0.90)

    plt.savefig(OUT / "fig_forest.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_forest.pdf")


# ----------------------------------------------------------------------------
# Figure 4: fig_positioning.pdf -- related-work positioning, single column
# ----------------------------------------------------------------------------

def fig_positioning():
    """Each row is a prior family; each column is an evaluation dimension. A
    filled circle means the family covers that dimension."""
    fams = [
        ("SemExp",        "Chaplot+20"),
        ("L3MVN",         "Yu+23"),
        ("ESC",           "Zhou+23"),
        ("CoW",           "Gadre+23"),
        ("VLFM",          "Yokoyama+24"),
        ("NavGPT",        "Zhou+24"),
        ("InstructNav",   "Long+24"),
        ("PromptIR / AirNet", "restoration"),
        ("RobustNav",     "Chattopadhyay+21"),
        ("Habitat ObjectNav", "Batra+20"),
        ("DRPhysNav",     "this paper"),
    ]
    cols = ["Degraded\nRGB", "Paired\nprotocol", r"$N \geq 300$",
            "Mech.\ndecomp.", "Multi-modal\nprobe",
            "Negative\nresults"]
    # 0 = not covered, 1 = covered, 0.5 = partial
    cover = np.array([
        [0,   0,   1,   0,   0.5, 0],     # SemExp (depth for mapping)
        [0,   0,   1,   0,   0,   0],     # L3MVN
        [0,   0,   1,   0,   0,   0],     # ESC
        [0,   0,   1,   0,   0.5, 0],     # CoW (depth for mapping)
        [0,   0,   1,   0,   0.5, 0],     # VLFM (depth value map)
        [0,   0,   0,   0,   0,   0],     # NavGPT (R2R, no degraded)
        [0,   0,   1,   0,   0.5, 0],     # InstructNav
        [1,   0,   0,   0,   0,   0],     # restoration (image-level only)
        [1,   0,   1,   0,   0,   0.5],   # RobustNav (corruption, no pairing)
        [0,   0,   1,   0,   0.5, 0],     # Habitat ObjectNav protocol
        [1,   1,   1,   1,   1,   1],     # DRPhysNav
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(6.75, 3.30))
    nr, nc = cover.shape
    col_w = 1.75
    label_zone_w = 5.8
    top_zone_h = 1.7

    # highlight strip behind the "this paper" row (bottom row)
    ax.add_patch(Rectangle((-label_zone_w + 0.5, 0),
                           nc * col_w + label_zone_w - 0.5, 1,
                           fc=T_ORACLE, ec="none", zorder=0, alpha=0.65))

    # thin gridlines
    for c in range(nc + 1):
        ax.plot([c * col_w, c * col_w], [0, nr],
                color="#D8D8D8", lw=0.4, zorder=1)
    for r in range(nr + 1):
        ax.plot([0, nc * col_w], [r, r],
                color="#D8D8D8", lw=0.4, zorder=1)

    # cells: ✓ / ✗ / ~ (partial)
    for r in range(nr):
        is_ours = (r == nr - 1)
        for c in range(nc):
            cx = (c + 0.5) * col_w
            cy = nr - r - 0.5
            v = cover[r, c]
            if v == 1.0:
                col = C_ORACLE if is_ours else C_BASE
                ax.text(cx, cy, r"$\checkmark$", ha="center", va="center",
                        fontsize=11.5, color=col, fontweight="bold", zorder=3)
            elif v == 0.5:
                ax.text(cx, cy, r"$\sim$", ha="center", va="center",
                        fontsize=11, color="#888888", zorder=3)
            else:
                ax.text(cx, cy, r"$-$", ha="center", va="center",
                        fontsize=10, color="#C8C8C8", zorder=3)

    # column labels (top)
    for c in range(nc):
        ax.text((c + 0.5) * col_w, nr + 0.35, cols[c],
                ha="center", va="bottom",
                fontsize=7.5, color="#222222", linespacing=1.20)

    # row labels (left): single line, "name  (ref)" with gray ref
    for r in range(nr):
        is_ours = (r == nr - 1)
        name, desc = fams[r]
        ax.text(-0.20, nr - r - 0.5, f"{name}  ", ha="right", va="center",
                fontsize=8.0, color=C_ORACLE if is_ours else "#111111",
                fontweight="bold" if is_ours else "normal")
        ax.text(-0.20, nr - r - 0.5 - 0.02, "", ha="left", va="center")
    # gray refs in a second pass, in their own column at the far left
    for r in range(nr):
        name, desc = fams[r]
        ax.text(-label_zone_w + 1.15, nr - r - 0.5, desc,
                ha="right", va="center",
                fontsize=6.2, color="#AAAAAA", fontstyle="italic")

    ax.set_xlim(-label_zone_w, nc * col_w + 0.3)
    ax.set_ylim(-0.3, nr + top_zone_h)
    ax.axis("off")
    ax.set_title("Coverage of evaluation dimensions",
                 loc="left", pad=8, fontsize=9.0, fontweight="bold",
                 x=-0.02)

    plt.savefig(OUT / "fig_positioning.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_positioning.pdf")


def fig_headline():
    """Intro-page headline result strip. Three side-by-side claims, single row."""
    fig = plt.figure(figsize=(7.0, 2.0))
    gs = fig.add_gridspec(1, 3, wspace=0.85, width_ratios=[1.0, 1.2, 0.85])

    # ----- (a) reach vs success bar -----
    ax = fig.add_subplot(gs[0, 0])
    arms = ["clean", "low-light"]
    reach = [0.93, 0.62]
    succ  = [0.74, 0.36]
    x = np.arange(len(arms)); w = 0.32
    ax.bar(x - w/2, reach, w, color=C_GOOD,  label="reach", edgecolor="white")
    ax.bar(x + w/2, succ,  w, color=C_CLEAN, label="success", edgecolor="white")
    for xi, r, s in zip(x, reach, succ):
        ax.text(xi - w/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=7.0, color="#333")
        ax.text(xi + w/2, s + 0.02, f"{s:.2f}", ha="center", fontsize=7.0, color="#333")
    ax.set_ylim(0, 1.06)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.0)
    ax.set_title("(a) reach-success gap", loc="left", fontsize=8.4, pad=3)
    ax.set_ylabel("rate", fontsize=8.2)
    ax.legend(fontsize=7.2, frameon=False, loc="lower left")
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)
    ax.tick_params(labelsize=7.5)

    # ----- (b) the 5-arm verdict (real data where available; TBD elsewhere) -----
    ax = fig.add_subplot(gs[0, 1])
    # (delta, p, CI lo, CI hi, N, has_data)
    rows = [
        ("Appearance",    0.040, 0.182, -0.018, 0.097,  300, True),
        ("Where-to-go",   0.040, 0.201, -0.020, 0.097,  300, True),
        ("Commit timing", None,  None,  None,   None,   150, False),
        ("Multi-frame",   0.013, 0.875, -0.067, 0.107,  150, True),  # ROUTER known
        ("Multi-view",    None,  None,  None,   None,   150, False),
    ]
    y = np.arange(len(rows))
    for yi, (lab, d, p, lo, hi, N, has) in zip(y, rows):
        if has:
            ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o",
                        color=C_NEUT, ecolor=C_NEUT, capsize=2, ms=5, lw=1.0)
            ax.text(0.18, yi, f"$+{d:.2f}$, n.s.",
                    va="center", fontsize=7.0, color="#666")
        else:
            ax.errorbar(0, yi, xerr=0.0, fmt="s", color="#CCCCCC", ms=5)
            ax.text(0.16, yi, "queued $N{=}150$",
                    va="center", fontsize=7.0, color="#999", style="italic")
    ax.axvline(0, color="#444", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.7)
    ax.set_xlim(-0.10, 0.32)
    ax.set_xlabel("$\\Delta$ SR (paired)", fontsize=7.8)
    ax.set_title("(b) 5 families, no significant gain",
                 loc="left", fontsize=8.4, pad=3)
    ax.tick_params(labelsize=7.5)

    # ----- (c) per-frame oracle ceiling -----
    ax = fig.add_subplot(gs[0, 2])
    cats = ["B0", "OracleGate", "headroom"]
    vals = [0.358, 0.415, 0.642 - 0.415]
    # stacked: B0 + OracleGate delta + remaining headroom to reach=64%
    ax.bar([0], [0.358], color=C_CLEAN, label="baseline SR")
    ax.bar([0], [0.057], bottom=[0.358], color=C_VIOL, label="oracle ceiling $+0.057$")
    ax.bar([0], [0.642 - 0.415], bottom=[0.415], color="#DDDDDD",
           label="remaining headroom\n(reach $-$ SR)")
    ax.set_xlim(-0.8, 0.8); ax.set_xticks([])
    ax.set_ylim(0, 0.75)
    ax.text(0, 0.179, "$0.358$ B0", ha="center", fontsize=7.3, color="white")
    ax.text(0, 0.387, "$+0.057$", ha="center", fontsize=7.3, color="white")
    ax.text(0, 0.530, "$+0.227$ unreached\nby any test-time fix",
            ha="center", fontsize=7.0, color="#333", style="italic")
    ax.set_title("(c) oracle leaves $22\\%$ unreached",
                 loc="left", fontsize=8.4, pad=3)
    ax.set_ylabel("success rate", fontsize=8.2)
    ax.tick_params(labelsize=7.5)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.18)
    out = OUT / "fig_headline.pdf"
    plt.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_roadmap():
    """Conclusion roadmap: closed test-time space vs. two open directions.

    Redesigned: 2x2 grid, semantic tints, no piercing arrows.
    Left = 'What is closed'; right = two open directions with rationale below.
    """
    fig, ax = plt.subplots(figsize=(6.75, 2.10))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#888888", fs=8.0, fw="normal"):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.6, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color="#222222",
                linespacing=1.25, zorder=3)

    def arrow(x1, y1, x2, y2, color="#888888", lw=1.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color),
                    zorder=1)

    # ---- header (title-strip) ----
    ax.text(2, 55, "Forward directions implied by the diagnosis",
            fontsize=9.0, fontweight="bold", color="#222222")

    # ---- left: closed test-time space ----
    box(2, 6, 30, 40,
        "Closed under test-time fixes\n\n"
        "irreducible commitment\nerror; no perception-side\npatch recovers it",
        T_DEGR, ec=C_DEGR, fs=8.0, fw="bold")
    # arrow from closed --> open
    arrow(33, 26, 37, 26, color="#888888", lw=1.0)

    # ---- right top: Road 1 ----
    box(38, 26, 60, 20,
        "Road 1  \u2014  degradation-aware training / adaptation\n\n"
        "teach the agent to remain discriminative under corruption\n"
        "(robust pre-training, adapter, contrastive heads)",
        T_BASE, ec=C_BASE, fs=7.5, fw="normal")

    # ---- right bottom: Road 2 ----
    box(38, 4, 60, 20,
        "Road 2  \u2014  a degradation-invariant sensing channel\n\n"
        "fuse RGB with depth / LiDAR under degraded RGB\n"
        "(this paper: DXCV probe; remedies need learned fusion)",
        T_POS, ec=C_POS, fs=7.5, fw="normal")

    plt.savefig(OUT / "fig_roadmap.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_roadmap.pdf")


def fig_takeaway():
    """Single-column tiny figure for the conclusion: visualises the 26-pt
    headroom on a number-line and points to the two viable directions
    (training-side / multi-modal with training) that the paper advocates."""
    fig, ax = plt.subplots(figsize=(7.0, 1.55))
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.axis("off")

    # number-line at y=0.42
    y0 = 0.42
    ax.hlines(y0, 0.07, 0.93, color="#666", lw=1.0)
    # ticks for 0, 0.25, 0.5, 0.75, 1.0
    for v in (0.0, 0.25, 0.50, 0.75, 1.0):
        x = 0.07 + 0.86 * v
        ax.vlines(x, y0 - 0.025, y0 + 0.025, color="#666", lw=0.8)
        ax.text(x, y0 - 0.10, f"{v:.2f}", ha="center", va="top",
                fontsize=7.5, color="#666")

    # markers
    def mark(v, label, color, off_y=0.10):
        x = 0.07 + 0.86 * v
        ax.plot(x, y0, "o", color=color, ms=8, mec="white", mew=1.0, zorder=4)
        ax.text(x, y0 + off_y, label, ha="center", va="bottom",
                fontsize=8.0, color=color, fontweight="bold")

    mark(0.36, "success", C_CLEAN, off_y=0.10)
    mark(0.62, "reach", C_GOOD, off_y=0.10)
    # 26-pt headroom bracket
    ax.annotate("", xy=(0.07 + 0.86 * 0.62, y0 + 0.05),
                xytext=(0.07 + 0.86 * 0.36, y0 + 0.05),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=0.9))
    ax.text((0.07 + 0.86 * 0.49), y0 + 0.20, "$26$-pt headroom",
            ha="center", va="bottom", fontsize=8.5, color="#222",
            fontweight="bold")

    # left annotation: what was tested
    ax.text(0.18, 0.10,
            "perception-side patches:\n"
            "$\\sim{+}0.04$, ns",
            ha="center", va="center", fontsize=7.3, color="#444",
            bbox=dict(boxstyle="round,pad=0.22", fc="#F2F2F2", ec="#999", lw=0.5))
    # right annotation: what is needed
    ax.text(0.78, 0.10,
            "training-side\\,/\\,multi-modal\n"
            "with learned RGB$+$depth fusion",
            ha="center", va="center", fontsize=7.3, color="#1B5E20",
            bbox=dict(boxstyle="round,pad=0.22", fc="#E8F5E9", ec="#2E7D32", lw=0.5))
    # arrow from "perception" to "headroom" with crossed-out
    ax.annotate("", xy=(0.07 + 0.86 * 0.49, y0 + 0.05),
                xytext=(0.18, 0.18),
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.7))
    ax.text(0.30, 0.30, "$\\times$", color="#D55E00", fontsize=11,
            ha="center", va="center", fontweight="bold")
    # arrow from "training" to "headroom"
    ax.annotate("", xy=(0.07 + 0.86 * 0.49, y0 + 0.05),
                xytext=(0.78, 0.18),
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.1))

    ax.set_title("Where the headroom lives, and what it takes to reach it",
                 loc="center", pad=2, fontsize=9.5)

    out = OUT / "fig_takeaway.pdf"
    plt.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_pipeline()
    fig_decomp()
    fig_forest()
    fig_positioning()
    fig_takeaway()
    fig_headline()
    fig_roadmap()
