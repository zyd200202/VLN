"""fig_topdown_traj.py -- multi-arm top-down trajectory vignette (appendix).

Three paired episodes from the low-light sev4 traj_vignette run (N=12,
seed 0, deterministic decoding on Qwen2-VL-7B), drawn over photoreal
bird's-eye orthographic renders of the actual HM3D scenes (ceiling
clipped; produced by InstructNav/render_topdown_rgb.py and world-aligned
via an empirically calibrated px-per-meter scale). Every trace is the
real per-step agent world position.

Layout:
  (a) ep 6 -- RES rescues:   B0 wanders 292 steps; RES converges in 91.
  (b) ep 7 -- Only oracle:   B0 fails fast; RES also fails; only the
                             non-deployable per-frame appearance oracle
                             reaches the goal.
  (c) ep 4 -- Nothing helps: the majority pattern in the N=300 sweep --
                             all three arms miss the goal.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Circle

from paper_style import C_BASE, C_DEGR, C_ORACLE, C_HL, apply_style

apply_style()

HERE = Path(__file__).resolve().parent
RUNS = Path("/root/autodl-tmp/DRPhysNav/runs/traj_vignette")
RGB = RUNS / "topdown_rgb"

ARMS = [
    ("B0",         C_DEGR),
    ("RES",        C_BASE),
    ("ORACLEGATE", C_ORACLE),
]
ARM_LABEL = {"B0": "B0", "RES": "RES", "ORACLEGATE": "OracleGate"}

CASES = [
    (6, "RES rescues",     "sofa",
     "B0 wanders 292 st; RES converges in 91 --\nrestoration $\\mathit{can}$ rescue (~4/100 eps)"),
    (7, "only oracle",     "toilet",
     "RES fails (304 st); only the non-\ndeployable oracle reaches the goal"),
    (4, "nothing helps",   "bed",
     "the majority pattern: no arm\nchanges the outcome (all miss)"),
]

SUCCESS_RADIUS_M = 1.0
VOID_GRAY = 0.92

META = json.load(open(RGB / "meta.json"))

STROKE = [pe.Stroke(linewidth=3.0, foreground="white", alpha=0.9),
          pe.Normal()]


def _load(ep, arm):
    for line in open(RUNS / f"tv_{arm}_low_light_s4_seed0_traj.jsonl"):
        d = json.loads(line)
        if d["ep"] == ep:
            return d
    raise KeyError((ep, arm))


def world_to_px(m, xz):
    xz = np.atleast_2d(np.asarray(xz, float))
    col = (xz[:, 0] - m["center_x"]) * m["px_per_m"] + m["res"] / 2
    row = (xz[:, 1] - m["center_z"]) * m["px_per_m"] + m["res"] / 2
    return col, row


def main():
    fig, axs = plt.subplots(1, 3, figsize=(6.75, 3.05),
                            gridspec_kw=dict(wspace=0.08))

    for col, (ep, tag, goal_name, note) in enumerate(CASES):
        ax = axs[col]
        m = META[str(ep)]
        eps = {arm: _load(ep, arm) for arm, _ in ARMS}
        ppm = m["px_per_m"]

        img = np.load(RGB / f"ep{ep:03d}_rgb.npy").astype(float) / 255.0
        void = img.sum(axis=2) < 0.04
        img[void] = VOID_GRAY
        ax.imshow(img, interpolation="bilinear", zorder=0)

        # ---- goal instances the episode is judged against
        goals_w = np.asarray(eps["B0"]["goals_world"], float)[:, [0, 2]]
        ends_w = np.array([np.asarray(eps[a]["traj_world"], float)[-1][[0, 2]]
                           for a, _ in ARMS])
        active_idx = {int(np.argmin(np.min(
            np.linalg.norm(goals_w[None, :, :] - ends_w[:, None, :], axis=2),
            axis=0)))}
        for i, (arm, _) in enumerate(ARMS):
            if eps[arm]["success"]:
                active_idx.add(int(np.argmin(
                    np.linalg.norm(goals_w - ends_w[i], axis=1))))
        active = goals_w[sorted(active_idx)]

        all_c, all_r = [], []
        gc, gr = world_to_px(m, active)
        all_c.append(gc); all_r.append(gr)

        # ---- goal stars + 1 m rings
        for (c,), (r,) in [world_to_px(m, g) for g in active]:
            ax.add_patch(Circle((c, r), SUCCESS_RADIUS_M * ppm,
                                fc=C_HL, ec="none", alpha=0.28, zorder=2))
            ax.add_patch(Circle((c, r), SUCCESS_RADIUS_M * ppm,
                                ec=C_HL, fc="none", lw=1.0, ls=(0, (3, 2)),
                                zorder=5, alpha=0.9))
            ax.plot(c, r, marker="*", color=C_HL, ms=11, mec="white",
                    mew=0.8, zorder=6)

        # ---- trajectories, white halo underneath for readability
        for arm, col_arm in ARMS:
            t = np.asarray(eps[arm]["traj_world"], float)[:, [0, 2]]
            xs, ys = world_to_px(m, t)
            all_c.append(xs); all_r.append(ys)
            ax.plot(xs, ys, color=col_arm, lw=1.4, solid_capstyle="round",
                    zorder=4, alpha=0.95, path_effects=STROKE)
            ax.plot(xs[0], ys[0], "o", color="white", ms=6,
                    mec=col_arm, mew=1.4, zorder=7)
            succ = bool(eps[arm]["success"])
            ax.plot(xs[-1], ys[-1], "o" if succ else "X", color=col_arm,
                    ms=6.5 if succ else 7, mec="white", mew=0.9, zorder=8)

        # ---- square crop to content + margin
        cols = np.concatenate(all_c); rows = np.concatenate(all_r)
        c0, c1 = cols.min(), cols.max()
        r0, r1 = rows.min(), rows.max()
        pad = 0.08 * max(c1 - c0, r1 - r0) + 1.2 * ppm
        ccen, rcen = (c0 + c1) / 2, (r0 + r1) / 2
        half = max(c1 - c0, r1 - r0) / 2 + pad
        ax.set_xlim(ccen - half, ccen + half)
        ax.set_ylim(rcen + half, rcen - half)

        # ---- 2 m scale bar, avoiding the busier bottom corner
        bar_px = 2.0 * ppm
        by = rcen + half * 0.86
        busy_l = np.any((rows >= rcen + half * 0.45) &
                        (cols <= ccen - half * 0.25))
        bx = (ccen + half * 0.84 - bar_px) if busy_l else (ccen - half * 0.84)
        ax.plot([bx, bx + bar_px], [by, by], color="#222222", lw=1.6,
                solid_capstyle="butt", zorder=9,
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white",
                                        alpha=0.9), pe.Normal()])
        ax.text(bx + bar_px / 2, by - half * 0.04, "2 m", ha="center",
                va="bottom", fontsize=6.2, color="#222222", zorder=9,
                path_effects=[pe.withStroke(linewidth=1.6,
                                            foreground="white")])

        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_color("#AAAAAA")

        ax.set_title(f"({chr(97+col)}) ep {ep} $\\cdot$ {goal_name} $\\cdot$ {tag}",
                     loc="left", fontsize=8.0, fontweight="bold", pad=28)
        ax.text(0.0, 1.02, note, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.4, color="#444444",
                style="italic", linespacing=1.30)
        # step counts BELOW the axes so they never fight goal markers
        x_cursor = 0.0
        for arm, col_arm in ARMS:
            n = len(eps[arm]["traj_world"])
            ax.text(x_cursor, -0.03, f"{ARM_LABEL[arm]}: {n}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.2, color=col_arm, fontweight="bold")
            x_cursor += 0.34

    # global legend on top
    handles = [plt.Line2D([], [], color=c, lw=1.8, label=ARM_LABEL[n])
               for n, c in ARMS]
    handles += [
        plt.Line2D([], [], color="white", marker="o", ls="none",
                   ms=6, mec="#666666", mew=1.2, label="start"),
        plt.Line2D([], [], color=C_HL, marker="*", ls="none",
                   ms=10, mec="white", mew=0.8, label="goal (1 m ring)"),
        plt.Line2D([], [], color="#666666", marker="o", ls="none",
                   ms=6, mec="white", mew=0.9, label="ends: success"),
        plt.Line2D([], [], color="#666666", marker="X", ls="none",
                   ms=7, mec="white", mew=0.9, label="ends: failure"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=7,
               frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 1.005),
               handlelength=1.3, handletextpad=0.35, columnspacing=1.1)

    fig.subplots_adjust(left=0.010, right=0.990, top=0.780, bottom=0.045)
    fig.savefig(HERE / "fig_topdown_traj.pdf")
    fig.savefig(HERE / "fig_topdown_traj.png", dpi=180)
    plt.close(fig)
    print("saved fig_topdown_traj.pdf")


if __name__ == "__main__":
    main()
