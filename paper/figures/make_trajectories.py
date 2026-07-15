"""fig_trajectories.py -- top-down multi-arm trajectory vignettes on
photoreal bird's-eye renders of the actual HM3D scenes.

Backgrounds are orthographic RGB renders (ceiling clipped by the near
plane) produced by InstructNav/render_topdown_rgb.py, world-aligned via
an empirically calibrated px-per-meter scale. Trajectories are REAL
step-level world-coordinate logs (DRPN_TRAJ_LOG=1 runs), so curves sit
exactly on the floors they were walked on.

Four paired episodes from the low-light sev-4 seed-0 vignette batch:
  (a) ep 6  <sofa>   B0 stops 0.10 m from goal yet FAILS (reach != success)
  (b) ep 2  <chair>  restoration pulls the agent onto the goal
  (c) ep 7  <toilet> only the appearance oracle finds the way
  (d) ep 8  <chair>  greedy per-frame gating walks the oracle away
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np

from paper_style import C_BASE, C_DEGR, C_ORACLE, C_HL, apply_style

apply_style()

HERE = Path(__file__).resolve().parent
RUNS = Path("/root/autodl-tmp/DRPhysNav/runs/traj_vignette")
RGB = RUNS / "topdown_rgb"

ARMS = [
    ("B0",         C_DEGR),
    ("RES",        C_BASE),
    ("OracleGate", C_ORACLE),
]
ARM_FILES = {
    "B0":         RUNS / "tv_B0_low_light_s4_seed0_traj.jsonl",
    "RES":        RUNS / "tv_RES_low_light_s4_seed0_traj.jsonl",
    "OracleGate": RUNS / "tv_ORACLEGATE_low_light_s4_seed0_traj.jsonl",
}

CASES = [
    (6, "(a) reach $\\neq$ success", "$\\langle$sofa$\\rangle$: B0 stops 0.10 m\nout, never commits"),
    (2, "(b) restoration rescues",   "$\\langle$chair$\\rangle$: RES/OG converge;\nB0 stalls 1.8 m away"),
    (7, "(c) oracle-only rescue",    "$\\langle$toilet$\\rangle$: only OracleGate\nfinds the room"),
    (8, "(d) gating hurts",          "$\\langle$chair$\\rangle$: greedy gate\nwalks away (2.8 m)"),
]

SUCCESS_RADIUS_M = 1.0
VOID_GRAY = 0.92          # paint out-of-mesh black background this gray

META = json.load(open(RGB / "meta.json"))

STROKE = [pe.Stroke(linewidth=2.1, foreground="white", alpha=0.85),
          pe.Normal()]


def load_ep(arm, ep):
    for line in open(ARM_FILES[arm]):
        d = json.loads(line)
        if d["ep"] == ep:
            return d
    raise KeyError(f"ep {ep} not in {arm}")


def world_to_px(m, xz):
    """World (x, z) -> image (col, row) for the calibrated ortho render."""
    xz = np.atleast_2d(np.asarray(xz, float))
    col = (xz[:, 0] - m["center_x"]) * m["px_per_m"] + m["res"] / 2
    row = (xz[:, 1] - m["center_z"]) * m["px_per_m"] + m["res"] / 2
    return col, row


def draw_panel(ax, ep, title, note):
    m = META[str(ep)]
    recs = {arm: load_ep(arm, ep) for arm, _ in ARMS}
    any_rec = recs["B0"]

    img = np.load(RGB / f"ep{ep:03d}_rgb.npy").astype(float) / 255.0
    void = img.sum(axis=2) < 0.04          # pure-black out-of-mesh pixels
    img[void] = VOID_GRAY
    ax.imshow(img, interpolation="bilinear", zorder=0)

    ppm = m["px_per_m"]

    # ---- goal instances actually converged to (success is at ANY instance)
    goals_w = np.asarray(any_rec["goals_world"], float)[:, [0, 2]]
    ends_w = np.array([np.asarray(recs[a]["traj_world"], float)[-1][[0, 2]]
                       for a, _ in ARMS])
    active_idx = {int(np.argmin(np.min(
        np.linalg.norm(goals_w[None, :, :] - ends_w[:, None, :], axis=2),
        axis=0)))}
    for i, (arm, _) in enumerate(ARMS):
        if recs[arm]["success"]:
            active_idx.add(int(np.argmin(
                np.linalg.norm(goals_w - ends_w[i], axis=1))))
    active = goals_w[sorted(active_idx)]

    # ---- trajectories (white-stroked for contrast on the photo background)
    all_cols, all_rows = [], []
    gc, gr = world_to_px(m, active)
    all_cols.append(gc); all_rows.append(gr)
    for arm, color in ARMS:
        t = np.asarray(recs[arm]["traj_world"], float)[:, [0, 2]]
        c, r = world_to_px(m, t)
        all_cols.append(c); all_rows.append(r)
        ax.plot(c, r, color=color, lw=1.15, alpha=0.95, zorder=3,
                solid_capstyle="round", path_effects=STROKE)
        succ = bool(recs[arm]["success"])
        ax.plot(c[-1], r[-1], "o" if succ else "X",
                color=color, ms=4.6 if succ else 5.2,
                mec="white", mew=0.6, zorder=5)

    # shared start
    s = np.asarray(any_rec["traj_world"], float)[0][[0, 2]]
    sc, sr = world_to_px(m, s)
    ax.plot(sc, sr, "o", ms=5, mfc="white", mec="#222222", mew=1.0, zorder=6)

    # ---- goal stars + 1 m success discs
    for (c,), (r,) in [world_to_px(m, g) for g in active]:
        ax.add_patch(mpatches.Circle((c, r), SUCCESS_RADIUS_M * ppm,
                                     fc=C_HL, ec="none", alpha=0.28, zorder=2))
        ax.add_patch(mpatches.Circle((c, r), SUCCESS_RADIUS_M * ppm,
                                     fc="none", ec=C_HL, lw=1.0,
                                     ls=(0, (3, 2)), zorder=2))
        ax.plot(c, r, "*", ms=9, color=C_HL, mec="#7a5a1c", mew=0.5, zorder=6)

    # ---- square crop to content + margin
    cols = np.concatenate(all_cols); rows = np.concatenate(all_rows)
    c0, c1 = cols.min(), cols.max()
    r0, r1 = rows.min(), rows.max()
    pad = 0.10 * max(c1 - c0, r1 - r0) + 1.2 * ppm
    cc, cr = (c0 + c1) / 2, (r0 + r1) / 2
    half = max(c1 - c0, r1 - r0) / 2 + pad
    ax.set_xlim(cc - half, cc + half)
    ax.set_ylim(cr + half, cr - half)

    # ---- 2 m scale bar: bottom-left, or bottom-right if that corner is busy
    pts_c, pts_r = cols, rows
    bar_px = 2.0 * ppm
    by = cr + half * 0.84
    busy_l = np.any((pts_r >= cr + half * 0.45) &
                    (pts_c <= cc - half * 0.25))
    bx = (cc + half * 0.82 - bar_px) if busy_l else (cc - half * 0.82)
    ax.plot([bx, bx + bar_px], [by, by], color="#222222", lw=1.6,
            solid_capstyle="butt", zorder=7,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="white",
                                    alpha=0.9), pe.Normal()])
    ax.text(bx + bar_px / 2, by - half * 0.045, "2 m", ha="center",
            va="bottom", fontsize=6.4, color="#222222", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_edgecolor("#999999")
    ax.set_title(title, fontsize=8.6, pad=22)
    ax.text(0.5, 1.035, note, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.4, style="italic", color="#555555")


def main():
    fig, axs = plt.subplots(1, 4, figsize=(6.75, 2.28))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.745, bottom=0.105,
                        wspace=0.06)

    for ax, (ep, title, note) in zip(axs, CASES):
        draw_panel(ax, ep, title, note)

    handles = [Line2D([], [], color=c, lw=1.6, label=n) for n, c in ARMS]
    handles += [
        Line2D([], [], marker="o", ls="none", mfc="white", mec="#222222",
               mew=1.0, ms=5, label="start"),
        Line2D([], [], marker="*", ls="none", color=C_HL, mec="#7a5a1c",
               mew=0.5, ms=9, label="goal + 1 m radius"),
        Line2D([], [], marker="o", ls="none", color="#555555", mec="white",
               mew=0.5, ms=4.6, label="end (success)"),
        Line2D([], [], marker="X", ls="none", color="#555555", mec="white",
               mew=0.5, ms=5.2, label="end (failure)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=6.6, bbox_to_anchor=(0.5, -0.015),
               handletextpad=0.45, columnspacing=1.1)

    out = HERE / "fig_trajectories.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    main()
