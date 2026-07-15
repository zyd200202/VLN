"""fig_case_studies.py -- episode-level decision traces (UniGoal Fig-4 style).

Three representative paired episodes from the N=300 low-light sev4 seed0 cell,
one per outcome bucket:

  RESCUED  ep 43  <toilet>  B0/RES fail, OracleGate converges to 0.17 m
  BROKEN   ep 117 <chair>   RES succeeds fast; OracleGate walks away to 7.6 m
  TIED     ep 17  <toilet>  B0 parks 0.02 m from goal yet still fails (commit)

Each column: top = distance-to-goal trace for the three arms;
             bottom = per-step blind reliability r for the same arms.
All curves are REAL per-step logs from the paired runs (motiv JSONL).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_style import (
    C_BASE, C_DEGR, C_POS, C_ORACLE, C_NEUT, C_HL,
    apply_style,
)

apply_style()

HERE = Path(__file__).resolve().parent
RUNS = Path("/root/autodl-tmp/DRPhysNav/runs")

ARMS = [
    ("B0",         RUNS / "redesign_n300/n300_B0_low_light_s4_seed0_motiv.jsonl",  C_DEGR),
    ("RES",        RUNS / "redesign_n300/n300_RES_low_light_s4_seed0_motiv.jsonl", C_BASE),
    ("OracleGate", RUNS / "oracle/or_ORACLEGATE_low_light_s4_seed0_motiv.jsonl",   C_ORACLE),
]

CASES = [
    (43,  "rescued", "toilet",
     "only the per-frame appearance\noracle converges (0.17 m)"),
    (117, "broken",  "chair",
     "greedy per-frame view choice\nwalks the oracle away (7.6 m)"),
    (17,  "tied",    "toilet",
     "B0 parks 0.02 m from goal yet\nnever commits: reach $\\neq$ success"),
]

SUCCESS_THRESH = 1.0   # ObjectNav success radius (m)


def load_ep(path, ep):
    for line in open(path):
        d = json.loads(line)
        if d["ep"] == ep:
            return d
    raise KeyError(f"ep {ep} not in {path}")


def main():
    fig, axs = plt.subplots(2, 3, figsize=(6.75, 3.55),
                            gridspec_kw=dict(height_ratios=[1.55, 1.0],
                                             hspace=0.24, wspace=0.28))

    for col, (ep, bucket, goal, note) in enumerate(CASES):
        ax_d, ax_r = axs[0, col], axs[1, col]

        max_step = 0
        for name, path, color in ARMS:
            d = load_ep(path, ep)
            steps = [s["step"] for s in d["steps"]]
            dist  = [s["dist"] for s in d["steps"]]
            rel   = [s["r"]    for s in d["steps"]]
            max_step = max(max_step, steps[-1])
            succ = bool(d["success"])

            ax_d.plot(steps, dist, color=color, lw=1.2, zorder=3,
                      alpha=0.95, solid_capstyle="round")
            # end-marker: filled circle = success, x = failure
            ax_d.plot(steps[-1], dist[-1],
                      "o" if succ else "X",
                      color=color, ms=5.5 if succ else 6,
                      mec="white", mew=0.7, zorder=5)
            # raw r is per-frame noisy: show faint raw + smoothed overlay
            ax_r.plot(steps, rel, color=color, lw=0.5, alpha=0.22, zorder=2)
            if len(rel) >= 9:
                k = 9
                kern = np.ones(k) / k
                sm = np.convolve(np.asarray(rel), kern, mode="same")
                # fix boundary bias of 'same' convolution
                norm = np.convolve(np.ones(len(rel)), kern, mode="same")
                sm = sm / norm
                ax_r.plot(steps, sm, color=color, lw=1.1, alpha=0.95, zorder=3)
            else:
                ax_r.plot(steps, rel, color=color, lw=1.1, alpha=0.95, zorder=3)

        # success band on distance panel; its label sits BELOW the axes
        # (in the gap above the reliability panel) so it never covers curves
        ax_d.axhspan(0, SUCCESS_THRESH, color="#E6F3E6", zorder=0)
        ax_d.text(0.0, -0.045, "success radius (1 m)",
                  transform=ax_d.transAxes, ha="left", va="top",
                  fontsize=5.8, color="#5A8A5A")

        ax_d.set_xlim(0, max_step * 1.02)
        ax_d.set_ylim(0, None)
        ax_d.set_xticklabels([])
        ax_d.set_title(f"({chr(97+col)}) {bucket} — ep {ep}, goal: {goal}",
                       loc="left", fontsize=8.0, fontweight="bold", pad=26)
        ax_d.grid(axis="y", zorder=1)
        if col == 0:
            ax_d.set_ylabel("distance to goal (m)", fontsize=7.8)

        # annotation (the diagnosis, one per case) ABOVE the axes, under the
        # panel title, so it never covers the traces
        ax_d.text(0.0, 1.06, note, transform=ax_d.transAxes,
                  ha="left", va="bottom", fontsize=6.4, color="#444444",
                  style="italic", linespacing=1.25)

        # reliability panel
        ax_r.set_xlim(0, max_step * 1.02)
        ax_r.set_ylim(0, 1.05)
        ax_r.set_yticks([0, 0.5, 1.0])
        ax_r.set_xlabel("step", fontsize=7.8)
        ax_r.grid(axis="y", zorder=1)
        if col == 0:
            ax_r.set_ylabel("reliability $r$", fontsize=7.8)

    # one legend for the whole figure, top-center
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=n)
               for n, _, c in ARMS]
    handles += [
        plt.Line2D([], [], color="#666666", marker="o", ls="none",
                   ms=5, mec="white", label="ends: success"),
        plt.Line2D([], [], color="#666666", marker="X", ls="none",
                   ms=6, mec="white", label="ends: failure"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.5, 1.015),
               handlelength=1.6, handletextpad=0.4, columnspacing=1.3)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.795, bottom=0.105)
    fig.savefig(HERE / "fig_case_studies.pdf")
    fig.savefig(HERE / "fig_case_studies.png", dpi=180)
    print("saved fig_case_studies.pdf")


if __name__ == "__main__":
    main()
