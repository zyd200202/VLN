"""
Fill the unified N=150 sweep results into the paper.

Reads (low-light sev4 seed0, val):
    /root/autodl-tmp/DRPhysNav/runs/maintable/mt_B0_low_light_s4_seed0.csv      (paired baseline)
    /root/autodl-tmp/DRPhysNav/runs/unified/u_{REVOKE,CRV,MUAP,FUSE,DXCV}_low_light_s4_seed0_N150.csv

Updates (in-place):
    /root/VLA_papers/AAAI_DRPhysNav/sections/50_experiments.tex
        - Tab. 3 placeholders for REVOKE / CRV / MUAP / FUSE / DXCV
        - Sec. 5.5 (DXCV) prose placeholders (verdict + interpretation)

Run AFTER /root/autodl-tmp/InstructNav/run_unified_n150.sh finishes:
    python fill_unified.py
"""
from __future__ import annotations

import csv
import os
import sys
from math import comb


RUNS = "/root/autodl-tmp/DRPhysNav/runs/unified"
B0 = "/root/autodl-tmp/DRPhysNav/runs/maintable/mt_B0_low_light_s4_seed0.csv"
TEX = "/root/VLA_papers/AAAI_DRPhysNav/sections/50_experiments.tex"

ARMS = ["REVOKE", "CRV", "MUAP", "FUSE", "DXCV"]


def arm_csv(arm: str) -> str:
    return f"{RUNS}/u_{arm}_low_light_s4_seed0_N150.csv"


def load_csv(p: str):
    with open(p) as f:
        return [r for r in list(csv.reader(f))[1:] if r]


def mcnemar_paired(b, c):
    n = min(len(b), len(c))
    b = b[:n]
    c = c[:n]
    sb = sum(float(r[0]) for r in b) / n
    sc = sum(float(r[0]) for r in c) / n
    b01 = sum(1 for i in range(n) if float(b[i][0]) == 0 and float(c[i][0]) == 1)
    b10 = sum(1 for i in range(n) if float(b[i][0]) == 1 and float(c[i][0]) == 0)
    ntot = b01 + b10
    k = min(b01, b10)
    p = sum(comb(ntot, i) for i in range(0, k + 1)) / (2 ** ntot) * 2 if ntot > 0 else 1.0
    p = min(1.0, p)
    return n, sb, sc, sc - sb, b01, b10, p


def fmt_delta(d):
    return f"{d:+.3f}"


def dxcv_prose(d: float, p: float) -> tuple[str, str]:
    if d > 0 and p < 0.05:
        verdict = "a statistically significant gain"
        interp = (
            "This is the first time a perception-side intervention significantly "
            "lifts SR; the gain is consistent with the multi-modal-sensing "
            "direction proposed by the conclusion and confirms that depth, as a "
            "degradation-invariant channel, can recover part of the commitment "
            "uncertainty caused by the RGB degradation."
        )
    elif d > 0:
        verdict = "a non-significant positive gain"
        interp = (
            "The direction agrees with the multi-modal-sensing hypothesis of the "
            "conclusion, but the magnitude is not significant: depth can reject "
            "geometrically implausible commits, but at inference time on a frozen "
            "agent it cannot \\emph{inject} the missing discriminative signal "
            "lost in degraded RGB."
        )
    elif abs(d) < 1e-9:
        verdict = "no detectable change"
        interp = (
            "Even a degradation-invariant additional channel, used at inference "
            "time on a frozen agent, lands at the same $\\sim 0$ as the other "
            "five families: depth can geometrically veto commits but, without "
            "retraining, cannot \\emph{inject} the discriminative signal removed "
            "by the RGB degradation. This is exactly the prediction of the "
            "irreducible-uncertainty diagnosis."
        )
    else:
        verdict = "a non-significant negative effect"
        interp = (
            "Vetoing geometrically implausible commits removes a small number of "
            "true positives in addition to false ones; without learning the joint "
            "RGB$+$depth representation, the verifier cannot tell them apart. "
            "This is consistent with the irreducible-uncertainty diagnosis."
        )
    return verdict, interp


def main() -> int:
    if not os.path.exists(B0):
        print(f"MISSING baseline: {B0}", file=sys.stderr)
        return 2
    b = load_csv(B0)

    results: dict[str, tuple[float, float]] = {}
    for arm in ARMS:
        p = arm_csv(arm)
        if not os.path.exists(p):
            print(f"WARN: arm {arm} csv missing: {p} -- skipping", file=sys.stderr)
            continue
        c = load_csv(p)
        n, sb, sc, d, b01, b10, pval = mcnemar_paired(b, c)
        print(f"{arm:7s} n={n} B0_SR={sb:.3f} ARM_SR={sc:.3f} dSR={d:+.3f} rescued={b01} broke={b10} p={pval:.3f}")
        results[arm] = (d, pval)

    with open(TEX) as f:
        src = f.read()

    for arm, (d, pval) in results.items():
        # Tab 3 placeholder pattern (matches what was written in the .tex file)
        if arm == "DXCV":
            row_old = f"Multimodal probe   & DXCV$^{{\\flat}}$ & $\\Delta_{{\\text{{{arm}}}}}$ & $p_{{\\text{{{arm}}}}}$ & $150$  \\\\"
            row_new = f"Multimodal probe   & DXCV$^{{\\flat}}$ & ${fmt_delta(d)}$ & ${pval:.3f}$ & $150$  \\\\"
        else:
            family = {"REVOKE": "Commit timing", "CRV": "Forced commit",
                      "MUAP": "Temporal     ", "FUSE": "Multi-view   "}[arm]
            # match the placeholder exactly as it appears in the .tex
            placeholders = {
                "REVOKE": "Commit timing      & REVOKE     & $\\Delta_{\\text{REVOKE}}$ & $p_{\\text{REVOKE}}$ & $150$ \\\\",
                "CRV":    "Forced commit      & CRV        & $\\Delta_{\\text{CRV}}$    & $p_{\\text{CRV}}$    & $150$ \\\\",
                "MUAP":   "Temporal           & MUAP       & $\\Delta_{\\text{MUAP}}$   & $p_{\\text{MUAP}}$   & $150$ \\\\",
                "FUSE":   "Multi-view         & FUSE       & $\\Delta_{\\text{FUSE}}$   & $p_{\\text{FUSE}}$   & $150$ \\\\",
            }
            replacements = {
                "REVOKE": f"Commit timing      & REVOKE     & ${fmt_delta(d)}$ & ${pval:.3f}$ & $150$ \\\\",
                "CRV":    f"Forced commit      & CRV        & ${fmt_delta(d)}$    & ${pval:.3f}$    & $150$ \\\\",
                "MUAP":   f"Temporal           & MUAP       & ${fmt_delta(d)}$   & ${pval:.3f}$   & $150$ \\\\",
                "FUSE":   f"Multi-view         & FUSE       & ${fmt_delta(d)}$   & ${pval:.3f}$   & $150$ \\\\",
            }
            row_old = placeholders[arm]
            row_new = replacements[arm]
        if row_old not in src:
            print(f"WARN: tab row placeholder for {arm} not found (already filled?)")
        else:
            src = src.replace(row_old, row_new)

    # DXCV prose
    if "DXCV" in results:
        d, pval = results["DXCV"]
        verdict, interp = dxcv_prose(d, pval)
        prose_old = ("$\\Delta\\text{SR}{=}\\Delta_{\\mathrm{DXCV}}$ (McNemar $p{=}p_{\\mathrm{DXCV}}$),\n"
                     "\\textsc{verdict-dxcv-placeholder}. \\textsc{interp-dxcv-placeholder}")
        prose_new = ("$\\Delta\\text{SR}{=}%s$ (McNemar $p{=}%.3f$), %s. %s" %
                     (fmt_delta(d), pval, verdict, interp))
        if prose_old in src:
            src = src.replace(prose_old, prose_new)
        else:
            print("WARN: DXCV prose placeholder not found (already filled?)")

    with open(TEX, "w") as f:
        f.write(src)
    print(f"\nFilled {len(results)} arms into: {TEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
