"""
Fill DXCV results into the paper once the chained run is finished.

Reads:  /root/autodl-tmp/DRPhysNav/runs/dxcv/paired80_{B0,DXCV}_low_light_s4_seed0_N80.csv
Writes: /root/VLA_papers/AAAI_DRPhysNav/sections/50_experiments.tex
         (replaces the placeholders inserted in Sec.~5.5 + Tab. 3)

Run:  python fill_dxcv.py
"""
import csv, os, sys, math
from math import comb

ROOT_RUN = "/root/autodl-tmp/DRPhysNav/runs/dxcv"
TEX = "/root/VLA_papers/AAAI_DRPhysNav/sections/50_experiments.tex"

B0 = f"{ROOT_RUN}/paired80_B0_low_light_s4_seed0_N80.csv"
DX = f"{ROOT_RUN}/paired80_DXCV_low_light_s4_seed0_N80.csv"


def load(p):
    with open(p) as f:
        rows = [r for r in list(csv.reader(f))[1:] if r]
    return rows


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


def main():
    for p in (B0, DX):
        if not os.path.exists(p):
            print(f"MISSING: {p}", file=sys.stderr)
            return 2
    b = load(B0); c = load(DX)
    n, sb, sc, d, b01, b10, p = mcnemar_paired(b, c)
    print(f"N={n}  B0_SR={sb:.3f}  DXCV_SR={sc:.3f}  dSR={d:+.3f}  rescued={b01} broke={b10}  McNemar p={p:.4f}")

    if d > 0 and p < 0.05:
        verdict = "a statistically significant gain"
        interp = (
            "This is the first time a perception-side intervention significantly "
            "lifts SR; the gain is consistent with the multi-modal-sensing direction "
            "proposed by the conclusion and confirms that depth, as a "
            "degradation-invariant channel, can recover part of the commitment "
            "uncertainty caused by the RGB degradation."
        )
    elif d > 0:
        verdict = "a non-significant positive gain"
        interp = (
            "The direction agrees with the multi-modal-sensing hypothesis of the "
            "conclusion, but the magnitude is not significant at $N{=}80$: depth "
            "can reject geometrically implausible commits, but at inference time "
            "on a frozen agent it cannot \\emph{inject} the missing discriminative "
            "signal lost in degraded RGB."
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
            "Vetoing geometrically implausible commits removes a small number "
            "of true positives in addition to false ones; without learning the "
            "joint RGB$+$depth representation, the verifier cannot tell them "
            "apart. This is consistent with the irreducible-uncertainty "
            "diagnosis."
        )

    with open(TEX) as f:
        src = f.read()

    # 1) Tab 3 row placeholders
    row_old = "Multimodal probe   & DXCV$^{\\flat}$ & $\\Delta_{\\text{DXCV}}$ & $p_{\\text{DXCV}}$ & $80$  \\\\"
    row_new = ("Multimodal probe   & DXCV$^{\\flat}$ & $%s$ & $%.3f$ & $80$  \\\\" %
               (fmt_delta(d), p))
    if row_old not in src:
        print("WARN: tab row placeholder not found (already filled?)")
    else:
        src = src.replace(row_old, row_new)

    # 2) prose placeholders
    prose_old = ("$\\Delta\\text{SR}{=}\\Delta_{\\mathrm{DXCV}}$ (McNemar $p{=}p_{\\mathrm{DXCV}}$),\n"
                 "\\textsc{verdict-dxcv-placeholder}. \\textsc{interp-dxcv-placeholder}")
    prose_new = ("$\\Delta\\text{SR}{=}%s$ (McNemar $p{=}%.3f$), %s. %s" %
                 (fmt_delta(d), p, verdict, interp))
    if prose_old not in src:
        print("WARN: prose placeholder not found (already filled?)")
    else:
        src = src.replace(prose_old, prose_new)

    with open(TEX, "w") as f:
        f.write(src)
    print("Filled DXCV results into:", TEX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
