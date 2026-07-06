"""Fill the Road-1 [TBD-Sat] placeholders in A0_appendix.tex once
runs/road1_eval/road1_R1_low_light_s4_seed0.csv lands.

Also fills the matching main-paper Table-1 row if the marker
``%ROAD1_ROW`` is found in sections/50_experiments.tex.

Idempotent: prints "no Road-1 CSV yet" and exits 0 if data isn't ready.
"""
from __future__ import annotations
import csv, os, re, sys, json
from math import comb

ROOT = "/root/autodl-tmp/DRPhysNav"
APP  = "/root/VLA_papers/AAAI_DRPhysNav/sections/A0_appendix.tex"
EXP  = "/root/VLA_papers/AAAI_DRPhysNav/sections/50_experiments.tex"

B0  = f"{ROOT}/runs/maintable/mt_B0_low_light_s4_seed0.csv"
R1  = f"{ROOT}/runs/road1_eval/road1_R1_low_light_s4_seed0.csv"
LOG = f"{ROOT}/road1/ckpt/train.log"


def load_succ(p):
    if not os.path.isfile(p):
        return None
    rows = list(csv.reader(open(p)))[1:]
    return [float(r[0]) for r in rows if r]


def mcnemar(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    b01 = sum(1 for i in range(n) if x[i] < .5 and y[i] > .5)
    b10 = sum(1 for i in range(n) if x[i] > .5 and y[i] < .5)
    nd, k = b01 + b10, min(b01, b10)
    p = min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / (2**nd)) if nd > 0 else 1.0
    return b01, b10, p


def parse_l1(path):
    if not os.path.isfile(path):
        return None, None, None
    pre = post = None
    with open(path) as f:
        for line in f:
            m = re.search(r"hold-out L1:\s*pre=([\d.eE+\-]+)\s+post=([\d.eE+\-]+)", line)
            if m:
                pre, post = float(m.group(1)), float(m.group(2))
    if pre is None:
        return None, None, None
    red = 1 - post / max(pre, 1e-9)
    return pre, post, red


def main():
    sb = load_succ(B0)
    sr = load_succ(R1)
    if sr is None or len(sr) < 150:
        print(f"no Road-1 CSV yet (or <150 eps); {R1}")
        sys.exit(0)
    n = min(len(sb), len(sr))
    b01, b10, p = mcnemar(sb, sr)  # b01: B0 fail + R1 succeed
    dsr = (sum(sr[:n]) - sum(sb[:n])) / n
    pre, post, red = parse_l1(LOG)

    repl = {
        "$\\ell_1^{\\text{pre}}$~$=$~\\textbf{[TBD-Sat]}":
            f"$\\ell_1^{{\\text{{pre}}}}$~$=$~\\textbf{{{pre:.3f}}}" if pre is not None
            else "$\\ell_1^{{\\text{pre}}}$~$=$~\\textbf{n/a}",
        "$\\ell_1^{\\text{post}}$~$=$~\\textbf{[TBD-Sat]} (reduction\n\\textbf{[TBD-Sat]}~\\%)":
            (f"$\\ell_1^{{\\text{{post}}}}$~$=$~\\textbf{{{post:.3f}}} (reduction "
             f"\\textbf{{{red*100:.1f}}}~\\%)") if post is not None
            else "$\\ell_1^{{\\text{post}}}$~$=$~\\textbf{n/a} (reduction \\textbf{n/a})",
        "$\\Delta\\text{SR}_{\\text{R1}}{=}\\textbf{[TBD-Sat]}":
            f"$\\Delta\\text{{SR}}_{{\\text{{R1}}}}{{=}}\\textbf{{{dsr:+.3f}}}",
        "$p{=}\\textbf{[TBD-Sat]}":     f"$p{{=}}\\textbf{{{p:.3f}}}",
        "$b_{01}{=}\\textbf{[TBD-Sat]}": f"$b_{{01}}{{=}}\\textbf{{{b01}}}",
        "$b_{10}{=}\\textbf{[TBD-Sat]}": f"$b_{{10}}{{=}}\\textbf{{{b10}}}",
    }

    with open(APP) as f:
        s = f.read()
    for k, v in repl.items():
        s = s.replace(k, v)
    with open(APP, "w") as f:
        f.write(s)

    # Try also to insert a Tab.1 row if marker exists
    with open(EXP) as f:
        exp = f.read()
    row = (f"        Train side       & Road~1 (adapter) & ${dsr:+.3f}$ "
           f"& $\\mathbf{{{p:.3f}}}$ & ${n}$ \\\\\n")
    if "%ROAD1_ROW" in exp and "Road~1 (adapter)" not in exp:
        exp = exp.replace("%ROAD1_ROW", row + "        %ROAD1_ROW")
        with open(EXP, "w") as f:
            f.write(exp)
        print(f"inserted Road-1 Tab.1 row: dSR={dsr:+.3f}, p={p:.3f}, N={n}")
    print(f"appendix updated: dSR={dsr:+.3f}, p={p:.3f}, b01={b01}, b10={b10},",
          f"L1 reduction={red*100:.1f}%" if red is not None else "no L1 log")


if __name__ == "__main__":
    main()
