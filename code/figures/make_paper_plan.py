"""Build PAPER_PLAN.pdf for AAAI submission (deadline 2026-07-28).

The output is a single multi-page PDF combining:
  1. Cover page (title, deadline, status, abstract)
  2. Paper writing skeleton (existing fig_paper_mindmap.pdf)
  3. Reasoning chain - Introduction + Motivation
  4. Reasoning chain - Method + Experiments
  5. Reasoning chain - Conclusion + Roadmap
  6. Project status mind-map (existing fig_mindmap.pdf)
  7. 30-day milestone Gantt

Output: /root/VLA_papers/AAAI_DRPhysNav/PAPER_PLAN.pdf
"""
from __future__ import annotations
import os
from datetime import date, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import fitz                                    # PyMuPDF
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -------- palette ---------------------------------------------------------
C_TITLE  = "#FFF59D"
C_PROB   = "#FFCDD2"   # question / pain point (red)
C_EVID   = "#BBDEFB"   # evidence / observation (blue)
C_CONCL  = "#C8E6C9"   # conclusion / claim (green)
C_NOTE   = "#FFE0B2"   # side note / figure / table (orange)
C_NEXT   = "#E1BEE7"   # next step / future work (purple)

E_TITLE = "#F57F17"
E_PROB  = "#B71C1C"
E_EVID  = "#0D47A1"
E_CONCL = "#1B5E20"
E_NOTE  = "#E65100"
E_NEXT  = "#4A148C"

# -------- helpers ---------------------------------------------------------
def box(ax, x, y, w, h, txt, fc, ec, fontsize=9.5, fontweight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.18,rounding_size=0.18",
                                fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, zorder=3,
            linespacing=1.25, wrap=True)


def arrow(ax, p_from, p_to, color="#444", lw=1.1, style="-|>",
          rad=0.0, label=None, label_pos=None, label_color="#444"):
    ax.add_patch(FancyArrowPatch(p_from, p_to, arrowstyle=style,
                                 lw=lw, color=color,
                                 mutation_scale=12,
                                 connectionstyle=f"arc3,rad={rad}",
                                 zorder=1.5))
    if label:
        if label_pos is None:
            mx, my = (p_from[0]+p_to[0])/2, (p_from[1]+p_to[1])/2
        else:
            mx, my = label_pos
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8.5, fontstyle="italic", color=label_color,
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec=label_color, lw=0.6), zorder=3)


def page_setup(ax, title, subtitle=None, xlim=(0,100), ylim=(0,70)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    mid_x = (xlim[0] + xlim[1]) / 2
    ax.text(mid_x, ylim[1]-2.4, title,
            ha="center", va="top", fontsize=20, fontweight="bold",
            color="#222")
    if subtitle:
        ax.text(mid_x, ylim[1]-5.5, subtitle,
                ha="center", va="top", fontsize=11, color="#555",
                style="italic")


# =============== PAGE 1 : COVER ==========================================
def page_cover(ax):
    ax.set_xlim(0,100); ax.set_ylim(0,70); ax.axis("off")

    # masthead band
    ax.add_patch(FancyBboxPatch((4, 56), 92, 11,
                                boxstyle="round,pad=0.5,rounding_size=0.8",
                                fc=C_TITLE, ec=E_TITLE, lw=1.6))
    ax.text(50, 64.2,
            "DRPhysNav  \u2014  AAAI Submission Plan",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="#5C4400")
    ax.text(50, 60.6,
            "Restoring the Image Does Not Restore Navigation: Diagnosing\n"
            "Degradation-Robust Object-Goal Navigation as a Commitment Problem",
            ha="center", va="center", fontsize=11.5, color="#5C4400",
            style="italic", linespacing=1.3)

    # countdown banner
    today = date.today()                       # 2026-06-28 in this env
    deadline = date(2026, 7, 28)
    days_left = (deadline - today).days

    ax.add_patch(FancyBboxPatch((4, 48), 92, 6,
                                boxstyle="round,pad=0.3,rounding_size=0.5",
                                fc="#FFEBEE", ec="#B71C1C", lw=1.4))
    ax.text(50, 51.0,
            f"Target venue: AAAI-2027  \u00b7  Submission deadline: 2026-07-28  "
            f"\u00b7  T-minus  {days_left}  days  (today {today.isoformat()})",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="#B71C1C")

    # 3 columns of status / scope / risk
    col_w, col_h = 28, 28
    col_y = 17

    # --- col 1 : status snapshot ---
    box(ax, 4, col_y, col_w, col_h, "", "#F1F8E9", "#558B2F")
    ax.text(4+col_w/2, col_y+col_h-2.0, "Status snapshot",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color="#33691E")
    snap = [
        "[done]  N=300 main split  (B0, RES, OracleGate, M1g)",
        "[done]  N=150 paired  (REVOKE, CRV, MUAP, FUSE, ROUTER)",
        "[done]  DXCV N=150 done  06-28 22:56  \u0394=\u22120.173, p=0.0002",
        "[done]  Six causal families covered in Tab. 1",
        "[done]  Paired McNemar + bootstrap CI infra",
        "[done]  Mind-map + writing skeleton frozen",
        "[todo]  Run fill_unified.py \u2192 fill Tab 1 \u00d7 5 cells",
        "[todo]  VLFM N=150 paired (~15h, next)",
        "[todo]  Road 1 pilot (GLEE-DA head) planned",
    ]
    yy = col_y + col_h - 4.5
    for s in snap:
        ax.text(4+1.6, yy, s, ha="left", va="top", fontsize=9.0,
                color="#1B5E20", linespacing=1.25)
        yy -= 2.55

    # --- col 2 : page-by-page contents ---
    box(ax, 36, col_y, col_w, col_h, "", "#E3F2FD", "#1565C0")
    ax.text(36+col_w/2, col_y+col_h-2.0, "Contents of this PDF",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color="#0D47A1")
    contents = [
        "p.1   This cover sheet",
        "p.2   Paper writing skeleton",
        "        (10 blocks: Title \u2192 Appendix)",
        "p.3   Reasoning chain  \u2014  Sec 1 + Sec 3",
        "        problem \u2192 evidence \u2192 finding",
        "p.4   Reasoning chain  \u2014  Sec 4 + Sec 5",
        "        prediction \u2192 protocol \u2192 sweep",
        "p.5   Reasoning chain  \u2014  Sec 6 + Roads",
        "        irreducible \u2192 two forward roads",
        "p.6   Project status mind-map",
        "        9 branches around DRPhysNav root",
        "p.7   30-day milestones (T\u221230 \u2192 T)",
        "        weeks 1\u20134, Gantt + checkpoints",
    ]
    yy = col_y + col_h - 4.5
    for s in contents:
        ax.text(36+1.6, yy, s, ha="left", va="top", fontsize=9.0,
                color="#0D47A1", linespacing=1.25,
                fontweight=("bold" if s.startswith("p.") else "normal"))
        yy -= 1.85

    # --- col 3 : top risks + decisions ---
    box(ax, 68, col_y, col_w, col_h, "", "#FFF3E0", "#E65100")
    ax.text(68+col_w/2, col_y+col_h-2.0, "Top risks  &  decisions",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color="#BF360C")
    risks = [
        "R1  DXCV doesn't finish in time",
        "       \u2192 fallback: ship 4/5 arms + grey row",
        "R2  Road 1 head fine-tune won't converge",
        "       \u2192 LoRA rank-32 fallback (Plan B)",
        "       \u2192 feature-restoration MLP (Plan C)",
        "R3  Single-backbone reviewer concern",
        "       \u2192 VLFM N=150 paired (~15h);",
        "          50 ep would self-contradict Sec. 5.7",
        "R4  OracleGate sig p=0.04 lucky?",
        "       \u2192 qualitative on 20 episodes",
        "R5  Page-limit overshoot",
        "       \u2192 already 7p + appendix split",
        "",
        "Decision: ship at p.4 of T\u22125,",
        "freeze writing at T\u22122.",
    ]
    yy = col_y + col_h - 4.5
    for s in risks:
        ax.text(68+1.6, yy, s, ha="left", va="top", fontsize=9.0,
                color="#BF360C", linespacing=1.25,
                fontweight=("bold" if s.startswith("Decision") else "normal"))
        yy -= 1.65

    # footer abstract
    ax.add_patch(FancyBboxPatch((4, 4), 92, 11,
                                boxstyle="round,pad=0.4,rounding_size=0.5",
                                fc="#FAFAFA", ec="#9E9E9E", lw=0.8))
    ax.text(50, 13.0, "One-paragraph abstract",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#424242")
    ax.text(50, 8.4,
            "Zero-shot VLM ObjectNav loses much of its SR under low light, fog or motion blur. The default remedy \u2014 a "
            "restoration\nfront-end \u2014 targets the wrong layer: degradation does not change whether the agent acts "
            "(commit rate ~93%) but it\ncorrupts the quality of an irreversible commitment. At paired N=300, the agent reaches "
            "within 1 m of the goal in 62% of episodes\nyet succeeds in only 36% (a 26-pt reach-to-success gap); it commits "
            "at frame reliability 0.39 vs 0.95 on clean. We sweep six\ncausal families; every deployable arm caps at ~0, only "
            "the non-deployable appearance upper bound is significant (+0.057).",
            ha="center", va="center", fontsize=9.2, color="#333",
            linespacing=1.4)


# =============== PAGE 3 : Sec 1 + Sec 3 reasoning chain ==================
def page_chain_intro_motivation(ax):
    page_setup(ax,
        "Reasoning Chain  \u2014  Introduction (Sec. 1) + Motivation (Sec. 3)",
        "How each section walks the reader from the problem to the diagnosis."
        "  Red = pain point, Blue = evidence, Green = claim, Orange = artefact.")

    # ===== top half: SECTION 1 chain (left-to-right) =====
    ax.add_patch(FancyBboxPatch((2.5, 33), 95, 25.5,
                                boxstyle="round,pad=0.4,rounding_size=0.4",
                                fc="#FAFAFA", ec="#9E9E9E", lw=0.8, zorder=0))
    ax.text(50, 57.0,
            "Section 1  \u2014  Introduction",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#0D47A1")

    n = [
        # (x, y, w, h, txt, fill, edge)
        (4.0, 47.5, 16, 6.5,
         "P1  Problem\nVLM-driven nav under\ndegraded camera\nloses SR",
         C_PROB, E_PROB),
        (22.5, 47.5, 16, 6.5,
         "P2  Default remedy\nrestoration front-end\n(Zero-DCE, URetinex);\n'clean look = clean act'",
         C_PROB, E_PROB),
        (41.0, 47.5, 17, 6.5,
         "P3  Our instrument\nbenchmark + paired\nN=300 + blind r +\nMcNemar exact",
         C_NOTE, E_NOTE),
        (60.5, 47.5, 17, 6.5,
         "P4  3 observations\n(i) deg. hurts (ii) appearance\nnot lever (iii) commit\nquality collapses",
         C_EVID, E_EVID),
        (79.5, 47.5, 17, 6.5,
         "P5  Negative results\nRES +0.04 p=.18\nOracleGate +0.057 p=.04\nM1g +0.04 p=.20",
         C_EVID, E_EVID),
        # bottom row of S1
        (40, 36.5, 30, 7.5,
         "P6  Contributions (i)\u2013(iv)\n"
         "(i) benchmark + paired protocol  \u2022  (ii) mechanism diagnosis at N=300\n"
         "(iii) six-family negative-result sweep  \u2022  (iv) sign-flip warning",
         C_CONCL, E_CONCL),
    ]
    for x, y, w, h, t, fc, ec in n:
        box(ax, x, y, w, h, t, fc, ec, fontsize=8.5)

    # arrows along the top row
    for x_from, x_to, y in [
        (20.0, 22.5, 50.75),
        (38.5, 41.0, 50.75),
        (58.0, 60.5, 50.75),
        (77.5, 79.5, 50.75),
    ]:
        arrow(ax, (x_from, y), (x_to, y))
    # converging arrows into P6
    arrow(ax, (30,  47.5), (50, 44.0), rad=0.20)
    arrow(ax, (70,  47.5), (60, 44.0), rad=-0.20)

    # ===== bottom half: SECTION 3 motivation chain (vertical waterfall) =====
    ax.add_patch(FancyBboxPatch((2.5, 1.5), 95, 30,
                                boxstyle="round,pad=0.4,rounding_size=0.4",
                                fc="#FAFAFA", ec="#9E9E9E", lw=0.8, zorder=0))
    ax.text(50, 30.0,
            "Section 3  \u2014  Why Navigation Fails Under Degradation  (M0 \u2192 M5)",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#0D47A1")

    chain = [
        ("Q0  Does degradation actually hurt SR?",                                                                                    "Yes.  Low-light: SR 0.43 \u2192 0.22  (\u0394=\u22120.21, paired CI excludes 0)"),
        ("Q1  Is the cause appearance?",                                                                                              "No.  +8.4 dB PSNR  but  SR stays within \u00b10.10 seed band; per-frame oracle no better"),
        ("Q2  Then what does failure look like?",                                                                                     "Walk-aways 78\u219288%, irreversible 73\u219279%, false stops 0.51\u21920.78"),
        ("Q3  Is reliability r informative?",                                                                                          "Yes.  Bucketing commits by r is monotone; mean r 0.92\u21920.41 over severities"),
        ("Q4  When exactly does the commitment fire?",                                                                                "On a SINGLE low-r frame.  r at trigger 0.95\u21920.41,  locked ~21 steps after"),
        ("Q5  Holds at the N=300 main split?",                                                                                        "Yes.  reach 0.62, SR 0.36 \u21d2 26-pt gap.  r at commit 0.39 vs 0.95."),
    ]
    base_x = 6.0
    y0 = 24.5
    dy = 3.8
    for i, (q, a) in enumerate(chain):
        y = y0 - i*dy
        box(ax, base_x,     y, 38, 3.0, q, C_PROB, E_PROB, fontsize=9.0)
        box(ax, base_x+42,  y, 50, 3.0, a, C_EVID, E_EVID, fontsize=9.0)
        if i > 0:
            arrow(ax, (base_x+19, y0 - (i-1)*dy), (base_x+19, y+3.0))
            arrow(ax, (base_x+67, y0 - (i-1)*dy), (base_x+67, y+3.0))

    # final summary box
    box(ax, 18, 2.5, 64, 2.7,
        "Summary  \u2192  perception ok, action ok; loss sits in reach-to-success conversion.\n"
        "Predicts perception fixes won't move SR \u21d2 tested in Sec.\u00a05.",
        C_CONCL, E_CONCL, fontsize=9.0, fontweight="bold")
    arrow(ax, (base_x+19, y0 - 5*dy), (50, 5.2), rad=-0.10)
    arrow(ax, (base_x+67, y0 - 5*dy), (50, 5.2), rad=0.10)


# =============== PAGE 4 : Sec 4 + Sec 5 reasoning chain ==================
def page_chain_method_experiments(ax):
    page_setup(ax,
        "Reasoning Chain  \u2014  Method (Sec. 4) + Experiments (Sec. 5)",
        "The diagnosis makes a falsifiable prediction; Sec. 4 builds the candidates, "
        "Sec. 5 tests them under a paired N=300 protocol.")

    # ===== top half : Section 4 - from prediction to candidates =====
    ax.add_patch(FancyBboxPatch((2.5, 38), 95, 21,
                                boxstyle="round,pad=0.4,rounding_size=0.4",
                                fc="#FAFAFA", ec="#9E9E9E", lw=0.8, zorder=0))
    ax.text(50, 57.5,
            "Section 4  \u2014  From Diagnosis to Candidate Interventions",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#0D47A1")

    box(ax, 4, 50, 26, 5,
        "Diagnosis claim (from Sec. 3)\nloss is in the reach-to-success\nconversion at low-r frames",
        C_CONCL, E_CONCL, fontsize=9.0)
    box(ax, 36, 50, 28, 5,
        "Falsifiable prediction\nappearance/where-to-go interventions\nshould NOT move SR",
        C_PROB, E_PROB, fontsize=9.0)
    box(ax, 70, 50, 26, 5,
        "Build candidates\nspan pixel + perception + decision\nplus a non-deployable UB",
        C_NOTE, E_NOTE, fontsize=9.0)
    arrow(ax, (30, 52.5), (36, 52.5))
    arrow(ax, (64, 52.5), (70, 52.5))

    # candidate row
    cy, ch = 39.5, 8.0
    cw = 17.0
    spans = [
        (3.0,  "4.1 Instrument\nblind r \u2208 [0,1]\nluma+contrast+Laplacian\n+ dark-channel + noise\n(training-free)", C_NOTE, E_NOTE),
        (21.0, "4.2 Pixel (Candidate 1)\nRES blanket restoration\nOracleGate per-frame UB\n(non-deployable ceiling)", C_NOTE, E_NOTE),
        (39.0, "4.3 Decision (Candidate 2)\nM1g: v = (1\u2212\u03b2)v_sem + \u03b2v_frontier\n\u03b2(r) = min(\u03b2_max, \u03b1(1\u2212r))\nstop iff r\u2265\u03c4_stop=0.6", C_NOTE, E_NOTE),
        (57.0, "Commit-timing\nREVOKE\n(stall window K, displacement \u03b5)", C_NOTE, E_NOTE),
        (75.0, "Temporal / Multi-view\nMUAP \u00b7 ROUTER \u00b7 FUSE\n+ multimodal probe DXCV\n(depth-geom sanity check)", C_NOTE, E_NOTE),
    ]
    for x, t, fc, ec in spans:
        box(ax, x, cy, cw, ch, t, fc, ec, fontsize=8.3)
        arrow(ax, (x+cw/2, 50), (x+cw/2, cy+ch), rad=0.0,
              color="#777", lw=0.9)

    # ===== bottom half : Section 5 - protocol + sweep + verdict =====
    ax.add_patch(FancyBboxPatch((2.5, 1.5), 95, 35,
                                boxstyle="round,pad=0.4,rounding_size=0.4",
                                fc="#FAFAFA", ec="#9E9E9E", lw=0.8, zorder=0))
    ax.text(50, 34.5,
            "Section 5  \u2014  Experiments  \u00b7  six causal families  \u00b7  paired against the same baseline",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#0D47A1")

    # 5.1 protocol
    box(ax, 4, 26, 30, 5,
        "5.1 Protocol\nInstructNav (Qwen2-VL + GLEE +\nvalue-map). N=300 main split;\nsame-seed same-episode pairing",
        C_NOTE, E_NOTE, fontsize=8.6)
    # 5.2 effect
    box(ax, 35, 26, 30, 5,
        "5.2 The degradation effect\nLow-light n=90: 0.43\u21920.22, \u0394=\u22120.21\nN=300 baseline SR = 0.37\nmotion-blur reported as non-sig",
        C_EVID, E_EVID, fontsize=8.6)
    # 5.3 two negatives
    box(ax, 66, 26, 30, 5,
        "5.3 Two negatives at N=300\nRES \u0394=+0.04 p=.18\nOracleGate UB +0.057 p=.04*\nM1g \u0394=+0.04 p=.20",
        C_EVID, E_EVID, fontsize=8.6)

    arrow(ax, (34, 28.5), (35, 28.5))
    arrow(ax, (65, 28.5), (66, 28.5))

    # 5.4 mechanism (centre)
    box(ax, 18, 16.5, 64, 6.5,
        "5.4 Mechanism  \u2014  why every candidate caps at ~+0.04\n"
        "reach ~0.62 and commit/stop loss ~0.25 are INVARIANT across B0/RES/M1g.\n"
        "RES lifts r at commit 0.39\u21920.89, but P(success | commit) only 0.40\u21920.42.\n"
        "\u21d2  perception\u2013decision decoupling   \u21d2  +0.04 ceiling explained",
        C_CONCL, E_CONCL, fontsize=9.0, fontweight="bold")
    arrow(ax, (80, 26),    (60, 23.0), rad=-0.20)
    arrow(ax, (50, 26),    (50, 23.0))
    arrow(ax, (19, 26),    (40, 23.0), rad=0.20)

    # 5.5 sweep
    box(ax, 4, 8.5, 92, 6.5,
        "5.5  Bottleneck resists the FULL intervention space (Tab. 1, six families)\n"
        "Appearance \u00b7 Appearance-UB \u00b7 Where-to-go  +  Commit-timing (REVOKE)  +  Forced (CRV)\n"
        " +  Temporal (MUAP, ROUTER)  +  Multi-view (FUSE)  +  Multimodal probe (DXCV).\n"
        "Every DEPLOYABLE arm sits at non-sig ~0; only the non-deployable UB clears p<.05 and even it caps at +0.057.",
        C_EVID, E_EVID, fontsize=9.0)
    arrow(ax, (50, 16.5), (50, 15))

    # 5.6 + 5.7
    box(ax, 4, 2.5, 44, 5,
        "5.6 Multimodal direction (DXCV)\ndepth sanity-check veto on low-r commits.\n"
        "Train-free probe \u2014 not the remedy; remedy must LEARN to fuse.",
        C_NOTE, E_NOTE, fontsize=8.6)
    box(ax, 52, 2.5, 44, 5,
        "5.7 Sample size & scope (the warning)\n"
        "Gaussian noise N=40 \u0394=\u22120.125  vs  N=150 \u0394=+0.087.\n"
        "Sign flip under-powered \u21d2 all main conclusions at N=300.",
        C_PROB, E_PROB, fontsize=8.6)


# =============== PAGE 5 : Sec 6 + Roadmap reasoning ======================
def page_chain_conclusion_roadmap(ax):
    page_setup(ax,
        "Reasoning Chain  \u2014  Conclusion (Sec. 6) + Roads Forward",
        "The negative result is not the end:  it points to one of two training-side directions.")

    # ===== top: Sec 6 conclusion logic =====
    ax.add_patch(FancyBboxPatch((2.5, 38), 95, 22,
                                boxstyle="round,pad=0.4,rounding_size=0.4",
                                fc="#FAFAFA", ec="#9E9E9E", lw=0.8, zorder=0))
    ax.text(50, 58.5,
            "Section 6  \u2014  Conclusion  (five logical steps)",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#0D47A1")

    steps = [
        (4,  50, 18, 6,
         "Step 1  Restate fact\ncommit ~93% &\nexplore-fail ~6%\nunchanged from clean",
         C_EVID, E_EVID),
        (24, 50, 18, 6,
         "Step 2  Sweep recap\n6 families,\nevery deployable \u2248 0,\nonly UB p<.05 (+0.057)",
         C_EVID, E_EVID),
        (44, 50, 18, 6,
         "Step 3  Decoupling\nRES lifts r 0.39\u21920.89\nbut decision quality\nunmoved",
         C_EVID, E_EVID),
        (64, 50, 18, 6,
         "Step 4  Verdict\nirreducible commitment\nerror at test time\non a frozen agent",
         C_CONCL, E_CONCL),
        (84, 50, 12, 6,
         "Step 5\ntwo roads\nforward",
         C_NEXT, E_NEXT),
    ]
    for x,y,w,h,t,fc,ec in steps:
        box(ax, x, y, w, h, t, fc, ec, fontsize=8.5)
    for x_from, x_to in [(22,24),(42,44),(62,64),(82,84)]:
        arrow(ax, (x_from, 53), (x_to, 53))

    # release statement
    box(ax, 14, 40, 72, 6,
        "Release pledge\n"
        "benchmark configs \u00b7 paired-protocol scaffold \u00b7 per-episode CSVs\n"
        "\u00b7 per-arm monkey-patches \u00b7 blind-r estimator + calibration probe \u00b7 figure code + .bib",
        C_NOTE, E_NOTE, fontsize=9.0)

    # ===== bottom: roads forward =====
    ax.add_patch(FancyBboxPatch((2.5, 1.5), 95, 34,
                                boxstyle="round,pad=0.4,rounding_size=0.4",
                                fc="#FAFAFA", ec="#9E9E9E", lw=0.8, zorder=0))
    ax.text(50, 33.5,
            "Two Roads Forward  (Conclusion + Appendix F)  \u2014  motivation for the next milestones",
            ha="center", va="center", fontsize=13.0, fontweight="bold",
            color="#4A148C")

    # --- Road 1 (left big card) ---
    ax.add_patch(FancyBboxPatch((4, 14), 44, 17,
                                boxstyle="round,pad=0.3,rounding_size=0.4",
                                fc="#F3E5F5", ec=E_NEXT, lw=1.2, zorder=2))
    ax.text(26, 29.8,
            "Road 1  \u2014  Degradation-aware training & adaptation",
            ha="center", va="center", fontsize=11.0, fontweight="bold",
            color="#4A148C", zorder=3)
    r1 = [
        "Why:  loss is commitment QUALITY on low-r frames.",
        "If perception/commit models stay discriminative under",
        "corruption \u21d2 +5\u201310 SR pts is plausible.",
        "",
        "Pilot: GLEE-SwinL head-only fine-tune",
        "   \u2022 freeze SwinL backbone + CLIP text encoder",
        "   \u2022 trainable: mask-DINO decoder head (tens of MB)",
        "   \u2022 self-supervised: GLEE(clean) is the teacher",
        "   \u2022 loss = KL(cls) + BCE(mask) + L1(box)",
        "   \u2022 ~30 train ep \u00d7 50 steps = ~1500 frames",
        "   \u2022 ~3h train + ~12h paired N=150 eval",
        "   \u2022 lands as final row in Tab. 1",
    ]
    yy = 28.0
    for s in r1:
        ax.text(5.2, yy, s, ha="left", va="top", fontsize=8.4,
                color="#4A148C", linespacing=1.20, zorder=3)
        yy -= 1.10

    # --- Road 2 (right big card) ---
    ax.add_patch(FancyBboxPatch((52, 14), 44, 17,
                                boxstyle="round,pad=0.3,rounding_size=0.4",
                                fc="#F3E5F5", ec=E_NEXT, lw=1.2, zorder=2))
    ax.text(74, 29.8,
            "Road 2  \u2014  Learned multimodal fusion under degraded RGB",
            ha="center", va="center", fontsize=11.0, fontweight="bold",
            color="#4A148C", zorder=3)
    r2 = [
        "Why:  RGB drops discriminative info under low light;",
        "depth / LiDAR do not.  But consulting depth only at",
        "INFERENCE time is bounded (DXCV is the probe).",
        "",
        "Pilot: detector that has LEARNED RGB + depth fusion",
        "   \u2022 GLEE-D variant with depth as 2nd modality",
        "   \u2022 trained on Habitat-rendered depth + degraded RGB",
        "   \u2022 evaluated through the same paired N=150 protocol",
        "   \u2022 contrasted against DXCV (train-free upper bound)",
        "   \u2022 if it beats DXCV \u21d2 fusion must be LEARNED",
        "   \u2022 second contribution beyond the Road-1 head pilot",
        "",
    ]
    yy = 28.0
    for s in r2:
        ax.text(53.2, yy, s, ha="left", va="top", fontsize=8.4,
                color="#4A148C", linespacing=1.20, zorder=3)
        yy -= 1.10

    # --- Fallback ladder (horizontal strip across the bottom) ---
    ax.add_patch(FancyBboxPatch((4, 6.5), 92, 6.5,
                                boxstyle="round,pad=0.3,rounding_size=0.4",
                                fc="#FFF8E1", ec="#F57F17", lw=1.2, zorder=2))
    ax.text(50, 12.0,
            "Fallback ladder for Road 1  (in case head fine-tune does not converge before T\u221215)",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="#BF360C", zorder=3)
    fb = [
        "Plan B  \u2014  LoRA rank-32 on the last 2 mask-DINO decoder layers (~5M params).  Cheaper, more likely to land a small but real \u0394SR.",
        "Plan C  \u2014  feature-restoration MLP adapter that maps degraded SwinL features toward clean SwinL features (~1h training).",
        "Plan D (paper-side)  \u2014  if no Road-1 datapoint by T\u22125, ship as Diagnosis & Benchmark paper; declare Road 1 as future work.",
    ]
    yy = 10.5
    for s in fb:
        ax.text(5.5, yy, s, ha="left", va="top", fontsize=8.7,
                color="#5D4037", linespacing=1.20, zorder=3)
        yy -= 1.20

    # --- bottom strip: long-term direction ---
    ax.add_patch(FancyBboxPatch((4, 2.0), 92, 3.5,
                                boxstyle="round,pad=0.2,rounding_size=0.3",
                                fc="#EDE7F6", ec=E_NEXT, lw=0.9, zorder=2))
    ax.text(50, 3.75,
            "Long-term backbone (Appendix F, fig_roadmap.pdf):  detector + policy jointly retrained on degraded scenes "
            "(degradation-aware joint training).\n"
            "This paper supplies the benchmark + paired-protocol instrument needed to MEASURE such a retrained model honestly.",
            ha="center", va="center", fontsize=9.0, color="#4A148C",
            linespacing=1.30, zorder=3)

    # tie-in arrow: Sec 6 step 5 -> roads block
    arrow(ax, (90, 50), (90, 32), color="#4A148C", lw=1.4)


# =============== PAGE 7 : 30-day milestones (Gantt) ======================
def page_milestones(ax):
    page_setup(ax,
        "30-Day Milestones  \u2014  AAAI 2027 Submission",
        "T = 2026-07-28 deadline   \u00b7   green = done, orange = active, blue = planned, red = hard checkpoint.",
        xlim=(0, 130), ylim=(0, 70))

    today = date.today()
    deadline = date(2026, 7, 28)
    days_total = (deadline - today).days + 1

    # ----- timeline axis -----
    x0, x1 = 38, 125
    y_axis = 8
    ax.plot([x0, x1], [y_axis, y_axis], color="#444", lw=1.2, zorder=2)
    # tick marks every 5 days
    for i in range(0, days_total+1, 5):
        t = today + timedelta(days=i)
        xp = x0 + (x1-x0) * (i/days_total)
        ax.plot([xp, xp], [y_axis-0.4, y_axis+0.4], color="#444", lw=1.0)
        ax.text(xp, y_axis-1.6,
                f"{t.strftime('%m-%d')}\n(T\u2212{days_total-1-i})",
                ha="center", va="top", fontsize=8.0, color="#444",
                linespacing=1.2)
    # deadline marker
    xp_dl = x1
    ax.plot([xp_dl, xp_dl], [y_axis-0.6, 60], color="#B71C1C",
            lw=1.0, ls="--", alpha=0.5, zorder=1)
    ax.text(xp_dl-0.5, 60, "AAAI deadline\n2026-07-28",
            ha="right", va="top", fontsize=9.5, color="#B71C1C",
            fontweight="bold")

    # ----- tasks (week, label, fc, ec, day_start, day_end, owner) -----
    # day indices relative to today=2026-06-28 (T-30)
    tasks = [
        # Week 1 : finish queue, fill table, baseline VLFM
        ("DXCV N=150 finishes; fill_unified.py;\nback-fill Tab. 1 + Fig. 5 forest",  C_NOTE, E_NOTE,  0,  3),
        ("VLFM N=150 paired @ low-light s4\n(closes single-backbone W1, ~15h\n\u2014 50 ep would sign-flip per Sec. 5.7)", C_PROB, E_PROB,  2,  6),
        ("OracleGate qualitative\n20 oracle-rescued episodes (W4)",                 C_PROB, E_PROB,  4,  7),
        ("Motion-blur N=300 baseline\n(closes single-primary-degradation gap)",     C_PROB, E_PROB,  3,  9),

        # Week 2 : Road 1 pilot (the make-or-break add)
        ("Road 1 stage A  \u2014  data gen\n~30 train ep \u00d7 50 steps = ~1500 frames",       C_NEXT, E_NEXT,  7,  9),
        ("Road 1 stage B  \u2014  GLEE head fine-tune\n~3h, periodic checkpoints",              C_NEXT, E_NEXT,  9, 12),
        ("Road 1 stage C  \u2014  paired N=150 eval\nfill final Tab. 1 row",                    C_NEXT, E_NEXT, 12, 15),

        # Week 3 : writing freeze & polishing
        ("Update Sec.\u00a05 + Sec.\u00a06 + Tab.\u00a01 prose\nintegrate VLFM, OracleGate-qual, motion-blur, Road 1", C_EVID, E_EVID, 15, 19),
        ("Re-generate all figures with final numbers\nfig_decomp, fig_forest, fig_teaser",       C_EVID, E_EVID, 18, 21),
        ("Internal full read \u00b7 reviewer pass #1\nclarity, claims, citation completeness",   C_EVID, E_EVID, 19, 22),

        # Week 4 : freeze, bib, format, ship
        ("WRITING FREEZE checkpoint\nno new experiments after this",                            "#FFCDD2", "#B71C1C", 22, 23),
        ("Polish abstract + Intro + Conclusion\nshrink to 7 pages, push extras to Appendix",   C_EVID, E_EVID, 22, 26),
        ("Bibliography + .bib double-check\nremove duplicates, fix arXiv \u2192 published",     C_EVID, E_EVID, 24, 26),
        ("Format check + supplementary zip\nAAAI submission portal upload",                    C_EVID, E_EVID, 26, 29),
        ("Final submission  \u2014  T\u22120 (2026-07-28)",                                     "#FFCDD2", "#B71C1C", 29, 30),
    ]

    bar_h = 1.8
    spacing = 0.45
    row = 0
    base_y = y_axis + 4.0
    for label, fc, ec, d0, d1 in tasks:
        xpa = x0 + (x1-x0) * (d0/days_total)
        xpb = x0 + (x1-x0) * (d1/days_total)
        ypa = base_y + row*(bar_h + spacing)
        ax.add_patch(FancyBboxPatch((xpa, ypa), max(xpb-xpa, 0.6), bar_h,
                                    boxstyle="round,pad=0.05,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=0.9, zorder=3))
        ax.text(x0 - 1, ypa + bar_h/2, label,
                ha="right", va="center", fontsize=8.0,
                color="#222", linespacing=1.2)
        row += 1

    # week separators
    week_breaks = [7, 14, 21]
    for wb in week_breaks:
        xp = x0 + (x1-x0) * (wb/days_total)
        ax.plot([xp, xp], [y_axis, base_y + len(tasks)*(bar_h+spacing)+1],
                color="#9E9E9E", lw=0.6, ls=":", zorder=1)
    # week labels
    for wb, name in [(3.5,"W1  finish queue, plug gaps"),
                     (10.5,"W2  Road 1 pilot"),
                     (17.5,"W3  writing + figures"),
                     (26,"W4  freeze, polish, ship")]:
        xp = x0 + (x1-x0) * (wb/days_total)
        ax.text(xp, base_y + len(tasks)*(bar_h+spacing) + 1.6, name,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#0D47A1")

    # ----- hard checkpoints text -----
    ax.add_patch(FancyBboxPatch((4, 1), 122, 5,
                                boxstyle="round,pad=0.3,rounding_size=0.4",
                                fc="#FFEBEE", ec="#B71C1C", lw=1.0))
    ax.text(65, 5.0,
            "Hard checkpoints",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#B71C1C")
    ax.text(65, 3.0,
            "T\u221229  DXCV done   \u00b7   "
            "T\u221221  VLFM N=150 + OracleGate-qual done   \u00b7   "
            "T\u221215  Road 1 pilot eval done (or Plan B/C taken)   \u00b7   "
            "T\u22128   writing freeze   \u00b7   "
            "T\u22122   format upload   \u00b7   "
            "T\u22120   submit",
            ha="center", va="center", fontsize=9.5, color="#B71C1C")


# =============== MAIN: build pages & merge ===============================
def main():
    here = "/root/VLA_papers/AAAI_DRPhysNav"
    fig_dir = f"{here}/figures"
    tmp_pdf = f"{fig_dir}/_plan_pages.pdf"
    out_pdf = f"{here}/PAPER_PLAN.pdf"

    # 5 matplotlib pages -> one multi-page intermediate PDF
    pages_fn = [
        ("Cover",                page_cover),
        ("Chain: Intro+Motiv",   page_chain_intro_motivation),
        ("Chain: Method+Exp",    page_chain_method_experiments),
        ("Chain: Concl+Roads",   page_chain_conclusion_roadmap),
        ("Milestones",           page_milestones),
    ]
    with PdfPages(tmp_pdf) as pp:
        for name, fn in pages_fn:
            fig, ax = plt.subplots(figsize=(16.5, 11.7))  # A3 landscape ish
            fn(ax)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
            pp.savefig(fig, dpi=180)
            plt.close(fig)
            print(f"  rendered: {name}")

    # Now merge: cover + skeleton + chain pages + status mind-map + milestones
    skeleton = f"{fig_dir}/fig_paper_mindmap.pdf"
    status   = f"{fig_dir}/fig_mindmap.pdf"
    if not os.path.exists(skeleton):
        raise FileNotFoundError(skeleton)
    if not os.path.exists(status):
        raise FileNotFoundError(status)

    out = fitz.open()
    plan = fitz.open(tmp_pdf)
    skl = fitz.open(skeleton)
    sta = fitz.open(status)

    # p1: cover                  -> plan[0]
    out.insert_pdf(plan, from_page=0, to_page=0)
    # p2: writing skeleton       -> skeleton (1 page)
    out.insert_pdf(skl)
    # p3-5: 3 chain pages        -> plan[1..3]
    out.insert_pdf(plan, from_page=1, to_page=3)
    # p6: status mind-map        -> status (1 page)
    out.insert_pdf(sta)
    # p7: milestones             -> plan[4]
    out.insert_pdf(plan, from_page=4, to_page=4)

    out.save(out_pdf, garbage=4, deflate=True)
    out.close(); plan.close(); skl.close(); sta.close()

    # cleanup intermediate
    os.remove(tmp_pdf)
    print(f"\n[OK] wrote {out_pdf}")
    print(f"     pages: {fitz.open(out_pdf).page_count}")


if __name__ == "__main__":
    main()
