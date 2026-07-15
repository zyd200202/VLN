"""Paper-writing skeleton mind map (graphviz, clustered grid).

Output: /root/VLA_papers/AAAI_DRPhysNav/figures/fig_paper_mindmap.pdf

Layout strategy: each top-level paper block (Title, Abstract, Sec 1..6,
Bibliography, Appendix) is a graphviz CLUSTER. Inside each cluster the
section header is at the top, then its child nodes stack below.  dot
arranges the clusters in a 2D grid via the implicit ROOT -> section edges.

Colour key (also shown in legend):
  yellow      centre / paper root
  blue        section header
  green       paragraph / text content
  orange      figure (fig_*.pdf)
  purple      table
  pink        equation / key quantitative result
  grey        appendix / bibliography
"""
from graphviz import Digraph

C_ROOT = "#FFF59D"
C_SEC  = "#BBDEFB"
C_PAR  = "#C8E6C9"
C_FIG  = "#FFE0B2"
C_TAB  = "#E1BEE7"
C_EQ   = "#FFCDD2"
C_APP  = "#ECEFF1"

E_ROOT = "#F57F17"
E_SEC  = "#0D47A1"
E_PAR  = "#1B5E20"
E_FIG  = "#E65100"
E_TAB  = "#4A148C"
E_EQ   = "#B71C1C"
E_APP  = "#37474F"


def node(g, nid, label, fill, edge, fontsize="9", penwidth="1.0"):
    g.node(nid, label="<" + label + ">", style="rounded,filled",
           fillcolor=fill, color=edge, fontname="Helvetica",
           fontsize=fontsize, fontcolor="black", shape="box",
           margin="0.12,0.06", penwidth=penwidth)


def add_section(g, sec_id, sec_label, children, cluster_label=None):
    """Plain section: header node + sub-nodes (no cluster box).
    With rankdir=TB this lays out as a one-section column that drops
    downward from the section header.
    """
    node(g, sec_id, sec_label, C_SEC, E_SEC, fontsize="11", penwidth="1.4")
    g.edge("ROOT", sec_id, color="gray45", penwidth="1.0")
    # Chain the children vertically by adding invisible same-section
    # ranking. We do this by giving each child a sequential edge from
    # the previous child (group="sec_id"), which dot uses to keep them
    # in a vertical line under the section header.
    prev = sec_id
    for cid, lbl, fc, ec in children:
        node(g, cid, lbl, fc, ec, fontsize="8.5")
        g.edge(prev, cid, color="gray70", penwidth="0.7",
               weight="3" if prev == sec_id else "5")
        prev = cid


def main():
    g = Digraph("DRPhysNav_PaperSkeleton", format="pdf")
    g.attr(layout="dot", rankdir="TB", splines="line",
           nodesep="0.18", ranksep="0.20", pad="0.5",
           bgcolor="white", newrank="true",
           label=("<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'>"
                  "<TR><TD ALIGN='CENTER'><FONT POINT-SIZE='24'><B>"
                  "DRPhysNav \u2014 Paper Writing Skeleton</B></FONT></TD></TR>"
                  "<TR><TD ALIGN='CENTER'><FONT POINT-SIZE='12' COLOR='gray30'><I>"
                  "Restoring the Image Does Not Restore Navigation:<BR/>"
                  "Diagnosing Degradation-Robust Object-Goal Navigation as a Commitment Problem"
                  "</I></FONT></TD></TR>"
                  "<TR><TD ALIGN='CENTER'><FONT POINT-SIZE='10' COLOR='gray30'>"
                  "AAAI submission template \u00b7 7-page main + unlimited appendix \u00b7 ~25 references"
                  "</FONT></TD></TR>"
                  "</TABLE>>"),
           labelloc="t", labeljust="c")
    g.attr("graph", fontname="Helvetica")
    g.attr("edge",  color="gray55", arrowsize="0.45", penwidth="0.8")
    g.attr("node",  fontname="Helvetica")

    # ----- root -----
    node(g, "ROOT",
         "<B><FONT POINT-SIZE='14'>PAPER</FONT></B><BR/>"
         "<FONT POINT-SIZE='9'>main.tex</FONT><BR/>"
         "<FONT POINT-SIZE='8' COLOR='gray35'>9 top-level blocks</FONT>",
         C_ROOT, E_ROOT, fontsize="11", penwidth="1.6")

    # ======= 0. Title block + Teaser =======
    add_section(g, "S0",
        "<B>0. Title Block</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>main.tex L39\u2013L65</FONT>",
        [
            ("S0_title",
             "<B>Title</B><BR/>"
             "&ldquo;Restoring the Image Does Not<BR/>"
             "Restore Navigation: Diagnosing<BR/>"
             "Degradation-Robust ObjectNav<BR/>"
             "as a Commitment Problem&rdquo;", C_PAR, E_PAR),
            ("S0_author",
             "<B>Author block</B><BR/>"
             "Anonymous submission", C_PAR, E_PAR),
            ("S0_teaser",
             "<B>Fig. 1  fig_teaser.pdf</B><BR/>"
             "<FONT POINT-SIZE='7' COLOR='gray35'>3-panel wide banner forced on p.1<BR/>"
             "via \\@maketitle hijack</FONT><BR/>"
             "(a) PSNR 7.4\u219215.9 dB &middot; r 0.39\u21920.89<BR/>"
             "(b) \u0394SR=+0.04, p=0.182, N=300<BR/>"
             "(c) HM3D ep. trajectory + 26-pt gap", C_FIG, E_FIG),
        ])

    # ======= Abstract =======
    add_section(g, "SA",
        "<B>Abstract</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>00_abstract.tex \u00b7 1 para, ~210 words</FONT>",
        [
            ("SA_a", "Problem: zero-shot VLM nav loses SR<BR/>"
                     "under low-light / fog / motion blur", C_PAR, E_PAR),
            ("SA_b", "Standard remedy = restoration front-end<BR/>"
                     "<I>but targets the wrong layer</I>", C_PAR, E_PAR),
            ("SA_c", "Setup: InstructNav on HM3D ObjectNav<BR/>"
                     "paired protocol \u00b7 blind r \u00b7 N=300", C_PAR, E_PAR),
            ("SA_d", "Finding 1: commit rate ~93% unchanged<BR/>"
                     "<B>quality</B> of commitment collapses", C_PAR, E_PAR),
            ("SA_e", "Finding 2: reach 0.62 vs SR 0.36<BR/>"
                     "<B>26-pt reach-to-success gap</B>", C_PAR, E_PAR),
            ("SA_f", "Finding 3: r at commit 0.39 vs 0.95<BR/>"
                     "once committed, locked ~12 steps", C_PAR, E_PAR),
            ("SA_g", "Sweep over <B>six causal families</B><BR/>"
                     "every deployable family caps at ~0", C_PAR, E_PAR),
            ("SA_h", "Methodological warning:<BR/>"
                     "<B>sign flip N=40 vs N=150</B>", C_PAR, E_PAR),
            ("SA_i", "Conclusion: degradation-aware training<BR/>"
                     "+ multimodal sensing", C_PAR, E_PAR),
        ])

    # ======= Section 1. Introduction =======
    add_section(g, "S1",
        "<B>1. Introduction</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>10_intro.tex \u00b7 ~1 page</FONT>",
        [
            ("S1_p1", "<B>P1</B>  Problem statement<BR/>"
                      "ObjectNav closed perception-action loop<BR/>"
                      "VLM-driven nav under degraded camera", C_PAR, E_PAR),
            ("S1_p2", "<B>P2</B>  Natural remedy<BR/>"
                      "restoration front-end (Zero-DCE, URetinex)<BR/>"
                      "\u21d2 assumption: clean look = clean behaviour", C_PAR, E_PAR),
            ("S1_p3", "<B>P3</B>  Benchmark paragraph<BR/>"
                      "4 corruptions \u00b7 several severities<BR/>"
                      "paired McNemar \u00b7 bootstrap CI \u00b7 blind r", C_PAR, E_PAR),
            ("S1_p4", "<B>P4</B>  &ldquo;Degradation breaks commitments&rdquo;<BR/>"
                      "3 observations \u2192 Fig. 2<BR/>"
                      "SR = reach \u2212 commit/stop loss", C_PAR, E_PAR),
            ("S1_p5", "<B>P5</B>  Negative results paragraph<BR/>"
                      "RES +0.04 p=0.18<BR/>"
                      "OracleGate UB +0.057 p=0.04<BR/>"
                      "M1g +0.04 p=0.20 \u00b7 six families null", C_PAR, E_PAR),
            ("S1_p6", "<B>P6  Contributions (i)\u2013(iv)</B><BR/>"
                      "(i) benchmark + paired protocol<BR/>"
                      "(ii) mechanism diagnosis at N=300<BR/>"
                      "(iii) six-family negative-result sweep<BR/>"
                      "(iv) sign-flip methodological warning", C_PAR, E_PAR),
        ])

    # ======= Section 2. Related Work =======
    add_section(g, "S2",
        "<B>2. Related Work</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>20_related.tex \u00b7 ~1 page</FONT>",
        [
            ("S2_p1", "<B>Zero-shot ObjectNav w/ foundation models</B><BR/>"
                      "ESC \u00b7 VLFM \u00b7 NavGPT \u00b7 InstructNav<BR/>"
                      "&rarr; we study <I>where</I> they break", C_PAR, E_PAR),
            ("S2_p2", "<B>Robustness to visual corruption</B><BR/>"
                      "RobustNav \u00b7 data augmentation<BR/>"
                      "pixel-layer restoration (Zero-DCE etc.)<BR/>"
                      "&rArr; pixel layer is wrong layer", C_PAR, E_PAR),
            ("S2_p3", "<B>Uncertainty / commitment / exec. control</B><BR/>"
                      "blind IQA \u00b7 active perception<BR/>"
                      "we localise to commitment-timing", C_PAR, E_PAR),
            ("S2_p4", "<B>Evaluation of stochastic agents</B><BR/>"
                      "clean SR varies 0.27\u20130.47 / seed<BR/>"
                      "paired N=300 + McNemar + bootstrap<BR/>"
                      "links to App. B (positioning)", C_PAR, E_PAR),
        ])

    # ======= Section 3. Motivation =======
    add_section(g, "S3",
        "<B>3. Why Navigation Fails</B><BR/>"
        "<B>Under Degradation</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>30_motivation.tex \u00b7 ~1.5 pages</FONT>",
        [
            ("S3_fig", "<B>Fig 2  fig_motivation.pdf</B>  (4 panels)<BR/>"
                       "(a) M0: \u0394SR=\u22120.21 CI excludes 0<BR/>"
                       "(b) M2: walk-away 78\u219288%, false-stop 0.51\u21920.78<BR/>"
                       "(c) M3: severity sweep, irrev. sat. 95%<BR/>"
                       "(d) M4: per-commit r, mean 0.95\u21920.41", C_FIG, E_FIG),
            ("S3_setup", "<B>Setup</B><BR/>"
                         "InstructNav val_mini seeds {0,1,2}<BR/>"
                         "n=90 pooled \u00b7 bootstrap B=5000", C_PAR, E_PAR),
            ("S3_m01", "<B>M0\u2013M1</B>  Degradation hurts,<BR/>"
                       "appearance not the lever<BR/>"
                       "+8.4 dB PSNR, SR within \u00b10.10 band", C_PAR, E_PAR),
            ("S3_m2", "<B>M2</B>  Failures are irreversible<BR/>"
                      "walk-away vs false-stop split<BR/>"
                      "r at turning point 0.93 \u2192 0.40", C_PAR, E_PAR),
            ("S3_m34", "<B>M3\u2013M4</B>  r informative;<BR/>"
                       "<B>single low-r frame triggers commit</B><BR/>"
                       "r 0.95\u21920.41 \u00b7 locked ~21 steps", C_PAR, E_PAR),
            ("S3_m5", "<B>M5  Reachability decomposition (N=300)</B><BR/>"
                      "<FONT POINT-SIZE='8' COLOR='#B71C1C'>"
                      "SR = reach \u2212 commit/stop loss</FONT><BR/>"
                      "reach 0.623 \u00b7 SR 0.358 \u00b7 loss 0.265<BR/>"
                      "false-commit 55%\u219276% \u00b7 locked ~12 steps", C_EQ, E_EQ),
            ("S3_sum", "<B>Summary</B>  the chain<BR/>"
                       "perception ok, action ok<BR/>"
                       "\u21d2 loss is in reach-to-success conversion<BR/>"
                       "\u21d2 perception fixes won't move SR", C_PAR, E_PAR),
        ])

    # ======= Section 4. Method =======
    add_section(g, "S4",
        "<B>4. From Diagnosis to</B><BR/>"
        "<B>Candidate Interventions</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>40_method.tex \u00b7 ~0.75 page</FONT>",
        [
            ("S4_overview", "<B>Overview</B><BR/>"
                            "diagnosis \u21d2 falsifiable prediction<BR/>"
                            "pixel + perception + decision layers<BR/>"
                            "paired N=300 instrument (App. C)", C_PAR, E_PAR),
            ("S4_r", "<B>4.1 Per-step reliability r</B><BR/>"
                     "luma + contrast + Laplacian +<BR/>"
                     "dark-channel + noise (training-free)<BR/>"
                     "<I>r is an INSTRUMENT, not a method</I>", C_PAR, E_PAR),
            ("S4_calib", "<B>Fig 3  fig_calibration.pdf</B><BR/>"
                         "clean 1.00 \u00b7 LL 0.65 \u00b7 blur 0.47<BR/>"
                         "fog 0.36 \u00b7 noise 0.20<BR/>"
                         "<I>r\u22650.9 no-op gate</I>", C_FIG, E_FIG),
            ("S4_c1", "<B>4.2 Candidate 1 (Pixel)</B><BR/>"
                      "RES blanket restoration<BR/>"
                      "OracleGate per-frame UB<BR/>"
                      "(non-deployable ceiling)", C_PAR, E_PAR),
            ("S4_c2", "<B>4.3 Candidate 2 (Decision)</B><BR/>"
                      "<FONT POINT-SIZE='8' COLOR='#B71C1C'>"
                      "v(a)=(1\u2212\u03b2) v_vlm+sem + \u03b2 v_frontier</FONT><BR/>"
                      "<FONT POINT-SIZE='8' COLOR='#B71C1C'>"
                      "\u03b2(r)=min(\u03b2_max, \u03b1(1\u2212r))</FONT><BR/>"
                      "stop iff r\u2265\u03c4_stop=0.6", C_EQ, E_EQ),
            ("S4_extra", "<B>Additional families probed in Exp.</B><BR/>"
                         "REVOKE \u00b7 CRV \u00b7 MUAP/ROUTER<BR/>"
                         "FUSE \u00b7 DXCV (multimodal probe)", C_PAR, E_PAR),
        ])

    # ======= Section 5. Experiments =======
    add_section(g, "S5",
        "<B>5. Experiments</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>50_experiments.tex \u00b7 ~2.0 pages</FONT>",
        [
            ("S5_proto", "<B>5.1 Protocol</B><BR/>"
                         "InstructNav (Qwen2-VL + GLEE + value-map)<BR/>"
                         "N=300 main split \u00b7 same-seed same-ep<BR/>"
                         "SR, SPL, \u0394SR, McNemar p, 95% CI", C_PAR, E_PAR),
            ("S5_eff", "<B>5.2 The degradation effect</B><BR/>"
                       "low-light n=90: 0.433\u21920.222, \u0394=\u22120.211<BR/>"
                       "N=300 baseline SR=0.37<BR/>"
                       "motion-blur reported as non-sig", C_PAR, E_PAR),
            ("S5_neg", "<B>5.3 Two negative results (N=300)</B><BR/>"
                       "RES \u0394=+0.040 p=0.182<BR/>"
                       "OracleGate UB \u0394=+0.057 <B>p=0.043*</B><BR/>"
                       "M1g \u0394=+0.040 p=0.201<BR/>"
                       "\u21d2 all converge to ~+0.04 ceiling", C_PAR, E_PAR),
            ("S5_mech", "<B>5.4 Mechanism: why candidates cap</B><BR/>"
                        "reach ~0.62, loss ~0.25 INVARIANT<BR/>"
                        "RES r 0.39\u21920.89 but P(success|commit) 0.40\u21920.42<BR/>"
                        "<B>perception-decision decoupling</B>", C_PAR, E_PAR),
            ("S5_decomp", "<B>Fig 4  fig_decomp.pdf</B><BR/>"
                          "(a) bars: reach / commit-loss / SR<BR/>"
                          "(b) r at commit vs P(success|commit)<BR/>"
                          "highlights 26-pt invariant gap", C_FIG, E_FIG),
            ("S5_sweep", "<B>5.5 Six-family sweep</B><BR/>"
                         "appearance / where-to-go / commit-timing<BR/>"
                         "forced-commit / temporal+multi-view / multimodal<BR/>"
                         "all deployable arms cap at ~0", C_PAR, E_PAR),
            ("S5_tab", "<B>Tab 1  \u0394SR sweep</B><BR/>"
                       "9 rows: family / arm / \u0394SR / p / N<BR/>"
                       "<I>only OracleGate (UB) p&lt;0.05</I><BR/>"
                       "5 N=150 placeholders (queue)", C_TAB, E_TAB),
            ("S5_forest", "<B>Fig 5  fig_forest.pdf</B><BR/>"
                          "paired \u0394SR + bootstrap 95% CI<BR/>"
                          "every deployable arm CI crosses 0", C_FIG, E_FIG),
            ("S5_r2", "<B>5.6 Probing the multimodal direction</B><BR/>"
                      "DXCV: depth-median + IQR sanity check<BR/>"
                      "vetoes VLM commit on low-r frames<BR/>"
                      "<I>train-free probe, not the remedy</I>", C_PAR, E_PAR),
            ("S5_size", "<B>5.7 Sample size &amp; scope</B><BR/>"
                        "<FONT COLOR='#B71C1C'><B>sign flip</B>:</FONT><BR/>"
                        "Gaussian-noise N=40 \u0394=\u22120.125<BR/>"
                        "vs N=150 \u0394=+0.087<BR/>"
                        "single backbone, 1 primary degradation", C_PAR, E_PAR),
        ])

    # ======= Section 6. Conclusion =======
    add_section(g, "S6",
        "<B>6. Conclusion</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>60_conclusion.tex \u00b7 1 para, ~0.4 page</FONT>",
        [
            ("S6_a", "<B>Restate finding</B><BR/>"
                     "commit ~93%, explore-fail ~6% \u2014 unchanged<BR/>"
                     "what collapses is commitment <I>quality</I>", C_PAR, E_PAR),
            ("S6_b", "<B>Six-family sweep recap</B><BR/>"
                     "every deployable arm \u2248 0<BR/>"
                     "only non-deployable UB sig (+0.057)", C_PAR, E_PAR),
            ("S6_c", "<B>Perception-decision decoupling</B><BR/>"
                     "r at commit 0.39 \u2192 0.89<BR/>"
                     "decision quality unmoved", C_PAR, E_PAR),
            ("S6_d", "<B>Irreducible commitment error</B><BR/>"
                     "inference-time patches on frozen,<BR/>"
                     "clean-trained agent cannot recover it", C_PAR, E_PAR),
            ("S6_e", "<B>Two roads forward</B><BR/>"
                     "Road 1: degradation-aware training/adapt<BR/>"
                     "Road 2: learned multimodal fusion<BR/>"
                     "(DXCV is the probe, not the cure)", C_PAR, E_PAR),
            ("S6_f", "<B>Release pledge</B><BR/>"
                     "benchmark + protocol + neg. results<BR/>"
                     "as instrument for retrained agents", C_PAR, E_PAR),
        ])

    # ======= Bibliography =======
    add_section(g, "SB",
        "<B>Bibliography</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>main.bbl \u00b7 ~25 entries</FONT>",
        [
            ("SB_a", "<B>Navigation backbones</B><BR/>"
                     "InstructNav, VLFM, NavGPT, ESC,<BR/>"
                     "CoW, L3MVN, SemExp, GOAT,<BR/>"
                     "R2R, VLN-CE, ObjectNav", C_APP, E_APP),
            ("SB_b", "<B>Detectors / VLMs</B><BR/>"
                     "GLEE \u00b7 Qwen2-VL", C_APP, E_APP),
            ("SB_c", "<B>Restoration / corruption</B><BR/>"
                     "Zero-DCE \u00b7 URetinex<BR/>"
                     "Hendrycks ImageNet-C<BR/>"
                     "RobustNav \u00b7 THDA", C_APP, E_APP),
            ("SB_d", "<B>Uncertainty / eval</B><BR/>"
                     "Abdar uncertainty \u00b7 Su blind IQA<BR/>"
                     "Ramakrishnan exploration<BR/>"
                     "Anderson SPL evaluation", C_APP, E_APP),
            ("SB_e", "<B>Training / sim</B><BR/>"
                     "HM3D \u00b7 HM3D-SEM<BR/>"
                     "DD-PPO \u00b7 PIRLNav", C_APP, E_APP),
        ])

    # ======= Appendix A..G =======
    add_section(g, "SAPP",
        "<B>Appendix A\u2013G</B><BR/>"
        "<FONT POINT-SIZE='7' COLOR='gray35'>A0_appendix.tex \u00b7 unlimited length</FONT>",
        [
            ("SAPP_A", "<B>A. Sample size + sign-flip</B><BR/>"
                       "<FONT COLOR='#B71C1C'>N=40 \u22120.125 vs N=150 +0.087</FONT><BR/>"
                       "+ scope (single backbone, primary deg.)", C_APP, E_APP),
            ("SAPP_B", "<B>B. Where the paper sits</B><BR/>"
                       "Fig fig_positioning.pdf<BR/>"
                       "dot-matrix vs 5 prior families \u00d7<BR/>"
                       "6 evaluation dimensions", C_APP, E_APP),
            ("SAPP_C", "<B>C. Paired protocol &amp; stats</B><BR/>"
                       "Fig fig_pipeline.pdf<BR/>"
                       "<FONT POINT-SIZE='7' COLOR='#B71C1C'>"
                       "p = 2 \u03a3 C(b01+b10, i) / 2^(b01+b10)</FONT><BR/>"
                       "bootstrap B=5000 seed=42", C_APP, E_APP),
            ("SAPP_D", "<B>D. Hyperparameters of probed arms</B><BR/>"
                       "RES \u00b7 M1g (\u03b1, \u03b2_max, \u03c4_gate, \u03c4_stop)<BR/>"
                       "OracleGate \u00b7 REVOKE (K, \u03b5, \u2026)<BR/>"
                       "CRV \u00b7 MUAP \u00b7 FUSE \u00b7 DXCV", C_APP, E_APP),
            ("SAPP_E", "<B>E. Detailed per-arm counts</B><BR/>"
                       "rescue b01 / break b10<BR/>"
                       "RES (24,12) \u00b7 OracleGate (28,11)<BR/>"
                       "M1g (27,15) \u00b7 ROUTER (10,8)<BR/>"
                       "REVOKE/CRV/MUAP/FUSE/DXCV TBD", C_APP, E_APP),
            ("SAPP_F", "<B>F. Forward directions</B><BR/>"
                       "Fig fig_roadmap.pdf<BR/>"
                       "Road 1 train-side \u00b7 Road 2 multimodal-learned<BR/>"
                       "(future-work backbone fine-tune: <B>GLEE-DA head</B>)", C_APP, E_APP),
            ("SAPP_G", "<B>G. Release artefacts</B><BR/>"
                       "(i) corrupted HM3D configs<BR/>"
                       "(ii) paired protocol scaffold<BR/>"
                       "(iii) per-episode CSVs<BR/>"
                       "(iv) 8 arm monkey-patches<BR/>"
                       "(v) blind r + calibration probe<BR/>"
                       "(vi) make_figures.py + .bib", C_APP, E_APP),
        ])

    # Force all section headers into the same rank (one horizontal row)
    with g.subgraph() as same:
        same.attr(rank="same")
        for sid in ["S0", "SA", "S1", "S2", "S3", "S4", "S5", "S6", "SB", "SAPP"]:
            same.node(sid)

    # ======= Colour legend =======
    with g.subgraph(name="cluster_legend") as lg:
        lg.attr(label="Legend  (node colour = block type)",
                style="rounded", color="gray60",
                fontname="Helvetica", fontsize="11", labeljust="l")
        lg.attr("node", shape="box", style="rounded,filled",
                fontname="Helvetica", fontsize="9.5",
                margin="0.08,0.04", penwidth="0.9")
        for nid, lbl, fc, ec in [
            ("LG1", "Section header",          C_SEC, E_SEC),
            ("LG2", "Paragraph / text",        C_PAR, E_PAR),
            ("LG3", "Figure (fig_*.pdf)",      C_FIG, E_FIG),
            ("LG4", "Table",                   C_TAB, E_TAB),
            ("LG5", "Equation / key result",   C_EQ,  E_EQ),
            ("LG6", "Bibliography / appendix", C_APP, E_APP),
        ]:
            lg.node(nid, lbl, fillcolor=fc, color=ec)
        lg.edge("LG1", "LG2", style="invis")
        lg.edge("LG2", "LG3", style="invis")
        lg.edge("LG3", "LG4", style="invis")
        lg.edge("LG4", "LG5", style="invis")
        lg.edge("LG5", "LG6", style="invis")
        lg.attr(rank="same")

    out = "/root/VLA_papers/AAAI_DRPhysNav/figures/fig_paper_mindmap"
    g.render(out, format="pdf", cleanup=True)
    print(f"wrote {out}.pdf")


if __name__ == "__main__":
    main()
