"""Generate a paper mind-map PDF via graphviz (twopi radial layout).

Output: /root/VLA_papers/AAAI_DRPhysNav/figures/fig_mindmap.pdf

Colour key:
    green  done / in paper
    orange running in the unified N=150 queue
    blue   planned (Road 1 pilot, VLFM, ...)
    red    reviewer concern
    purple paper artefact
    yellow centre
"""
from graphviz import Digraph

C_DONE   = "#C8E6C9"
C_QUEUE  = "#FFE0B2"
C_PLAN   = "#BBDEFB"
C_RISK   = "#FFCDD2"
C_ART    = "#E1BEE7"
C_CENTER = "#FFF59D"

E_DONE   = "#1B5E20"
E_QUEUE  = "#E65100"
E_PLAN   = "#0D47A1"
E_RISK   = "#B71C1C"
E_ART    = "#4A148C"
E_CENTER = "#F57F17"


def node(g, nid, label, fill, edge, fontsize="10", shape="box", width=None):
    attrs = dict(
        label=label, style="rounded,filled", fillcolor=fill, color=edge,
        fontname="Helvetica", fontsize=fontsize, fontcolor="black",
        shape=shape, margin="0.18,0.10", penwidth="1.2",
    )
    if width:
        attrs["width"] = width
    g.node(nid, **attrs)


def main():
    g = Digraph("DRPhysNav_MindMap", format="pdf")
    # twopi = radial; the centre node is ROOT
    g.attr(layout="twopi", root="ROOT", overlap="false",
           ranksep="2.4 equally", pad="0.5", splines="curved",
           bgcolor="white",
           label=("<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'>"
                  "<TR><TD ALIGN='CENTER'><FONT POINT-SIZE='28'><B>"
                  "DRPhysNav &mdash; Paper Mind Map</B></FONT></TD></TR>"
                  "<TR><TD ALIGN='CENTER'><FONT POINT-SIZE='14' COLOR='gray30'><I>"
                  "Restoring the image does not restore navigation &mdash; "
                  "status, evidence, and the next-step plan"
                  "</I></FONT></TD></TR>"
                  "</TABLE>>"),
           labelloc="t", labeljust="c")
    g.attr("graph", fontname="Helvetica", fontsize="14")
    g.attr("edge",  color="gray45", arrowsize="0.55", penwidth="0.9")
    g.attr("node",  fontname="Helvetica")

    # =============== Centre node ===============
    node(g, "ROOT",
         "<<B><FONT POINT-SIZE='18'>DRPhysNav</FONT></B><BR/>"
         "<FONT POINT-SIZE='10'>Diagnosing Degradation-Robust<BR/>"
         "Object-Goal Navigation as a<BR/>Commitment Problem</FONT><BR/>"
         "<FONT POINT-SIZE='8' COLOR='#7D6500'><I>N=300 paired McNemar &middot; "
         "InstructNav &middot; HM3D ObjectNav</I></FONT>>",
         C_CENTER, E_CENTER, fontsize="12")

    # =============== Branch 1: Problem & Question ===============
    node(g, "B1", "<<B>1. Problem &amp; Question</B>>", C_DONE, E_DONE, fontsize="11")
    g.edge("ROOT", "B1")
    for nid, txt in [
        ("B1a", "Zero-shot VLM nav<BR/>(InstructNav, VLFM, NavGPT)<BR/>clean &rarr; degraded RGB"),
        ("B1b", "Standard remedy:<BR/>restoration front-end<BR/>(Zero-DCE, URetinex)"),
        ("B1c", "Hidden assumption:<BR/>clean-looking image<BR/>&rArr; clean behaviour"),
        ("B1d", "<B>Our claim</B>:<BR/>restoration aims at<BR/>the wrong layer"),
    ]:
        node(g, nid, "<" + txt + ">", C_DONE, E_DONE, fontsize="9")
        g.edge("B1", nid)

    # =============== Branch 2: Diagnosis M0-M5 ===============
    node(g, "B2", "<<B>2. Diagnosis M0&ndash;M5</B><BR/><I>Sec.3 (n=90, M5 at N=300)</I>>",
         C_DONE, E_DONE, fontsize="11")
    g.edge("ROOT", "B2")
    for nid, txt in [
        ("B2a", "<B>M0</B>  low-light<BR/>&Delta;SR = &minus;0.21<BR/>paired CI excludes 0"),
        ("B2b", "<B>M1</B>  appearance NOT the lever<BR/>+8.4 dB PSNR<BR/>SR moves +0.07 only"),
        ("B2c", "<B>M2</B>  failures become<BR/>irreversible commitments<BR/>walk-away 78&rarr;88%"),
        ("B2d", "<B>M3</B>  reliability r<BR/>monotone over severities<BR/>0.92&rarr;0.41"),
        ("B2e", "<B>M4</B>  single low-r frame<BR/>triggers commit<BR/>r=0.95&rarr;0.41"),
        ("B2f", "<B>M5</B>  N=300 decomp.<BR/>reach 0.62 vs SR 0.36<BR/><B>26-pt gap</B>"),
    ]:
        node(g, nid, "<" + txt + ">", C_DONE, E_DONE, fontsize="9")
        g.edge("B2", nid)

    # =============== Branch 3: Instrument ===============
    node(g, "B3", "<<B>3. Diagnosis Instrument</B><BR/><I>Sec.4 + App.C</I>>",
         C_DONE, E_DONE, fontsize="11")
    g.edge("ROOT", "B3")
    for nid, txt in [
        ("B3a", "Paired protocol<BR/>same seed, same episodes"),
        ("B3b", "McNemar exact two-sided<BR/>on discordant pairs"),
        ("B3c", "Bootstrap 95% CI<BR/>B=5000, RNG seed 42"),
        ("B3d", "Blind reliability r<BR/>luma + contrast + Laplacian +<BR/>dark-channel + noise"),
        ("B3e", "Calibration (probe set)<BR/>clean 1.00 &middot; LL 0.65 &middot;<BR/>blur 0.47 &middot; fog 0.36 &middot; noise 0.20"),
    ]:
        node(g, nid, "<" + txt + ">", C_DONE, E_DONE, fontsize="9")
        g.edge("B3", nid)

    # =============== Branch 4: Sweep ===============
    node(g, "B4", "<<B>4. Intervention Sweep</B><BR/><I>Six causal families &middot; Tab.1 / Fig.5</I>>",
         C_ART, E_ART, fontsize="11")
    g.edge("ROOT", "B4")

    # families subgrouped under B4 -- N=300 done first
    for nid, txt, fill, edge in [
        ("B4_RES",    "<B>Appearance &middot; RES</B><BR/>&Delta;=+0.040 &middot; p=0.18<BR/>N=300 &middot; <B>done</B>",                       C_DONE, E_DONE),
        ("B4_OG",     "<B>Appearance UB &middot; OracleGate</B><BR/>&Delta;=+0.057 &middot; <B>p=0.043</B>*<BR/>N=300 &middot; <B>done</B><BR/><I>non-deployable upper bound</I>",   C_DONE, E_DONE),
        ("B4_M1g",    "<B>Where-to-go &middot; M1g</B><BR/>&Delta;=+0.040 &middot; p=0.20<BR/>N=300 &middot; <B>done</B>",                       C_DONE, E_DONE),
        ("B4_ROUTER", "<B>Temporal &middot; ROUTER</B><BR/>&Delta;=+0.013 &middot; p=0.87<BR/>N=150 &middot; <B>done</B>",                       C_DONE, E_DONE),
        ("B4_REVOKE", "<B>Commit timing &middot; REVOKE</B><BR/>&Delta;<SUB>REVOKE</SUB> &middot; p<SUB>REVOKE</SUB><BR/>N=150 &middot; <B>running now</B>",       C_QUEUE, E_QUEUE),
        ("B4_CRV",    "<B>Forced commit &middot; CRV</B><BR/>&Delta;<SUB>CRV</SUB> &middot; p<SUB>CRV</SUB><BR/>N=150 &middot; queued",          C_QUEUE, E_QUEUE),
        ("B4_MUAP",   "<B>Temporal &middot; MUAP</B><BR/>&Delta;<SUB>MUAP</SUB> &middot; p<SUB>MUAP</SUB><BR/>N=150 &middot; queued",            C_QUEUE, E_QUEUE),
        ("B4_FUSE",   "<B>Multi-view &middot; FUSE</B><BR/>&Delta;<SUB>FUSE</SUB> &middot; p<SUB>FUSE</SUB><BR/>N=150 &middot; queued",          C_QUEUE, E_QUEUE),
        ("B4_DXCV",   "<B>Multimodal probe &middot; DXCV</B><BR/>smoke n=10 SR=0.30 (works)<BR/>&Delta;<SUB>DXCV</SUB> N=150 queued",            C_QUEUE, E_QUEUE),
    ]:
        node(g, nid, "<" + txt + ">", fill, edge, fontsize="9")
        g.edge("B4", nid)

    # =============== Branch 5: Mechanism & Negative Result ===============
    node(g, "B5", "<<B>5. Mechanism &amp; Negative Result</B><BR/><I>Sec.5 + Fig.4</I>>",
         C_DONE, E_DONE, fontsize="11")
    g.edge("ROOT", "B5")
    for nid, txt in [
        ("B5a", "<B>Decomposition</B><BR/>SR = reach &minus; commit/stop loss"),
        ("B5b", "reach ~0.62, loss ~0.25<BR/><B>INVARIANT</B> across B0/RES/M1g"),
        ("B5c", "<B>Decoupling</B>: r at commit<BR/>0.39 &rarr; 0.89 (RES)<BR/>but SR moves only +0.047"),
        ("B5d", "&rArr; every <B>deployable</B><BR/>family caps at ~0<BR/>only non-deployable UB is sig"),
        ("B5e", "&rArr; \"irreducible\"<BR/>commitment error<BR/>at test time"),
    ]:
        node(g, nid, "<" + txt + ">", C_DONE, E_DONE, fontsize="9")
        g.edge("B5", nid)

    # =============== Branch 6: Two Roads Forward ===============
    node(g, "B6", "<<B>6. Two Roads Forward</B><BR/><I>Conclusion + App.F</I>>",
         C_DONE, E_DONE, fontsize="11")
    g.edge("ROOT", "B6")
    for nid, txt, fill, edge in [
        ("B6_R1",     "<B>Road 1</B><BR/>degradation-aware<BR/>training &amp; adaptation<BR/><I>ADVOCATED &middot; not tested yet</I>",   C_DONE, E_DONE),
        ("B6_R2",     "<B>Road 2</B><BR/>multimodal sensing<BR/>(<B>learned</B> RGB + depth fusion)<BR/><I>DXCV is the probe, not the cure</I>", C_DONE, E_DONE),
    ]:
        node(g, nid, "<" + txt + ">", fill, edge, fontsize="9")
        g.edge("B6", nid)

    # =============== Branch 7: TO DO ===============
    node(g, "B7", "<<B>7. TO DO &mdash; Make-or-Break Extensions</B>>",
         C_PLAN, E_PLAN, fontsize="11")
    g.edge("ROOT", "B7")
    for nid, txt in [
        ("B7_R1",    "<B>Road 1 pilot</B>  <FONT COLOR='#0D47A1'>(HIGHEST PRIORITY)</FONT><BR/>"
                     "GLEE-SwinL head-only fine-tune<BR/>"
                     "freeze backbone &middot; ~3h train<BR/>"
                     "self-supervised: GLEE(clean) as label<BR/>"
                     "loss: KL(cls) + BCE(mask) + L1(box)<BR/>"
                     "then N=150 paired eval (~12h)"),
        ("B7_VLFM",  "<B>VLFM second backbone</B><BR/>50 episode paired<BR/>low-light sev 4 seed 0<BR/>~5h &middot; closes single-backbone W1"),
        ("B7_OG",    "<B>OracleGate qualitative</B><BR/>20 oracle-rescued episodes<BR/>verify restoration recovers conf<BR/>closes W4 (\"lucky routing\")"),
        ("B7_MB",    "<B>Motion-blur at N=300</B><BR/>co-report at main split<BR/>closes single-degradation gap"),
        ("B7_FB",    "<B>Plan B / Plan C</B> (fallback if R1 head fails)<BR/>B: LoRA rank-32 on last 2 layers (~5M params)<BR/>C: feature-restoration MLP adapter"),
    ]:
        node(g, nid, "<" + txt + ">", C_PLAN, E_PLAN, fontsize="9")
        g.edge("B7", nid)

    # =============== Branch 8: Contributions & Release ===============
    node(g, "B8", "<<B>8. Contributions &amp; Release Artefacts</B>>",
         C_ART, E_ART, fontsize="11")
    g.edge("ROOT", "B8")
    for nid, txt in [
        ("B8_c1", "<B>(i)</B>  benchmark + paired protocol<BR/>(McNemar exact &middot; bootstrap CI &middot; blind r)"),
        ("B8_c2", "<B>(ii)</B>  method-indep. mechanism diagnosis<BR/>at N=300 &middot; reframe failure<BR/>as irreversible commitment"),
        ("B8_c3", "<B>(iii)</B>  negative-result sweep over six families<BR/>only non-deployable UB is significant<BR/>(+0.057)"),
        ("B8_c4", "<B>(iv)</B>  methodological sign-flip warning<BR/>N=40 vs N=150 on Gaussian noise"),
        ("B8_rel","<B>Released</B>: corrupted HM3D configs &middot;<BR/>paired protocol &middot; per-episode CSVs &middot;<BR/>9 arm monkey-patches &middot; blind-r code &middot;<BR/>make_figures.py + bib"),
    ]:
        node(g, nid, "<" + txt + ">", C_ART, E_ART, fontsize="9")
        g.edge("B8", nid)

    # =============== Branch 9: Reviewer concerns + venue ===============
    node(g, "B9", "<<B>9. Reviewer Concerns &amp; Submission Strategy</B>>",
         C_RISK, E_RISK, fontsize="11")
    g.edge("ROOT", "B9")
    for nid, txt in [
        ("B9_W",   "<B>Current weaknesses</B><BR/>"
                   "W1 single backbone<BR/>"
                   "W2 5 placeholders<BR/>"
                   "W3 \"irreducible\" claim too strong<BR/>"
                   "W4 OracleGate sig &middot; W5 Road-1 untested<BR/>"
                   "&rArr; rating now: <B>5 borderline reject</B>"),
        ("B9_F",   "<B>After fixes</B> (Road-1 pilot + VLFM + qual.)<BR/>"
                   "Soundness 3/4 &rarr; 4/4<BR/>"
                   "Contribution 2/4 &rarr; 3/4<BR/>"
                   "Overall &rarr; <B>6&ndash;7 (weak accept &rarr; accept)</B>"),
        ("B9_V1",  "<B>PRIMARY:</B> NeurIPS 27 D&amp;B Track<BR/>after fixes ~65&ndash;75% accept"),
        ("B9_V2",  "<B>FALLBACK:</B><BR/>CoRL 27  ~50&ndash;60%<BR/>CVPR 27 / ICCV 27  ~30&ndash;40%<BR/>AAAI 27  ~35&ndash;45%"),
    ]:
        node(g, nid, "<" + txt + ">", C_RISK, E_RISK, fontsize="9")
        g.edge("B9", nid)

    # ----- Colour legend (an isolated subgraph, not attached to ROOT) -----
    with g.subgraph(name="cluster_legend") as lg:
        lg.attr(label="Legend / Status", style="rounded", color="gray60",
                fontname="Helvetica", fontsize="12", labeljust="l")
        lg.attr("node", shape="box", style="rounded,filled",
                fontname="Helvetica", fontsize="10",
                margin="0.10,0.05", penwidth="1.0")
        lg.node("LG1", "Done / in paper",       fillcolor=C_DONE,  color=E_DONE)
        lg.node("LG2", "Running (N=150 queue)", fillcolor=C_QUEUE, color=E_QUEUE)
        lg.node("LG3", "Planned next",          fillcolor=C_PLAN,  color=E_PLAN)
        lg.node("LG4", "Reviewer concern",      fillcolor=C_RISK,  color=E_RISK)
        lg.node("LG5", "Paper artefact",        fillcolor=C_ART,   color=E_ART)
        # invisible chain so they line up horizontally
        lg.edge("LG1", "LG2", style="invis")
        lg.edge("LG2", "LG3", style="invis")
        lg.edge("LG3", "LG4", style="invis")
        lg.edge("LG4", "LG5", style="invis")
        lg.attr(rank="same")

    # write file
    out = "/root/VLA_papers/AAAI_DRPhysNav/figures/fig_mindmap"
    g.render(out, format="pdf", cleanup=True)
    print(f"wrote {out}.pdf")


if __name__ == "__main__":
    main()
