# DRPhysNav — Overleaf-ready source

Import this whole folder as an Overleaf project (New Project → Upload
Project → drop the ZIP of this directory).

## Compile settings on Overleaf

- Compiler: **pdfLaTeX**
- Main document: `main.tex`
- TeX Live: 2023 or newer (AAAI-25 style needs a recent `natbib`/`caption`)

Overleaf will run BibTeX automatically on the first compile. If the
References page comes up empty, click *Recompile* once more or set
Menu → Settings → Compiler → `pdflatex + bibtex + pdflatex + pdflatex`.

## Layout

```
main.tex                     top-level document + teaser \maketitle hook
references.bib               bibliography database (BibTeX)
main.bbl                     pre-built bibliography (delete if you want
                             Overleaf to regenerate from .bib)
aaai25.sty, aaai25.bst       AAAI-25 template (anonymous submission mode)

sections/
  00_abstract.tex            abstract
  10_intro.tex               introduction
  20_related.tex             related work
  30_motivation.tex          motivation / benchmark instrument
  40_method.tex              method / diagnosis
  50_experiments.tex         experiments (Tables 1, 2, Figures 5, 6, 7)
  60_conclusion.tex          conclusion + limitations
  A0_appendix.tex            appendix (Sections A–I)
  _sweep_block.tex           auto-generated Table 1 (intervention sweep)
  _cross_backbone_block.tex  auto-generated Table 2 (cross-corruption)
  _qwen_sweep_block.tex      appendix cross-corruption prose block
  _og_vignettes_block.tex    appendix OracleGate case vignettes table

figures/                     all 11 PDF figures cited from the .tex sources
  fig_teaser.pdf             Figure 1  (in title macro)
  fig_pipeline.pdf           Figure 2a (paper-side of merged fig)
  fig_corruption_gallery.pdf Figure 2b (corruption gallery)
  fig_motivation.pdf         Figure 3  (motivation / reach-vs-success)
  fig_case_studies.pdf       Figure 5  (episode-level traces)
  fig_trajectories.pdf       Figure 6  (top-down multi-arm trajectories)
  fig_decomp.pdf             Figure 7  (commitment bottleneck decomposition)
  fig_calibration.pdf        appendix   (reliability calibration)
  fig_topdown_traj.pdf       appendix   (extra topdown vignette)
  fig_road1.pdf              appendix J (Road 1 adapter pipeline)
  fig_roadmap.pdf            appendix   (forward directions roadmap)
```

## Reproducing this build locally

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Should produce a 15-page PDF (10-page main body + references + 4-page
appendix), no undefined citations or references.
