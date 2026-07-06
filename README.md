# DRPhysNav — Restoring the Image Does Not Restore Navigation

Code, experiment logs and paper source for the AAAI submission:

> **Restoring the Image Does Not Restore Navigation:
> Diagnosing Degradation-Robust Object-Goal Navigation as a Commitment Problem**

We diagnose why VLM-driven ObjectNav agents fail under visual
degradation: the agent still *reaches* the goal but never *commits* to
it. A six-family test-time intervention sweep confirms every deployable
patch is either non-significant or significantly negative, and a first
training-side pilot (Road 1) already matches the best deployable arm
while measurably re-aligning degraded features.

---

## Repository layout

```
paper/                  Overleaf-ready LaTeX source + compiled PDF
├── main.tex, sections/, figures/, references.bib, aaai25.sty, aaai25.bst
├── main.pdf            15-page compiled paper
└── README.md           Overleaf import instructions

code/
├── instructnav_patch/  our modifications to InstructNav (Python only)
│   ├── objnav_benchmark.py    driver with paired protocol + DRPN_TRAJ_LOG
│   ├── promptir_restore.py    PromptIR wrapper (RES arm)
│   ├── render_topdown_rgb.py  bird's-eye scene renders for Figs. 6 & 9
│   ├── llm_utils/qwen_backend.py   Qwen2-VL-7B local backend
│   ├── llm_utils/road1_adapter.py  Road-1 adapter monkey-patch
│   └── run_*.sh                    per-arm launchers
├── road1_adapter/      Road-1 feature adapter training
│   ├── stage_a_gen_data.py   (clean, degraded) SwinL feature pairs
│   ├── stage_b_train_head.py 2-layer residual conv adapter, L1 loss
│   ├── train.log              full training log (13.5% L1 reduction)
│   └── PLAN.md
└── figures/            all figure generators
    ├── make_teaser.py, make_figures.py, make_more_figures.py
    ├── make_case_studies.py, make_trajectories.py, make_topdown_traj.py
    ├── paper_style.py         palette / rcParams
    └── fill_tables.py, fill_road1.py  autogen LaTeX tables from CSVs

data/
├── per_episode_csv/    every SR/SPL/DTG number behind Tables 1 & 2
│   ├── maintable/, redesign_n300/, oracle/    N=300 main split
│   ├── sweep_signmap/, unified/, crv/         N=150 sweep arms
│   ├── traj_vignette/                         N=12 vignette runs
│   └── road1_eval/                            Road-1 held-out eval
├── motiv_jsonl/        step-level logs used by Figs. 2, 5, 7
└── vignette/           12-episode trajectory bundle for Figs. 6 & 9
    ├── tv_{B0,RES,ORACLEGATE}_*_traj.jsonl    world+grid trajectories
    ├── tv_*_motiv.jsonl                       per-step reliability r
    ├── traj_maps/*.npy                        Habitat top-down occupancy
    └── topdown_rgb/                           photoreal orthographic renders
                                               (bird's-eye, ceiling clipped)

checkpoints/
└── adapter_low_light_s4.pth   Road-1 head adapter (~10⁶ params, 24 MB)

scripts/
└── reproduce_figures.sh       one-command figure regeneration
```

## What is *not* included (and why)

- **HM3D scenes** (~30 GB, license-restricted): download from
  [habitat-matterport-3d-dataset](https://github.com/facebookresearch/habitat-matterport3d-dataset).
- **PromptIR / Qwen2-VL-7B / SwinL weights** (~40 GB): pull from the
  respective upstream repos.
- **`.pyc` files that ship with InstructNav's upstream release**: we
  include *only our own* Python patches; run against a clean InstructNav
  checkout and drop the files in `code/instructnav_patch/` on top.
- **Bulky per-arm run artefacts** (~15 GB of intermediate JSONL, top-down
  maps for the full N=300 grid, and rejected experiments):
  contact us if you need the raw dumps.

## Reproducing the paper

**Compile the PDF (Overleaf or local pdflatex):**

```
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Regenerate every figure from the shipped data:**

```
bash scripts/reproduce_figures.sh
```

Each `data/*.csv` is the source of truth for the corresponding table
cell; each figure PDF depends only on files under `data/`. See the
docstring at the top of every script for exact provenance.

**Re-run the Road-1 pilot** (needs Habitat + HM3D val + SwinL):

```
cd code/road1_adapter
python stage_a_gen_data.py --split val --corruption low_light --severity 4
python stage_b_train_head.py --steps 2000 --lr 1e-4
```

**Re-run any sweep arm** (needs the full InstructNav stack):

```
cd code/instructnav_patch
bash run_crv150_res.sh          # e.g. the RES paired baseline at N=150
```

## Statistical protocol

Paired McNemar exact tests on shared episodes; 95% bootstrap CIs with
B=5000 over paired episode differences; α=0.05. Every ΔSR / p-value /
CI in the paper is computed from `data/per_episode_csv/` by the scripts
in `code/figures/` — nothing is hand-entered.

## Citation

Redacted for anonymous review. A citation will be added once the paper
is de-anonymised.

## License

MIT (see [`LICENSE`](LICENSE)).
