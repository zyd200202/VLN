"""Paper-wide figure style for DRPhysNav.

Import at the top of every figure generator:

    from paper_style import *
    apply_style()

This locks the palette + typography + rcParams so all figures share one
visual language. Colors are *semantic*: use C_BASE only for baseline/clean,
C_DEGR only for degraded/negative, etc.
"""
from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------- palette
# Semantic paper palette (Nature-inspired, high-contrast, WCAG-safe).
# Each color has a fixed meaning; do not use for other roles.
C_BASE   = "#3B547A"   # deep cool blue    -- baseline / clean / reference
C_DEGR   = "#C0392B"   # deep red          -- degraded / negative effect / harm
C_POS    = "#2A9D8F"   # deep teal         -- our-positive / restoration / success
C_ORACLE = "#8E44AD"   # deep purple       -- non-deployable ceiling (OracleGate, UB)
C_NEUT   = "#7F8C8D"   # neutral gray      -- reference lines / non-significant
C_HL     = "#E9B44C"   # amber             -- annotation / callout / highlight

# Ancillary tints (use sparingly, only for fills behind primary color)
T_BASE   = "#DDE4F0"   # light tint of C_BASE
T_DEGR   = "#F4D9D6"   # light tint of C_DEGR
T_POS    = "#D3EFEA"   # light tint of C_POS
T_ORACLE = "#E9DBF2"   # light tint of C_ORACLE
T_NEUT   = "#ECEEEF"   # light tint of C_NEUT

# ---------------------------------------------------------------- typography
# Match paper body font (Times) exactly. Panel titles are short labels.
BASE_FONTSIZE = 8.5    # axis labels / annotations
TICK_FONTSIZE = 7.5    # axis tick labels
TITLE_FONTSIZE = 9.0   # panel titles (short label style: "(a) Reach vs. success")
LEGEND_FONTSIZE = 7.5


def apply_style():
    """Apply rcParams once at generator start."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # STIX pairs well with Times
        "axes.labelsize": BASE_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType, editable in Illustrator
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------- size presets
# AAAI-25 double column: single col = 3.35", double col = 6.75"
SIZE_DOUBLE_SHORT = (6.75, 2.15)   # figure* horizontal — teaser, motivation, forest
SIZE_DOUBLE_MED   = (6.75, 2.80)   # figure* — pipeline, positioning
SIZE_DOUBLE_TALL  = (6.75, 4.20)   # figure* two-row — motivation 4-panel
SIZE_SINGLE_SHORT = (3.35, 1.90)   # figure — calibration, small
SIZE_SINGLE_MED   = (3.35, 2.50)   # figure — decomp single panel
