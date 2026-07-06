#!/usr/bin/env bash
# Regenerate every figure PDF used in the paper from the shipped
# data/ artefacts. Requires only a matplotlib + numpy + PyMuPDF stack;
# no simulator or GPU needed.
#
# The figure scripts hard-code the data paths under
# /root/autodl-tmp/DRPhysNav/runs/ from our authoring machine; the
# symlink below transparently redirects those paths to the shipped
# data/ tree.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Redirect legacy authoring paths -> shipped data
DRP=/root/autodl-tmp/DRPhysNav/runs
if [[ ! -e "$DRP" ]]; then
    sudo mkdir -p "$(dirname "$DRP")"
    sudo ln -sfn "$REPO/data/per_episode_csv" "$DRP"
    # figure scripts also read motiv_jsonl by campaign name; symlink each
    for camp in oracle redesign_n300; do
        for f in "$REPO/data/motiv_jsonl"/*.jsonl; do
            base=$(basename "$f")
            case "$base" in
                or_*)   sudo ln -sfn "$f" "$DRP/oracle/$base" ;;
                n300_*) sudo ln -sfn "$f" "$DRP/redesign_n300/$base" ;;
            esac
        done
    done
    # vignette bundle
    sudo ln -sfn "$REPO/data/vignette/traj_maps" "$DRP/traj_vignette/traj_maps"
    sudo ln -sfn "$REPO/data/vignette/topdown_rgb" "$DRP/traj_vignette/topdown_rgb"
    for f in "$REPO/data/vignette"/tv_*.jsonl; do
        sudo ln -sfn "$f" "$DRP/traj_vignette/$(basename "$f")"
    done
fi

cd "$REPO/code/figures"
python make_figures.py
python make_more_figures.py
python make_case_studies.py
python make_trajectories.py
python make_topdown_traj.py
python make_teaser.py

echo
echo "Figures written under code/figures/*.pdf"
ls *.pdf | sed 's/^/  /'
