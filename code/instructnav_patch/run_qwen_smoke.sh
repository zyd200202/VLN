#!/bin/bash
# Smoke test: InstructNav + Qwen2-VL-7B as VLM backbone, low_light s4, 2 episodes.
# Verifies end-to-end (sim + GLEE + Qwen) pipeline before launching the N=150 cross-backbone sweep.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- VLM backbone toggle ---
export DRPN_BACKBONE=qwen2vl
export QWEN2VL_MODEL=/root/autodl-tmp/models/Qwen2-VL-7B-Instruct
export QWEN2VL_MAX_NEW=768

# --- Match maintable B0 baseline arm (no restore, no DDA, no router) ---
export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
export DRPN_USE_FUSE=0 DRPN_FUSE_ORACLE=0 DRPN_RESTORER=""
export DRPN_MOTIV_LOG=1 DRPN_SEED=0 DRPN_MIX=0

export DRPN_DEGRADE_TYPE=low_light
export DRPN_DEGRADE_SEVERITY=4

# Disable wandb for smoke
export DRPN_WANDB=0

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/qwen_smoke
LOG="$OUTDIR/qwen_smoke.log"
mkdir -p "$OUTDIR"

echo "===== QWEN SMOKE START $(date) =====" | tee "$LOG"
python objnav_benchmark.py \
  --split val \
  --eval_episodes 2 \
  --out_csv "$OUTDIR/qwen_smoke.csv" \
  --save_traj 0 2>&1 | tee -a "$LOG"
echo "===== QWEN SMOKE DONE $(date) =====" | tee -a "$LOG"
