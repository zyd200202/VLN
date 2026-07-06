#!/bin/bash
# One-shot blind-feature measurement: dump (luma/median/p10/darkfrac/sigma/lap/dark) per frame for
# each degradation type, so router thresholds are set from REAL data instead of guesses.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/root/autodl-tmp/DRPhysNav/runs/rodump; mkdir -p "$OUT"; LOG="$OUT/dump.log"; : > "$LOG"
export DRPN_WANDB=0 DRPN_MIX=0 DRPN_SEED=0
export DRPN_USE_ROUTER=1 DRPN_ROUTER_DUMP=1 DRPN_ROUTER_LOG=0
# disable fusion to keep it fast (we only need _ro_blind features)
export DRPN_USE_FUSE=0 DRPN_RESTORER=""
for COND in low_light:4 gaussian_noise:4 motion_blur:4 fog:4 low_light:1 low_light:2; do
  T=${COND%%:*}; S=${COND##*:}
  echo "########## RODUMP $T s$S ##########" | tee -a "$LOG"
  export DRPN_DEGRADE_TYPE="$T" DRPN_DEGRADE_SEVERITY="$S"
  python objnav_benchmark.py --split val --eval_episodes 1 \
      --out_csv "$OUT/rod_${T}_s${S}.csv" --save_traj 0 2>&1 | grep -aE "RODUMP" | head -40 | tee -a "$LOG"
done
echo "########## RODUMP DONE ##########" | tee -a "$LOG"
