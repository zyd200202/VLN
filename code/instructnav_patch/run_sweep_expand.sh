#!/bin/bash
# SWEEP-EXPAND (实验1: 补齐符号图谱 + 加N + 多seed). Per 蓝图『下一步·路由器实验计划』.
# Goal: (a) make per-cell benefit/harm judgments closer to significant (N=100, +seed1),
#       (b) locate the zero-crossing severity for low_light (add sev3),
#       (c) round out the TYPE axis at sev4 (add fog) to confirm type-dependent sign flip.
# Paired B0/RES per (type,severity,seed), same episodes. Skip-aware: stop anytime, the most
# decisive cells run FIRST. N=100 overwrites the earlier N=40 seed0 CSVs (N=100 supersedes).
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/sweep_signmap
LOG="$OUTDIR/sweep_expand.log"; mkdir -p "$OUTDIR"
EPISODES="${SWEEP_EPISODES:-100}"; SPLIT=val
export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=sweep_expand
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_MOTIV_LOG=1
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG DRPN_TARG_ORACLE
}
run_one() {  # args: arm cond sev seed
  local ARM="$1" COND="$2" SEV="$3" SEED="$4"; base_env
  export DRPN_SEED="$SEED" DRPN_DEGRADE_SEVERITY="$SEV"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    B0)  : ;;
    RES) export DRPN_USE_RESTORE=1 ;;
    *) echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  # seed0 keeps the legacy name (overwrites N=40); seed>=1 gets a seed-tagged name
  local CSV
  if [ "$SEED" = "0" ]; then CSV="$OUTDIR/sw_${ARM}_${COND}_s${SEV}_seed0.csv"
  else CSV="$OUTDIR/sw_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"; fi
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$EPISODES" ]; then echo "SKIP(done) $(basename $CSV)"|tee -a "$LOG"; return; fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$EPISODES ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$EPISODES" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}
# ===== QUEUE: most-decisive FIRST (stop anytime; earliest = most valuable) =====
QUEUE=(
  # -- Tier 1a: low_light flip-curve, seed0, N=100 (hero degradation; sev3 is NEW zero-crossing point)
  "B0 low_light 3 0"      "RES low_light 3 0"      # NEW: fills the gap between harmful sev1 and beneficial sev2/4
  "B0 low_light 1 0"      "RES low_light 1 0"      # upgrade strongest HARM signal to N=100
  "B0 low_light 4 0"      "RES low_light 4 0"      # upgrade strongest BENEFIT signal to N=100
  "B0 low_light 2 0"      "RES low_light 2 0"      # upgrade mid point to N=100
  # -- Tier 1b: TYPE axis at sev4, seed0, N=100 (confirm type-dependent harm + add fog)
  "B0 gaussian_noise 4 0" "RES gaussian_noise 4 0"
  "B0 motion_blur 4 0"    "RES motion_blur 4 0"
  "B0 fog 4 0"            "RES fog 4 0"            # NEW type (structure-preserving haze)
  # -- Tier 2: seed1 replication of the 4 anchor cells (multi-seed robustness)
  "B0 low_light 1 1"      "RES low_light 1 1"
  "B0 low_light 4 1"      "RES low_light 4 1"
  "B0 gaussian_noise 4 1" "RES gaussian_noise 4 1"
  "B0 motion_blur 4 1"    "RES motion_blur 4 1"
)
echo "########## SWEEP-EXPAND START $(date) N=$EPISODES ##########" | tee -a "$LOG"
for ((i=0;i<${#QUEUE[@]};i+=2)); do
  run_one ${QUEUE[i]}; run_one ${QUEUE[i+1]}
done
echo "ALL_DONE sweep_expand $(date)" | tee -a "$LOG"
