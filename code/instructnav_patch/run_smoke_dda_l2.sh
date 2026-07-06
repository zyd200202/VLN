#!/bin/bash
# ============================================================================
# SMOKE TEST -- Reliability-gated Directional Decision Arbitration (DDA).
# Idea: under visual degradation, distrust the gpt4v (degraded-image) direction vote
# and pick direction from image-quality-independent cues (semantic memory + commonsense
# action + history geometry). Targets the DOMINANT failure (67% never get close).
#   - DDA : DRPN_USE_DDA=1  (uexp enabled with MAX=0 ONLY to trigger r computation; inert otherwise)
# Compared (paired) against the already-run B0sm (smoke_thf_uap, N=40, same config/episodes).
# low_light, sev4, seed0, N=40.
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/smoke_dda_l2
LOG="$OUTDIR/smoke.log"
mkdir -p "$OUTDIR"
EPISODES="${SMOKE_EPISODES:-40}"
SEV=4
SPLIT=val

export DRPN_WANDB=1
export DRPN_WANDB_PROJECT=DRPhysNav
export DRPN_WANDB_GROUP=smoke_dda_l2
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb
mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0
  export DRPN_DEGRADE_SEVERITY=$SEV
  export DRPN_MOTIV_LOG=1
}
en_dda() {
  # DDA active; uexp enabled with MAX=0/ALPHA=0 ONLY so the agent computes frame_reliability
  # (-> mapper.uap_reliability), which DDA reads. uexp itself contributes 0 (inert).
  export DRPN_USE_DDA=1 DRPN_DDA_LEVEL=2
  export DRPN_USE_UEXP=1 DRPN_UEXP_MAX=0.0 DRPN_UEXP_ALPHA=0.0
}

run_one() {  # args: arm cond seed
  local ARM="$1" COND="$2" SEED="$3"
  base_env
  export DRPN_SEED="$SEED"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    B0sm) : ;;
    DDA)  en_dda ;;
    *)    echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM"
  export DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/smoke_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$EPISODES" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$EPISODES -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$EPISODES" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s  n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

QUEUE=(
  "DDA low_light 0"
)
echo "############ SMOKE DDA-L2 START $(date) split=$SPLIT N=$EPISODES sev=$SEV ############" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do
  run_one $item
done
echo "ALL_DONE smoke_dda_l2 $(date)" | tee -a "$LOG"
