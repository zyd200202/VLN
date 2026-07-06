#!/bin/bash
# ============================================================================
# SMOKE TEST (per 方案蓝图 重定位): does the *active* decision layer that targets
# the DOMINANT failure mode (missed/false-stop perception under degradation) beat B0?
#   - B0sm : InstructNav vanilla (baseline)              -- paired control
#   - MUAP : THF (reliability-adaptive detection thresh + temporal hysteresis)
#            + UAP-reobserve (commitment deferral on STOP)  -- the redesigned method
# Same config/seed/N -> habitat samples IDENTICAL episodes -> paired McNemar.
# low_light, sev4, seed0, N=40 (quick directional check before full N=300).
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/smoke_thf_uap
LOG="$OUTDIR/smoke.log"
mkdir -p "$OUTDIR"
EPISODES="${SMOKE_EPISODES:-40}"
SEV=4
SPLIT=val

export DRPN_WANDB=1
export DRPN_WANDB_PROJECT=DRPhysNav
export DRPN_WANDB_GROUP=smoke_thf_uap
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb
mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_DEGRADE_SEVERITY=$SEV
  export DRPN_MOTIV_LOG=1
}
en_thf() {  # reliability-adaptive detection: lower GLEE thresh under low r + temporal hysteresis
  export DRPN_USE_THF=1 DRPN_THF_LOW_CONF=0.12 DRPN_THF_BASE_CONF=0.25 \
         DRPN_THF_REL_GATE=0.9 DRPN_THF_WINDOW=8 DRPN_THF_MIN_FRAMES=3 DRPN_THF_MATCH_DIST=0.6
}
en_uap() {  # commitment deferral: defer irreversible STOP until reliability-weighted evidence accrues
  export DRPN_USE_UAP=1 DRPN_UAP_MODE=reobserve DRPN_UAP_REL_GATE=0.9 \
         DRPN_UAP_COMMIT_TAU=0.9 DRPN_UAP_MAX_REOBS=3 DRPN_UAP_CONF_MARGIN=0.35
}

run_one() {  # args: arm cond seed
  local ARM="$1" COND="$2" SEED="$3"
  base_env
  export DRPN_SEED="$SEED"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    B0sm) : ;;                       # baseline
    MUAP) en_thf; en_uap ;;          # redesigned active decision layer
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
  "B0sm low_light 0"
  "MUAP low_light 0"
)
echo "############ SMOKE THF+UAP START $(date) split=$SPLIT N=$EPISODES sev=$SEV ############" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do
  run_one $item
done
echo "ALL_DONE smoke_thf_uap $(date)" | tee -a "$LOG"
