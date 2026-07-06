#!/bin/bash
# ============================================================================
# RESTORE PROBE (per 方案蓝图 §D/§E) -- test the V-path improvement.
# Arms: RES (USE_RESTORE only) and RESM1g (USE_RESTORE + uexp), low_light, seed0, N=300.
# Compared against already-done B0 (0.370) and M1g (0.410) on the SAME seed0 300-episode set
# (np.random.choice seeded by DRPN_SEED=0 -> identical episodes -> 4-way paired). greedy + wandb.
# Skip-aware. Pre-registered: N=300, sev4, low_light, seed0.
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/redesign_n300
LOG="$OUTDIR/restore_probe.log"
mkdir -p "$OUTDIR"
EPISODES="${N300_EPISODES:-300}"
SEV=4
SPLIT=val

export DRPN_WANDB=1
export DRPN_WANDB_PROJECT=DRPhysNav
export DRPN_WANDB_GROUP=restore_probe
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb
mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_DEGRADE_SEVERITY=$SEV
  export DRPN_MOTIV_LOG=1
}
en_uexp()    { export DRPN_USE_UEXP=1 DRPN_UEXP_ALPHA=1.0 DRPN_UEXP_MAX=0.6; }
en_restore() { export DRPN_USE_RESTORE=1; }

run_one() {  # args: arm cond seed
  local ARM="$1" COND="$2" SEED="$3"
  base_env
  export DRPN_SEED="$SEED"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    RES)    en_restore ;;            # V-path: blind restoration before perception
    RESM1g) en_restore; en_uexp ;;   # V+N: restoration + reliability-gated reweighting
    *)      echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM"
  export DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/n300_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
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
  "RES low_light 0"
  # "RESM1g low_light 0"   # CANCELLED 2026-06-16: V+N (restore+reweight) route judged dead
  #                         (M1g insignificant +0.04 p=0.20; combo has no working component). Kept as negative result only.
)
echo "############ RESTORE PROBE START $(date) split=$SPLIT N=$EPISODES sev=$SEV ############" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do
  run_one $item
done
echo "ALL_DONE restore_probe $(date)" | tee -a "$LOG"
