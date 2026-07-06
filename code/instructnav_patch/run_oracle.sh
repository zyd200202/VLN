#!/bin/bash
# ORACLE upper-bound experiments (settle root cause A vs B for the gating story). Paired, seed0, N=40.
#   CLEANCEIL : degradation OFF -> perfect-restoration ceiling. Total headroom = SR(clean) - SR(B0).
#   ORACLEGATE: low_light sev4, per-frame pick deg/res using the CLEAN frame as judge -> ceiling of
#               ANY per-frame deg/res gate. If ORACLEGATE <= RES, per-frame gating is a dead end.
# Compare against B0sm=0.350 and RES=0.450 (already measured, same seed/episodes).
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/oracle
LOG="$OUTDIR/oracle.log"; mkdir -p "$OUTDIR"
EPISODES="${ORACLE_EPISODES:-40}"; SEV=4; SPLIT=val
export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=oracle
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_MOTIV_LOG=1
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG DRPN_TARG_ORACLE
}
run_one() {
  local ARM="$1" COND="$2" SEED="$3"; base_env
  export DRPN_SEED="$SEED" DRPN_DEGRADE_SEVERITY=$SEV
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    CLEANCEIL)  : ;;                                              # degradation off (COND=clean)
    ORACLEGATE) export DRPN_USE_TARG=1 DRPN_TARG_ORACLE=1 ;;      # clean-judged per-frame deg/res gate
    *) echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/or_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$EPISODES" ]; then echo "SKIP $(basename $CSV)"|tee -a "$LOG"; return; fi
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
# ORACLEGATE first: it is THE critical negative result (pairs with B0@300=0.370 / RES@300=0.410).
QUEUE=("ORACLEGATE low_light 0" "CLEANCEIL clean 0")
echo "########## ORACLE UPPER-BOUND START $(date) N=$EPISODES ##########" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do run_one $item; done
echo "ALL_DONE oracle $(date)" | tee -a "$LOG"
