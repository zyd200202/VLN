#!/bin/bash
# SMOKE: TARG-v2 (default-restore gate) vs degraded baseline (B0sm), paired same-seed.
# Diagnosis of TARG-v1: margin=0.05 on a tiny utility (U~0.13) over-rejected -> only 13% restored
# adopted, even though U_res>U_deg in 82% of episodes -> TARG-v1 collapsed toward the degraded baseline.
# TARG-v2 flips the default: KEEP the restored frame unless it demonstrably HURTS (U_res < U_deg - margin),
# with margin=0. Expect adoption ~50-80% and SR to climb toward RES (0.450). Confirms failure was the
# conservative gate, not the method.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/smoke_targ_v2
LOG="$OUTDIR/smoke.log"; mkdir -p "$OUTDIR"
EPISODES="${SMOKE_EPISODES:-40}"; SEV=4; SPLIT=val
export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=smoke_targ_v2
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG
}
run_one() {
  local ARM="$1" COND="$2" SEED="$3"; base_env
  export DRPN_SEED="$SEED"; export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    TARGv2) export DRPN_USE_TARG=1 DRPN_TARG_DEFAULT=res DRPN_TARG_MARGIN=0.0 DRPN_TARG_LOG=1 ;;
    *) echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/smoke_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
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
# Reuse existing B0sm baseline for paired analysis.
cp -n /root/autodl-tmp/DRPhysNav/runs/smoke_thf_uap/smoke_B0sm_low_light_s4_seed0.csv \
      "$OUTDIR/smoke_B0sm_low_light_s4_seed0.csv" 2>/dev/null
QUEUE=("TARGv2 low_light 0")
echo "########## SMOKE TARG-v2 START $(date) N=$EPISODES ##########" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do run_one $item; done
echo "ALL_DONE smoke_targ_v2 $(date)" | tee -a "$LOG"
