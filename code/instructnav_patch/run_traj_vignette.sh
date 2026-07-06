#!/bin/bash
# ============================================================================
# Top-down multi-arm trajectory vignette runs (for the paper's trajectory
# comparison figure). Paired N=12 on low_light s4, GPT-4o backbone (same as
# main table), arms:
#   B0         : degraded baseline, no restoration
#   RES        : always-on PromptIR restoration
#   ORACLEGATE : clean-judged per-frame deg/res gate (upper bound)
# DRPN_TRAJ_LOG=1 records per-step agent world position, goal positions and a
# top-down occupancy map per episode -> runs/traj_vignette/*_traj.jsonl.
# Expected wall time ~2-2.5 h total (~3.5 min/episode with GPT-4o).
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
export HOST="${HOST:-x86_64-conda_cos6-linux-gnu}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N="${TRAJ_EPISODES:-12}"; SPLIT=val; SEED=0
COND=low_light; SEV=4
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/traj_vignette
LOG="$OUTDIR/traj_vignette.log"
mkdir -p "$OUTDIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
  export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
  export DRPN_USE_FUSE=0 DRPN_FUSE_ORACLE=0 DRPN_RESTORER=""
  export DRPN_MOTIV_LOG=1 DRPN_TRAJ_LOG=1 DRPN_SEED=$SEED DRPN_MIX=0
  export DRPN_WANDB=0
  unset DRPN_BACKBONE DRPN_ROAD1_ADAPTER   # GPT-4o pipeline, no adapter
}

run_one() {
  local ARM="$1"; base_env
  export DRPN_DEGRADE_TYPE="$COND" DRPN_DEGRADE_SEVERITY="$SEV"
  case "$ARM" in
    B0)         : ;;
    RES)        export DRPN_USE_RESTORE=1 ;;
    ORACLEGATE) export DRPN_USE_TARG=1 DRPN_TARG_ORACLE=1 ;;
    *) echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  local CSV="$OUTDIR/tv_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then
    echo "SKIP $(basename $CSV)" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$N ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" \
      --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f"%(sys.argv[1].split('/')[-1],n,sr))
PY
}

echo "########## TRAJ VIGNETTE START $(date) N=$N ##########" | tee -a "$LOG"
for ARM in B0 RES ORACLEGATE; do run_one "$ARM"; done
echo "ALL_DONE traj_vignette $(date)" | tee -a "$LOG"
