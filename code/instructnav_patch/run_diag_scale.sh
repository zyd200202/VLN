#!/bin/bash
# ============================================================================
# DIAGNOSIS @ SCALE -- cross-type process-data top-up for the "true cause" argument.
# Goal: reproduce the commitment/reachability decomposition (M2/M3/M4) at val N=300
# for the degradation types we currently lack at scale, so the mechanism claim
# ("failure = irreversible low-reliability commitment; bottleneck = reach->success
#  conversion, not exploration") generalizes across ALL types, not just low_light.
#
# Already have @scale (>=150, do NOT rerun): low_light s4 @300 (B0/RES/M1g),
#   gaussian_noise s4 @150 (B0/RES/ROUTER). clean@300 + ORACLEGATE@300 come from run_oracle.sh.
# Gaps filled here: fog s4 @300, motion_blur s4 @300, gaussian_noise s4 @300 (upgrade 150->300).
#
# Pure B0 (bare InstructNav) + DRPN_MOTIV_LOG=1 (per-step r/dist/found/action). No method arms.
# Chains AFTER run_oracle.sh (waits for "ALL_DONE oracle") so it never contends for GPU.
# Skip-aware, seeded -> same val episodes/order as redesign_n300 within a seed (paired-comparable).
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/diag_scale
LOG="$OUTDIR/diag_scale.log"
mkdir -p "$OUTDIR"
EPISODES="${DIAG_EPISODES:-300}"
SEV=4
SPLIT=val
ORACLE_LOG=/root/autodl-tmp/DRPhysNav/runs/oracle/oracle.log

export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=diag_scale
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"

base_env() {  # bare InstructNav, every method OFF, per-step process log ON
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0
  export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG DRPN_TARG_ORACLE
}

run_one() {  # args: cond seed
  local COND="$1" SEED="$2"; base_env
  export DRPN_SEED="$SEED"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  export DRPN_WANDB_ARM="B0" DRPN_WANDB_NAME="B0_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/ds_B0_${COND}_s${SEV}_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$EPISODES" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [B0] $COND sev$SEV seed$SEED N=$EPISODES -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$EPISODES" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

# ---- chain: do not start until the running oracle job FULLY exits (shared GPU) ----
# NB: do NOT grep oracle.log for "ALL_DONE oracle" -- a stale marker from a prior N=40
# oracle run is already in the log and would trigger immediately. Instead, block while the
# run_oracle.sh process (and any objnav_benchmark on the oracle CSV) is still alive.
echo "[chain] waiting for run_oracle.sh AND run_commit_layer.sh to exit ... $(date)" | tee -a "$LOG"
while pgrep -f "run_oracle.sh" >/dev/null 2>&1 \
   || pgrep -f "run_commit_layer.sh" >/dev/null 2>&1 \
   || pgrep -f "or_ORACLEGATE_low_light_s4_seed0.csv" >/dev/null 2>&1 \
   || pgrep -f "or_CLEANCEIL_clean_s4_seed0.csv" >/dev/null 2>&1 \
   || pgrep -f "cl_MUAP_low_light_s4_seed0.csv" >/dev/null 2>&1; do
  sleep 120
done
echo "[chain] oracle + commit_layer fully exited -> starting diag_scale $(date)" | tee -a "$LOG"

# Priority: fog (real degradation, total gap) -> motion_blur (gap, only n=32) -> noise upgrade 150->300.
QUEUE=(
  "fog 0"
  "motion_blur 0"
  "gaussian_noise 0"
)
echo "############ DIAG-SCALE START $(date) split=$SPLIT N=$EPISODES sev=$SEV ############" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do run_one $item; done
echo "ALL_DONE diag_scale $(date)" | tee -a "$LOG"
