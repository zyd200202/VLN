#!/bin/bash
# ============================================================================
# REDESIGN v3 -- N=300 pre-registered re-run (per AeroDream-Nav_方案蓝图.xlsx §10c/§11).
# Locked: N=300, --split val (np.random.choice seeded by DRPN_SEED => same 300 eps, same order,
# across all arms within a seed -> clean paired McNemar). greedy decoding. wandb online (process data).
# Phase 1 (headline): B0/M1g x low_light x seed{0,1}.
# Phase 2 (generalization + break-circularity): B0/M1g x fog x s0 ; B0 x clean x s0.
# Skip-aware. DO NOT change design mid-run (pre-registration).
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
LOG="$OUTDIR/redesign_n300.log"
mkdir -p "$OUTDIR"
EPISODES="${N300_EPISODES:-300}"
SEV=4
SPLIT=val

# ---- wandb: online process logging on every arm (key already in ~/.netrc) ----
export DRPN_WANDB=1
export DRPN_WANDB_PROJECT=DRPhysNav
export DRPN_WANDB_GROUP=redesign_n300
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb
mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_DEGRADE_SEVERITY=$SEV
  # per-step process data (r / geodesic dist / found / action) for wandb on EVERY arm (policy no-op)
  export DRPN_MOTIV_LOG=1
}
en_uexp() { export DRPN_USE_UEXP=1 DRPN_UEXP_ALPHA=1.0 DRPN_UEXP_MAX=0.6; }

run_one() {  # args: arm cond seed
  local ARM="$1" COND="$2" SEED="$3"
  base_env
  export DRPN_SEED="$SEED"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    B0)  ;;                       # bare InstructNav (control)
    M1g) en_uexp ;;               # module 1, reliability-gated value-map reweighting (main core)
    *)   echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
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
  # M1g arms CANCELLED 2026-06-16: reliability-gated reweighting judged dead
  # (N=300 SR=0.410 vs B0=0.370, +0.04 ΔSR, p~=0.20 -> not significant). Kept as negative result only.
  # B0 baselines retained (still needed as the control for the TARG comparison).
  # ===== Phase 1: baselines (low_light, seed 0 & 1) =====
  "B0 low_light 0"   # "M1g low_light 0"  (cancelled)
  "B0 low_light 1"   # "M1g low_light 1"  (cancelled)
  # ===== Phase 2: generalization (fog) + break-circularity (clean r distribution) =====
  "B0 fog 0"         # "M1g fog 0"        (cancelled)
  "B0 clean 0"
)
echo "############ REDESIGN N=300 START $(date) split=$SPLIT N=$EPISODES sev=$SEV ############" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do
  run_one $item
done
echo "ALL_DONE redesign_n300 $(date)" | tee -a "$LOG"
