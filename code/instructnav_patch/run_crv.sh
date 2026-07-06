#!/bin/bash
# ============================================================================
# CRV -- Close-Range recall-triggered Commit. The ONE diagnosis-aligned lever
# nobody has pulled: the commit is VLM-gated (found_goal = chainon_answer['Flag']),
# and the GLEE threshold is ALREADY reliability-adaptive (so THF/MUAP capped at +0.04).
# Failure decomp (motiv): ~60-70% of near-miss failures = agent <1.5m from target but
# VLM won't flag 'found' on the degraded frame -> a RECALL failure at the commit layer.
# CRV forces Flag=True when a low-thresh GLEE pass sees the TARGET class with a LARGE mask
# (close) PERSISTING M frames under low reliability. Precision guards: target-class only,
# mask-area proximity, M-frame vote, conf floor, r<gate (clean no-op).
#
# Proven where there is headroom: gaussian_noise s4 (clean 0.437 vs B0 0.220 -> 0.217 room;
# 55/150 episodes reach <=1m yet fail, 33 of them recall-type). Paired B0 control reused from
# maintable/mt_B0_gaussian_noise_s4_seed0.csv (same split/seed/episode-order -> head-N identical).
# Fail-fast: smoke@40 MUST fire (see [CRV] counters); CRV@150 must be paired-positive before N=300.
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/crv
LOG="$OUTDIR/crv.log"
mkdir -p "$OUTDIR"
SEV=4; SPLIT=val; SEED=0; COND=gaussian_noise

export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=crv
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0 DRPN_USE_REVOKE=0
  export DRPN_USE_CRV=0
  export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG DRPN_TARG_ORACLE
}
en_restore() { export DRPN_USE_RESTORE=1; }
en_crv() { export DRPN_USE_CRV=1 DRPN_CRV_REL_GATE=0.85 DRPN_CRV_CONF=0.10 \
                  DRPN_CRV_FLOOR=0.15 DRPN_CRV_AREA=9000 DRPN_CRV_FRAMES=2 DRPN_CRV_MAX=3; }

run_one() {  # args: arm N
  local ARM="$1" EPISODES="$2"; base_env
  export DRPN_SEED="$SEED" DRPN_DEGRADE_TYPE="$COND"
  case "$ARM" in
    B0)     : ;;
    CRV)    en_crv ;;
    RESCRV) en_restore; en_crv ;;
    RES)    en_restore ;;
    *) echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/crv_${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$EPISODES" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$EPISODES -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$EPISODES" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  grep -E "\[CRV\]" "$LOG" | tail -n 4
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

# ---- chain: wait until any current shared-GPU job (diag_scale / oracle / commit_layer) exits ----
echo "[chain] waiting for diag_scale/oracle/commit_layer + their python to exit ... $(date)" | tee -a "$LOG"
while pgrep -f "run_diag_scale.sh" >/dev/null 2>&1 || pgrep -f "run_oracle.sh" >/dev/null 2>&1 || pgrep -f "run_commit_layer.sh" >/dev/null 2>&1 || pgrep -f "objnav_benchmark.py" >/dev/null 2>&1; do
  sleep 120
done
echo "[chain] GPU free -> starting CRV $(date)" | tee -a "$LOG"

# Fail-fast: smoke MUST fire; then paired @150 vs maintable B0; combined RES+CRV last.
QUEUE=(
  "CRV 40"      # smoke: assert it triggers (see [CRV] forced_commits) + direction
  "CRV 150"     # paired vs maintable/mt_B0_gaussian_noise_s4_seed0.csv (head-150 identical)
  "RESCRV 150"  # repair + commit-recovery combined (the 'fix + decision together' test)
)
echo "############ CRV START $(date) split=$SPLIT cond=$COND sev=$SEV ############" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do run_one $item; done
echo "ALL_DONE crv $(date)" | tee -a "$LOG"
