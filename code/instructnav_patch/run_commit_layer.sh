#!/bin/bash
# ============================================================================
# COMMIT-LAYER METHOD CHECK -- does the diagnosis-aligned lever survive at N=150?
# Diagnosis (N=300): bottleneck = commit/stop policy (reach 0.62 vs SR 0.36), NOT
# perception. Perception fixes (RES/M1g) capped at +0.04 because they never touched
# this layer. Literature (ConsistNav +8~11% same-stack; temporal-aggregation; 3DGSNav
# -32.8% w/o re-verify) shows the commit/stop executive HAS headroom.
#
# MUAP = THF (reliability-adaptive detection thresh + temporal hysteresis)
#      + UAP-reobserve (defer irreversible STOP until reliability-weighted evidence accrues).
# This is modules 1+2 (temporal aggregation + reliability gate). It EXISTS but was only
# smoke-tested at N=23 (showed -0.13, but N<=40 is unreliable per our own sign-flip lesson).
# HONEST: this is a real test, not a re-run of a dead method (M1g != MUAP; M1g was value-map
# reweighting). Module 3 (confirm-or-REVOKE recovery) is NOT here yet -> separate, after smoke.
#
# N=150, val, low_light s4, seed0, MOTIV on. B0 paired control = redesign_n300 B0 seed0
# (identical split/seed/episode-order -> head-150 are the SAME episodes; analyzed offline).
# Chains AFTER run_oracle.sh fully exits (shared GPU). Skip-aware.
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/commit_layer
LOG="$OUTDIR/commit_layer.log"
mkdir -p "$OUTDIR"
SEV=4; SPLIT=val; SEED=0; COND=low_light

export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=commit_layer
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0 DRPN_USE_REVOKE=0
  export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG DRPN_TARG_ORACLE
}
en_thf() { export DRPN_USE_THF=1 DRPN_THF_LOW_CONF=0.12 DRPN_THF_BASE_CONF=0.25 \
                  DRPN_THF_REL_GATE=0.9 DRPN_THF_WINDOW=8 DRPN_THF_MIN_FRAMES=3 DRPN_THF_MATCH_DIST=0.6; }
en_uap() { export DRPN_USE_UAP=1 DRPN_UAP_MODE=reobserve DRPN_UAP_REL_GATE=0.9 \
                  DRPN_UAP_COMMIT_TAU=0.9 DRPN_UAP_MAX_REOBS=3 DRPN_UAP_CONF_MARGIN=0.35; }
en_revoke() { export DRPN_USE_REVOKE=1 DRPN_REVOKE_STALL_K=6 DRPN_REVOKE_EPS=0.15 \
                     DRPN_REVOKE_FAR=0.8 DRPN_REVOKE_MAX=2 DRPN_REVOKE_COOLDOWN=2.0; }

run_one() {  # args: arm N
  local ARM="$1" EPISODES="$2"; base_env
  export DRPN_SEED="$SEED" DRPN_DEGRADE_TYPE="$COND"
  case "$ARM" in
    B0)         : ;;
    MUAP)       en_thf; en_uap ;;
    REVOKE)     en_revoke ;;
    MUAPREVOKE) en_thf; en_uap; en_revoke ;;
    *) echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/cl_${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$EPISODES" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$EPISODES -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$EPISODES" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  # surface REVOKE activity (banner/counters) right after each arm
  grep -E "REVOKE\]" "$LOG" | tail -n 3
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

# ---- chain: wait until run_oracle.sh + its python fully exit (do NOT grep stale ALL_DONE) ----
echo "[chain] waiting for run_oracle.sh / ORACLEGATE / CLEANCEIL to exit ... $(date)" | tee -a "$LOG"
while pgrep -f "run_oracle.sh" >/dev/null 2>&1 || pgrep -f "or_ORACLEGATE_low_light_s4_seed0.csv" >/dev/null 2>&1 || pgrep -f "or_CLEANCEIL_clean_s4_seed0.csv" >/dev/null 2>&1; do
  sleep 120
done
echo "[chain] oracle fully exited -> starting commit_layer $(date)" | tee -a "$LOG"

# B0 paired control reused from redesign_n300 (same seed/episodes -> head-N identical).
# Pre-registered: smoke REVOKE@40 (must FIRE, see [REVOKE] counters) -> then 3 arms @150 paired.
QUEUE=(
  "REVOKE 40"        # smoke: verify the new module actually triggers (no-op assert) + directional
  "MUAP 150"         # defer-only (existing); smoke@23 was -0.13, re-check at scale + via motiv
  "REVOKE 150"       # revoke-only (the diagnosis-aligned NEW lever)
  "MUAPREVOKE 150"   # defer + revoke combined
)
echo "############ COMMIT-LAYER START $(date) split=$SPLIT sev=$SEV ############" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do run_one $item; done
echo "ALL_DONE commit_layer $(date)" | tee -a "$LOG"
