#!/bin/bash
# DARE -- diagnosis-driven Degradation-Aware Restoration decision layer, on the MIXED-degradation
# benchmark (per-episode random (type,severity), unknown to the agent; paired across arms).
# Arms (all DRPN_MIX=1, same seed/episodes => identical per-episode degradation => paired):
#   SMOKE     : DARE blind, N=3, ROUTER_LOG=1 -> confirm wiring (banner + per-branch counts) before the long run
#   BYPASS    : always use degraded frame (no restore)              [= naive robust lower ref]
#   RESTORE   : always blanket-restore (USE_RESTORE)                [= naive "intuitive" strong baseline]
#   DARE      : blind router + specialized restore + chunk hysteresis (our method)
#   ORACLE    : router with TRUE (type,sev) -> per-condition routing UPPER BOUND
# Compare pooled across the mixed episodes (paired McNemar + bootstrap CI).
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/dare
LOG="$OUTDIR/dare.log"; mkdir -p "$OUTDIR"
EPISODES="${DARE_EPISODES:-300}"; SPLIT=val; SEED="${DARE_SEED:-0}"
export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=dare
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"
# realistic mixture: low-light dominant (varying darkness) + occasional destructive ops
export DRPN_MIX=1
export DRPN_MIX_SPEC="${DRPN_MIX_SPEC:-low_light:1,2,4;low_light:2,4;gaussian_noise:4;motion_blur:4}"
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
  export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
  export DRPN_MOTIV_LOG=1 DRPN_SEED="$SEED"
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG
}
run_one() {  # args: arm N [extra]
  local ARM="$1" N="$2"; base_env
  case "$ARM" in
    SMOKE)   export DRPN_USE_ROUTER=1 DRPN_ROUTER_LOG=1 ;;
    BYPASS)  : ;;                                  # mixed degradation, no restore
    RESTORE) export DRPN_USE_RESTORE=1 ;;          # mixed degradation, blanket restore
    DARE)    export DRPN_USE_ROUTER=1 ;;           # blind router (method)
    ORACLE)  export DRPN_USE_ROUTER=1 DRPN_ROUTER_ORACLE=1 ;;  # routing upper bound
    *) echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_mix_seed${SEED}_N${N}"
  local CSV="$OUTDIR/dare_${ARM}_mix_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then echo "SKIP(done) $(basename $CSV)"|tee -a "$LOG"; return; fi
  echo "===== [$ARM] mixed seed$SEED N=$N ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}
echo "########## DARE START $(date) N=$EPISODES seed=$SEED mix=$DRPN_MIX_SPEC ##########" | tee -a "$LOG"
run_one SMOKE 3        # wiring check first: look for [DRPhysNav] ROUTER enabled + [ROUTER]... + branch hit-counts
echo "----- SMOKE done; verify ROUTER banner & branch counts above before trusting the rest -----" | tee -a "$LOG"
run_one BYPASS  "$EPISODES"
run_one RESTORE "$EPISODES"
run_one DARE    "$EPISODES"
run_one ORACLE  "$EPISODES"
echo "ALL_DONE dare $(date)" | tee -a "$LOG"
