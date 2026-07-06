#!/bin/bash
# SWEEP: map where blanket restoration's benefit FLIPS SIGN (helpful -> harmful), to establish the
# operating regime that justifies TARG (per-frame adaptive restoration).
# Hypotheses:
#   H1 (severity): restoration helps at HIGH severity (much to recover) but HURTS at LOW severity
#                  (over-processes near-clean frames -> artifacts/OOD). Test low_light sev1/sev2.
#   H2 (noise/op): restoration HURTS for destructive ops -- gaussian_noise (denoise removes texture)
#                  and motion_blur (sharpen amplifies noise), at sev4.
# Each condition runs B0 (degraded) and RES (blanket restore) paired (same seed/episodes), N=40.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/sweep_signmap
LOG="$OUTDIR/sweep.log"; mkdir -p "$OUTDIR"
EPISODES="${SWEEP_EPISODES:-40}"; SPLIT=val
export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=sweep_signmap
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_MOTIV_LOG=1
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG
}
run_one() {  # args: arm cond sev seed
  local ARM="$1" COND="$2" SEV="$3" SEED="$4"; base_env
  export DRPN_SEED="$SEED" DRPN_DEGRADE_SEVERITY="$SEV"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    B0)  : ;;
    RES) export DRPN_USE_RESTORE=1 ;;
    *) echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}"
  local CSV="$OUTDIR/sw_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
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
# Paired B0/RES per condition. Order: cheap/decisive first (low severity, then destructive types).
QUEUE=(
  "B0 low_light 1 0"      "RES low_light 1 0"      # H1: very low severity -> expect restore to HURT
  "B0 low_light 2 0"      "RES low_light 2 0"      # H1: low-mid severity
  "B0 gaussian_noise 4 0" "RES gaussian_noise 4 0" # H2: high noise -> denoise destroys texture
  "B0 motion_blur 4 0"    "RES motion_blur 4 0"    # H2: blur -> sharpen amplifies noise/ringing
)
echo "########## SWEEP SIGN-MAP START $(date) N=$EPISODES ##########" | tee -a "$LOG"
for item in "${QUEUE[@]}"; do run_one $item; done
echo "ALL_DONE sweep_signmap $(date)" | tee -a "$LOG"
