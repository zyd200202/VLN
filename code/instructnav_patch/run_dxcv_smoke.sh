#!/bin/bash
# Road 2 -- DXCV smoke n=10 to verify the depth cross-modal commitment veto fires.
# If the SUMMARY shows checks>0 and (vetoed>0 OR no_det_veto>0), the hook is wired.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/dxcv
LOG="$OUTDIR/smoke.log"
mkdir -p "$OUTDIR"
SEV=4; SPLIT=val; SEED=0; COND=low_light; N=10

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0 DRPN_USE_REVOKE=0
  export DRPN_USE_CRV=0 DRPN_USE_DXCV=0
  export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  export DRPN_SEED="$SEED" DRPN_DEGRADE_TYPE="$COND"
}
en_dxcv() {
  export DRPN_USE_DXCV=1 \
         DRPN_DXCV_REL_GATE=0.85 \
         DRPN_DXCV_DMIN=0.30 DRPN_DXCV_DMAX=3.50 DRPN_DXCV_IQR_MAX=0.60 \
         DRPN_DXCV_CONF=0.10 DRPN_DXCV_FLOOR=0.12 DRPN_DXCV_AREA=2500 \
         DRPN_DXCV_LOG=1
}

ARM=DXCV
base_env; en_dxcv
CSV="$OUTDIR/smoke_${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}.csv"
echo "===== [$ARM] smoke $COND s$SEV seed$SEED N=$N -> $(basename "$CSV") $(date '+%H:%M:%S') =====" | tee "$LOG"
python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
echo "----- DXCV markers -----" | tee -a "$LOG"
grep -E "\[DRPhysNav\]\[DXCV\]" "$LOG" | tail -n 50 | tee -a "$LOG"
echo "----- SR -----" | tee -a "$LOG"
python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
echo "smoke_done $(date)" | tee -a "$LOG"
