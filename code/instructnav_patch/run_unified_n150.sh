#!/bin/bash
# ============================================================================
# UNIFIED N=150 SWEEP -- bring all non-N=300 Tab-3 arms to a common scale.
# All on low-light sev4 seed0, val split, paired against
#   ./maintable/mt_B0_low_light_s4_seed0.csv  (existing N=150 B0).
# Queue (chained AFTER any current objnav_benchmark.py exits, so the running
# RESCRV gaussian-noise N=150 finishes first):
#   1) DXCV smoke n=10  (verify the new Road-2 patch fires + doesn't crash)
#   2) REVOKE  N=150
#   3) CRV     N=150
#   4) MUAP    N=150
#   5) FUSE    N=150
#   6) DXCV    N=150
# Each arm: log SUMMARY SR + paired McNemar vs mt_B0.
# Tail:  tail -f /root/autodl-tmp/DRPhysNav/runs/unified/unified.log
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/unified
LOG="$OUTDIR/unified.log"
mkdir -p "$OUTDIR"
SEV=4; SPLIT=val; SEED=0; COND=low_light; N=150
B0_REF=/root/autodl-tmp/DRPhysNav/runs/maintable/mt_B0_low_light_s4_seed0.csv

export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=unified_n150
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0 DRPN_USE_REVOKE=0
  export DRPN_USE_CRV=0 DRPN_USE_FUSE=0 DRPN_USE_DXCV=0
  export DRPN_FUSE_ORACLE=0 DRPN_ROUTER_ORACLE=0 DRPN_TARG_ORACLE=0
  export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  export DRPN_SEED="$SEED" DRPN_DEGRADE_TYPE="$COND"
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG
  unset DRPN_CRV_REL_GATE DRPN_CRV_CONF DRPN_CRV_FLOOR DRPN_CRV_AREA DRPN_CRV_FRAMES DRPN_CRV_MAX DRPN_CRV_CENTER DRPN_CRV_DIAG
  unset DRPN_DXCV_REL_GATE DRPN_DXCV_DMIN DRPN_DXCV_DMAX DRPN_DXCV_IQR_MAX
  unset DRPN_DXCV_CONF DRPN_DXCV_FLOOR DRPN_DXCV_AREA DRPN_DXCV_LOG
  unset DRPN_RESTORER
}
en_revoke() { export DRPN_USE_REVOKE=1 DRPN_REVOKE_STALL_K=6 DRPN_REVOKE_EPS=0.15 \
                     DRPN_REVOKE_FAR=0.8 DRPN_REVOKE_MAX=2 DRPN_REVOKE_COOLDOWN=2.0; }
en_crv()    { export DRPN_USE_CRV=1 DRPN_CRV_REL_GATE=0.90 DRPN_CRV_CONF=0.10 \
                     DRPN_CRV_FLOOR=0.12 DRPN_CRV_AREA=5000 DRPN_CRV_FRAMES=1 \
                     DRPN_CRV_MAX=3 DRPN_CRV_CENTER=0.18 DRPN_CRV_DIAG=0; }
en_thf()    { export DRPN_USE_THF=1 DRPN_THF_LOW_CONF=0.12 DRPN_THF_BASE_CONF=0.25 \
                     DRPN_THF_REL_GATE=0.9 DRPN_THF_WINDOW=8 DRPN_THF_MIN_FRAMES=3 DRPN_THF_MATCH_DIST=0.6; }
en_uap()    { export DRPN_USE_UAP=1 DRPN_UAP_MODE=reobserve DRPN_UAP_REL_GATE=0.9 \
                     DRPN_UAP_COMMIT_TAU=0.9 DRPN_UAP_MAX_REOBS=3 DRPN_UAP_CONF_MARGIN=0.35; }
en_muap()   { en_thf; en_uap; }
en_fuse()   { export DRPN_USE_FUSE=1 DRPN_FUSE_BETA=0.10 DRPN_FUSE_IOU=0.5 \
                     DRPN_FUSE_REL_FLOOR=0.2 DRPN_FUSE_IDENTITY_MAD=1.0 \
                     DRPN_FUSE_ORACLE=0 DRPN_FUSE_LOG=0; }
en_dxcv()   { export DRPN_USE_DXCV=1 DRPN_DXCV_REL_GATE=0.85 \
                     DRPN_DXCV_DMIN=0.30 DRPN_DXCV_DMAX=3.50 DRPN_DXCV_IQR_MAX=0.60 \
                     DRPN_DXCV_CONF=0.10 DRPN_DXCV_FLOOR=0.12 DRPN_DXCV_AREA=2500 \
                     DRPN_DXCV_LOG=0; }

run_one() {  # args: ARM N TAG
  local ARM="$1" EPISODES="$2" TAG="${3:-}"
  base_env
  case "$ARM" in
    B0)     : ;;
    REVOKE) en_revoke ;;
    CRV)    en_crv ;;
    MUAP)   en_muap ;;
    FUSE)   en_fuse ;;
    DXCV)   en_dxcv ;;
    DXCVdbg) en_dxcv; export DRPN_DXCV_LOG=1 ;;
    *) echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}${TAG:+_$TAG}"
  local CSV="$OUTDIR/u_${ARM}_${COND}_s${SEV}_seed${SEED}_N${EPISODES}${TAG:+_$TAG}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$EPISODES" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$EPISODES -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$EPISODES" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  echo "----- $ARM markers -----" | tee -a "$LOG"
  grep -E "\[DRPhysNav\]\[(REVOKE|CRV|MUAP|UAP|THF|FUSE|DXCV|ROUTER)\]" "$LOG" | tail -n 6 | tee -a "$LOG"
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

paired_mcnemar() {  # args: BCSV CCSV LABEL ARM
  local BCSV="$1"; local CCSV="$2"; local LBL="$3"; local ARM="$4"
  if [ ! -f "$BCSV" ] || [ ! -f "$CCSV" ]; then
    echo "PAIRED[$LBL/$ARM]: missing csv ($BCSV or $CCSV)" | tee -a "$LOG"; return
  fi
  python - "$BCSV" "$CCSV" "$LBL" "$ARM" <<'PY' | tee -a "$LOG"
import sys, csv
def load(p): return [r for r in list(csv.reader(open(p)))[1:] if r]
b=load(sys.argv[1]); c=load(sys.argv[2])
n=min(len(b),len(c)); b=b[:n]; c=c[:n]
sb=sum(float(r[0]) for r in b)/n; sc=sum(float(r[0]) for r in c)/n
b01=sum(1 for i in range(n) if float(b[i][0])==0 and float(c[i][0])==1)
b10=sum(1 for i in range(n) if float(b[i][0])==1 and float(c[i][0])==0)
from math import comb
ntot=b01+b10; k=min(b01,b10)
p=sum(comb(ntot,i) for i in range(0,k+1))/(2**ntot)*2 if ntot>0 else 1.0
p=min(1.0,p)
print("PAIRED[%s/%s] n=%d  B0_SR=%.3f  ARM_SR=%.3f  dSR=%+.3f"%(sys.argv[3],sys.argv[4],n,sb,sc,sc-sb))
print("  discordant: rescued(0->1)=%d  broke(1->0)=%d  net=%+d  McNemar exact p=%.4f"%(b01,b10,b01-b10,p))
print("  VERDICT:", "SIGNIFICANT +" if (sc>sb and p<0.05) else ("positive but n.s." if sc>sb else ("no gain" if sc==sb else "negative")))
PY
}

# 1) wait for current bench to exit
echo "[chain] $(date) -- waiting for any running objnav_benchmark.py (PID list below) ..." | tee -a "$LOG"
pgrep -af "objnav_benchmark.py" | tee -a "$LOG"
while pgrep -f "objnav_benchmark.py" >/dev/null 2>&1; do sleep 60; done
echo "[chain] $(date) -- GPU free, starting unified queue" | tee -a "$LOG"

# 2) DXCV smoke n=10 with verbose log to verify the new Road-2 patch
echo "===== [SMOKE] DXCV n=10 with LOG=1 ($(date)) =====" | tee -a "$LOG"
run_one DXCVdbg 10 smoke
if ! grep -q "\[DRPhysNav\]\[DXCV\]" "$LOG"; then
  echo "[chain] DXCV never logged anything -> patch broken; aborting paired runs." | tee -a "$LOG"
  exit 2
fi

# 3) Unified N=150 sweep
echo "[chain] $(date) -- starting unified N=$N sweep (5 arms)" | tee -a "$LOG"
QUEUE=(REVOKE CRV MUAP FUSE DXCV)
for ARM in "${QUEUE[@]}"; do
  run_one "$ARM" "$N"
done

# 4) Paired McNemar vs the existing B0_N150 reference
echo "[chain] $(date) -- paired McNemar vs $B0_REF" | tee -a "$LOG"
for ARM in "${QUEUE[@]}"; do
  ARM_CSV="$OUTDIR/u_${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}.csv"
  paired_mcnemar "$B0_REF" "$ARM_CSV" "N${N}_${COND}_s${SEV}_seed${SEED}" "$ARM"
done

echo "ALL_DONE unified_n150 $(date)" | tee -a "$LOG"
