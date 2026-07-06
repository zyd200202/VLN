#!/bin/bash
# Road 2 -- DXCV chained launcher. Waits for the existing benchmark process to exit,
# then runs (1) smoke n=10, (2) paired B0 + DXCV N=80 on low_light s4 seed0, then
# computes the paired McNemar comparison. Tail the LOG below for status.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/dxcv
LOG="$OUTDIR/chain.log"
mkdir -p "$OUTDIR"
SEV=4; SPLIT=val; SEED=0; COND=low_light

export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=dxcv
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0 DRPN_USE_REVOKE=0
  export DRPN_USE_CRV=0 DRPN_USE_DXCV=0
  export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  export DRPN_SEED="$SEED" DRPN_DEGRADE_TYPE="$COND"
  unset DRPN_DXCV_REL_GATE DRPN_DXCV_DMIN DRPN_DXCV_DMAX DRPN_DXCV_IQR_MAX
  unset DRPN_DXCV_CONF DRPN_DXCV_FLOOR DRPN_DXCV_AREA DRPN_DXCV_LOG
}
en_dxcv() {
  export DRPN_USE_DXCV=1 \
         DRPN_DXCV_REL_GATE=0.85 \
         DRPN_DXCV_DMIN=0.30 DRPN_DXCV_DMAX=3.50 DRPN_DXCV_IQR_MAX=0.60 \
         DRPN_DXCV_CONF=0.10 DRPN_DXCV_FLOOR=0.12 DRPN_DXCV_AREA=2500 \
         DRPN_DXCV_LOG=0
}

run_one() {  # args: ARM N TAG
  local ARM="$1"; local N="$2"; local TAG="$3"
  base_env
  case "$ARM" in
    B0)   : ;;
    DXCV) en_dxcv ;;
    *) echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_${TAG}"
  local CSV="$OUTDIR/${TAG}_${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $TAG $COND s$SEV seed$SEED N=$N -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  echo "----- DXCV summary -----" | tee -a "$LOG"
  grep -E "\[DRPhysNav\]\[DXCV\]" "$LOG" | tail -n 6 | tee -a "$LOG"
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

paired_mcnemar() {  # args: BCSV CCSV LABEL
  local BCSV="$1"; local CCSV="$2"; local LBL="$3"
  python - "$BCSV" "$CCSV" "$LBL" <<'PY' | tee -a "$LOG"
import sys, csv
def load(p):
    return [r for r in list(csv.reader(open(p)))[1:] if r]
try:
    b=load(sys.argv[1]); c=load(sys.argv[2])
except Exception as e:
    print("PAIRED: missing csv:", e); raise SystemExit
n=min(len(b),len(c)); b=b[:n]; c=c[:n]
sb=sum(float(r[0]) for r in b)/n; sc=sum(float(r[0]) for r in c)/n
b01=sum(1 for i in range(n) if float(b[i][0])==0 and float(c[i][0])==1)  # DXCV rescued
b10=sum(1 for i in range(n) if float(b[i][0])==1 and float(c[i][0])==0)  # DXCV broke
from math import comb
ntot=b01+b10; k=min(b01,b10)
p=sum(comb(ntot,i) for i in range(0,k+1))/(2**ntot)*2 if ntot>0 else 1.0
p=min(1.0,p)
lbl=sys.argv[3]
print("PAIRED[%s] n=%d  B0_SR=%.3f  DXCV_SR=%.3f  dSR=%+.3f"%(lbl,n,sb,sc,sc-sb))
print("  discordant: DXCV_rescued(0->1)=%d  DXCV_broke(1->0)=%d  net=%+d  McNemar exact p=%.4f"%(b01,b10,b01-b10,p))
print("  VERDICT:", "SIGNIFICANT +" if (sc>sb and p<0.05) else ("positive but n.s." if sc>sb else "no gain"))
PY
}

# 1) wait for any running benchmark to exit
echo "[chain] $(date) -- waiting for any objnav_benchmark.py to exit ..." | tee -a "$LOG"
while pgrep -f "objnav_benchmark.py" >/dev/null 2>&1; do sleep 60; done
echo "[chain] $(date) -- GPU free, starting DXCV smoke" | tee -a "$LOG"

# 2) smoke n=10 with verbose log to verify the hook fires
export DRPN_DXCV_LOG=1
en_dxcv; export DRPN_DXCV_LOG=1
SMOKE_CSV="$OUTDIR/smoke_DXCV_${COND}_s${SEV}_seed${SEED}_N10.csv"
if [ -f "$SMOKE_CSV" ] && [ "$(wc -l < "$SMOKE_CSV")" -gt 10 ]; then
  echo "=== SKIP smoke (done): $(basename "$SMOKE_CSV") ===" | tee -a "$LOG"
else
  echo "===== [SMOKE] DXCV n=10 with LOG=1 ($(date)) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes 10 --out_csv "$SMOKE_CSV" --save_traj 0 >> "$LOG" 2>&1
  echo "----- smoke DXCV events -----" | tee -a "$LOG"
  grep -E "\[DRPhysNav\]\[DXCV\]" "$LOG" | tail -n 30 | tee -a "$LOG"
fi
unset DRPN_DXCV_LOG
HOOK_OK=$(grep -c "\[DRPhysNav\]\[DXCV\] (veto|pass|SUMMARY|enabled)" "$LOG")
if [ "$(grep -c '\[DRPhysNav\]\[DXCV\]' "$LOG")" = "0" ]; then
  echo "[chain] DXCV never logged ANYTHING -> patch broken; aborting paired." | tee -a "$LOG"
  exit 2
fi

# 3) paired N=80
echo "[chain] $(date) -- starting paired N=80 B0 vs DXCV" | tee -a "$LOG"
run_one B0   80 paired80
run_one DXCV 80 paired80

# 4) McNemar
paired_mcnemar \
  "$OUTDIR/paired80_B0_${COND}_s${SEV}_seed${SEED}_N80.csv" \
  "$OUTDIR/paired80_DXCV_${COND}_s${SEV}_seed${SEED}_N80.csv" \
  N80

echo "ALL_DONE dxcv chain $(date)" | tee -a "$LOG"
