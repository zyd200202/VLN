#!/bin/bash
# ============================================================================
# CRV paired N=150 on gaussian_noise s4 seed0 (the high-headroom condition).
# Runs a FRESH B0 and the improved CRV on the SAME code/seed/episode-order so
# the comparison is exactly paired (McNemar). Improvements over the smoke:
#   - central-band spatial guard (DRPN_CRV_CENTER) -> target must be in front,
#     rejecting "saw a bed across a doorway" false commits.
#   - moderate-aggressive fire config informed by the diag smoke (clean +flips,
#     zero harm at n=8): area=5000 frames=1 conf/floor=0.10/0.12 rel_gate=0.90.
# Also emits the [CRV][RECALL] aggregate (can-it-ever-work denominator).
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
LOG="$OUTDIR/crv150.log"
mkdir -p "$OUTDIR"
SEV=4; SPLIT=val; SEED=0; COND=gaussian_noise; N=150

export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=crv150
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0 DRPN_USE_REVOKE=0
  export DRPN_USE_CRV=0
  export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
  export DRPN_SEED="$SEED" DRPN_DEGRADE_TYPE="$COND"
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG DRPN_TARG_ORACLE
  unset DRPN_CRV_REL_GATE DRPN_CRV_CONF DRPN_CRV_FLOOR DRPN_CRV_AREA DRPN_CRV_FRAMES DRPN_CRV_MAX DRPN_CRV_CENTER DRPN_CRV_DIAG
}
en_crv() { export DRPN_USE_CRV=1 DRPN_CRV_REL_GATE=0.90 DRPN_CRV_CONF=0.10 \
                  DRPN_CRV_FLOOR=0.12 DRPN_CRV_AREA=5000 DRPN_CRV_FRAMES=1 \
                  DRPN_CRV_MAX=3 DRPN_CRV_CENTER=0.18 DRPN_CRV_DIAG=0; }

run_one() {  # args: arm
  local ARM="$1"; base_env
  case "$ARM" in
    B0)  : ;;
    CRV) en_crv ;;
    *) echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}"
  local CSV="$OUTDIR/c150_${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$N -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  grep -E "\[CRV\]\[RECALL\]|forced_commits|CRV\]\[WARN" "$LOG" | tail -n 3
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

# wait for any current GPU job
echo "[chain] waiting for any objnav_benchmark to exit ... $(date)" | tee -a "$LOG"
while pgrep -f "objnav_benchmark.py" >/dev/null 2>&1; do sleep 60; done
echo "[chain] GPU free -> starting CRV150 $(date)" | tee -a "$LOG"

echo "############ CRV150 START $(date) cond=$COND sev=$SEV N=$N ############" | tee -a "$LOG"
run_one B0
run_one CRV

# paired McNemar B0 vs CRV (head-min identical order)
python - "$OUTDIR/c150_B0_${COND}_s${SEV}_seed${SEED}_N${N}.csv" \
          "$OUTDIR/c150_CRV_${COND}_s${SEV}_seed${SEED}_N${N}.csv" <<'PY' | tee -a "$LOG"
import sys, csv
def load(p):
    return [r for r in list(csv.reader(open(p)))[1:] if r]
try:
    b=load(sys.argv[1]); c=load(sys.argv[2])
except Exception as e:
    print("PAIRED: missing csv:", e); raise SystemExit
n=min(len(b),len(c)); b=b[:n]; c=c[:n]
sb=sum(float(r[0]) for r in b)/n; sc=sum(float(r[0]) for r in c)/n
b01=sum(1 for i in range(n) if float(b[i][0])==0 and float(c[i][0])==1)  # CRV rescued
b10=sum(1 for i in range(n) if float(b[i][0])==1 and float(c[i][0])==0)  # CRV broke
# exact two-sided McNemar (binomial on discordant pairs)
from math import comb
ntot=b01+b10; k=min(b01,b10)
p=sum(comb(ntot,i) for i in range(0,k+1))/(2**ntot)*2 if ntot>0 else 1.0
p=min(1.0,p)
print("PAIRED n=%d  B0_SR=%.3f  CRV_SR=%.3f  d=%+.3f"%(n,sb,sc,sc-sb))
print("  discordant: CRV_rescued(0->1)=%d  CRV_broke(1->0)=%d  net=%+d  McNemar exact p=%.4f"%(b01,b10,b01-b10,p))
print("  VERDICT:", "SIGNIFICANT +" if (sc>sb and p<0.05) else ("positive but n.s." if sc>sb else "no gain"))
PY
echo "ALL_DONE crv150 $(date)" | tee -a "$LOG"
