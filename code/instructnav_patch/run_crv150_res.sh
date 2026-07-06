#!/bin/bash
# ============================================================================
# Follow-up to run_crv150.sh: add the paper's counter-example arm RES (blanket
# denoising/restoration) and the combined RES+CRV, on the SAME seed/order so the
# 4-arm table (B0 / RES / CRV / RES+CRV) is exactly paired. Chained to start only
# after run_crv150.sh (B0 then CRV) and its python have exited.
# Story under test: appearance fix (RES) ~= B0, decision fix (CRV) > B0.
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
en_restore() { export DRPN_USE_RESTORE=1; }
en_crv() { export DRPN_USE_CRV=1 DRPN_CRV_REL_GATE=0.90 DRPN_CRV_CONF=0.10 \
                  DRPN_CRV_FLOOR=0.12 DRPN_CRV_AREA=5000 DRPN_CRV_FRAMES=1 \
                  DRPN_CRV_MAX=3 DRPN_CRV_CENTER=0.18 DRPN_CRV_DIAG=0; }

run_one() {  # args: arm
  local ARM="$1"; base_env
  case "$ARM" in
    RES)    en_restore ;;
    RESCRV) en_restore; en_crv ;;
    *) echo "unknown arm $ARM" | tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}"
  local CSV="$OUTDIR/c150_${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then
    echo "=== SKIP (done): $(basename "$CSV") ===" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND sev$SEV seed$SEED N=$N -> $(basename "$CSV") ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}

# chain: wait for run_crv150.sh + its python to finish
echo "[chain-res] waiting for crv150 (B0+CRV) to finish ... $(date)" | tee -a "$LOG"
while pgrep -f "run_crv150.sh" >/dev/null 2>&1 || pgrep -f "objnav_benchmark.py" >/dev/null 2>&1; do sleep 90; done
echo "[chain-res] free -> starting RES + RES+CRV $(date)" | tee -a "$LOG"

run_one RES
run_one RESCRV

# ---- 4-arm paired table (B0 / RES / CRV / RES+CRV), same order ----
python - "$OUTDIR" "$COND" "$SEV" "$SEED" "$N" <<'PY' | tee -a "$LOG"
import sys, csv, os
from math import comb
OUT,COND,SEV,SEED,N=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
def load(arm):
    p=os.path.join(OUT,"c150_%s_%s_s%s_seed%s_N%s.csv"%(arm,COND,SEV,SEED,N))
    if not os.path.exists(p): return None
    return [float(r[0]) for r in list(csv.reader(open(p)))[1:] if r]
arms={a:load(a) for a in ["B0","RES","CRV","RESCRV"]}
B=arms["B0"]
def mcnemar(b,c):
    n=min(len(b),len(c)); b,c=b[:n],c[:n]
    r=sum(1 for i in range(n) if b[i]==0 and c[i]==1)
    w=sum(1 for i in range(n) if b[i]==1 and c[i]==0)
    nt=r+w; k=min(r,w)
    p=min(1.0,(sum(comb(nt,i) for i in range(k+1))/2**nt*2) if nt>0 else 1.0)
    return n,r,w,p
print("======== 4-ARM PAIRED TABLE (gaussian_noise s%s seed%s) ========"%(SEV,SEED))
for a in ["B0","RES","CRV","RESCRV"]:
    v=arms[a]
    if v is None: print("  %-7s MISSING"%a); continue
    sr=sum(v)/len(v)
    if a=="B0":
        print("  %-7s n=%d SR=%.3f  (baseline)"%(a,len(v),sr))
    else:
        n,r,w,p=mcnemar(B,v)
        print("  %-7s n=%d SR=%.3f  d=%+.3f vs B0  rescued=%d broke=%d net=%+d  McNemar p=%.4f %s"
              %(a,len(v),sr,sr-sum(B[:n])/n,r,w,r-w,p,"*" if p<0.05 else ""))
print("Story check: RES~=B0 (appearance no-op) AND CRV>B0 (decision fix) => thesis holds.")
PY
echo "ALL_DONE crv150_res $(date)" | tee -a "$LOG"
