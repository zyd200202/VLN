#!/bin/bash
# ============================================================================================
# MAIN TABLE — per-condition (per type x severity), the way robustness reviewers expect
# (ImageNet-C / RobustNav style): NO mixed average. Each (type,severity) is a SEPARATE fixed
# degradation; arms B0 / RES / ROUTER are paired (same seed => same episodes) and compared
# per cell with McNemar significance.
#
#   B0     : degraded input, no restoration (lower ref)
#   RES    : blanket handcrafted restoration (shows the sign-flip: helps low_light/fog, hurts noise/blur)
#   ROUTER : type-aware blind routing (METHOD) -> restore where beneficial, bypass where harmful.
#            Per condition this validates the BLIND classifier in-situ: ROUTER should ~= RES on
#            low_light/fog and ~= B0 on noise/blur => "always the right action per type".
#
# Hero-first ordering: stop anytime; the earliest cells carry the argument.
# ============================================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N="${MAIN_EPISODES:-300}"; SPLIT=val; SEED="${MAIN_SEED:-0}"
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/maintable; LOG="$OUTDIR/maintable.log"; mkdir -p "$OUTDIR"
export DRPN_WANDB="${DRPN_WANDB:-1}" DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=maintable
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"
export DRPN_MIX=0

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
  export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
  export DRPN_USE_FUSE=0 DRPN_FUSE_ORACLE=0 DRPN_RESTORER=""
  export DRPN_MOTIV_LOG=1 DRPN_SEED="$SEED"
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG
}
run_one() {  # args: ARM COND SEV
  local ARM="$1" COND="$2" SEV="$3"; base_env
  export DRPN_DEGRADE_SEVERITY="$SEV"
  export DRPN_DEGRADE_TYPE="$COND"; [ "$COND" = "clean" ] && export DRPN_DEGRADE_TYPE="none"
  case "$ARM" in
    B0)     : ;;
    RES)    export DRPN_USE_RESTORE=1 ;;
    ROUTER) export DRPN_USE_ROUTER=1 ;;
    *) echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_${COND}_s${SEV}_seed${SEED}_N${N}"
  local CSV="$OUTDIR/mt_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then echo "SKIP(done) $(basename $CSV)"|tee -a "$LOG"; return; fi
  echo "===== [$ARM] $COND s$SEV seed$SEED N=$N ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
}
cell() {  # args: COND SEV  -> run 3 arms + paired McNemar
  local COND="$1" SEV="$2"
  run_one B0     "$COND" "$SEV"
  run_one RES    "$COND" "$SEV"
  run_one ROUTER "$COND" "$SEV"
  python - "$OUTDIR" "$COND" "$SEV" "$SEED" <<'PY' | tee -a "$LOG"
import sys, csv, os
od, cond, sev, seed = sys.argv[1:5]
def load(a):
    p=f"{od}/mt_{a}_{cond}_s{sev}_seed{seed}.csv"
    if not os.path.exists(p): return None
    return [float(r[0]) for r in list(csv.reader(open(p)))[1:] if r]
def mcnemar(x,y):
    n=min(len(x),len(y)); x,y=x[:n],y[:n]
    b=sum(1 for i in range(n) if x[i]>.5 and y[i]<.5)  # x win,y lose
    c=sum(1 for i in range(n) if x[i]<.5 and y[i]>.5)
    # exact two-sided binomial on discordant pairs
    from math import comb
    nd=b+c; k=min(b,c)
    p=min(1.0, 2*sum(comb(nd,i) for i in range(k+1))/(2**nd)) if nd>0 else 1.0
    return b,c,p
b=load("B0"); r=load("RES"); ro=load("ROUTER")
print(f"---- CELL {cond} s{sev} (N={len(b) if b else 0}) ----")
for name,v in [("B0",b),("RES",r),("ROUTER",ro)]:
    if v: print(f"   {name:7s} SR={sum(v)/len(v):.3f}")
if b and ro:
    bb,cc,p=mcnemar(ro,b); print(f"   ROUTER vs B0 : +{bb}/-{cc} discordant, McNemar p={p:.3f}")
if b and r:
    bb,cc,p=mcnemar(r,b);  print(f"   RES    vs B0 : +{bb}/-{cc} discordant, McNemar p={p:.3f}")
PY
}
echo "########## MAIN TABLE START $(date) N=$N seed=$SEED ##########" | tee -a "$LOG"
# ---- MVP (minimal publishable): 3 decisive cells fully prove the sign-flip + router-picks-right ----
cell low_light      4     # restoration BENEFIT  -> ROUTER should match RES, both > B0
cell gaussian_noise 4     # restoration HARM     -> ROUTER should bypass (== B0), avoiding RES's drop
cell motion_blur    4     # restoration HARM     -> ROUTER == B0
echo "########## MVP MAIN TABLE DONE $(date) ##########" | tee -a "$LOG"
# ---- slow supplements (run later if results hold): uncomment as needed ----
# cell fog            4
# cell low_light      2
# cell low_light      1
# cell low_light      3
