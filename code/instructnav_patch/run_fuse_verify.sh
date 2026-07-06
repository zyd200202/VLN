#!/bin/bash
# CHEAP pre-flight verification BEFORE committing to the expensive N=300 main table.
# Runs the paired pair {B0, FUSE_ROUTER(hero)} at small N on the SAME mixed episodes and checks:
#   (1) wiring: FUSE + ROUTER banners fire, restored-path actually contributes (add/corroborate > 0),
#       bypass works on harmful degradations (bypass_identity > 0);
#   (2) NO REGRESSION: count episodes where B0 succeeded but HERO failed (must be ~0 / <= gains);
#   (3) effect direction: HERO SR >= B0 SR - tol.
# Prints PASS/FAIL. Use this to decide whether the 4-day N=300 run is worth launching.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/fuse_verify
LOG="$OUTDIR/verify.log"; mkdir -p "$OUTDIR"; : > "$LOG"   # clear stale verdicts so the gate reads only THIS run
N="${VERIFY_N:-40}"; SPLIT=val; SEED="${VERIFY_SEED:-0}"; TOL="${VERIFY_TOL:-0.0}"
export DRPN_WANDB=0
export DRPN_MIX=1
export DRPN_MIX_SPEC="${DRPN_MIX_SPEC:-low_light:1,2,4;low_light:2,4;gaussian_noise:4;motion_blur:4}"
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
  export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
  export DRPN_USE_FUSE=0 DRPN_FUSE_ORACLE=0 DRPN_FUSE_LOG=0 DRPN_RESTORER=""
  export DRPN_MOTIV_LOG=1 DRPN_SEED="$SEED"
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG
}
run_one() {
  local ARM="$1"; base_env
  case "$ARM" in
    B0)          : ;;
    FUSE_ROUTER) export DRPN_USE_FUSE=1 DRPN_USE_ROUTER=1 DRPN_FUSE_LOG=0 ;;
  esac
  local CSV="$OUTDIR/v_${ARM}_seed${SEED}_N${N}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then echo "SKIP(done) $(basename $CSV)"|tee -a "$LOG"; return; fi
  echo "===== [verify $ARM] N=$N seed$SEED ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
}
echo "########## FUSE VERIFY START $(date) N=$N seed=$SEED ##########" | tee -a "$LOG"
run_one B0
run_one FUSE_ROUTER
echo "----- branch activity (grep from log) -----" | tee -a "$LOG"
grep -E "\[DRPhysNav\] (FUSE|ROUTER)|\[DRPhysNav\]\[FUSE\]|\[DRPhysNav\]\[ROUTER\]" "$LOG" | tail -8 | tee -a "$LOG"
echo "----- paired gate -----" | tee -a "$LOG"
python - "$OUTDIR" "$SEED" "$N" "$TOL" <<'PY' | tee -a "$LOG"
import sys, csv, os
out, seed, N, tol = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
def load(a):
    p=f"{out}/v_{a}_seed{seed}_N{N}.csv"
    if not os.path.exists(p): return None
    return [float(r[0]) for r in list(csv.reader(open(p)))[1:] if r]
b=load("B0"); h=load("FUSE_ROUTER")
if not b or not h:
    print("VERIFY: FAIL (missing runs)"); sys.exit(2)
n=min(len(b),len(h)); b,h=b[:n],h[:n]
srb=sum(b)/n; srh=sum(h)/n
reg=sum(1 for i in range(n) if b[i]>0.5 and h[i]<0.5)   # B0 win, HERO lose -> regression
gain=sum(1 for i in range(n) if h[i]>0.5 and b[i]<0.5)  # HERO win, B0 lose -> gain
print("VERIFY n=%d | B0 SR=%.3f | HERO SR=%.3f | dSR=%+.3f | gains=%d regressions=%d"%(n,srb,srh,srh-srb,gain,reg))
ok_noreg = (reg <= gain)                 # fusion should not destroy more than it creates
ok_dir   = (srh >= srb - tol)            # effect direction non-negative within tol
verdict = "PASS" if (ok_noreg and ok_dir) else "FAIL"
print("VERIFY: %s (no-regression=%s, direction=%s)"%(verdict, ok_noreg, ok_dir))
print("  -> PASS: worth launching N=300.  FAIL: fusion not helping / regressing -> revisit before 300.")
PY
echo "########## FUSE VERIFY DONE $(date) ##########" | tee -a "$LOG"
