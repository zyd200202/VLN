#!/bin/bash
# Road 1 Stage C — paired N=150 eval on low_light s4, GLEE-feature-adapter on,
# alongside the matching B0 baseline (no adapter) for paired McNemar.
# Inherits same seed/split/severity as the main paper table.
# no `set -u`: conda's binutils-activate references $HOST without default
cd /root/autodl-tmp/InstructNav
export HOST="${HOST:-x86_64-conda_cos6-linux-gnu}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Road 1 adapter checkpoint (set by Stage B; this is the default location) ---
export DRPN_ROAD1_ADAPTER="${DRPN_ROAD1_ADAPTER:-/root/autodl-tmp/DRPhysNav/road1/ckpt/adapter_low_light_s4.pth}"

base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
  export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
  export DRPN_USE_FUSE=0 DRPN_FUSE_ORACLE=0 DRPN_RESTORER=""
  export DRPN_MOTIV_LOG=1 DRPN_SEED=0 DRPN_MIX=0
  export DRPN_WANDB=0
  unset DRPN_BACKBONE  # Road 1 uses GPT-4o pipeline as the main paper
}

N="${MAIN_EPISODES:-150}"; SPLIT=val
COND=low_light; SEV=4
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/road1_eval
LOG="$OUTDIR/road1_eval.log"
mkdir -p "$OUTDIR"

run_one() {
  local ARM="$1"; base_env
  export DRPN_DEGRADE_TYPE="$COND" DRPN_DEGRADE_SEVERITY="$SEV"
  case "$ARM" in
    R1)   : ;;  # Road-1 adapter ON (DRPN_ROAD1_ADAPTER already set)
    *)    echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  local CSV="$OUTDIR/road1_${ARM}_${COND}_s${SEV}_seed0.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then
    echo "SKIP(done) $(basename $CSV)" | tee -a "$LOG"; return
  fi
  echo "===== [$ARM] $COND s$SEV N=$N $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  echo "===== [$ARM] DONE $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
}

echo "########## ROAD-1 EVAL START $(date) ckpt=$DRPN_ROAD1_ADAPTER ##########" | tee -a "$LOG"
run_one R1

# Paired McNemar against the main-table B0 (same seed, same N, same cell)
python - <<'PY' | tee -a "$LOG"
import csv, os
od = "/root/autodl-tmp/DRPhysNav/runs/road1_eval"
b0 = "/root/autodl-tmp/DRPhysNav/runs/maintable/mt_B0_low_light_s4_seed0.csv"
r1 = f"{od}/road1_R1_low_light_s4_seed0.csv"

def load(p):
    if not os.path.exists(p): return None
    return [float(r[0]) for r in list(csv.reader(open(p)))[1:] if r]

def mcnemar(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    b = sum(1 for i in range(n) if x[i] > .5 and y[i] < .5)
    c = sum(1 for i in range(n) if x[i] < .5 and y[i] > .5)
    from math import comb
    nd = b + c; k = min(b, c)
    p = min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / (2**nd)) if nd > 0 else 1.0
    return b, c, p

bv = load(b0); rv = load(r1)
print(f"---- ROAD-1 EVAL low_light s4 N={len(bv) if bv else 0} ----")
if bv: print(f"   B0  (main table)        SR={sum(bv)/len(bv):.3f}")
if rv: print(f"   R1  (adapter)           SR={sum(rv)/len(rv):.3f}")
if bv and rv:
    bb, cc, p = mcnemar(rv, bv)
    n = min(len(bv), len(rv))
    print(f"   R1 vs B0 (paired N={n}): +{bb}/-{cc} discordant, "
          f"McNemar p={p:.3f}, dSR={(sum(rv[:n])-sum(bv[:n]))/n:+.3f}")
PY

echo "########## ROAD-1 EVAL DONE $(date) ##########" | tee -a "$LOG"
