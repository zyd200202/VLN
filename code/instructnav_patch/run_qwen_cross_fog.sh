#!/bin/bash
# ============================================================================
# Cross-backbone replication: InstructNav with Qwen2-VL-7B replacing GPT-4o.
# Paired N=150 on low_light s4 (the headline cell), arms = B0 + RES.
# Produces two CSVs that can be directly compared to the GPT-4o N=150 from
# /root/autodl-tmp/DRPhysNav/runs/maintable/mt_{B0,RES}_low_light_s4_seed0.csv
#
# Expected wall time: ~10.5 h/cell  -> ~21 h total. Run with nohup.
# ============================================================================
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- VLM backbone toggle (Qwen2-VL-7B replaces GPT-4o) ---
export DRPN_BACKBONE=qwen2vl
export QWEN2VL_MODEL=/root/autodl-tmp/models/Qwen2-VL-7B-Instruct
export QWEN2VL_MAX_NEW=256   # cap "Thinking Process" length, ~2x speedup vs default
export QWEN2VL_GREEDY=1      # deterministic, matches paired-eval discipline

# --- Identical to maintable B0 base settings: no restore, no DDA, no router ---
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
  export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
  export DRPN_USE_FUSE=0 DRPN_FUSE_ORACLE=0 DRPN_RESTORER=""
  export DRPN_MOTIV_LOG=1 DRPN_SEED=0 DRPN_MIX=0
  export DRPN_WANDB=0  # offline; we read CSVs directly
}

N="${MAIN_EPISODES:-150}"; SPLIT=val; SEED=0
COND=fog; SEV=4
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/qwen_cross_fog
LOG="$OUTDIR/qwen_cross.log"
mkdir -p "$OUTDIR"

run_one() {
  local ARM="$1"; base_env
  export DRPN_DEGRADE_TYPE="$COND" DRPN_DEGRADE_SEVERITY="$SEV"
  case "$ARM" in
    B0)  : ;;
    RES) export DRPN_USE_RESTORE=1 ;;
    *)   echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  local CSV="$OUTDIR/qwen_${ARM}_${COND}_s${SEV}_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then
    echo "SKIP(done) $(basename $CSV)" | tee -a "$LOG"; return
  fi
  echo "===== [Qwen-$ARM] $COND s$SEV seed$SEED N=$N ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  echo "===== [Qwen-$ARM] DONE $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
}

echo "########## QWEN2VL CROSS-BACKBONE START $(date) N=$N seed=$SEED ##########" | tee -a "$LOG"
run_one B0
run_one RES

# paired McNemar at the end
python - "$OUTDIR" "$COND" "$SEV" "$SEED" <<'PY' | tee -a "$LOG"
import sys, csv, os
od, cond, sev, seed = sys.argv[1:5]
def load(a):
    p = f"{od}/qwen_{a}_{cond}_s{sev}_seed{seed}.csv"
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
b = load("B0"); r = load("RES")
print(f"---- Qwen2-VL CROSS-BACKBONE CELL {cond} s{sev} (N={len(b) if b else 0}) ----")
for name, v in [("B0", b), ("RES", r)]:
    if v: print(f"   {name:7s} SR={sum(v)/len(v):.3f}")
if b and r:
    bb, cc, p = mcnemar(r, b)
    print(f"   RES vs B0 : +{bb}/-{cc} discordant, McNemar p={p:.3f}, "
          f"dSR={(sum(r)-sum(b))/len(b):+.3f}")
print("---- compare GPT-4o N=150 (same cell, same seed) -----")
gpt_b0  = "/root/autodl-tmp/DRPhysNav/runs/maintable/mt_B0_low_light_s4_seed0.csv"
gpt_res = "/root/autodl-tmp/DRPhysNav/runs/maintable/mt_RES_low_light_s4_seed0.csv"
def load_(p):
    return [float(r[0]) for r in list(csv.reader(open(p)))[1:] if r]
b2 = load_(gpt_b0); r2 = load_(gpt_res)
print(f"   GPT-4o B0  SR={sum(b2)/len(b2):.3f}")
print(f"   GPT-4o RES SR={sum(r2)/len(r2):.3f}")
bb, cc, p = mcnemar(r2, b2)
print(f"   GPT-4o RES vs B0: +{bb}/-{cc} discordant, McNemar p={p:.3f}, "
      f"dSR={(sum(r2)-sum(b2))/len(b2):+.3f}")
PY

echo "########## QWEN2VL CROSS-BACKBONE DONE $(date) ##########" | tee -a "$LOG"
