#!/bin/bash
# FUSE -- DARE component C: uncertainty-weighted dual-path detection FUSION, on the MIXED-degradation
# benchmark (per-episode random (type,severity), unknown to agent; paired across arms).
# Borrowed methods: Perception Matters (2024) inverse-uncertainty weighted aggregation;
#                   Robust Bayesian Semantic Mapping (2023) confidence-weighted fusion + overconf reg.
# Arms (all DRPN_MIX=1, SAME seed/episodes/mix-spec as run_dare.sh => identical per-episode
#       degradation => PAIRED with dare's BYPASS(B0)/RESTORE(RES) -> reuse those, don't recompute):
#   SMOKE       : FUSE blind, N=3, FUSE_LOG=1 -> verify wiring (banner + per-branch counts) first
#   FUSE        : dual-path uncertainty-weighted fusion (our method, blind)
#   FUSE_ORACLE : trust restored fully (beta=0,r=1) -> fusion UPPER BOUND
# Compare {B0, RES} (from run_dare.sh) vs {FUSE, FUSE_ORACLE} pooled, paired McNemar + bootstrap CI.
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTDIR=/root/autodl-tmp/DRPhysNav/runs/fuse
DAREDIR=/root/autodl-tmp/DRPhysNav/runs/dare           # reuse paired B0/RES from here
LOG="$OUTDIR/fuse.log"; mkdir -p "$OUTDIR"
EPISODES="${FUSE_EPISODES:-300}"; SPLIT=val; SEED="${FUSE_SEED:-0}"
export DRPN_WANDB=1 DRPN_WANDB_PROJECT=DRPhysNav DRPN_WANDB_GROUP=fuse
export WANDB_DIR=/root/autodl-tmp/DRPhysNav/wandb; mkdir -p "$WANDB_DIR"
# IDENTICAL mixture + seed as run_dare.sh -> per-episode (type,sev) matches -> paired with dare B0/RES
export DRPN_MIX=1
export DRPN_MIX_SPEC="${DRPN_MIX_SPEC:-low_light:1,2,4;low_light:2,4;gaussian_noise:4;motion_blur:4}"
base_env() {
  export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
  export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
  export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_TARG_ORACLE=0
  export DRPN_USE_ROUTER=0 DRPN_ROUTER_ORACLE=0 DRPN_ROUTER_LOG=0
  export DRPN_USE_FUSE=0 DRPN_FUSE_ORACLE=0 DRPN_FUSE_LOG=0
  export DRPN_RESTORER=""
  export DRPN_MOTIV_LOG=1 DRPN_SEED="$SEED"
  unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG
}
run_one() {  # args: arm N
  local ARM="$1" N="$2"; base_env
  case "$ARM" in
    SMOKE)         export DRPN_USE_FUSE=1 DRPN_USE_ROUTER=1 DRPN_FUSE_LOG=1 DRPN_ROUTER_LOG=1 ;;
    B0)            : ;;                                          # pure degraded (lower ref); paired w/ all arms
    FUSE_ROUTER)   export DRPN_USE_FUSE=1 DRPN_USE_ROUTER=1 ;;   # ★HERO: A(sign-map bypass)+C(fusion), handcrafted specialized
    FUSE_ROUTER_ORA) export DRPN_USE_FUSE=1 DRPN_USE_ROUTER=1 DRPN_ROUTER_ORACLE=1 ;; # route by TRUE type (A upper bound)+C
    FUSE)          export DRPN_USE_FUSE=1 ;;                     # ablation: C only (blind blanket restorer, no routing)
    FUSE_ORACLE)   export DRPN_USE_FUSE=1 DRPN_FUSE_ORACLE=1 ;;  # fusion upper bound (trust restored fully)
    RES_PIR)       export DRPN_USE_RESTORE=1 DRPN_RESTORER=promptir ;;  # ablation: learned blanket restore (PromptIR can't do low_light)
    FUSE_PIR)      export DRPN_USE_FUSE=1 DRPN_RESTORER=promptir ;;      # ablation: PromptIR(B)+fusion(C)
    *) echo "unknown arm $ARM"|tee -a "$LOG"; return ;;
  esac
  export DRPN_WANDB_ARM="$ARM" DRPN_WANDB_NAME="${ARM}_mix_seed${SEED}_N${N}"
  local CSV="$OUTDIR/fuse_${ARM}_mix_seed${SEED}.csv"
  if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt "$N" ]; then echo "SKIP(done) $(basename $CSV)"|tee -a "$LOG"; return; fi
  echo "===== [$ARM] mixed seed$SEED N=$N ($(date '+%m-%d %H:%M')) =====" | tee -a "$LOG"
  python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
  python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
}
echo "########## FUSE START $(date) N=$EPISODES seed=$SEED mix=$DRPN_MIX_SPEC ##########" | tee -a "$LOG"
run_one SMOKE 3        # wiring: expect [FUSE] banner + [ROUTER] banner + restored-path/bypass counts
echo "----- SMOKE done; verify FUSE+ROUTER banners & branch counts above -----" | tee -a "$LOG"
# ---- main 300 table (all same seed/mix => paired; B0/RES reused from DARE stage) ----
run_one FUSE_ROUTER     "$EPISODES"   # ★HERO (A+C)
run_one FUSE            "$EPISODES"   # ablation: C only
run_one FUSE_ROUTER_ORA "$EPISODES"   # routing upper bound (A oracle + C)
run_one FUSE_ORACLE     "$EPISODES"   # fusion upper bound
run_one RES_PIR         "$EPISODES"   # ablation: learned blanket restore
run_one FUSE_PIR        "$EPISODES"   # ablation: PromptIR(B)+C
# paired table (B0 from here; RES handcrafted reused from dare, identical seed/mix => same degradation)
echo "----- paired table -----" | tee -a "$LOG"
python - "$DAREDIR" "$OUTDIR" "$SEED" <<'PY' | tee -a "$LOG"
import sys, csv, os
dare, fuse, seed = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    if not os.path.exists(p): return None
    rows=[r for r in list(csv.reader(open(p)))[1:] if r]
    return [float(r[0]) for r in rows]
arms={"B0":f"{dare}/dare_BYPASS_mix_seed{seed}.csv","RES(handcraft)":f"{dare}/dare_RESTORE_mix_seed{seed}.csv",
      "FUSE_ROUTER(HERO)":f"{fuse}/fuse_FUSE_ROUTER_mix_seed{seed}.csv","FUSE(C-only)":f"{fuse}/fuse_FUSE_mix_seed{seed}.csv",
      "FUSE_ROUTER_ORA":f"{fuse}/fuse_FUSE_ROUTER_ORA_mix_seed{seed}.csv","FUSE_ORACLE":f"{fuse}/fuse_FUSE_ORACLE_mix_seed{seed}.csv",
      "RES_PIR":f"{fuse}/fuse_RES_PIR_mix_seed{seed}.csv","FUSE_PIR(B+C)":f"{fuse}/fuse_FUSE_PIR_mix_seed{seed}.csv"}
print("%-12s %5s %7s"%("arm","n","SR"))
for a,p in arms.items():
    s=load(p)
    if s: print("%-12s %5d %7.3f"%(a,len(s),sum(s)/len(s)))
    else: print("%-12s   (missing: %s)"%(a,os.path.basename(p)))
PY
echo "ALL_DONE fuse $(date)" | tee -a "$LOG"
