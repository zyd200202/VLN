#!/bin/bash
# CRV diagnostic smoke@40 on gaussian_noise s4 seed0.
# Goal: answer "can CRV ever work?" via the [CRV][RECALL] aggregate, and get the
# aggressive-fire SR direction in ONE run.
#  - rel_gate=0.99  -> log almost every VLM-declined frame (recall denominator)
#  - area=2500 frames=1 conf/floor=0.10 -> fire aggressively (so SR shows help/harm)
# Paired control: maintable/mt_B0_gaussian_noise_s4_seed0.csv (head-40 identical order).
cd /root/autodl-tmp/InstructNav
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTDIR=/root/autodl-tmp/DRPhysNav/runs/crv
LOG="$OUTDIR/crv_diag.log"
mkdir -p "$OUTDIR"
SEV=4; SPLIT=val; SEED=0; COND=gaussian_noise; N=40

# base env (all DRPN modules off)
export DRPN_USE_CNP=0 DRPN_USE_RESTORE=0 DRPN_USE_UAP=0 DRPN_USE_THF=0
export DRPN_USE_ARBITER=0 DRPN_USE_UEXP=0 DRPN_UEXP_NOGATE=0 DRPN_ARBITER_REL_GATE=0
export DRPN_USE_DDA=0 DRPN_USE_TARG=0 DRPN_USE_ROUTER=0 DRPN_MIX=0 DRPN_USE_REVOKE=0
export DRPN_DEGRADE_SEVERITY=$SEV DRPN_MOTIV_LOG=1
export DRPN_SEED="$SEED" DRPN_DEGRADE_TYPE="$COND"
unset DRPN_TARG_DEFAULT DRPN_TARG_MARGIN DRPN_TARG_LOG DRPN_TARG_ORACLE

# CRV: aggressive fire + full recall logging
export DRPN_USE_CRV=1 DRPN_CRV_REL_GATE=0.99 DRPN_CRV_CONF=0.10 \
       DRPN_CRV_FLOOR=0.10 DRPN_CRV_AREA=2500 DRPN_CRV_FRAMES=1 DRPN_CRV_MAX=3 DRPN_CRV_DIAG=0

export DRPN_WANDB=0
CSV="$OUTDIR/crvdiag_gaussian_noise_s4_seed0_N${N}.csv"
echo "############ CRV-DIAG START $(date) cond=$COND sev=$SEV N=$N ############" | tee -a "$LOG"
python objnav_benchmark.py --split "$SPLIT" --eval_episodes "$N" --out_csv "$CSV" --save_traj 0 >> "$LOG" 2>&1
echo "==== RECALL / FIRE summary ====" | tee -a "$LOG"
grep -E "\[CRV\]\[RECALL\]|forced_commits|force-commit|CRV\]\[WARN" "$LOG" | tail -n 8 | tee -a "$LOG"
python - "$CSV" <<'PY' | tee -a "$LOG"
import sys, csv
rows=[r for r in list(csv.reader(open(sys.argv[1])))[1:] if r]
if rows:
    n=len(rows); sr=sum(float(r[0]) for r in rows)/n; spl=sum(float(r[1]) for r in rows)/n
    print("SUMMARY %s n=%d SR=%.3f SPL=%.3f"%(sys.argv[1].split('/')[-1],n,sr,spl))
PY
echo "ALL_DONE crv_diag $(date)" | tee -a "$LOG"
