#!/bin/bash
# Road 1 master runner. Chains Stage A (data gen) -> Stage B (adapter train) ->
# Stage C (paired N=150 eval). Each stage skips if its output already exists,
# so this can be safely re-launched after partial failures.
# NOTE: no `set -u` -- conda's activate-binutils_linux-64.sh references $HOST
# without a default, which would trip `set -u` and abort the whole runner.
DRPN_ROOT=/root/autodl-tmp/DRPhysNav
ROAD1=$DRPN_ROOT/road1
LOG=$ROAD1/road1_master.log
mkdir -p "$ROAD1/data" "$ROAD1/ckpt"
echo "########## ROAD-1 MASTER START $(date) ##########" | tee -a "$LOG"

# Provide a safe fallback so the binutils activation hook doesn't die on $HOST.
export HOST="${HOST:-x86_64-conda_cos6-linux-gnu}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/habitat
export CUDA_HOME=/root/autodl-tmp/envs/habitat
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# ---- Stage A: gather train-split pair shards ----
H5=$ROAD1/data/frames.h5
if [ -f "$H5" ] && [ "$(/root/autodl-tmp/envs/habitat/bin/python -c \
    "import h5py,sys; sys.exit(0 if h5py.File(sys.argv[1])['clean'].shape[0]>1000 else 1)" "$H5" 2>/dev/null; echo $?)" -eq 0 ]; then
  echo "[stageA] SKIP -- $H5 already has >1000 pairs" | tee -a "$LOG"
else
  echo "===== Stage A: data gen $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
  cd $ROAD1
  # NOTE: default --split is val + 6-scene deny list; adapter data is drawn
  # only from val scenes that the N=150 paired eval never touches.
  /root/autodl-tmp/envs/habitat/bin/python stage_a_gen_data.py \
      --num_frames 1500 --max_episodes 30 --stride 10 \
      --deg_type low_light --deg_sev 4 \
      --out_dir "$ROAD1/data" >> "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ] || [ ! -f "$H5" ]; then
    echo "===== Stage A: FAILED (rc=$RC) $(date '+%m-%d %H:%M') -- aborting queue =====" | tee -a "$LOG"
    exit 1
  fi
  echo "===== Stage A: DONE $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
fi

# ---- Stage B: train residual adapter on the pair shards ----
CKPT=$ROAD1/ckpt/adapter_low_light_s4.pth
if [ -f "$CKPT" ]; then
  echo "[stageB] SKIP -- $CKPT exists" | tee -a "$LOG"
else
  echo "===== Stage B: adapter train $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
  cd $ROAD1
  /root/autodl-tmp/envs/habitat/bin/python stage_b_train_head.py \
      --data_dir "$ROAD1/data" --ckpt_dir "$ROAD1/ckpt" \
      --steps 2000 --batch 4 --lr 1e-4 --save_every 500 >> "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ] || [ ! -f "$CKPT" ]; then
    echo "===== Stage B: FAILED (rc=$RC) $(date '+%m-%d %H:%M') -- aborting queue =====" | tee -a "$LOG"
    exit 1
  fi
  echo "===== Stage B: DONE $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
fi

# ---- Stage C: paired N=150 eval with adapter ----
if [ -f "$DRPN_ROOT/runs/road1_eval/road1_R1_low_light_s4_seed0.csv" ]; then
  echo "[stageC] SKIP -- eval CSV exists" | tee -a "$LOG"
else
  echo "===== Stage C: paired N=150 eval $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
  export DRPN_ROAD1_ADAPTER="$CKPT"
  bash /root/autodl-tmp/InstructNav/run_road1_eval.sh >> "$LOG" 2>&1
  echo "===== Stage C: DONE $(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
fi

echo "########## ROAD-1 MASTER DONE $(date) ##########" | tee -a "$LOG"
