#!/bin/bash
# Waits for the running road1 eval / master queue to finish, then launches
# the trajectory-vignette runs (B0/RES/ORACLEGATE, N=12, low_light s4).
LOG=/root/autodl-tmp/DRPhysNav/runs/traj_vignette/launcher.log
mkdir -p $(dirname "$LOG")
echo "[traj-launcher] waiting for road1/master queue to finish $(date)" | tee -a "$LOG"
while pgrep -f "objnav_benchmark.py.*road1_" > /dev/null || pgrep -f "master_queue.sh" > /dev/null; do
  sleep 120
done
echo "[traj-launcher] GPU free -- starting traj vignette $(date)" | tee -a "$LOG"
bash /root/autodl-tmp/InstructNav/run_traj_vignette.sh >> "$LOG" 2>&1
echo "[traj-launcher] DONE $(date)" | tee -a "$LOG"
