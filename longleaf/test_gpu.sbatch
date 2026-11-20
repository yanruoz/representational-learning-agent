#!/bin/bash
# Test GPU job: checks CUDA, then runs BiomedCLIP on a few real ROCOv2 image-caption pairs.
# Submitted by longleaf/setup_longleaf.sh (which sets WORK_DIR, REPO, partition, and log path).
# Resubmit by hand:
#   sbatch -p l40-gpu --output=$WORK_DIR/logs/test_gpu_%j.log \
#          --export=ALL,WORK_DIR=$WORK_DIR,REPO=$PWD longleaf/test_gpu.sbatch
#SBATCH --job-name=repagent-gputest
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail

: "${WORK_DIR:?set WORK_DIR}"
: "${REPO:?set REPO}"
export HF_HOME="$WORK_DIR/hf_cache"
export HF_HUB_OFFLINE=1   # weights were cached by the setup script; fail loudly if not

echo "host: $(hostname)  job: $SLURM_JOB_ID  partition: $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

"$WORK_DIR/envs/repagent/bin/python" "$REPO/longleaf/gpu_smoke.py" \
  --rocov2-dir "$WORK_DIR/data/rocov2" \
  --out "$WORK_DIR/logs/gpu_smoke_${SLURM_JOB_ID}.json"
