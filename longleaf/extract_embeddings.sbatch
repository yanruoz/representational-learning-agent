#!/bin/bash
# Frozen-encoder embeddings for one dataset on a Longleaf GPU. Same script the Mac runs on MPS.
# Compute nodes run offline, so first cache the weights on a login node (once per encoder):
#   HF_HOME=$WORK_DIR/hf_cache $WORK_DIR/envs/repagent/bin/python scripts/extract_embeddings.py \
#          --dataset rocov2 --download-only
# Then submit from the repo root (ENCODER is optional; default: every encoder that applies):
#   WORK_DIR=/work/users/${USER:0:1}/${USER:1:1}/$USER/rep-agent
#   sbatch -p l40-gpu --output=$WORK_DIR/logs/extract_%j.log \
#          --export=ALL,WORK_DIR=$WORK_DIR,DATASET=rocov2,ENCODER=dinov2_vitb14_frozen longleaf/extract_embeddings.sbatch
# Output: $WORK_DIR/cache/<dataset>/<encoder>/ (copy back to the Mac's cache/ with rsync)
#SBATCH --job-name=repagent-extract
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail

: "${WORK_DIR:?set WORK_DIR}"
: "${DATASET:?set DATASET}"
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
export HF_HOME="$WORK_DIR/hf_cache"
export HF_HUB_OFFLINE=1   # weights were cached by setup_longleaf.sh

echo "host: $(hostname)  job: $SLURM_JOB_ID  partition: $SLURM_JOB_PARTITION  commit: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

"$WORK_DIR/envs/repagent/bin/python" "$REPO/scripts/extract_embeddings.py" \
  --dataset "$DATASET" \
  --data-root "$WORK_DIR/data" \
  --cache-root "$WORK_DIR/cache" \
  --device cuda \
  --batch-size 128 \
  ${ENCODER:+--encoder "$ENCODER"}
