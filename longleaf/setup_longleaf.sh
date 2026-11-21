#!/usr/bin/env bash
# One-time Longleaf setup for the GPU steps (encoder embedding extraction, fine-tuning).
# Run on a Longleaf login node from the repo root:
#
#   bash longleaf/setup_longleaf.sh                # env + data + model weights + test GPU job
#   PARTITION=a100-gpu bash longleaf/setup_longleaf.sh   # use another GPU partition for the test
#
# What it does (each step is skipped if already done, so rerunning is safe):
#   1. Picks a /work directory:    $WORK_DIR (default /work/users/<o>/<n>/<onyen>/rep-agent)
#   2. Conda env at                $WORK_DIR/envs/repagent (Python 3.12, longleaf/requirements-gpu.txt)
#   3. Writes the exact versions to longleaf/requirements-gpu.lock.txt
#   4. Downloads both datasets to  $WORK_DIR/data/<dataset>/ (checksums verified)
#   5. Caches every applicable encoder's weights in $WORK_DIR/hf_cache (compute nodes run offline)
#   6. Submits longleaf/test_gpu.sbatch; its log and JSON land in $WORK_DIR/logs/
#
# Home has a small quota, so the env, data, and model cache all live in /work.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ONYEN="${USER}"
WORK_DIR="${WORK_DIR:-/work/users/${ONYEN:0:1}/${ONYEN:1:1}/${ONYEN}/rep-agent}"
ENV_DIR="$WORK_DIR/envs/repagent"
PARTITION="${PARTITION:-l40-gpu}"

parent="$(dirname "$WORK_DIR")"
[[ -d "$parent" ]] || { echo "ERROR: $parent does not exist. Set WORK_DIR to your /work directory and rerun." >&2; exit 1; }
mkdir -p "$WORK_DIR"/{data,hf_cache,logs,outputs,envs}
echo "[setup] repo:     $REPO"
echo "[setup] work dir: $WORK_DIR"

# ── 1. conda ──
if ! command -v conda >/dev/null; then
  module load anaconda 2>/dev/null || module load miniforge 2>/dev/null || true
fi
if ! command -v conda >/dev/null; then
  MF="$WORK_DIR/miniforge3"
  if [[ ! -x "$MF/bin/conda" ]]; then
    echo "[setup] no conda module found; installing Miniforge into $MF"
    curl -fsSL -o "$WORK_DIR/miniforge.sh" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$WORK_DIR/miniforge.sh" -b -p "$MF"
  fi
  export PATH="$MF/bin:$PATH"
fi
echo "[setup] conda: $(command -v conda)"

# ── 2. environment ──
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[setup] creating env $ENV_DIR"
  conda create -y -p "$ENV_DIR" python=3.12 pip
fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r "$REPO/longleaf/requirements-gpu.txt"
"$PY" -m pip freeze > "$REPO/longleaf/requirements-gpu.lock.txt"
echo "[setup] installed; exact versions in longleaf/requirements-gpu.lock.txt"

# ── 3. data ──
for d in pbmc_multiome rocov2; do
  "$PY" "$REPO/scripts/fetch_data.py" --dataset "$d" --data-root "$WORK_DIR/data"
done

# ── 4. model weights (download only, on CPU) ──
export HF_HOME="$WORK_DIR/hf_cache"
for d in pbmc_multiome rocov2; do
  "$PY" "$REPO/scripts/extract_embeddings.py" --dataset "$d" --download-only
done

# ── 5. test GPU job ──
JOB=$(sbatch --parsable \
  --partition="$PARTITION" \
  --output="$WORK_DIR/logs/test_gpu_%j.log" \
  --export=ALL,WORK_DIR="$WORK_DIR",REPO="$REPO" \
  "$REPO/longleaf/test_gpu.sbatch")
echo "[setup] submitted test job $JOB on $PARTITION"
echo "        check:  squeue -u $ONYEN"
echo "        result: cat $WORK_DIR/logs/test_gpu_$JOB.log $WORK_DIR/logs/gpu_smoke_$JOB.json"
