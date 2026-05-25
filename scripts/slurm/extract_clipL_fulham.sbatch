#!/bin/bash
#SBATCH --job-name=psm-extract-clipl-fulham
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/extract_clipl_fulham_%j.out
#SBATCH --error=logs/extract_clipl_fulham_%j.err
#
# Partition is auto-inferred from the QoS prefix on this cluster
# (h200_* -> h200 partition). Don't add --partition or sbatch will
# warn and ignore it.
#
# One-shot GPU job to fill the missing clip_l_features.h5 for session
# 1501677363692556 (Fulham). The other two sessions already have CLIP-L
# extractions; this closes the gap so the encoder-comparison sweep can
# cover all 3 sessions × 2 encoders.
#
# Submit with:
#   sbatch scripts/slurm/extract_clipL_fulham.sbatch
#
# Override defaults via sbatch --export, e.g.:
#   sbatch --export=ROOT=/checkpoint/dream/arjangt/video_retrieval/aria \
#          scripts/slurm/extract_clipL_fulham.sbatch
#
# Env knobs (with defaults):
#   ROOT     sessions root (default /checkpoint/dream/arjangt/video_retrieval/aria)
#   SID      session id (default 1501677363692556 — Fulham)
#   FPS      sample fps (default 3 — matches existing extractions)
#   SEGMENT  segment-sec (default 1)
#   CONDA_ENV  conda env name (default psm)
#
# Expected wall time on H200: ~10-15 min for a 15-min session at fps=3.
# Memory peak: ~12 GB GPU, ~30 GB system (h5py + ffmpeg frame cache).

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/aria}"
SID="${SID:-1501677363692556}"
FPS="${FPS:-3}"
SEGMENT="${SEGMENT:-1}"
CONDA_ENV="${CONDA_ENV:-psm}"
CHECKPOINT="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
OUT_BASENAME="clip_l_features.h5"

# --- Activate conda env ---
# Source the conda machinery; works on FAIR clusters where conda is in
# the user's PATH but not pre-activated in batch shells.
if [[ -z "${CONDA_SHLVL:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
echo "[slurm] ROOT=$ROOT SID=$SID FPS=$FPS SEGMENT=$SEGMENT"
echo "[slurm] checkpoint=$CHECKPOINT writing to $ROOT/$SID/$OUT_BASENAME"
echo

session_dir="$ROOT/$SID"
if [[ ! -d "$session_dir" ]]; then
  echo "[slurm] ERR: session dir $session_dir not found" >&2
  exit 1
fi

# Aria session layout: data.mp4 + gps.json + imu.json sit in the session dir.
python -m psm_extraction extract \
  --video      "$session_dir/data.mp4" \
  --output     "$session_dir/$OUT_BASENAME" \
  --models     clip \
  --checkpoint "clip:$CHECKPOINT" \
  --sample-fps "$FPS" --segment-sec "$SEGMENT" \
  --session-id "$SID"

echo
echo "[slurm] done. Verify with:"
echo "  h5dump -H -d /clip/embeddings $session_dir/$OUT_BASENAME"
