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
# Batch shells don't source ~/.bashrc, so conda's wrapper functions aren't
# defined. We hunt for conda.sh in the usual locations and source it
# explicitly. If $CONDA_BASE is exported at submit time, use that first.
activate_conda() {
  local base=""
  if [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    base="$CONDA_BASE"
  else
    for candidate in \
      "$HOME/miniforge3" \
      "$HOME/miniconda3" \
      "$HOME/anaconda3" \
      "/private/home/$USER/miniforge3" \
      "/private/home/$USER/miniconda3" \
      "/private/home/$USER/anaconda3" \
      "/opt/conda" \
      "/opt/homebrew/Caskroom/miniconda/base"; do
      if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
        base="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$base" ]]; then
    echo "[slurm] ERR: could not find conda.sh; set CONDA_BASE explicitly" >&2
    echo "        e.g. sbatch --export=CONDA_BASE=$HOME/miniforge3 ..." >&2
    exit 1
  fi
  echo "[slurm] sourcing $base/etc/profile.d/conda.sh"
  # shellcheck disable=SC1090,SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

activate_conda

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
