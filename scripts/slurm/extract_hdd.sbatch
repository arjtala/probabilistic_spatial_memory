#!/bin/bash
#SBATCH --job-name=psm-extract-hdd
#SBATCH --account=dream
#SBATCH --qos=h200_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-131
#
# Extract CLIP/SigLIP features from Honda HDD drives for PSM (F-HDD-2/3).
# One array task per drive; submit once per encoder via MODEL=.
#
# QoS: h200_dev is used to actually get scheduled on this cluster. It may cap
# concurrent array tasks, so a 132-task array trickles rather than bursts --
# that's fine. If dev rejects the array size, throttle with e.g.
# `--array=0-131%8`, or fall back to a bulk QoS for the full sweep.
#
# ── BEFORE the full array: run the embed-sanity gate (its own sbatch) ──
#   sbatch scripts/slurm/hdd_embed_sanity.sbatch
# Windshield frames (reflections, motion blur, sky) may collapse to near-
# identical embeddings, which would kill F-HDD-3's cross-session retrieval.
# The gate extracts ~50 frames from one drive and reports the cosine spread;
# only launch this array if the verdict is OK.
#
# Submit pattern:
#   sbatch                            scripts/slurm/extract_hdd.sbatch  # clip_l
#   sbatch --export=ALL,MODEL=clip_bigg  scripts/slurm/extract_hdd.sbatch
#   sbatch --export=ALL,MODEL=siglip2_l  scripts/slurm/extract_hdd.sbatch
#
# Env knobs:
#   ROOT      /checkpoint/dream/arjangt/video_retrieval/hdd/release_2019_07_08
#   OUT_ROOT  /checkpoint/dream/arjangt/video_retrieval/hdd/features
#   FPS       1     (raise for the memory-vs-area high-fps operating point)
#   MODEL     clip_l (default) | clip_bigg | siglip2_l
#   CONDA_ENV psm

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/hdd/release_2019_07_08}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/hdd/features}"
FPS="${FPS:-1}"
MODEL="${MODEL:-clip_l}"
CONDA_ENV="${CONDA_ENV:-psm}"

case "$MODEL" in
  clip_l)     FAMILY="clip";   CHECKPOINT="laion/CLIP-ViT-L-14-laion2B-s32B-b82K" ;;
  clip_bigg)  FAMILY="clip";   CHECKPOINT="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" ;;
  siglip2_l)  FAMILY="siglip"; CHECKPOINT="google/siglip2-large-patch16-256" ;;
  *) echo "[slurm] ERR: unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) model=$MODEL"

python scripts/extract_hdd_sessions.py \
  --root "$ROOT" \
  --out-root "$OUT_ROOT" \
  --fps "$FPS" \
  --model-family "$FAMILY" \
  --checkpoint "$CHECKPOINT" \
  --drive-index "$SLURM_ARRAY_TASK_ID"

echo "[slurm] done."
