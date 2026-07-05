#!/bin/bash
#SBATCH --job-name=psm-hdd-sanity
#SBATCH --account=dream
#SBATCH --qos=h200_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Embed-sanity GATE for Honda HDD (run this ONCE before the full extraction
# array). Front-facing windshield frames (reflections, motion blur, sky) are a
# different distribution from the wearable/LiDAR corpora; if they collapse to
# near-identical embeddings, F-HDD-3's cross-session retrieval signal dies. This
# extracts ~50 frames from one drive, embeds them, and prints the pairwise
# cosine-similarity spread + an OK/DEGENERATE/FAIL verdict. Only launch
# scripts/slurm/extract_hdd.sbatch if the verdict is OK.
#
# The stats JSON is written to captures/hdd/embed_sanity.json so the verdict
# survives after the SLURM log rotates. Job exit code is non-zero on
# FAIL/DEGENERATE, so `sbatch --wait` / dependency chains can gate on it.
#
# Submit:
#   sbatch                               scripts/slurm/hdd_embed_sanity.sbatch   # clip_l
#   sbatch --export=ALL,MODEL=siglip2_l  scripts/slurm/hdd_embed_sanity.sbatch
#   sbatch --export=ALL,DRIVE=201703061033 scripts/slurm/hdd_embed_sanity.sbatch # a specific drive
#
# Env knobs:
#   ROOT   /checkpoint/dream/arjangt/video_retrieval/hdd/release_2019_07_08
#   MODEL  clip_l (default) | clip_bigg | siglip2_l
#   DRIVE  (optional) specific drive_id; default = first discovered drive
#   CONDA_ENV  psm

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/hdd/release_2019_07_08}"
MODEL="${MODEL:-clip_l}"
CONDA_ENV="${CONDA_ENV:-psm}"

case "$MODEL" in
  clip_l)     FAMILY="clip";   CHECKPOINT="laion/CLIP-ViT-L-14-laion2B-s32B-b82K" ;;
  clip_bigg)  FAMILY="clip";   CHECKPOINT="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" ;;
  siglip2_l)  FAMILY="siglip"; CHECKPOINT="google/siglip2-large-patch16-256" ;;
  *) echo "[slurm] ERR: unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

# Robust conda activation for batch shells. --export=ALL preserves the caller's
# conda-init PATH; the conda.sh fallback covers the documented case where env
# overrides drop it (see extract_lookout.sbatch).
if ! source activate "$CONDA_ENV" 2>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/arjangt/.conda)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs captures/hdd

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} model=$MODEL drive=${DRIVE:-<first>}"

DRIVE_ARGS=()
if [[ -n "${DRIVE:-}" ]]; then
  DRIVE_ARGS=(--drives "$DRIVE")
fi

python scripts/extract_hdd_sessions.py \
  --root "$ROOT" \
  --model-family "$FAMILY" \
  --checkpoint "$CHECKPOINT" \
  --sanity-only \
  --sanity-out "captures/hdd/embed_sanity_${MODEL}.json" \
  "${DRIVE_ARGS[@]}"

echo "[slurm] done (verdict + stats above; JSON at captures/hdd/embed_sanity_${MODEL}.json)."
