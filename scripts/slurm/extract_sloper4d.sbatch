#!/bin/bash
#SBATCH --job-name=psm-extract-sloper4d
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_sloper4d_%A_%a.out
#SBATCH --error=logs/extract_sloper4d_%A_%a.err
#SBATCH --array=0-4
#
# Extract CLIP features from SLOPER4D urban sequences. One array task
# per sequence × encoder combination; submit twice (once for each
# encoder via MODEL=clip_l / MODEL=clip_bigg).
#
# Submit pattern:
#   sbatch                     scripts/slurm/extract_sloper4d.sbatch  # clip_l
#   sbatch --export=MODEL=clip_bigg scripts/slurm/extract_sloper4d.sbatch  # bigG
#
# Env knobs:
#   ROOT          /checkpoint/dream/arjangt/SLOPER4D-unzipped       (read source)
#   OUT_ROOT      /checkpoint/dream/arjangt/video_retrieval/sloper4d (write features)
#   FPS           1
#   MODEL         clip_l (default) | clip_bigg
#   ORIGIN_LAT    24.4381  (Xiamen University)
#   ORIGIN_LNG    118.0992
#   SEQUENCES     space-separated allowlist; defaults to the 5 street/r10+
#                 keepers (drops seq002_football_001 which is room-scale)
#   CONDA_ENV     psm
#
# Per-task wall time on H200 with warm HF cache:
#   - clip_l: ~1-3 min per sequence (~200-900 frames @ 1 fps)
#   - bigG:   ~3-8 min per sequence (5x more compute)

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/SLOPER4D-unzipped}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/sloper4d}"
FPS="${FPS:-1}"
MODEL="${MODEL:-clip_l}"
ORIGIN_LAT="${ORIGIN_LAT:-24.4381}"
ORIGIN_LNG="${ORIGIN_LNG:-118.0992}"
CONDA_ENV="${CONDA_ENV:-psm}"

# 5 r10/r11+ keepers (seq002_football_001 dropped — room-scale only,
# 22m bbox doesn't span any r10 cells).
DEFAULT_SEQS="seq003_street_002 seq005_library_002 seq007_garden_001 seq008_running_001 seq009_running_002"
SEQUENCES="${SEQUENCES:-$DEFAULT_SEQS}"

case "$MODEL" in
  clip_l)
    FAMILY="clip"
    CHECKPOINT="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    ;;
  clip_bigg)
    FAMILY="clip"
    CHECKPOINT="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    ;;
  siglip2_l)
    FAMILY="siglip"
    CHECKPOINT="google/siglip2-large-patch16-256"
    ;;
  *)
    echo "[slurm] ERR: unknown MODEL=$MODEL (expected clip_l|clip_bigg|siglip2_l)" >&2
    exit 1
    ;;
esac

# Pick the SEQUENCES[SLURM_ARRAY_TASK_ID]
read -ra SEQ_ARR <<< "$SEQUENCES"
N=${#SEQ_ARR[@]}
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]]; then
  echo "[slurm] task=$SLURM_ARRAY_TASK_ID >= $N sequences; nothing to do" >&2
  exit 0
fi
NAME="${SEQ_ARR[$SLURM_ARRAY_TASK_ID]}"

# Standard cluster pattern; works in non-interactive batch shells.
source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

# Compute nodes have no internet; force HF offline. Warm the cache
# from a login node first (see scripts/slurm/README.md).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "[slurm] sequence=$NAME model=$MODEL checkpoint=$CHECKPOINT"
echo "[slurm] root=$ROOT out_root=$OUT_ROOT fps=$FPS origin=$ORIGIN_LAT,$ORIGIN_LNG"

python scripts/extract_sloper4d_sessions.py \
  --root "$ROOT" \
  --out-root "$OUT_ROOT" \
  --fps "$FPS" \
  --origin-lat "$ORIGIN_LAT" \
  --origin-lng "$ORIGIN_LNG" \
  --model-family "$FAMILY" \
  --checkpoint "$CHECKPOINT" \
  --sequences "$NAME"

echo "[slurm] done."
