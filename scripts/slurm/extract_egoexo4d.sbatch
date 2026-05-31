#!/bin/bash
#SBATCH --job-name=psm-extract-egoexo4d
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/extract_egoexo4d_%A_%a.out
#SBATCH --error=logs/extract_egoexo4d_%A_%a.err
#SBATCH --array=0-695%32
#
# Partition auto-inferred from QoS prefix; don't pass --partition.
#
# Array job that runs the Ego-Exo4D extractor over all 696 takes whose
# atomic_descriptions provide >=1 ego-visible question (see
# /checkpoint/dream/arjangt/video_retrieval/egoexo4d_atomic/<take_uid>/
# questions.yaml, produced by scripts/egoexo4d_atomic_to_questions.py).
#
# Per task:
#   1. extract.py reads /datasets/egoexo4d/v2/takes/<take_uid>/ as an
#      Ego-Exo4D take directory.
#   2. Frames come from frame_aligned_videos/aria01_214-1.mp4 via the
#      existing ffmpeg extract_frames path (~30s/take at fps=1).
#   3. (lat, lng) comes from trajectory/closed_loop_trajectory.csv
#      projected at fake (0,0) origin -> track_mode=egoexo4d_slam.
#   4. CLIP-L embeddings land in
#      /checkpoint/.../egoexo4d_atomic/<take_uid>/$OUT_BASENAME.
#
# Submit with:
#   sbatch scripts/slurm/extract_egoexo4d.sbatch
#
# Subset a smaller range for smoke-test:
#   sbatch --array=0-3 scripts/slurm/extract_egoexo4d.sbatch
#
# Env knobs (with defaults):
#   TAKES_ROOT     /datasets/egoexo4d/v2/takes   (read-only source)
#   OUT_ROOT       /checkpoint/.../egoexo4d_atomic (writeable, has questions.yaml)
#   MANIFEST       $OUT_ROOT/.take_uids.txt     (one take_uid per line; auto-generated)
#   FPS            sample fps (default 1)
#   MODEL          clip | clip_l | clip_bigg (default clip_l)
#   OUT            output basename (default $MODEL_features.h5)
#   CONDA_ENV      conda env name (default psm)
#
# Wall time per task (H200, CLIP-L @ fps=1): ~1-3 min typically; the
# longest takes (1106 q across e.g. a 30-min take) hit ~5 min. 30 min
# ceiling is generous.

set -euo pipefail

TAKES_ROOT="${TAKES_ROOT:-/datasets/egoexo4d/v2/takes}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/egoexo4d_atomic}"
MANIFEST="${MANIFEST:-$OUT_ROOT/.take_uids.txt}"
FPS="${FPS:-1}"
SEGMENT="${SEGMENT:-1}"
MODEL="${MODEL:-clip_l}"
CONDA_ENV="${CONDA_ENV:-psm}"

case "$MODEL" in
  clip)
    FAMILY="clip"
    CHECKPOINT="openai/clip-vit-base-patch32"
    DEFAULT_OUT="clip_features.h5"
    ;;
  clip_l)
    FAMILY="clip"
    CHECKPOINT="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    DEFAULT_OUT="clip_l_features.h5"
    ;;
  clip_bigg)
    FAMILY="clip"
    CHECKPOINT="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    DEFAULT_OUT="clip_bigg_features.h5"
    ;;
  *)
    echo "[slurm] ERR: unknown MODEL=$MODEL (expected clip|clip_l|clip_bigg)" >&2
    exit 1
    ;;
esac
OUT="${OUT:-$DEFAULT_OUT}"

# Build manifest on first task; subsequent tasks read it. The manifest
# is the sorted list of take_uid directory names under $OUT_ROOT (one
# per take with a questions.yaml). Single-writer race avoided by
# guarding on the file existing first.
if [[ ! -f "$MANIFEST" ]]; then
  # Write atomically so concurrent tasks don't see a half-written file.
  tmp="${MANIFEST}.tmp.$$"
  ( cd "$OUT_ROOT" && ls -d */ 2>/dev/null | sed 's:/$::' ) | sort > "$tmp"
  mv -n "$tmp" "$MANIFEST"
  rm -f "$tmp" 2>/dev/null || true
fi
N_TAKES=$(wc -l < "$MANIFEST")
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_TAKES" ]]; then
  echo "[slurm] task=$SLURM_ARRAY_TASK_ID >= manifest size $N_TAKES; nothing to do" >&2
  exit 0
fi

NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
TAKE_DIR="$TAKES_ROOT/$NAME"
OUT_DIR="$OUT_ROOT/$NAME"

source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
echo "[slurm] NAME=$NAME FPS=$FPS MODEL=$MODEL OUT=$OUT"
echo "[slurm] TAKE_DIR=$TAKE_DIR OUT_DIR=$OUT_DIR"

if [[ ! -d "$TAKE_DIR" ]]; then
  echo "[slurm] ERR: take dir $TAKE_DIR not found" >&2
  exit 1
fi
if [[ ! -f "$TAKE_DIR/frame_aligned_videos/aria01_214-1.mp4" ]]; then
  echo "[slurm] ERR: $TAKE_DIR/frame_aligned_videos/aria01_214-1.mp4 missing" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# The orchestrator detects Ego-Exo4D take dirs via the egoexo4d_take
# helper; we just pass the take dir as --video.
python -m psm_extraction extract \
  --video      "$TAKE_DIR" \
  --output     "$OUT_DIR/$OUT" \
  --models     "$FAMILY" \
  --checkpoint "$FAMILY:$CHECKPOINT" \
  --sample-fps "$FPS" --segment-sec "$SEGMENT" \
  --session-id "$NAME" \
  --verbose

echo
echo "[slurm] done. Verify with:"
echo "  python -c \"import h5py; f=h5py.File('$OUT_DIR/$OUT'); print(dict(f.attrs)); print(dict(f['$FAMILY'].attrs))\""
