#!/bin/bash
#SBATCH --job-name=psm-extract-nymeria
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_nymeria_%A_%a.out
#SBATCH --error=logs/extract_nymeria_%A_%a.err
#SBATCH --array=0-29%16
#
# Partition is auto-inferred from QoS prefix (h200_* -> h200); don't
# pass --partition.
#
# Array job over the 30 Nymeria sessions on the cluster at
# /checkpoint/.../nymeria_partial/. Per task:
#   1. extract.py reads <session>/recording_head/data/data.vrs via
#      the existing VRS reader path (Nymeria/AEA layout).
#   2. Frames decoded at FPS from the RGB camera stream (214-1).
#   3. (lat, lng) projected from
#      <session>/recording_head/mps/slam/closed_loop_trajectory.csv
#      at fake origin (0, 0). track_mode='vrs_slam'.
#   4. CLIP-L embeddings land in
#      $OUT_ROOT/<session_id>/$OUT_BASENAME.
#
# Concurrency capped at 16 (vs 32 for Aria/Ego-Exo4D) because each
# Nymeria session decode is heavier — 8.7 GB VRS, full-session frame
# extraction takes ~10 min vs Ego-Exo4D's 30 s.
#
# Submit with:
#   sbatch scripts/slurm/extract_nymeria.sbatch
#
# Subset for smoke-test (e.g. high-mobility first):
#   sbatch --array=0-3 scripts/slurm/extract_nymeria.sbatch
#
# Env knobs:
#   ROOT          /checkpoint/.../nymeria_partial    (read-only source root)
#   OUT_ROOT      /checkpoint/.../nymeria_atomic     (writeable, has questions.yaml)
#   MANIFEST      $OUT_ROOT/.session_ids.txt         (one session_id per line; auto-built)
#   FPS           1                                  (sample fps)
#   MODEL         clip_l                             (clip|clip_l|clip_bigg)
#   OUT           $MODEL_features.h5
#   CONDA_ENV     psm
#
# Wall time per task (H200, CLIP-L @ fps=1):
#   - VRS decode + JPEG: ~5-10 min per ~15-min session
#   - CLIP-L embed: ~30 s for ~900 frames
#   Total: ~6-11 min/task; 2h ceiling is well generous.

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_partial}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
MANIFEST="${MANIFEST:-$OUT_ROOT/.session_ids.txt}"
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
  siglip2_l)
    FAMILY="siglip"
    CHECKPOINT="google/siglip2-large-patch16-256"
    DEFAULT_OUT="siglip2_l_features.h5"
    ;;
  *)
    echo "[slurm] ERR: unknown MODEL=$MODEL (expected clip|clip_l|clip_bigg|siglip2_l)" >&2
    exit 1
    ;;
esac
OUT="${OUT:-$DEFAULT_OUT}"

# Build manifest on first task; subsequent tasks just read it. Atomic
# write so concurrent tasks don't see a half-written file.
if [[ ! -f "$MANIFEST" ]]; then
  tmp="${MANIFEST}.tmp.$$"
  mkdir -p "$OUT_ROOT"
  ( cd "$OUT_ROOT" && ls -d */ 2>/dev/null | sed 's:/$::' ) | sort > "$tmp"
  mv -n "$tmp" "$MANIFEST"
  rm -f "$tmp" 2>/dev/null || true
fi
N_SESSIONS=$(wc -l < "$MANIFEST")
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_SESSIONS" ]]; then
  echo "[slurm] task=$SLURM_ARRAY_TASK_ID >= manifest size $N_SESSIONS; nothing to do" >&2
  exit 0
fi

NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
SESSION_DIR="$ROOT/$NAME"
OUT_DIR="$OUT_ROOT/$NAME"

source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
echo "[slurm] NAME=$NAME FPS=$FPS MODEL=$MODEL OUT=$OUT"
echo "[slurm] SESSION_DIR=$SESSION_DIR OUT_DIR=$OUT_DIR"

if [[ ! -d "$SESSION_DIR" ]]; then
  echo "[slurm] ERR: session dir $SESSION_DIR not found" >&2
  exit 1
fi
if [[ ! -f "$SESSION_DIR/recording_head/data/data.vrs" ]]; then
  echo "[slurm] ERR: $SESSION_DIR/recording_head/data/data.vrs missing" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# Pass session dir; orchestrator detects Nymeria/AEA layout via
# _locate_vrs_file's recording_head/data/data.vrs candidate.
python -m psm_extraction extract \
  --video      "$SESSION_DIR" \
  --output     "$OUT_DIR/$OUT" \
  --models     "$FAMILY" \
  --checkpoint "$FAMILY:$CHECKPOINT" \
  --sample-fps "$FPS" --segment-sec "$SEGMENT" \
  --session-id "$NAME" \
  --verbose

echo
echo "[slurm] done. Verify track_mode + frame count with:"
echo "  python -c \"import h5py; f=h5py.File('$OUT_DIR/$OUT'); print(dict(f['$FAMILY'].attrs))\""
