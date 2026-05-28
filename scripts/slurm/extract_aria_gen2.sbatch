#!/bin/bash
#SBATCH --job-name=psm-extract-aria-gen2
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/extract_aria_gen2_%A_%a.out
#SBATCH --error=logs/extract_aria_gen2_%A_%a.err
#SBATCH --array=0-11
#
# Partition is auto-inferred from the QoS prefix on this cluster
# (h200_* -> h200 partition). Don't add --partition or sbatch will
# warn and ignore it.
#
# Array job that runs the VRS extractor over all 12 Aria Gen 2 Pilot
# sequences (see journal/manifests/aria_gen2_subset_v1.yaml). One
# array task per session, parallelism limited by H200 availability
# in the shared QoS. Each task:
#   1. Reads VRS frames from $ROOT/$NAME/video.vrs at FPS (default 1)
#   2. Resolves per-frame (lat, lng) via GPS stream 281-2 when
#      present, else SLAM closed_loop_trajectory.csv (track_mode
#      reported in features.h5).
#   3. Runs CLIP-L over frames and writes $ROOT/$NAME/$OUT_BASENAME.
#
# Submit with:
#   sbatch scripts/slurm/extract_aria_gen2.sbatch
#
# Re-run a subset by overriding --array, e.g. only walk_*:
#   sbatch --array=0,1 scripts/slurm/extract_aria_gen2.sbatch
#
# Env knobs (with defaults):
#   ROOT     sessions root (default /checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot)
#   FPS      sample fps (default 1 — matches v2 reference)
#   SEGMENT  segment-sec (default 1)
#   MODEL    one of clip | clip_l | clip_bigg (default clip_l)
#   OUT      output basename (default $MODEL_features.h5)
#   CONDA_ENV  conda env name (default psm)
#
# Wall time estimate per session (H200, CLIP-L @ fps=1):
#   - VRS decode + JPEG: ~3-5 min for the 2-3 GB recordings
#   - CLIP-L embed: ~30 s for the ~300-1200 frames per session
#   Total: ~5-10 min per task; the 1h30 ceiling is generous.

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot}"
FPS="${FPS:-1}"
SEGMENT="${SEGMENT:-1}"
MODEL="${MODEL:-clip_l}"
CONDA_ENV="${CONDA_ENV:-psm}"

# Map MODEL knob -> (family arg, checkpoint, default output basename).
# Keep these in sync with scripts/extract_*_all.sh — same names, same
# checkpoints, so eval scripts can find features by glob.
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

# Names in the same order as the manifest's `sessions:` list (and the
# downloader's menu indices). SLURM_ARRAY_TASK_ID 0..11 picks one.
SESSIONS=(
  walk_0
  walk_1
  cook_0
  clean_0
  eat_0
  eat_1
  eat_2
  eat_3
  play_0
  play_1
  play_2
  play_3
)

NAME="${SESSIONS[$SLURM_ARRAY_TASK_ID]}"

# Standard cluster pattern: `source activate <env>` works in batch
# shells without needing conda init.
source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

# Compute nodes have no internet. Force HF transformers to use the
# local cache only; warm the cache from a login node first if needed
# (see scripts/slurm/README.md).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
echo "[slurm] NAME=$NAME ROOT=$ROOT FPS=$FPS SEGMENT=$SEGMENT MODEL=$MODEL OUT=$OUT"

session_dir="$ROOT/$NAME"
if [[ ! -d "$session_dir" ]]; then
  echo "[slurm] ERR: session dir $session_dir not found — run the downloader first" >&2
  exit 1
fi
if [[ ! -f "$session_dir/video.vrs" ]]; then
  echo "[slurm] ERR: $session_dir/video.vrs missing — main_vrs not downloaded?" >&2
  exit 1
fi

# Pass the session dir (not video.vrs) — the orchestrator detects VRS
# layouts and routes via read_vrs_session.
python -m psm_extraction extract \
  --video      "$session_dir" \
  --output     "$session_dir/$OUT" \
  --models     "$FAMILY" \
  --checkpoint "$FAMILY:$CHECKPOINT" \
  --sample-fps "$FPS" --segment-sec "$SEGMENT" \
  --session-id "$NAME" \
  --verbose

echo
echo "[slurm] done. Verify track_mode with:"
echo "  python -c \"import h5py; f=h5py.File('$session_dir/$OUT'); print(dict(f.attrs)); print(dict(f['$FAMILY'].attrs))\""
