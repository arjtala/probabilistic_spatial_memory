#!/bin/bash
#SBATCH --job-name=psm-extract-lookout
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/extract_lookout_%A_%a.out
#SBATCH --error=logs/extract_lookout_%A_%a.err
#SBATCH --array=0-9
#
# Extract CLIP features from the LookOut top-10 v1 paper subset.
# Sequence list (sorted by bbox extent desc, stratified by location):
#   0 Mainquad_jan10            Stanford       321m
#   1 Sanmateopark_garage_jan11 San Mateo      318m
#   2 Fostersquare1_jan16       Foster City    260m
#   3 BurlingameDT5_feb5        Burlingame     240m
#   4 SanmateoDT2_Jan12         San Mateo      235m
#   5 Gates_to_mainquad_jan10   Stanford       224m
#   6 Huang_Gates_jan10         Stanford       206m
#   7 BurlingameDT4_feb5        Burlingame     199m
#   8 SSC3_jan17_               Stanford SSC   186m
#   9 Hillsdale6_jan14          Hillsdale      178m
#
# Submit twice for the clipL + bigG encoder ablation. NOTE: pass
# `--export=ALL,...` (not just `--export=...`) so the conda init
# environment from your login shell is preserved alongside the
# overrides; the script also falls back to sourcing conda.sh
# explicitly, but ALL is the cleanest signal of intent:
#
#   sbatch --export=ALL                       scripts/slurm/extract_lookout.sbatch
#   sbatch --export=ALL,MODEL=clip_bigg       scripts/slurm/extract_lookout.sbatch
#
# Env knobs:
#   ROOT       /checkpoint/dream/arjangt/LookOut-unzipped/aria_navigation_blurred
#   OUT_ROOT   /checkpoint/dream/arjangt/video_retrieval/lookout
#   FPS        1
#   MODEL      clip_l (default) | clip_bigg
#   SEQUENCES  space-separated allowlist; defaults to the 10 top-bbox keepers
#   CONDA_ENV  psm
#
# Per-task wall time on H200 with warm HF cache:
#   - clip_l: ~3-8 min (200-600 frames at 1 fps, depending on session length)
#   - bigG:   ~8-20 min (5× more compute per frame)

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/LookOut-unzipped/aria_navigation_blurred}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/lookout}"
FPS="${FPS:-1}"
MODEL="${MODEL:-clip_l}"
CONDA_ENV="${CONDA_ENV:-psm}"

# Top-10 v1 paper subset (sorted by bbox desc). Edit the SEQUENCES env
# var to override if running a different scope (e.g. all 46 r10-qualifying
# sequences for camera-ready).
DEFAULT_SEQS="Mainquad_jan10 Sanmateopark_garage_jan11 Fostersquare1_jan16 BurlingameDT5_feb5 SanmateoDT2_Jan12 Gates_to_mainquad_jan10 Huang_Gates_jan10 BurlingameDT4_feb5 SSC3_jan17_ Hillsdale6_jan14"
SEQUENCES="${SEQUENCES:-$DEFAULT_SEQS}"

case "$MODEL" in
  clip_l)
    ENCODER="clipL"
    ;;
  clip_bigg)
    ENCODER="bigG"
    ;;
  siglip2_l)
    ENCODER="siglip2L"
    ;;
  *)
    echo "[slurm] ERR: unknown MODEL=$MODEL (expected clip_l|clip_bigg|siglip2_l)" >&2
    exit 1
    ;;
esac

# Pick SEQUENCES[SLURM_ARRAY_TASK_ID].
read -ra SEQ_ARR <<< "$SEQUENCES"
N=${#SEQ_ARR[@]}
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]]; then
  echo "[slurm] task=$SLURM_ARRAY_TASK_ID >= $N sequences; nothing to do" >&2
  exit 0
fi
NAME="${SEQ_ARR[$SLURM_ARRAY_TASK_ID]}"

# Activate the conda env explicitly via the hook so we don't rely on
# PATH coming through SBATCH's env propagation. `source activate` (the
# old shortcut) silently fails with "activate: No such file or directory"
# when --export drops the conda-init PATH entries — which happened on
# the bigG array when the submit used `--export=MODEL=...` instead of
# `--export=ALL,MODEL=...`.
CONDA_BASE="${CONDA_BASE:-/home/arjangt/.conda}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  echo "[slurm] ERR: conda.sh not found; cannot activate env" >&2
  exit 1
fi
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "[slurm] sequence=$NAME encoder=$ENCODER fps=$FPS"
echo "[slurm] root=$ROOT out_root=$OUT_ROOT"

python scripts/extract_lookout_sessions.py \
  --root "$ROOT" \
  --out-root "$OUT_ROOT" \
  --encoder "$ENCODER" \
  --fps "$FPS" \
  --sequences "$NAME"

echo "[slurm] done."
