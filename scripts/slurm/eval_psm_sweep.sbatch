#!/bin/bash
#SBATCH --job-name=psm-eval-seedsweep
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_seedsweep_%j.out
#SBATCH --error=logs/eval_seedsweep_%j.err
#
# Partition is auto-inferred from the QoS prefix on this cluster
# (h200_* -> h200). Don't add --partition.
#
# Run the PSM evaluation seed sweep: for each (session, encoder, seed)
# combination, ingest the session into PSM, run all questions through
# `psm --search`, and write per-question IoU + Hit@k to a JSON. Outputs
# match the v2 naming convention so eval_aggregate.py pools them with
# the existing v2 numbers if those are present.
#
# This is the PSM half of the E11 baselines-vs-PSM comparison. The other
# half is `eval_baselines.sbatch` (which writes captures/baselines/).
#
# Submit with:
#   sbatch scripts/slurm/eval_psm_sweep.sbatch
#
# Common overrides:
#   # Just one encoder, fast turnaround:
#   sbatch --export=ENCODERS=bigG scripts/slurm/eval_psm_sweep.sbatch
#
#   # Specific seeds (e.g. only seed 0 for a smoke test):
#   sbatch --export=SEEDS=0 scripts/slurm/eval_psm_sweep.sbatch
#
# Env knobs (forwarded to scripts/eval_bigg_all.sh):
#   ROOT         sessions root (default cluster aria path)
#   CAPTURES     where JSONs land (default $REPO/captures)
#   SESSIONS     space-separated session ids
#   SEEDS        space-separated seeds (default "0 1 2 3 4")
#   ENCODERS     which encoders to sweep (default "bigG clipL")
#                each one re-runs the full session × seed loop
#   TOP          --top value passed to psm --search (default 5)
#   CODEC        --exemplar-codec (default raw)
#   CONDA_ENV    conda env name (default psm)
#
# Expected wall time: 3 sessions × 5 seeds × ~30s per (session, seed)
# PSM ingest+query = ~7.5 min per encoder. Two encoders = ~15 min total.
# Plus first-run HuggingFace text encoder warmup if cache is cold.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-psm}"

# Standard cluster pattern.
source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

# Compute nodes have no internet; force HF offline. Warm the cache from
# a login node first (see scripts/slurm/README.md "One-time setup").
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO="$(pwd)"
ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/aria}"
CAPTURES="${CAPTURES:-$REPO/captures}"
ENCODERS="${ENCODERS:-bigG clipL}"

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "[slurm] ROOT=$ROOT CAPTURES=$CAPTURES ENCODERS=$ENCODERS"

# Sanity check: psm binary must be built. eval_lookback.py defaults to
# $REPO/targets/psm; if missing, the sweep will fail at the first psm
# --search call.
if [[ ! -x "$REPO/targets/psm" ]]; then
  echo "[slurm] ERR: $REPO/targets/psm not found. Build first with:" >&2
  echo "        make cluster H3_PREFIX=\$CONDA_PREFIX HDF5_PREFIX=\$CONDA_PREFIX" >&2
  echo "        (cluster profile skips -flto, which conda's compilers don't wire up)." >&2
  exit 1
fi

# Map encoder key -> (features basename, HF checkpoint) the way
# eval_baselines_all.sh does. We re-run scripts/eval_bigg_all.sh per
# encoder rather than per (encoder, session) so the inner per-seed
# loop stays sequential and predictable.
encoder_features() {
  case "$1" in
    bigG)  echo "clip_bigg_features.h5" ;;
    clipL) echo "clip_l_features.h5" ;;
    clip)  echo "clip_features.h5" ;;
    *)     echo "" ;;
  esac
}
encoder_checkpoint() {
  case "$1" in
    bigG)  echo "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" ;;
    clipL) echo "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" ;;
    clip)  echo "openai/clip-vit-base-patch32" ;;
    *)     echo "" ;;
  esac
}

for enc in $ENCODERS; do
  feat_basename="$(encoder_features "$enc")"
  ckpt="$(encoder_checkpoint "$enc")"
  if [[ -z "$feat_basename" || -z "$ckpt" ]]; then
    echo "[slurm] WARN: unknown encoder key '$enc', skipping" >&2
    continue
  fi

  # TAG matches the naming convention in the v2 writeup so outputs slot
  # into the eval_aggregate globs (clipBigG for bigG, clipL for L).
  case "$enc" in
    bigG)  TAG="clipBigG" ;;
    clipL) TAG="clipL" ;;
    clip)  TAG="clipBase" ;;
  esac

  echo
  echo "=================================================================="
  echo "[slurm] encoder=$enc tag=$TAG features=$feat_basename"
  echo "=================================================================="

  ROOT="$ROOT" \
  CAPTURES="$CAPTURES" \
  CHECKPOINT="$ckpt" \
  FEATURES="$feat_basename" \
  TAG="$TAG" \
  bash scripts/eval_bigg_all.sh
done

echo
echo "[slurm] done. Aggregate cross-method (baselines + PSM) with:"
echo "  python scripts/eval_aggregate.py --by-seed --label-from-features \\"
echo "    $CAPTURES/eval_*_clipBigG_e128_s*.json \\"
echo "    $CAPTURES/baselines/*.json"
