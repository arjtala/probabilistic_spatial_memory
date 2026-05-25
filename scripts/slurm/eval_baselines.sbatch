#!/bin/bash
#SBATCH --job-name=psm-eval-baselines
#SBATCH --account=dream
#SBATCH --qos=cpu_lowest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_baselines_%j.out
#SBATCH --error=logs/eval_baselines_%j.err
#
# Partition is auto-inferred from the QoS prefix on this cluster
# (cpu_lowest -> cpu partition). Don't add --partition or sbatch will
# warn and ignore it.
#
# Run the three E11 retrieval baselines (brute-force CLIP, sliding-window
# CLIP, uniform-sample CLIP) across all sessions × encoders. CPU-only —
# the baselines rank pre-extracted embeddings via dot product, no GPU
# needed. The text encoder runs once per question for ~60 questions
# total at <1s wall clock.
#
# Output JSONs land under $REPO/captures/baselines/ and are schema-
# compatible with eval_lookback.py output so eval_aggregate.py can pool
# baselines + PSM in one table.
#
# Submit with:
#   sbatch scripts/slurm/eval_baselines.sbatch
#
# Override defaults via sbatch --export, e.g.:
#   sbatch --export=ROOT=/checkpoint/dream/arjangt/video_retrieval/aria,ENCODERS=bigG \
#          scripts/slurm/eval_baselines.sbatch
#
# Env knobs (with defaults — these are forwarded to eval_baselines_all.sh):
#   ROOT         sessions root
#   SESSIONS     space-separated session ids
#   ENCODERS     space-separated encoder keys ("bigG", "clipL", "clip")
#   OUT_DIR      where JSONs land (default $REPO/captures/baselines)
#   TOP          --top value (default 5)
#   WINDOW_SIZES sliding-window sweep (default "3,5,10")
#   SAMPLE_RATES uniform-sample sweep (default "30,75")
#   CONDA_ENV    conda env name (default psm)
#
# Expected wall time: ~5-10 min for 3 sessions × 2 encoders × 3 methods.
# If the text encoder downloads weights on first run, add ~2 min.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-psm}"

# Standard cluster pattern: `source activate <env>` works in batch
# shells without needing conda init.
source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname)"
echo "[slurm] forwarding to scripts/eval_baselines_all.sh"
echo

# Default ROOT to the cluster data path if not overridden. Falls back to
# the repo-local default the script already has if neither is set.
export ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/aria}"

bash scripts/eval_baselines_all.sh
