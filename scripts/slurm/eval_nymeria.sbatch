#!/bin/bash
#SBATCH --job-name=psm-eval-nymeria
#SBATCH --account=dream
#SBATCH --qos=cpu_lowest
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_nymeria_%j.out
#SBATCH --error=logs/eval_nymeria_%j.err
#
# Full Nymeria 30-session eval suite for the ECCV 2026 paper.
# CPU-only — CLIP-L text-encode + PSM search + brute-force cosine.
# No GPU needed.
#
# Three stages:
#   1. E11 baselines (brute-force, sliding-window, uniform-sample)
#   2. E12 hyperparameter sweep (30 sessions × 1 seed × 3 axes)
#   3. Item 7 latency benchmark (brute-force CLIP scaling curve)
#
# Total estimated wall time: ~2h (1h baselines + 45min hyperparam + 15min latency).
# The 3h limit provides headroom.
#
# Submit with:
#   mkdir -p logs
#   sbatch scripts/slurm/eval_nymeria.sbatch
#
# Override stages:
#   STAGES="baselines" sbatch scripts/slurm/eval_nymeria.sbatch
#   STAGES="hyperparam latency" sbatch scripts/slurm/eval_nymeria.sbatch

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# All Nymeria defaults baked in; override via env if needed.
export ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
export CAPTURES="${CAPTURES:-$REPO/captures/nymeria}"
export STAGES="${STAGES:-baselines hyperparam latency}"
export SEEDS="${SEEDS:-0}"

echo "[slurm] eval_nymeria started at $(date)"
echo "[slurm] REPO=$REPO"
echo "[slurm] ROOT=$ROOT"
echo "[slurm] CAPTURES=$CAPTURES"
echo "[slurm] STAGES=$STAGES"
echo

bash "$REPO/scripts/eval_nymeria_all.sh"

echo "[slurm] eval_nymeria finished at $(date)"
