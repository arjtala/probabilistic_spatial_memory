#!/bin/bash
#SBATCH --job-name=psm-hyperparam-sweep
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_hyperparam_%j.out
#SBATCH --error=logs/eval_hyperparam_%j.err
#
# Partition is auto-inferred from the QoS prefix on this cluster
# (h200_* -> h200). Don't add --partition.
#
# Run the E12 hyperparameter sensitivity sweep on the cluster. For each
# axis (H3 resolution, retention window, reservoir size), holds the
# other two at v2 defaults and varies the swept value across a
# representative range. Per-axis: 5 values x 3 sessions x 5 seeds = 75
# eval_lookback.py invocations. Three axes => 225 runs total.
#
# Per-call cost is dominated by `targets/psm --search` over the session
# (~1s) plus CLIP text encoding (~50ms/question x 20 questions). H200
# is overkill for the encoding side but the env is unified with the
# eval_psm_sweep job, so reuse the same QoS. Expected wall time:
# ~30-60 min total when HF cache is warm.
#
# Submit with:
#   sbatch scripts/slurm/eval_hyperparam.sbatch
#
# Override defaults via sbatch --export, e.g.:
#   # Just the H3 axis:
#   sbatch --export=AXES=h3_res scripts/slurm/eval_hyperparam.sbatch
#
#   # Both encoders (1 axis at a time keeps each job under wall-time):
#   sbatch --export=ENCODER=clipL scripts/slurm/eval_hyperparam.sbatch
#
# Env knobs (forwarded to eval_hyperparam_sweep.sh):
#   ROOT             sessions root (default cluster aria path)
#   CAPTURES         where JSONs land (default $REPO/captures/hyperparam)
#   SESSIONS         space-separated session ids
#   SEEDS            space-separated seeds (default "0 1 2 3 4")
#   ENCODER          bigG (default), clipL, or clip
#   AXES             which axes to sweep (default all three)
#   H3_RES_LIST      "8 9 10 11 12"
#   RETENTION_LIST   "30x24 45x16 75x12 150x6 300x3"
#   EXEMPLARS_LIST   "16 32 64 128 256"
#   CONDA_ENV        conda env name (default psm)

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

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"

# Sanity check: psm binary must be built.
if [[ ! -x "$REPO/targets/psm" ]]; then
  echo "[slurm] ERR: $REPO/targets/psm not found. Build first with:" >&2
  echo "        make cluster H3_PREFIX=\$CONDA_PREFIX HDF5_PREFIX=\$CONDA_PREFIX" >&2
  exit 1
fi

# Default ROOT to the cluster data path if not overridden.
export ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/aria}"

bash scripts/eval_hyperparam_sweep.sh

echo
echo "[slurm] done. Generate F3 sensitivity plot with:"
echo "  python scripts/eval_hyperparam_plot.py $REPO/captures/hyperparam"
