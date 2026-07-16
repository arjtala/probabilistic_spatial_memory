#!/bin/bash
#SBATCH --job-name=psm-fixed-budget
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fixed_budget_%j.out
#SBATCH --error=logs/fixed_budget_%j.err
#
# Partition is auto-inferred from the QoS prefix (h200_* -> h200). Don't
# add --partition.
#
# Fixed-exemplar-budget retention study (WearablesAI novelty experiment,
# journal/wearables_novelty_experiments.md). For each of the 14 street-scale
# sessions it runs eval_fixed_budget.py over the retention policies at a fixed
# GLOBAL exemplar budget M, matched by bytes and candidate count. Only the CLIP
# text encoder runs (frame features are precomputed), so this is light despite
# needing a GPU for query encoding.
#
# Submit:
#   sbatch scripts/slurm/fixed_budget_suite.sbatch
# Smoke (few sessions / seeds):
#   sbatch --export=SMOKE=1 scripts/slurm/fixed_budget_suite.sbatch
# Grid/null controls (set MODE):
#   sbatch --export=MODE=grid  scripts/slurm/fixed_budget_suite.sbatch
#   sbatch --export=MODE=null  scripts/slurm/fixed_budget_suite.sbatch
#
# Env knobs:
#   PSM_DATA_ROOT  dataset root (default /checkpoint/dream/arjangt)
#   BUDGETS        default 128 (contract primary); sweep "64,128,256,512"
#   SEEDS          default 0,1,2,3,4
#   H3_RESOLUTIONS default 12
#   OUT_ROOT       default captures/wearables_fixed_budget
#   MODE           base (default) | grid | null
#   SMOKE          1 -> 2 sessions, seeds 0,1 only

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-psm}"
source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PY:-python}"

export PSM_DATA_ROOT="${PSM_DATA_ROOT:-/checkpoint/dream/arjangt}"
VR="$PSM_DATA_ROOT/video_retrieval"
BUDGETS="${BUDGETS:-128}"
SEEDS="${SEEDS:-0,1,2,3,4}"
H3_RESOLUTIONS="${H3_RESOLUTIONS:-12}"
OUT_ROOT="${OUT_ROOT:-captures/wearables_fixed_budget}"
MODE="${MODE:-base}"
CLIP_DEVICE="${CLIP_DEVICE:-cuda}"

LOOKOUT=(Mainquad_jan10 Sanmateopark_garage_jan11 Fostersquare1_jan16 \
         BurlingameDT5_feb5 SanmateoDT2_Jan12 Gates_to_mainquad_jan10 \
         Huang_Gates_jan10 BurlingameDT4_feb5 SSC3_jan17_ Hillsdale6_jan14)
SLOPER4D=(seq003_street_002 seq008_running_001 seq009_running_002)
NYMERIA=(20230608_s0_shelby_arroyo_act0_3ciwl8)

# Method set: full for base; spatial-only for controls (per the protocol).
METHODS_BASE="global_reservoir,fifo,hybrid,uniform_time,semantic_kcenter,spatial_priority,spatial_balanced,visit_balanced"
METHODS_CTRL="spatial_priority,spatial_balanced,visit_balanced"

if [[ "${SMOKE:-0}" == "1" ]]; then
  LOOKOUT=(Mainquad_jan10); SLOPER4D=(seq003_street_002); NYMERIA=()
  SEEDS="0,1"
fi

run_corpus () {  # $1=root_subdir  $2..=sessions
  local sub="$1"; shift
  [[ $# -gt 0 ]] || return 0
  ROOT="$VR/$sub" PY="$PY" BUDGETS="$BUDGETS" SEEDS="$SEEDS" \
    H3_RESOLUTIONS="$H3_RESOLUTIONS" METHODS="$METHODS" \
    CLIP_DEVICE="$CLIP_DEVICE" OUT_ROOT="$OUT_DIR" \
    bash scripts/run_wearables_budget_suite.sh "$@"
}

case "$MODE" in
  base)
    METHODS="$METHODS_BASE"; OUT_DIR="$OUT_ROOT"
    run_corpus lookout "${LOOKOUT[@]}"
    run_corpus sloper4d "${SLOPER4D[@]}"
    run_corpus nymeria_atomic "${NYMERIA[@]}"
    ;;
  grid)
    # Grid translate/rotate boundary controls (spatial methods only).
    METHODS="$METHODS_CTRL"
    for variant in "--translate-east-m 4.5" "--translate-north-m 4.5" "--rotation-deg 30"; do
      tag=$(echo "$variant" | tr -d ' -' | tr 'A-Z' 'a-z')
      OUT_DIR="captures/grid_controls/$tag"
      EXTRA_ARGS="$variant" run_corpus lookout "${LOOKOUT[@]}"
      EXTRA_ARGS="$variant" run_corpus sloper4d "${SLOPER4D[@]}"
    done
    ;;
  null)
    # Broken-alignment coordinate-permutation null (spatial methods only).
    METHODS="$METHODS_CTRL"
    for pseed in 101 202 303; do
      OUT_DIR="captures/coordinate_null/p$pseed"
      EXTRA_ARGS="--coord-permutation-seed $pseed" run_corpus lookout "${LOOKOUT[@]}"
      EXTRA_ARGS="--coord-permutation-seed $pseed" run_corpus sloper4d "${SLOPER4D[@]}"
    done
    ;;
  *) echo "[FATAL] unknown MODE=$MODE" >&2; exit 2 ;;
esac

echo "[fixed_budget_suite] MODE=$MODE done."
