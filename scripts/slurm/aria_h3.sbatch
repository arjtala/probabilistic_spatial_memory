#!/bin/bash
#SBATCH --job-name=psm-aria-h3
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/aria_h3_%j.out
#SBATCH --error=logs/aria_h3_%j.err
#
# H3-resolution acceptance test for an Aria Gen 2 session (steps 2 + 3
# of scripts/aria_acceptance.sh). Step 1 (Gemini frame captioning) is
# intentionally NOT included — it needs internet access + GEMINI_API_KEY
# and is fast (~1-2 min) locally. Run that first so questions.yaml
# lands on disk; this job goes straight to the H3-res sweep.
#
# Submit:
#   sbatch --export=ALL,SESSION=walk_0 scripts/slurm/aria_h3.sbatch
#   sbatch --export=ALL,SESSION=walk_1 scripts/slurm/aria_h3.sbatch
#
# Env knobs (with defaults):
#   SESSION       walk_0
#   SEEDS         "0 1 2 3 4"
#   H3_RES_LIST   "8 9 10 11 12"
#   ENCODERS      "clipL"          (add bigG once an Aria bigG H5 exists)
#   OUT_ROOT      /checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot
#   CONDA_ENV     psm
#
# Wall time on h200 with warm HF cache: ~5-10 min total.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-psm}"
SESSION="${SESSION:-walk_0}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot}"
SEEDS="${SEEDS:-0 1 2 3 4}"
H3_RES_LIST="${H3_RES_LIST:-8 9 10 11 12}"
ENCODERS="${ENCODERS:-clipL}"

source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO="$(pwd)"
CAPTURES="$REPO/captures/aria_${SESSION}_h3"
QUESTIONS="$OUT_ROOT/$SESSION/questions.yaml"

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "[slurm] session=$SESSION encoders=$ENCODERS seeds=$SEEDS h3=$H3_RES_LIST"
echo "[slurm] questions=$QUESTIONS"
echo "[slurm] captures=$CAPTURES"

if [[ ! -f "$QUESTIONS" ]]; then
  echo "[slurm] ERR: $QUESTIONS does not exist. Caption the session first:" >&2
  echo "        GEMINI_API_KEY=... python scripts/aria_generate_questions.py \\" >&2
  echo "          --session-dir $OUT_ROOT/$SESSION \\" >&2
  echo "          --features    $OUT_ROOT/$SESSION/clip_l_features.h5 \\" >&2
  echo "          --out         $QUESTIONS --n-questions 30 --model gemini" >&2
  exit 2
fi

if [[ ! -x "$REPO/targets/psm" ]]; then
  echo "[slurm] ERR: $REPO/targets/psm not found. Build first." >&2
  exit 2
fi

mkdir -p "$CAPTURES"

for enc in $ENCODERS; do
  echo
  echo "━━━ $enc ━━━"
  ROOT="$OUT_ROOT" \
  CAPTURES="$CAPTURES" \
  SESSIONS="$SESSION" \
  SEEDS="$SEEDS" \
  ENCODER="$enc" \
  AXES="h3_res" \
  H3_RES_LIST="$H3_RES_LIST" \
  bash "$REPO/scripts/eval_hyperparam_sweep.sh"
done

for enc in $ENCODERS; do
  python "$REPO/scripts/eval_hyperparam_plot.py" "$CAPTURES" \
    --encoder "$enc" \
    --out "$REPO/journal/figures/aria_${SESSION}_h3_${enc}.svg" >/dev/null || true
done

echo
python "$REPO/scripts/h3_acceptance.py" \
  --captures "$CAPTURES" \
  --sequence "$SESSION"

echo
echo "[slurm] done."
