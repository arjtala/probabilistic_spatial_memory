#!/bin/bash
#SBATCH --job-name=psm-sloper4d-h3
#SBATCH --account=dream
#SBATCH --qos=h200_comm_shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sloper4d_h3_%j.out
#SBATCH --error=logs/sloper4d_h3_%j.err
#
# Run the SLOPER4D H3-resolution acceptance test (steps 2 + 3 of
# scripts/sloper4d_acceptance.sh). Step 1 (Gemini frame
# captioning) is intentionally NOT included here — it needs internet
# access for the proxy and a GEMINI_API_KEY. Run that locally first
# so questions.yaml lands on disk; this job then assumes the file
# exists and goes straight to the H3-res sweep.
#
# Sweep shape: 5 seeds × 5 H3 values × {clipL, bigG} = 50 eval_lookback
# invocations on questions.yaml (N questions × CLIP text encode). On
# h200 with a warm HF cache: ~10-15 min total wall time.
#
# Submit AFTER questions.yaml exists at $OUT_ROOT/$SEQ/questions.yaml
#
# Submit (defaults to seq009_running_002):
#   sbatch scripts/slurm/sloper4d_h3.sbatch
#
# For other sequences:
#   sbatch --export=SEQ=seq008_running_001 scripts/slurm/sloper4d_h3.sbatch
#   sbatch --export=SEQ=seq003_street_002  scripts/slurm/sloper4d_h3.sbatch

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-psm}"
SEQ="${SEQ:-seq009_running_002}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/sloper4d}"
SEEDS="${SEEDS:-0 1 2 3 4}"
H3_RES_LIST="${H3_RES_LIST:-8 9 10 11 12}"
ENCODERS="${ENCODERS:-clipL bigG}"

source activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

REPO="$(pwd)"
CAPTURES="$REPO/captures/sloper4d_${SEQ}_h3"
QUESTIONS="$OUT_ROOT/$SEQ/questions.yaml"

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "[slurm] seq=$SEQ encoders=$ENCODERS seeds=$SEEDS h3=$H3_RES_LIST"
echo "[slurm] questions=$QUESTIONS"
echo "[slurm] captures=$CAPTURES"

if [[ ! -f "$QUESTIONS" ]]; then
  echo "[slurm] ERR: $QUESTIONS does not exist. Caption the sequence first:" >&2
  echo "        GEMINI_API_KEY=... python scripts/sloper4d_generate_questions.py \\" >&2
  echo "          --features $OUT_ROOT/$SEQ/clip_l_features.h5 \\" >&2
  echo "          --video /checkpoint/dream/arjangt/SLOPER4D-unzipped/$SEQ/rgb_data/$SEQ.MP4 \\" >&2
  echo "          --out $QUESTIONS --n-questions 20 --model gemini" >&2
  exit 2
fi

# psm binary sanity check.
if [[ ! -x "$REPO/targets/psm" ]]; then
  echo "[slurm] ERR: $REPO/targets/psm not found. Build first." >&2
  exit 2
fi

mkdir -p "$CAPTURES"

# Run the H3 sweep for each encoder.
for enc in $ENCODERS; do
  echo
  echo "━━━ $enc ━━━"
  ROOT="$OUT_ROOT" \
  CAPTURES="$CAPTURES" \
  SESSIONS="$SEQ" \
  SEEDS="$SEEDS" \
  ENCODER="$enc" \
  AXES="h3_res" \
  H3_RES_LIST="$H3_RES_LIST" \
  bash "$REPO/scripts/eval_hyperparam_sweep.sh"
done

# Plots.
for enc in $ENCODERS; do
  python "$REPO/scripts/eval_hyperparam_plot.py" "$CAPTURES" \
    --encoder "$enc" \
    --out "$REPO/journal/figures/sloper4d_${SEQ}_h3_${enc}.svg" || true
done

# Acceptance check (curve monotonic + ≥5pp r10→r12 absolute lift,
# both encoders). Standalone script so the local + sbatch drivers
# share the same verdict logic.
echo
python "$REPO/scripts/sloper4d_h3_acceptance.py" \
  --captures "$CAPTURES" \
  --sequence "$SEQ"

echo
echo "[slurm] done."
