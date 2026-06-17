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
# Run the SLOPER4D seq009 H3-resolution acceptance test (steps 2 + 3
# of scripts/sloper4d_seq009_acceptance.sh). Step 1 (Gemini frame
# captioning) is intentionally NOT included here — it needs internet
# access for the proxy and a GEMINI_API_KEY. Run that locally first
# so questions.yaml lands on disk; this job then assumes the file
# exists and goes straight to the H3-res sweep.
#
# Sweep shape: 5 seeds × 5 H3 values × {clipL, bigG} = 50 eval_lookback
# invocations on questions.yaml (20 questions × CLIP text encode). On
# h200 with a warm HF cache: ~2-5 min total wall time.
#
# Submit AFTER questions.yaml exists at
#   /checkpoint/.../sloper4d/seq009_running_002/questions.yaml
#
# Submit:
#   sbatch scripts/slurm/sloper4d_seq009_h3.sbatch

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
CAPTURES="$REPO/captures/sloper4d_seq009_h3"
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
    --out "$REPO/journal/figures/sloper4d_seq009_h3_${enc}.svg" || true
done

# Acceptance check (same logic as the local acceptance script).
echo
echo "━━━ Acceptance check ━━━"
python - <<'PY'
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path

CAPS = Path("captures/sloper4d_seq009_h3")
fname_re = re.compile(
    r"^h3_res_(?P<label>[^_]+)_seq009_running_002_(?P<enc>clipL|bigG)_s(?P<seed>\d+)\.json$"
)

by_cell = defaultdict(list)
for p in sorted(CAPS.glob("h3_res_*.json")):
    m = fname_re.match(p.name)
    if not m:
        continue
    enc = m["enc"]
    res = int(m["label"])
    d = json.loads(p.read_text())
    rate = d.get("summary", {}).get("exemplar_hit_rate_at_5")
    if rate is None:
        continue
    by_cell[(enc, res)].append(rate)

print(f"{'encoder':6s}  {'r10 mean':9s}  {'r12 mean':9s}  {'ratio':7s}  verdict")
print("-" * 60)
all_pass = True
for enc in ("clipL", "bigG"):
    r10 = by_cell.get((enc, 10), [])
    r12 = by_cell.get((enc, 12), [])
    if not r10 or not r12:
        print(f"{enc:6s}  MISSING DATA r10={len(r10)} r12={len(r12)}")
        all_pass = False
        continue
    mean10 = sum(r10) / len(r10)
    mean12 = sum(r12) / len(r12)
    ratio = mean12 / mean10 if mean10 > 0 else float("inf")
    verdict = "PASS" if ratio >= 2.0 else "FAIL"
    if ratio < 2.0:
        all_pass = False
    print(f"{enc:6s}  {mean10*100:6.1f}%   {mean12*100:6.1f}%   {ratio:5.2f}x   {verdict}")
print()
if all_pass:
    print("ACCEPTANCE: PASS (Nymeria-street finding replicated on SLOPER4D)")
else:
    print("ACCEPTANCE: FAIL (consider whether seq009 alone is enough)")
PY

echo
echo "[slurm] done."
