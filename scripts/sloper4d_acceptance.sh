#!/usr/bin/env bash
# End-to-end H3-resolution acceptance test for a SLOPER4D sequence.
#
# Pipeline:
#   1. If questions.yaml missing: caption N frames via Gemini (or Claude
#      if --model claude). Requires GEMINI_API_KEY (or CLAUDE_API_KEY).
#   2. Run the H3-resolution axis of the hyperparam sweep:
#        - 5 seeds × 5 H3 values × {clipL, bigG}
#        - Outputs JSONs at captures/sloper4d_<SEQ>_h3/
#   3. Aggregate + check acceptance:
#        - curve strictly non-decreasing across r8…r12 for both encoders
#        - absolute lift r10 → r12 ≥ 5 percentage points for both encoders
#      Rationale: a ratio-based threshold penalizes stronger encoders
#      that already discriminate well at r10. Absolute-lift +
#      monotonicity tests the spatial-axis claim independent of
#      encoder baseline strength.
#
# Usage:
#   GEMINI_API_KEY=sk-... SEQ=seq009_running_002 bash scripts/sloper4d_acceptance.sh
#
# Env knobs (with defaults):
#   SEQ          seq009_running_002
#   N_QUESTIONS  30
#   MLLM_MODEL   gemini       gemini | claude
#   SEEDS        "0 1 2 3 4"
#   H3_RES_LIST  "8 9 10 11 12"
#   ENCODERS     "clipL bigG"
#   SRC_ROOT     /checkpoint/dream/arjangt/SLOPER4D-unzipped
#   OUT_ROOT     /checkpoint/dream/arjangt/video_retrieval/sloper4d

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEQ="${SEQ:-seq009_running_002}"
SRC_ROOT="${SRC_ROOT:-/checkpoint/dream/arjangt/SLOPER4D-unzipped}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/sloper4d}"
N_QUESTIONS="${N_QUESTIONS:-30}"
MLLM_MODEL="${MLLM_MODEL:-gemini}"
SEEDS="${SEEDS:-0 1 2 3 4}"
H3_RES_LIST="${H3_RES_LIST:-8 9 10 11 12}"
ENCODERS="${ENCODERS:-clipL bigG}"

SEQ_DIR="$OUT_ROOT/$SEQ"
QUESTIONS="$SEQ_DIR/questions.yaml"
VIDEO="$SRC_ROOT/$SEQ/rgb_data/$SEQ.MP4"
FEATURES_CLIPL="$SEQ_DIR/clip_l_features.h5"
CAPTURES="$REPO/captures/sloper4d_${SEQ}_h3"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SLOPER4D $SEQ — H3-resolution acceptance test"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  features (clipL): $FEATURES_CLIPL"
echo "║  questions:        $QUESTIONS"
echo "║  captures:         $CAPTURES"
echo "║  N questions:      $N_QUESTIONS  ($MLLM_MODEL)"
echo "║  H3 sweep:         $H3_RES_LIST"
echo "║  encoders:         $ENCODERS"
echo "║  seeds:            $SEEDS"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# Step 1 — caption frames if needed.
if [[ ! -f "$QUESTIONS" ]]; then
  echo "━━━ Step 1: captioning $N_QUESTIONS frames via $MLLM_MODEL ━━━"
  if [[ ! -f "$VIDEO" ]]; then
    echo "[ERR] missing source video: $VIDEO" >&2
    exit 2
  fi
  python "$REPO/scripts/sloper4d_generate_questions.py" \
    --features "$FEATURES_CLIPL" \
    --video "$VIDEO" \
    --out "$QUESTIONS" \
    --n-questions "$N_QUESTIONS" \
    --model "$MLLM_MODEL"
  echo
else
  echo "━━━ Step 1: questions.yaml already exists ($(grep -c '^- id:' "$QUESTIONS" || true) questions) ━━━"
  echo
fi

# Step 2 — H3-res sweep, both encoders. Reuses eval_hyperparam_sweep.sh.
mkdir -p "$CAPTURES"
for enc in $ENCODERS; do
  echo "━━━ Step 2.${enc}: H3-res sweep ($enc) ━━━"
  ROOT="$OUT_ROOT" \
  CAPTURES="$CAPTURES" \
  SESSIONS="$SEQ" \
  SEEDS="$SEEDS" \
  ENCODER="$enc" \
  AXES="h3_res" \
  H3_RES_LIST="$H3_RES_LIST" \
  bash "$REPO/scripts/eval_hyperparam_sweep.sh"
  echo
done

# Step 3 — plots + acceptance check via the shared verdict script.
echo "━━━ Step 3: aggregate + acceptance ━━━"
for enc in $ENCODERS; do
  python "$REPO/scripts/eval_hyperparam_plot.py" "$CAPTURES" \
    --encoder "$enc" \
    --out "$REPO/journal/figures/sloper4d_${SEQ}_h3_${enc}.svg" >/dev/null
done

echo
python "$REPO/scripts/sloper4d_h3_acceptance.py" \
  --captures "$CAPTURES" \
  --sequence "$SEQ"

echo
echo "═══ Done. ═══"
