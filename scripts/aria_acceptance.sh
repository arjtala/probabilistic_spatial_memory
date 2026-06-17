#!/usr/bin/env bash
# End-to-end H3-resolution acceptance test for an Aria Gen 2 session.
# Parallels scripts/sloper4d_acceptance.sh for Aria's VRS-source path
# and real-GPS track_mode.
#
# Pipeline:
#   1. If questions.yaml missing: caption N frames via Gemini, using
#      the VRS reader to source frames (no MP4 in Aria's stack).
#      Requires GEMINI_API_KEY (or CLAUDE_API_KEY).
#   2. Run the H3-resolution axis of the hyperparam sweep:
#        - 5 seeds × 5 H3 values × {clipL, bigG}  (bigG only if its H5
#          exists; defaults to clipL otherwise)
#   3. Aggregate + check acceptance (same rule as SLOPER4D):
#        - curve strictly non-decreasing across r8…r12
#        - absolute lift r10 → r12 ≥ 5 percentage points
#
# Usage:
#   GEMINI_API_KEY=sk-... SESSION=walk_0 bash scripts/aria_acceptance.sh
#
# Env knobs:
#   SESSION       walk_0
#   N_QUESTIONS   30
#   MLLM_MODEL    gemini  | claude
#   SEEDS         "0 1 2 3 4"
#   H3_RES_LIST   "8 9 10 11 12"
#   ENCODERS      "clipL"  (add "bigG" once an Aria bigG H5 exists)
#   OUT_ROOT      /checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SESSION="${SESSION:-walk_0}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot}"
N_QUESTIONS="${N_QUESTIONS:-30}"
MLLM_MODEL="${MLLM_MODEL:-gemini}"
SEEDS="${SEEDS:-0 1 2 3 4}"
H3_RES_LIST="${H3_RES_LIST:-8 9 10 11 12}"
ENCODERS="${ENCODERS:-clipL}"

SESS_DIR="$OUT_ROOT/$SESSION"
QUESTIONS="$SESS_DIR/questions.yaml"
FEATURES_CLIPL="$SESS_DIR/clip_l_features.h5"
CAPTURES="$REPO/captures/aria_${SESSION}_h3"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Aria Gen 2 $SESSION — H3-resolution acceptance test"
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

if [[ ! -f "$QUESTIONS" ]]; then
  echo "━━━ Step 1: captioning $N_QUESTIONS frames via $MLLM_MODEL ━━━"
  python "$REPO/scripts/aria_generate_questions.py" \
    --session-dir "$SESS_DIR" \
    --features    "$FEATURES_CLIPL" \
    --out         "$QUESTIONS" \
    --n-questions "$N_QUESTIONS" \
    --model       "$MLLM_MODEL"
  echo
else
  echo "━━━ Step 1: questions.yaml already exists ($(grep -c '^- id:' "$QUESTIONS" || true) questions) ━━━"
  echo
fi

mkdir -p "$CAPTURES"
for enc in $ENCODERS; do
  echo "━━━ Step 2.${enc}: H3-res sweep ($enc) ━━━"
  ROOT="$OUT_ROOT" \
  CAPTURES="$CAPTURES" \
  SESSIONS="$SESSION" \
  SEEDS="$SEEDS" \
  ENCODER="$enc" \
  AXES="h3_res" \
  H3_RES_LIST="$H3_RES_LIST" \
  bash "$REPO/scripts/eval_hyperparam_sweep.sh"
  echo
done

echo "━━━ Step 3: aggregate + acceptance ━━━"
for enc in $ENCODERS; do
  python "$REPO/scripts/eval_hyperparam_plot.py" "$CAPTURES" \
    --encoder "$enc" \
    --out "$REPO/journal/figures/aria_${SESSION}_h3_${enc}.svg" >/dev/null
done

echo
python "$REPO/scripts/h3_acceptance.py" \
  --captures "$CAPTURES" \
  --sequence "$SESSION"

echo
echo "═══ Done. ═══"
