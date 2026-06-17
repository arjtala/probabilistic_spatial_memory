#!/usr/bin/env bash
# End-to-end H3-resolution acceptance test for SLOPER4D seq009.
#
# Pipeline:
#   1. If questions.yaml missing: caption 20 frames via Gemini (or Claude
#      if --model claude). Requires GEMINI_API_KEY (or CLAUDE_API_KEY).
#   2. Run the H3-resolution axis of the hyperparam sweep:
#        - 5 seeds × 5 H3 values × {clipL, bigG}
#        - Outputs JSONs at captures/sloper4d_seq009_h3/
#   3. Aggregate + check acceptance:
#        - Hit@5 at r12 ≥ 2× Hit@5 at r10  for both encoders
#      Replicates the Nymeria street-scale finding (3.2% → 8.9% bigG,
#      a 2.8× lift) on a completely independent corpus (LiDAR-SLAM
#      vs Aria SLAM, DJI Action2 vs Aria cameras, Xiamen vs San
#      Francisco). If it replicates, the spatial-axis story holds
#      independent of hardware + positioning source.
#
# Usage:
#   GEMINI_API_KEY=sk-... bash scripts/sloper4d_seq009_acceptance.sh
#
# Defaults you can override via env:
#   N_QUESTIONS  20         questions generated per sequence
#   MLLM_MODEL   gemini     gemini | claude
#   SEEDS        "0 1 2 3 4"
#   H3_RES_LIST  "8 9 10 11 12"
#   OUT_ROOT     /checkpoint/dream/arjangt/video_retrieval/sloper4d
#   ENCODERS     "clipL bigG"

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEQ="seq009_running_002"
SRC_ROOT="${SRC_ROOT:-/checkpoint/dream/arjangt/SLOPER4D-unzipped}"
OUT_ROOT="${OUT_ROOT:-/checkpoint/dream/arjangt/video_retrieval/sloper4d}"
N_QUESTIONS="${N_QUESTIONS:-20}"
MLLM_MODEL="${MLLM_MODEL:-gemini}"
SEEDS="${SEEDS:-0 1 2 3 4}"
H3_RES_LIST="${H3_RES_LIST:-8 9 10 11 12}"
ENCODERS="${ENCODERS:-clipL bigG}"

SEQ_DIR="$OUT_ROOT/$SEQ"
QUESTIONS="$SEQ_DIR/questions.yaml"
VIDEO="$SRC_ROOT/$SEQ/rgb_data/$SEQ.MP4"
FEATURES_CLIPL="$SEQ_DIR/clip_l_features.h5"
CAPTURES="$REPO/captures/sloper4d_seq009_h3"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SLOPER4D seq009 — H3-resolution acceptance test           ║"
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

# Step 3 — aggregate, plot, and check the acceptance criterion.
echo "━━━ Step 3: aggregate + acceptance ━━━"
for enc in $ENCODERS; do
  echo
  echo "--- $enc ---"
  python "$REPO/scripts/eval_hyperparam_plot.py" "$CAPTURES" \
    --encoder "$enc" \
    --out "$REPO/journal/figures/sloper4d_seq009_h3_${enc}.svg"
done

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

# (encoder, h3_res) -> list of Hit@5
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
    verdict = "✓ PASS" if ratio >= 2.0 else "✗ FAIL"
    if ratio < 2.0:
        all_pass = False
    print(f"{enc:6s}  {mean10*100:6.1f}%   {mean12*100:6.1f}%   {ratio:5.2f}×   {verdict}")
print()
if all_pass:
    print("ACCEPTANCE: ✓ replicated Nymeria-street H3-resolution finding on SLOPER4D")
    sys.exit(0)
else:
    print("ACCEPTANCE: ✗ did not replicate (consider whether seq009 alone is enough)")
    sys.exit(1)
PY

echo
echo "═══ Done. ═══"
