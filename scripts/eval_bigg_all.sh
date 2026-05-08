#!/usr/bin/env bash
# Run eval_lookback over the three sessions x 5 seeds using the bigG
# features produced by scripts/extract_bigg_all.sh, then aggregate.
#
# Output JSONs land in captures/eval_<sid>_clipBigG_e128_s<seed>.json,
# matching the eval_<sid>_clipL_e128_s<seed>.json naming so a side-by-side
# comparison is just a glob change.
#
# Usage:
#   conda activate psm
#   bash scripts/eval_bigg_all.sh
#
# Env knobs:
#   CHECKPOINT    bigG checkpoint (must match what extract used)
#   FEATURES      input HDF5 basename (default clip_bigg_features.h5)
#   TAG           label suffix in output filenames (default clipBigG)
#   ROOT          datasets root (default ./datasets)
#   CAPTURES      captures dir (default ./captures)
#   SEEDS         space-separated seeds (default "0 1 2 3 4")
#   SESSIONS      space-separated session ids (default the 3 demo sessions)
#   TOP           --top value for psm --search (default 5; raise for counting)
#   TAG_TOP_SUFFIX  if "1", auto-suffix TAG with _top<TOP> (e.g. clipL_top20)

set -euo pipefail

ROOT="${ROOT:-datasets}"
CAPTURES="${CAPTURES:-captures}"
CHECKPOINT="${CHECKPOINT:-laion/CLIP-ViT-bigG-14-laion2B-39B-b160k}"
FEATURES="${FEATURES:-clip_bigg_features.h5}"
TAG="${TAG:-clipBigG}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SESSIONS="${SESSIONS:-1501677363692556 287142033569927 201703061033}"
TOP="${TOP:-5}"
TAG_TOP_SUFFIX="${TAG_TOP_SUFFIX:-0}"
if [[ "$TAG_TOP_SUFFIX" == "1" && "$TOP" != "5" ]]; then
  TAG="${TAG}_top${TOP}"
fi

mkdir -p "$CAPTURES"

echo "[eval] checkpoint=$CHECKPOINT features=$FEATURES tag=$TAG top=$TOP"
echo "[eval] sessions: $SESSIONS"
echo "[eval] seeds: $SEEDS"

for sid in $SESSIONS; do
  feat="$ROOT/$sid/$FEATURES"
  qs="$ROOT/$sid/questions.yaml"
  if [[ ! -f "$feat" ]]; then
    echo "[eval] missing $feat — run scripts/extract_bigg_all.sh first" >&2
    exit 1
  fi
  if [[ ! -f "$qs" ]]; then
    echo "[eval] missing $qs" >&2
    exit 1
  fi
  for seed in $SEEDS; do
    out="$CAPTURES/eval_${sid}_${TAG}_e128_s${seed}.json"
    echo "[eval] === $sid seed=$seed -> $out ==="
    python scripts/eval_lookback.py \
      "$feat" "$qs" \
      --top "$TOP" --time-window 75 --capacity 12 --exemplars 128 \
      --clip-checkpoint "$CHECKPOINT" \
      --seed "$seed" \
      --out "$out"
  done
done

agg_out="$CAPTURES/eval_seedsweep_${TAG}_e128.json"
echo "[eval] aggregating -> $agg_out"
python scripts/eval_aggregate.py --by-seed --label-from-features \
  $CAPTURES/eval_*_${TAG}_e128_s*.json \
  --out "$agg_out"

echo "[eval] done. Compare against the CLIP-L baseline with:"
echo "  python scripts/eval_aggregate.py --by-seed --label-from-features \\"
echo "    $CAPTURES/eval_*_clipL_e128_s*.json $CAPTURES/eval_*_${TAG}_e128_s*.json"
