#!/usr/bin/env bash
# Approach 2: pre-smooth the features.h5 embeddings, then run PSM
# unchanged. Each frame's embedding becomes the mean of its temporal
# neighbors within ±W/2 seconds; PSM's reservoir samples from these
# smoothed embeddings, so the cell-level candidate generation
# benefits from smoothing -- not just the rerank.
#
# Reference points (shelby_arroyo_act0):
#   sliding-window @ 10s = 15.0% Hit@5
#   PSM + W=30 query-pool rerank = 15.0% Hit@5 (approach 1 ceiling)
#   Target: lift Hit@5 above 15.0% by smoothing PSM's input.

set -euo pipefail

TAKE="${TAKE:-20230608_s0_shelby_arroyo_act0_3ciwl8}"
ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
CKPT="${CKPT:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"
SRC="$ROOT/$TAKE/clip_l_features.h5"
Q="$ROOT/$TAKE/questions.yaml"

mkdir -p captures

WINDOWS="${WINDOWS:-5 10 20 30 60}"

# Step 1: build smoothed feature files (one per window).
SMOOTHED_DIR="${SMOOTHED_DIR:-/tmp/psm_smoothed_features/$TAKE}"
mkdir -p "$SMOOTHED_DIR"
for W in $WINDOWS; do
  DST="$SMOOTHED_DIR/clip_l_features_pool${W}s.h5"
  if [ -f "$DST" ]; then
    echo "=== smoothed bank W=${W}s already exists: $DST ==="
  else
    echo "=== smoothing bank W=${W}s -> $DST ==="
    python scripts/smooth_features.py "$SRC" "$DST" --window "$W" --group clip
  fi
done

# Step 2: run PSM eval on each smoothed bank.
for W in $WINDOWS; do
  H5="$SMOOTHED_DIR/clip_l_features_pool${W}s.h5"
  OUT="captures/eval_${TAKE}_smoothed${W}s.json"
  echo "=== PSM eval on smoothed bank W=${W}s -> $OUT ==="
  python scripts/eval_lookback.py "$H5" "$Q" \
    --per-cell-cap 5 --top 5 \
    --h3-resolution 12 --time-window 30 --capacity 60 \
    --exemplars 1024 \
    --clip-device cpu --clip-checkpoint "$CKPT" \
    --out "$OUT"
done

echo
echo "=== summary ==="
for W in $WINDOWS; do
  OUT="captures/eval_${TAKE}_smoothed${W}s.json"
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1])); s = d['summary']
print(f\"smoothed W=${W}s  hit@5={s['exemplar_hit_rate_at_5']*100:5.1f}%  exemplar_mIoU@5={s['exemplar_miou_at_5']:.3f}  bucket_mIoU@5={s['bucket_miou_at_5']:.3f}\")
" "$OUT"
done
