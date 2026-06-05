#!/usr/bin/env bash
# Extended PSM + temporal-pooling rerank sweep.
#
# First sweep (windows 0..30s, rerank_pool=50) found 15.0% Hit@5 at W=30,
# matching sliding-window @ 10s exactly. The peak is at the boundary of
# the first sweep — push further to find the true peak, and try widening
# rerank_pool to see whether PSM's top-50 candidate set is the ceiling.

set -euo pipefail

TAKE="${TAKE:-20230608_s0_shelby_arroyo_act0_3ciwl8}"
ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
CKPT="${CKPT:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"
H5="$ROOT/$TAKE/clip_l_features.h5"
Q="$ROOT/$TAKE/questions.yaml"

mkdir -p captures

# Sweep two axes: W (pool window) and P (rerank pool size).
WINDOWS="${WINDOWS:-30 45 60 90 120 180}"
POOLS="${POOLS:-50 100 200}"

for P in $POOLS; do
  for W in $WINDOWS; do
    OUT="captures/eval_${TAKE}_pool${W}s_p${P}.json"
    echo "=== pool_window=${W}s rerank_pool=${P} -> $OUT ==="
    python scripts/eval_psm_temporal_pool.py "$H5" "$Q" \
      --pool-window "$W" --per-cell-cap 5 \
      --top 5 --rerank-pool "$P" \
      --h3-resolution 12 --time-window 30 --capacity 60 \
      --exemplars 1024 \
      --clip-device cpu --clip-checkpoint "$CKPT" \
      --out "$OUT"
  done
done

echo
echo "=== summary ==="
for P in $POOLS; do
  for W in $WINDOWS; do
    OUT="captures/eval_${TAKE}_pool${W}s_p${P}.json"
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1])); s = d['summary']
print(f\"P=${P} W=${W}s  hit@5={s['exemplar_hit_rate_at_5']*100:5.1f}%  exemplar_mIoU@5={s['exemplar_miou_at_5']:.3f}  bucket_mIoU@5={s['bucket_miou_at_5']:.3f}\")
" "$OUT"
  done
done
