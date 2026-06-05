#!/usr/bin/env bash
# Sweep PSM + temporal-pooling rerank over pool_window ∈ {0,3,5,10,15,30}
# on a chosen Nymeria session. Writes one JSON per window under captures/.
#
# Usage:
#   bash scripts/sweep_temporal_pool.sh
#   TAKE=<other_session_id> bash scripts/sweep_temporal_pool.sh
#
# Reference points (shelby_arroyo_act0):
#   sliding-window @ 10s = 15.0% Hit@5 (the target to beat)
#   per-frame brute-force = 13.4%
#   PSM @ cap=K, R=1024, W=0 (no pool) = 13.4% (baseline; matches brute-force)

set -euo pipefail

TAKE="${TAKE:-20230608_s0_shelby_arroyo_act0_3ciwl8}"
ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
CKPT="${CKPT:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"
H5="$ROOT/$TAKE/clip_l_features.h5"
Q="$ROOT/$TAKE/questions.yaml"

mkdir -p captures

WINDOWS="${WINDOWS:-0 3 5 10 15 30}"

for W in $WINDOWS; do
  OUT="captures/eval_${TAKE}_pool${W}s.json"
  echo "=== pool_window=${W}s -> $OUT ==="
  python scripts/eval_psm_temporal_pool.py "$H5" "$Q" \
    --pool-window "$W" --per-cell-cap 5 \
    --h3-resolution 12 --time-window 30 --capacity 60 \
    --exemplars 1024 \
    --clip-device cpu --clip-checkpoint "$CKPT" \
    --out "$OUT"
done

echo
echo "=== summary ==="
for W in $WINDOWS; do
  OUT="captures/eval_${TAKE}_pool${W}s.json"
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1])); s = d['summary']
print(f\"W=${W}s  hit@5={s['exemplar_hit_rate_at_5']*100:5.1f}%  exemplar_mIoU@5={s['exemplar_miou_at_5']:.3f}  bucket_mIoU@5={s['bucket_miou_at_5']:.3f}\")
" "$OUT"
done
