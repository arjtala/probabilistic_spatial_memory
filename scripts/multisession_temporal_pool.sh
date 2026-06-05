#!/usr/bin/env bash
# Run approach-1 (query-time temporal-pool rerank at W=30s) on all 4
# Nymeria sessions. Compares PSM-only, PSM+pool(W=30), and the
# sliding-window @ 10s baseline already in captures/.
#
# On shelby_arroyo_act0, approach 1 lifted Hit@5 from 13.4% to 15.0%,
# matching sliding-window @ 10s exactly. Generality check: does the
# lift hold across mobility tiers?

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
CKPT="${CKPT:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"

# Auto-discover sessions: any directory under $ROOT that contains both
# clip_l_features.h5 and questions.yaml. Override with SESSIONS env var
# to specify explicit names.
if [ -z "${SESSIONS:-}" ]; then
  SESSIONS="$(
    for d in "$ROOT"/*/; do
      sid="$(basename "$d")"
      if [ -f "$d/clip_l_features.h5" ] && [ -f "$d/questions.yaml" ]; then
        echo "$sid"
      fi
    done
  )"
fi
if [ -z "$SESSIONS" ]; then
  echo "!!! no sessions found under $ROOT with both clip_l_features.h5 + questions.yaml"
  exit 1
fi
echo "=== discovered $(echo "$SESSIONS" | wc -w) sessions under $ROOT"

W="${W:-30}"
P="${P:-50}"

mkdir -p captures

for SID in $SESSIONS; do
  H5="$ROOT/$SID/clip_l_features.h5"
  Q="$ROOT/$SID/questions.yaml"
  if [ ! -f "$H5" ] || [ ! -f "$Q" ]; then
    echo "!!! skipping $SID (missing features or questions)"
    continue
  fi
  OUT="captures/eval_${SID}_pool${W}s_p${P}.json"
  if [ -f "$OUT" ]; then
    echo "=== $SID exists: $OUT ==="
  else
    echo "=== $SID -> $OUT ==="
    python scripts/eval_psm_temporal_pool.py "$H5" "$Q" \
      --pool-window "$W" --per-cell-cap 5 \
      --top 5 --rerank-pool "$P" \
      --h3-resolution 12 --time-window 30 --capacity 60 \
      --exemplars 1024 \
      --clip-device cpu --clip-checkpoint "$CKPT" \
      --out "$OUT"
  fi
done

echo
echo "=== summary (PSM + pool W=${W}s) ==="
for SID in $SESSIONS; do
  OUT="captures/eval_${SID}_pool${W}s_p${P}.json"
  [ -f "$OUT" ] || continue
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1])); s = d['summary']
sid = sys.argv[2]
print(f\"{sid:50s}  hit@5={s['exemplar_hit_rate_at_5']*100:5.1f}%  exemplar_mIoU@5={s['exemplar_miou_at_5']:.3f}  bucket_mIoU@5={s['bucket_miou_at_5']:.3f}\")
" "$OUT" "$SID"
done
