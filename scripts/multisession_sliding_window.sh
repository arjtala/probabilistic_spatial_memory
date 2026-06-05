#!/usr/bin/env bash
# Run sliding-window CLIP across all extracted Nymeria sessions.
# Companion to multisession_temporal_pool.sh -- needed to verify
# whether the +1.6pp lift sliding-window @ 10s gave on shelby_arroyo
# generalizes (mirror of the approach-1 generality test).
#
# Sliding-window emits one JSON per window length (3s, 5s, 10s) per
# session, so this is a fan-out over (session × window).

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
CKPT="${CKPT:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"

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
  echo "!!! no sessions found"; exit 1
fi
echo "=== discovered $(echo "$SESSIONS" | wc -w) sessions"

mkdir -p captures

for SID in $SESSIONS; do
  H5="$ROOT/$SID/clip_l_features.h5"
  Q="$ROOT/$SID/questions.yaml"
  OUT_BASE="captures/eval_${SID}_sliding"
  # The sliding script writes _wXs.json per window internally; check
  # for one of them to short-circuit on resume.
  if [ -f "${OUT_BASE}_w10s.json" ]; then
    echo "=== $SID sliding exists ==="
  else
    echo "=== $SID sliding -> ${OUT_BASE}_w{3,5,10}s.json ==="
    python scripts/eval_sliding_window.py "$H5" "$Q" \
      --top 5 --clip-device cpu --clip-checkpoint "$CKPT" \
      --out "${OUT_BASE}.json"
  fi
done

echo
echo "=== summary (sliding-window @ 10s) ==="
for SID in $SESSIONS; do
  OUT="captures/eval_${SID}_sliding_w10s.json"
  [ -f "$OUT" ] || continue
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1])); s = d['summary']
sid = sys.argv[2]
print(f\"{sid:50s}  hit@5={s['exemplar_hit_rate_at_5']*100:5.1f}%  exemplar_mIoU@5={s['exemplar_miou_at_5']:.3f}\")
" "$OUT" "$SID"
done
