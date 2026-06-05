#!/usr/bin/env bash
# Run Long-CLIP text encoder over the existing CLIP-L image bank
# across all 30 Nymeria sessions. Long-CLIP-L shares CLIP-L's vision
# tower, so the image bank does NOT need re-extraction -- only text
# queries are re-encoded with the 248-token positional embedding.
#
# Setup (one-time on the cluster):
#   git clone https://github.com/beichenzbc/Long-CLIP.git ~/long-clip
#   # Download longclip-L.pt per the repo README (Google Drive link, ~1.7GB)
#   # Place at ~/long-clip/checkpoints/longclip-L.pt
#   export LONGCLIP_ROOT=~/long-clip
#
# Compare against the W=0 PSM baseline numbers we already have
# (mean Hit@5 = 8.95% across 30 sessions). The hypothesis: queries
# that lose their disambiguating verb to the 77-token cap will be
# scored correctly by Long-CLIP, lifting Hit@5 on long-narration
# sessions specifically.

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"

if [ -z "${LONGCLIP_ROOT:-}" ]; then
  echo "!!! LONGCLIP_ROOT not set. See setup steps in script header."
  exit 1
fi
if [ ! -f "$LONGCLIP_ROOT/checkpoints/longclip-L.pt" ]; then
  echo "!!! checkpoint missing: $LONGCLIP_ROOT/checkpoints/longclip-L.pt"
  exit 1
fi

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
echo "=== discovered $(echo "$SESSIONS" | wc -w) sessions"

mkdir -p captures

for SID in $SESSIONS; do
  H5="$ROOT/$SID/clip_l_features.h5"
  Q="$ROOT/$SID/questions.yaml"
  OUT="captures/eval_${SID}_longclip_text.json"
  if [ -f "$OUT" ]; then
    echo "=== $SID longclip exists ==="
    continue
  fi
  echo "=== $SID longclip -> $OUT ==="
  python scripts/eval_longclip_text.py "$H5" "$Q" \
    --device cpu --context-length 248 --out "$OUT"
done

echo
echo "=== summary (Long-CLIP text + CLIP-L image bank, brute-force) ==="
for SID in $SESSIONS; do
  OUT="captures/eval_${SID}_longclip_text.json"
  [ -f "$OUT" ] || continue
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1])); s = d['summary']
sid = sys.argv[2]
print(f\"{sid:50s}  hit@5={s['exemplar_hit_rate_at_5']*100:5.1f}%  exemplar_mIoU@5={s['exemplar_miou_at_5']:.3f}\")
" "$OUT" "$SID"
done
