#!/usr/bin/env bash
# 30-session per_cell_cap sweep (task #23): runs the existing
# multisession_per_cell_cap_sweep.py harness across all Nymeria
# sessions on disk, at cap ∈ {1, 2, 3, 5}. Output JSONs are written
# to captures/multisession_pcc_sweep/<session>/eval_<sid>_pcc<N>.json.
#
# We already have cap=K=5 numbers across 30 sessions (from
# eval_*_pool0s_p50.json in captures/), so this fills in the
# cap=1/2/3 columns that are currently marked "(cap sweep on 30
# sessions pending)" in §5.5 tab:multisession's last row.

set -euo pipefail

ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
CKPT="${CKPT:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"

# Auto-discover: any subdir with both clip_l_features.h5 and questions.yaml.
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
N_SESSIONS=$(echo "$SESSIONS" | wc -w)
echo "=== discovered $N_SESSIONS sessions"

python scripts/multisession_per_cell_cap_sweep.py \
  --root "$ROOT" \
  --sessions $SESSIONS \
  --caps 1 2 3 5 \
  --clip-device cpu \
  --clip-checkpoint "$CKPT"

# Aggregator follows
python scripts/aggregate_cap_sweep_30.py
