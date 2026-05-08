#!/usr/bin/env bash
# Re-extract the three localization-paradox sessions with OpenCLIP-bigG.
#
# Mirrors reproducibility.md §1 but swaps the encoder to
# laion/CLIP-ViT-bigG-14-laion2B-39B-b160k (~1B params, 1280-dim).
# Output files land alongside the existing clip_l_features.h5 as
# clip_bigg_features.h5 so the L-vs-bigG comparison is one flag away.
#
# Expected wall time on M4 Pro MPS: ~25-30 min per 15-min session at
# --sample-fps 3. Lower --batch-size if you OOM on MPS (default 16
# inside CLIPPyTorchRunner).
#
# Usage:
#   conda activate psm
#   bash scripts/extract_bigg_all.sh
#
# Env knobs:
#   FPS         override sample-fps (default 3)
#   SEGMENT     override segment-sec (default 1)
#   CHECKPOINT  override the bigG checkpoint
#   ROOT        override datasets root (default ./datasets)

set -euo pipefail

ROOT="${ROOT:-datasets}"
FPS="${FPS:-3}"
SEGMENT="${SEGMENT:-1}"
CHECKPOINT="${CHECKPOINT:-laion/CLIP-ViT-bigG-14-laion2B-39B-b160k}"
OUT_BASENAME="clip_bigg_features.h5"

echo "[extract] checkpoint=$CHECKPOINT fps=$FPS segment=$SEGMENT"
echo "[extract] writing to $ROOT/<sid>/$OUT_BASENAME"

# Aria sessions: data.mp4 + gps.json + imu.json sit in the session dir.
for sid in 1501677363692556 287142033569927; do
  echo "[extract] === $sid (Aria) ==="
  python -m psm_extraction extract \
    --video      "$ROOT/$sid/data.mp4" \
    --output     "$ROOT/$sid/$OUT_BASENAME" \
    --models     clip \
    --checkpoint "clip:$CHECKPOINT" \
    --sample-fps "$FPS" --segment-sec "$SEGMENT" \
    --session-id "$sid"
done

# Honda HDD session: video.mp4, no JSON sidecars; GPS lives in the
# original features.h5 (must already exist).
echo "[extract] === 201703061033 (Honda HDD) ==="
python -m psm_extraction extract \
  --video       "$ROOT/201703061033/video.mp4" \
  --output      "$ROOT/201703061033/$OUT_BASENAME" \
  --models      clip \
  --checkpoint  "clip:$CHECKPOINT" \
  --sample-fps  "$FPS" --segment-sec "$SEGMENT" \
  --gps-source  "$ROOT/201703061033/features.h5" \
  --session-id  201703061033

echo "[extract] done."
