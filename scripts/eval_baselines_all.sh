#!/usr/bin/env bash
# Run the three E11 retrieval baselines (brute-force CLIP, sliding-window
# CLIP, uniform-sample CLIP) across N sessions × M encoders. Output JSONs
# land under captures/baselines/ for downstream eval_aggregate pooling.
#
# This is the CPU-only sweep that follows E11 in EXPERIMENTS.md. None of
# these baselines call PSM; they're the "what if you skip the spatial
# decomposition entirely" comparison the paper needs.
#
# Usage:
#   conda activate psm
#   bash scripts/eval_baselines_all.sh
#
# Env knobs:
#   ROOT        sessions root (default ./datasets, override to
#               /checkpoint/dream/arjangt/video_retrieval/aria on cluster)
#   SESSIONS    space-separated session ids (default the 3 demo sessions)
#   ENCODERS    space-separated encoder keys (default "bigG clipL")
#               each key maps to a (features-basename, checkpoint) pair
#               via the case-block below
#   OUT_DIR     where JSONs land (default $REPO/captures/baselines)
#   TOP         --top value (default 5)
#   WINDOW_SIZES sliding-window sweep, comma-separated (default 3,5,10)
#   SAMPLE_RATES uniform-sample sweep, comma-separated (default 30,75)
#
# Skips sessions where the encoder's features file is missing (logs a
# warning) instead of failing the whole sweep — useful for sessions where
# only one encoder has been extracted.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$REPO/datasets}"
SESSIONS="${SESSIONS:-1501677363692556 287142033569927 201703061033}"
ENCODERS="${ENCODERS:-bigG clipL}"
OUT_DIR="${OUT_DIR:-$REPO/captures/baselines}"
TOP="${TOP:-5}"
WINDOW_SIZES="${WINDOW_SIZES:-3,5,10}"
SAMPLE_RATES="${SAMPLE_RATES:-30,75}"

mkdir -p "$OUT_DIR"

encoder_features() {
  case "$1" in
    bigG)   echo "clip_bigg_features.h5" ;;
    clipL)  echo "clip_l_features.h5" ;;
    clip)   echo "clip_features.h5" ;;
    *)      echo "" ;;
  esac
}

encoder_checkpoint() {
  case "$1" in
    bigG)   echo "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" ;;
    clipL)  echo "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" ;;
    clip)   echo "openai/clip-vit-base-patch32" ;;
    *)      echo "" ;;
  esac
}

echo "[baselines] root=$ROOT"
echo "[baselines] sessions: $SESSIONS"
echo "[baselines] encoders: $ENCODERS"
echo "[baselines] out: $OUT_DIR"
echo "[baselines] top=$TOP windows=$WINDOW_SIZES rates=$SAMPLE_RATES"
echo

for sid in $SESSIONS; do
  qs="$ROOT/$sid/questions.yaml"
  if [[ ! -f "$qs" ]]; then
    echo "[baselines] WARN: missing $qs, skipping session $sid" >&2
    continue
  fi

  for enc in $ENCODERS; do
    feat_basename="$(encoder_features "$enc")"
    ckpt="$(encoder_checkpoint "$enc")"
    if [[ -z "$feat_basename" || -z "$ckpt" ]]; then
      echo "[baselines] WARN: unknown encoder key '$enc', skipping" >&2
      continue
    fi

    feat="$ROOT/$sid/$feat_basename"
    if [[ ! -f "$feat" ]]; then
      echo "[baselines] WARN: missing $feat, skipping ($sid × $enc)" >&2
      continue
    fi

    echo "[baselines] === $sid × $enc ($feat_basename) ==="

    python "$REPO/scripts/eval_brute_force_clip.py" "$feat" "$qs" \
      --top "$TOP" \
      --clip-checkpoint "$ckpt" \
      --out "$OUT_DIR/brute_force_${sid}_${enc}.json"

    python "$REPO/scripts/eval_sliding_window.py" "$feat" "$qs" \
      --top "$TOP" \
      --window-sizes "$WINDOW_SIZES" \
      --clip-checkpoint "$ckpt" \
      --out "$OUT_DIR/sliding_${sid}_${enc}.json"

    python "$REPO/scripts/eval_uniform_sample.py" "$feat" "$qs" \
      --top "$TOP" \
      --rates "$SAMPLE_RATES" \
      --clip-checkpoint "$ckpt" \
      --out "$OUT_DIR/uniform_${sid}_${enc}.json"
  done
done

echo
echo "[baselines] done. Aggregate across methods + PSM with:"
echo "  python scripts/eval_aggregate.py --by-seed --label-from-features \\"
echo "    captures/eval_*_clipBigG_e128_s*.json \\"
echo "    $OUT_DIR/*.json"
