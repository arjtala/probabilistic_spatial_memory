#!/usr/bin/env bash
# Sweep PSM's three core hyperparameters one axis at a time (E12). For
# each axis, hold the other two at the v2/E11 defaults and vary the
# swept value across a representative range. Reuses eval_lookback.py
# as the inner loop so the IoU + Hit @k scorer is identical to the
# numbers reported in §1-§4 of journal/localization_paradox2.md and the
# E11 table in PAPER.md.
#
# Axis defaults (override with env knobs):
#   H3_RES_LIST        H3 resolution sweep             default: "8 9 10 11 12"
#   RETENTION_LIST     "<time_window>x<capacity>"     default: "30x24 45x16 75x12 150x6 300x3"
#                      Holds ~12-15 min effective horizon at each setting.
#   EXEMPLARS_LIST     reservoir capacity per tile    default: "16 32 64 128 256"
#
# Other env knobs (forwarded to eval_lookback.py):
#   ROOT          sessions root (default cluster aria path)
#   CAPTURES      where JSONs land (default $REPO/captures/hyperparam)
#   SESSIONS      space-separated session ids
#   SEEDS         space-separated reservoir-sampler seeds
#   ENCODER       single encoder key (bigG | clipL | clip). default: bigG
#   AXES          which axes to sweep ("h3_res retention exemplars")
#                 default: all three. drop axes to short-circuit.
#
# Defaults for held-constant knobs (match v2 + E11):
#   H3 resolution: 10
#   retention: 75s x 12
#   exemplars: 128
#
# Usage:
#   conda activate psm
#   bash scripts/eval_hyperparam_sweep.sh
#
# Submission via SLURM:
#   sbatch scripts/slurm/eval_hyperparam.sbatch

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$REPO/datasets}"
CAPTURES="${CAPTURES:-$REPO/captures/hyperparam}"
SESSIONS="${SESSIONS:-1501677363692556 287142033569927 201703061033}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ENCODER="${ENCODER:-bigG}"
AXES="${AXES:-h3_res retention exemplars}"

H3_RES_LIST="${H3_RES_LIST:-8 9 10 11 12}"
RETENTION_LIST="${RETENTION_LIST:-30x24 45x16 75x12 150x6 300x3}"
EXEMPLARS_LIST="${EXEMPLARS_LIST:-16 32 64 128 256}"

# Defaults held constant when an axis is NOT being swept.
DEFAULT_H3_RES=10
DEFAULT_TIME_WINDOW=75
DEFAULT_CAPACITY=12
DEFAULT_EXEMPLARS=128

# Encoder -> (features basename, checkpoint) maps mirror eval_baselines_all.sh.
encoder_features() {
  case "$1" in
    bigG)     echo "clip_bigg_features.h5" ;;
    clipL)    echo "clip_l_features.h5" ;;
    clip)     echo "clip_features.h5" ;;
    siglip2L) echo "siglip2_l_features.h5" ;;
    *)        echo "" ;;
  esac
}
encoder_checkpoint() {
  case "$1" in
    bigG)     echo "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" ;;
    clipL)    echo "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" ;;
    clip)     echo "openai/clip-vit-base-patch32" ;;
    siglip2L) echo "google/siglip2-large-patch16-256" ;;
    *)        echo "" ;;
  esac
}

FEAT_BASENAME="$(encoder_features "$ENCODER")"
CHECKPOINT="$(encoder_checkpoint "$ENCODER")"
if [[ -z "$FEAT_BASENAME" || -z "$CHECKPOINT" ]]; then
  echo "[hyperparam] ERR: unknown encoder '$ENCODER'" >&2
  exit 1
fi

mkdir -p "$CAPTURES"
echo "[hyperparam] root=$ROOT encoder=$ENCODER ($FEAT_BASENAME)"
echo "[hyperparam] sessions: $SESSIONS"
echo "[hyperparam] seeds: $SEEDS"
echo "[hyperparam] axes: $AXES"
echo "[hyperparam] out: $CAPTURES"
echo

run_one() {
  # $1=axis, $2=label (value rendered for the filename),
  # $3=h3_res, $4=time_window, $5=capacity, $6=exemplars, $7=sid, $8=seed
  local axis="$1" label="$2" h3_res="$3" time_window="$4" capacity="$5" \
        exemplars="$6" sid="$7" seed="$8"
  local feat="$ROOT/$sid/$FEAT_BASENAME"
  local qs="$ROOT/$sid/questions.yaml"
  if [[ ! -f "$feat" ]]; then
    echo "[hyperparam] WARN: missing $feat, skipping $sid" >&2
    return
  fi
  if [[ ! -f "$qs" ]]; then
    echo "[hyperparam] WARN: missing $qs, skipping $sid" >&2
    return
  fi
  local out="$CAPTURES/${axis}_${label}_${sid}_${ENCODER}_s${seed}.json"
  if [[ -f "$out" ]]; then
    echo "[hyperparam] skip (exists): $out"
    return
  fi
  echo "[hyperparam] $axis=$label  sid=$sid  seed=$seed  -> $(basename "$out")"
  python "$REPO/scripts/eval_lookback.py" "$feat" "$qs" \
    --top 5 \
    --h3-resolution "$h3_res" \
    --time-window "$time_window" \
    --capacity "$capacity" \
    --exemplars "$exemplars" \
    --clip-checkpoint "$CHECKPOINT" \
    --seed "$seed" \
    --out "$out"
}

sweep_h3_res() {
  echo "=== axis: h3_res (range: $H3_RES_LIST) ==="
  for h3 in $H3_RES_LIST; do
    for sid in $SESSIONS; do
      for seed in $SEEDS; do
        run_one h3_res "$h3" "$h3" \
          "$DEFAULT_TIME_WINDOW" "$DEFAULT_CAPACITY" "$DEFAULT_EXEMPLARS" \
          "$sid" "$seed"
      done
    done
  done
}

sweep_retention() {
  echo "=== axis: retention (range: $RETENTION_LIST) ==="
  for token in $RETENTION_LIST; do
    local tw="${token%x*}"
    local cap="${token#*x}"
    local label="${tw}x${cap}"
    for sid in $SESSIONS; do
      for seed in $SEEDS; do
        run_one retention "$label" "$DEFAULT_H3_RES" \
          "$tw" "$cap" "$DEFAULT_EXEMPLARS" \
          "$sid" "$seed"
      done
    done
  done
}

sweep_exemplars() {
  echo "=== axis: exemplars (range: $EXEMPLARS_LIST) ==="
  for ex in $EXEMPLARS_LIST; do
    for sid in $SESSIONS; do
      for seed in $SEEDS; do
        run_one exemplars "$ex" "$DEFAULT_H3_RES" \
          "$DEFAULT_TIME_WINDOW" "$DEFAULT_CAPACITY" "$ex" \
          "$sid" "$seed"
      done
    done
  done
}

for axis in $AXES; do
  case "$axis" in
    h3_res)    sweep_h3_res ;;
    retention) sweep_retention ;;
    exemplars) sweep_exemplars ;;
    *)
      echo "[hyperparam] WARN: unknown axis '$axis', skipping" >&2
      ;;
  esac
  echo
done

echo "[hyperparam] done. Plot the sensitivity curves with:"
echo "  python scripts/eval_hyperparam_plot.py $CAPTURES"
