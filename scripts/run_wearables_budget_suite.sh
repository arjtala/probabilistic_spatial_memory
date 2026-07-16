#!/usr/bin/env bash
# Run the fixed-exemplar-budget experiment over one or more session folders.
#
# Each session directory must contain FEATURE_NAME and QUESTIONS_NAME. Pass
# session IDs as positional arguments, or omit them to auto-discover every
# complete session directly below ROOT.
#
# Example:
#   ROOT=/data/nymeria_atomic CLIP_DEVICE=cuda \
#     bash scripts/run_wearables_budget_suite.sh session_a session_b

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-python}"
ROOT="${ROOT:-${PSM_DATA_ROOT:-}/video_retrieval/nymeria_atomic}"
OUT_ROOT="${OUT_ROOT:-captures/wearables_fixed_budget}"
FEATURE_NAME="${FEATURE_NAME:-clip_l_features.h5}"
QUESTIONS_NAME="${QUESTIONS_NAME:-questions.yaml}"
BUDGETS="${BUDGETS:-128}"
SEEDS="${SEEDS:-0,1,2,3,4}"
H3_RESOLUTIONS="${H3_RESOLUTIONS:-12}"
METHODS="${METHODS:-global_reservoir,fifo,hybrid,uniform_time,semantic_kcenter,spatial_priority,spatial_balanced,visit_balanced}"
CLIP_DEVICE="${CLIP_DEVICE:-auto}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-laion/CLIP-ViT-L-14-laion2B-s32B-b82K}"

if [[ -z "$ROOT" || "$ROOT" == "/video_retrieval/nymeria_atomic" ]]; then
  printf '[FATAL] Set ROOT or PSM_DATA_ROOT to the Nymeria extraction tree.\n' >&2
  exit 2
fi
if [[ ! -d "$ROOT" ]]; then
  printf '[FATAL] Dataset root does not exist: %s\n' "$ROOT" >&2
  exit 2
fi

sessions=("$@")
if [[ ${#sessions[@]} -eq 0 ]]; then
  while IFS= read -r session_dir; do
    sessions+=("$(basename "$session_dir")")
  done < <(
    find "$ROOT" -mindepth 1 -maxdepth 1 -type d \
      -exec test -f "{}/$FEATURE_NAME" \; \
      -exec test -f "{}/$QUESTIONS_NAME" \; \
      -print | sort
  )
fi
if [[ ${#sessions[@]} -eq 0 ]]; then
  printf '[FATAL] No complete sessions under %s.\n' "$ROOT" >&2
  exit 2
fi

extra=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra+=(--dry-run)
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  extra+=(--force)
fi

printf '[suite] root=%s sessions=%d budgets=%s seeds=%s resolutions=%s\n' \
  "$ROOT" "${#sessions[@]}" "$BUDGETS" "$SEEDS" "$H3_RESOLUTIONS"

completed=0
for session in "${sessions[@]}"; do
  features="$ROOT/$session/$FEATURE_NAME"
  questions="$ROOT/$session/$QUESTIONS_NAME"
  if [[ ! -f "$features" || ! -f "$questions" ]]; then
    printf '[WARN] Skip %s: missing %s or %s.\n' \
      "$session" "$FEATURE_NAME" "$QUESTIONS_NAME" >&2
    continue
  fi
  printf '\n[suite] === %s ===\n' "$session"
  "$PY" scripts/eval_fixed_budget.py \
    "$features" "$questions" \
    --budgets "$BUDGETS" \
    --seeds "$SEEDS" \
    --h3-resolutions "$H3_RESOLUTIONS" \
    --methods "$METHODS" \
    --clip-device "$CLIP_DEVICE" \
    --clip-checkpoint "$CLIP_CHECKPOINT" \
    --out-dir "$OUT_ROOT/$session" \
    "${extra[@]}"
  completed=$((completed + 1))
done

if [[ "${DRY_RUN:-0}" != "1" && $completed -gt 0 ]]; then
  "$PY" scripts/summarize_fixed_budget.py \
    "$OUT_ROOT" --recursive --strict \
    --out "$OUT_ROOT/summary.md"
fi

printf '\n[suite] completed %d/%d session(s).\n' "$completed" "${#sessions[@]}"
