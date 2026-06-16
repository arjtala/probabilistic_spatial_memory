#!/usr/bin/env bash
# Run the full Nymeria 30-session evaluation suite for the ECCV 2026
# Wearables AI paper. Wraps the existing eval scripts with Nymeria-
# specific defaults:
#   - CLIP-L encoder (only encoder extracted for Nymeria)
#   - captures/nymeria/ output tree (separate from Aria-internal results)
#   - 30 sessions auto-discovered from the cluster features dir
#   - Single seed (corpus breadth > seed variance at 30 sessions)
#
# Three stages, each independently skippable:
#   1. E11 baselines (brute-force, sliding-window, uniform-sample)
#   2. E12 hyperparameter sweep (H3 res, retention, exemplars)
#   3. Item 7 latency benchmark (brute-force CLIP latency scaling)
#
# Usage:
#   bash scripts/eval_nymeria_all.sh              # all three stages
#   STAGES="baselines" bash scripts/eval_nymeria_all.sh
#   STAGES="hyperparam" bash scripts/eval_nymeria_all.sh
#   STAGES="latency" bash scripts/eval_nymeria_all.sh
#   STAGES="baselines hyperparam" bash scripts/eval_nymeria_all.sh
#
# Env knobs (override any default):
#   ROOT          Nymeria features root
#   CAPTURES      output root (default captures/nymeria)
#   STAGES        which stages to run (default "baselines hyperparam latency")
#   SEEDS         seeds for hyperparam sweep (default "0" — single seed)
#   SESSIONS      override auto-discovery (space-separated session ids)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic}"
CAPTURES="${CAPTURES:-$REPO/captures/nymeria}"
STAGES="${STAGES:-baselines hyperparam latency}"
SEEDS="${SEEDS:-0}"

# ── Auto-discover sessions from the cluster features dir ──────────
if [[ -z "${SESSIONS:-}" ]]; then
  if [[ -f "$ROOT/.session_ids.txt" ]]; then
    SESSIONS="$(tr '\n' ' ' < "$ROOT/.session_ids.txt")"
  else
    SESSIONS="$(cd "$ROOT" && ls -d */ 2>/dev/null | sed 's:/$::' | tr '\n' ' ')"
  fi
fi

N_SESSIONS=$(echo "$SESSIONS" | wc -w | tr -d ' ')
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PSM Nymeria 30-session evaluation (ECCV 2026)         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  root:     $ROOT"
echo "║  captures: $CAPTURES"
echo "║  sessions: $N_SESSIONS"
echo "║  encoder:  clipL (CLIP-ViT-L-14-laion2B)"
echo "║  seeds:    $SEEDS"
echo "║  stages:   $STAGES"
echo "╚══════════════════════════════════════════════════════════╝"
echo

mkdir -p "$CAPTURES"

# ── Stage 1: E11 baselines ────────────────────────────────────────
if [[ "$STAGES" == *baselines* ]]; then
  echo "━━━ Stage 1: E11 baselines (brute-force, sliding-window, uniform-sample) ━━━"
  ROOT="$ROOT" \
  SESSIONS="$SESSIONS" \
  ENCODERS="clipL" \
  OUT_DIR="$CAPTURES/baselines" \
    bash "$REPO/scripts/eval_baselines_all.sh"
  echo
fi

# ── Stage 2: E12 hyperparameter sweep ─────────────────────────────
if [[ "$STAGES" == *hyperparam* ]]; then
  echo "━━━ Stage 2: E12 hyperparameter sweep (30 sessions × 1 seed) ━━━"
  ROOT="$ROOT" \
  SESSIONS="$SESSIONS" \
  SEEDS="$SEEDS" \
  ENCODER="clipL" \
  CAPTURES="$CAPTURES/hyperparam" \
    bash "$REPO/scripts/eval_hyperparam_sweep.sh"
  echo
fi

# ── Stage 3: Item 7 latency benchmark ────────────────────────────
if [[ "$STAGES" == *latency* ]]; then
  echo "━━━ Stage 3: Item 7 brute-force CLIP latency scaling (30 sessions) ━━━"
  LATENCY_DIR="$CAPTURES/latency"
  mkdir -p "$LATENCY_DIR"
  for sid in $SESSIONS; do
    h5="$ROOT/$sid/clip_l_features.h5"
    if [[ ! -f "$h5" ]]; then
      echo "[latency] WARN: missing $h5, skipping $sid" >&2
      continue
    fi
    python3 "$REPO/scripts/bench_brute_force_clip.py" \
      "$h5" \
      --group clip \
      --clip-checkpoint "laion/CLIP-ViT-L-14-laion2B-s32B-b82K" \
      --out "$LATENCY_DIR/bench_brute_force_${sid}.json"
  done
  echo "[latency] Done. $N_SESSIONS sessions → $LATENCY_DIR/"
  echo
fi

echo "═══ All stages complete. Outputs under $CAPTURES/ ═══"
