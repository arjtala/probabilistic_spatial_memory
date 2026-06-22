#!/usr/bin/env bash
# reproduce_paper.sh — one-shot reproduction of the PSM paper headlines.
#
# Regenerates from the captured JSON/MP4 artifacts that ship in this repo:
#   - Table 2 H3 acceptance verdicts (13/14 PASS claim across LookOut +
#     SLOPER4D + Nymeria, three encoders).
#   - F2 (PSM vs vanilla Gemini-3.1-Pro at K=8) table + PDF/SVG.
#   - F3 (H3 sensitivity, 3 encoders, 14-session mean ±1σ) PDF/SVG.
#   - F6 (memory + latency vs session length) PDF/SVG.
#   - Bootstrap 95 % CIs on the four-session Aria detail (mIoU + Hit@5).
#
# What this script DOES NOT do (and cannot, without external resources):
#   - Encoder feature extraction. The CLIP-L/CLIP-bigG/SigLIP-2L feature
#     H5s under /checkpoint/dream/arjangt/video_retrieval/... require a
#     cluster GPU pass (~12 GPU-hours wall for the full 14-session set).
#   - MLLM baseline rerun. captures/mllm_baseline/*_gemini.json each
#     embed ~30 Gemini 3.1 Pro completions; regenerating them needs
#     GEMINI_API_KEY and ~3000 API calls.
#   - H3-resolution sweep rerun. captures/aria_*_h3/, captures/sloper4d_*_h3/
#     are the output of `scripts/eval_hyperparam_sweep.sh` after features
#     and questions are on disk; rerunning is gated on the two items above.
# Use scripts/aria_acceptance.sh or scripts/sloper4d_acceptance.sh to drive
# the full pipeline end-to-end when the prerequisites are met.
#
# Where the captured artifacts came from:
#   - captures/aria_*_h3/         scripts/aria_acceptance.sh (LookOut MPS)
#   - captures/sloper4d_*_h3/     scripts/sloper4d_acceptance.sh (SLOPER4D
#                                   + Nymeria; the shelby_arroyo session
#                                   reuses the SLOPER4D harness)
#   - captures/mllm_baseline/     scripts/eval_mllm_baseline.py --mllm gemini
#   - benchmarks/nymeria/         scripts/bench_brute_force_clip.py (30 sess.)
#
# Usage:
#   conda activate psm                      # or any Python 3.12 env with
#                                           # `pip install -r requirements-paper.txt`
#   bash scripts/reproduce_paper.sh
#
# Dependencies for this driver are pinned in requirements-paper.txt at
# the repo root (numpy / h3 / h5py). The full extraction-side stack
# (torch, transformers, projectaria-tools, ...) lives in
# extraction/pyproject.toml and is only needed to regenerate captures.
#
# Idempotent: re-running overwrites the generated PDFs/SVGs in place.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Resolve python: prefer the user-active `python` only if it's the psm env,
# otherwise fall back to the absolute psm-env interpreter.
PSM_PY_FALLBACK="/home/arjangt/.conda/envs/psm/bin/python"
if command -v python >/dev/null 2>&1; then
  PY_PATH="$(command -v python)"
  case "$PY_PATH" in
    *"/envs/psm/"*) PY="$PY_PATH" ;;
    *) PY="$PSM_PY_FALLBACK" ;;
  esac
else
  PY="$PSM_PY_FALLBACK"
fi

if [[ ! -x "$PY" ]]; then
  printf '[FATAL] No usable python interpreter found (tried %s).\n' "$PY" >&2
  printf '         Activate the psm conda env: `conda activate psm`.\n' >&2
  exit 2
fi

# Total step count for the [N/M] headers.
TOTAL=5

printf '╔════════════════════════════════════════════════════════════╗\n'
printf '║  PSM paper — one-shot reproduction                          ║\n'
printf '╠════════════════════════════════════════════════════════════╣\n'
printf '║  repo:   %-50s║\n' "$REPO"
printf '║  python: %-50s║\n' "$PY"
printf '╚════════════════════════════════════════════════════════════╝\n'
printf '\n'

# ----------------------------------------------------------------------
# Step 1/5 — sanity check (env + captures on disk)
# ----------------------------------------------------------------------
printf '[1/%d] Sanity check — interpreter + captures layout\n' "$TOTAL"
printf '       python: %s\n' "$PY"
"$PY" -c "import numpy, h3, h5py" 2>/dev/null \
  || printf '       [WARN] one of numpy/h3/h5py not importable in this interpreter.\n'

for d in captures captures/mllm_baseline benchmarks; do
  if [[ -d "$REPO/$d" ]]; then
    printf '       OK     %s\n' "$d"
  else
    printf '       MISSING %s (downstream steps may be partial)\n' "$d"
  fi
done
printf '\n'

# ----------------------------------------------------------------------
# Step 2/5 — H3 acceptance verdicts (reproduces Table 2)
# ----------------------------------------------------------------------
# 14 v1 sessions: (sequence_id, captures_dir). Order matches plot_f3.
V1_SESSIONS=(
  "Mainquad_jan10                          captures/aria_Mainquad_jan10_h3"
  "Sanmateopark_garage_jan11               captures/aria_Sanmateopark_garage_jan11_h3"
  "Fostersquare1_jan16                     captures/aria_Fostersquare1_jan16_h3"
  "BurlingameDT5_feb5                      captures/aria_BurlingameDT5_feb5_h3"
  "SanmateoDT2_Jan12                       captures/aria_SanmateoDT2_Jan12_h3"
  "Gates_to_mainquad_jan10                 captures/aria_Gates_to_mainquad_jan10_h3"
  "Huang_Gates_jan10                       captures/aria_Huang_Gates_jan10_h3"
  "BurlingameDT4_feb5                      captures/aria_BurlingameDT4_feb5_h3"
  "SSC3_jan17_                             captures/aria_SSC3_jan17__h3"
  "Hillsdale6_jan14                        captures/aria_Hillsdale6_jan14_h3"
  "seq003_street_002                       captures/sloper4d_seq003_street_002_h3"
  "seq008_running_001                      captures/sloper4d_seq008_running_001_h3"
  "seq009_running_002                      captures/sloper4d_seq009_h3"
  "20230608_s0_shelby_arroyo_act0_3ciwl8   captures/sloper4d_20230608_s0_shelby_arroyo_act0_3ciwl8_h3"
)

printf '[2/%d] H3 acceptance verdicts — Table 2 reproduction (14 sessions)\n' "$TOTAL"
n_pass=0
n_fail=0
n_skip=0
for entry in "${V1_SESSIONS[@]}"; do
  read -r seq caps_rel <<< "$entry"
  caps_abs="$REPO/$caps_rel"
  if [[ ! -d "$caps_abs" ]]; then
    printf '       SKIP    %-45s (missing %s)\n' "$seq" "$caps_rel"
    n_skip=$((n_skip + 1))
    continue
  fi
  # h3_acceptance.py exits 0 on PASS / 1 on FAIL. We want to continue
  # the loop in both cases — `if !` guards `set -e`.
  if "$PY" "$REPO/scripts/h3_acceptance.py" \
        --captures "$caps_abs" \
        --sequence "$seq" >/dev/null 2>&1; then
    printf '       PASS    %s\n' "$seq"
    n_pass=$((n_pass + 1))
  else
    printf '       FAIL    %s\n' "$seq"
    n_fail=$((n_fail + 1))
  fi
done
printf '\n'
printf '       Summary: %d PASS / %d FAIL / %d SKIP (paper claim: 13/14 PASS)\n' \
  "$n_pass" "$n_fail" "$n_skip"
printf '       (for full per-encoder breakdown re-run h3_acceptance.py without redirect)\n'
printf '\n'

# ----------------------------------------------------------------------
# Step 3/5 — vanilla MLLM aggregation (F2 table)
# ----------------------------------------------------------------------
printf '[3/%d] Vanilla MLLM aggregation — F2 table (PSM vs Gemini-3.1-Pro @ K=8)\n' "$TOTAL"
if [[ ! -d "$REPO/captures/mllm_baseline" ]] || \
   ! ls "$REPO/captures/mllm_baseline"/*_gemini.json >/dev/null 2>&1; then
  printf '       SKIP — captures/mllm_baseline/*_gemini.json missing.\n'
else
  # plot_f2 prints the apples-to-apples per-session table on stdout
  # and writes the SVG. Stdout is the F2 reproduction artifact for
  # reviewers reading the script log.
  if ! "$PY" "$REPO/scripts/plot_f2_psm_vs_mllm.py"; then
    printf '       [WARN] plot_f2_psm_vs_mllm.py exited non-zero; continuing.\n'
  fi
fi
printf '\n'

# ----------------------------------------------------------------------
# Step 4/5 — figure regeneration (F2 / F3 / F6 PDFs + SVGs)
# ----------------------------------------------------------------------
printf '[4/%d] Figure regeneration — F2 / F3 / F6 SVG + PDF\n' "$TOTAL"

run_plot () {
  local label="$1"; shift
  local script="$1"; shift
  if [[ ! -f "$REPO/scripts/$script" ]]; then
    printf '       SKIP %s — missing scripts/%s\n' "$label" "$script"
    return
  fi
  if ! "$PY" "$REPO/scripts/$script" "$@"; then
    printf '       [WARN] %s exited non-zero; continuing.\n' "$label"
  fi
}

run_plot "F2" plot_f2_psm_vs_mllm.py
run_plot "F3" plot_f3_multi_corpus.py
run_plot "F6" plot_f6_memory_latency.py

# Try to rsvg-convert the SVGs to PDF (the plot scripts only print the
# command they would run). If rsvg-convert isn't installed, leave the
# SVG as the canonical artifact.
if command -v rsvg-convert >/dev/null 2>&1; then
  for stem in f2_psm_vs_mllm f3_multi_corpus_h3 f6_memory_latency; do
    svg="$REPO/journal/figures/${stem}.svg"
    pdf="$REPO/journal/figures/${stem}.pdf"
    if [[ -f "$svg" ]]; then
      if rsvg-convert -f pdf -o "$pdf" "$svg" 2>/dev/null; then
        printf '       PDF    %s\n' "journal/figures/${stem}.pdf"
      else
        printf '       [WARN] rsvg-convert failed for %s; SVG is still up-to-date.\n' "$stem"
      fi
    fi
  done
else
  printf '       (rsvg-convert not on PATH — SVGs are regenerated; PDFs left untouched.\n'
  printf '        Install librsvg, or render manually: rsvg-convert -f pdf -o X.pdf X.svg)\n'
fi
printf '\n'

# ----------------------------------------------------------------------
# Step 5/5 — bootstrap CIs on the headline 4-session Aria detail
# ----------------------------------------------------------------------
printf '[5/%d] Bootstrap 95%% CIs — 4-session Aria detail (mIoU + Hit@5)\n' "$TOTAL"
if [[ ! -f "$REPO/scripts/bootstrap_ci.py" ]]; then
  printf '       SKIP — scripts/bootstrap_ci.py not present.\n'
else
  # Headline comparison: clipBigG eval over the four detail sessions
  # (the file pattern matches the captures already on disk for the
  # main per_cell_cap=K table).
  shopt -s nullglob
  bigg_files=( "$REPO"/captures/eval_*_clipBigG_e128_s*.json )
  shopt -u nullglob
  if [[ ${#bigg_files[@]} -eq 0 ]]; then
    printf '       SKIP — no captures/eval_*_clipBigG_e128_s*.json on disk.\n'
  else
    printf '       Aggregating %d eval JSONs (clipBigG, e128, all seeds)…\n' \
      "${#bigg_files[@]}"
    if ! "$PY" "$REPO/scripts/bootstrap_ci.py" --aggregate "${bigg_files[@]}"; then
      printf '       [WARN] bootstrap_ci.py exited non-zero; continuing.\n'
    fi
  fi
fi
printf '\n'

printf '═════ Done. Generated artifacts:\n'
printf '       journal/figures/f2_psm_vs_mllm.{svg,pdf}\n'
printf '       journal/figures/f3_multi_corpus_h3.{svg,pdf}\n'
printf '       journal/figures/f6_memory_latency.{svg,pdf}\n'
printf '       (Tables 2/F2 + bootstrap CIs printed above.)\n'
