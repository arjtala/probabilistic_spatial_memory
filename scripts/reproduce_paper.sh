#!/usr/bin/env bash
# reproduce_paper.sh — one-shot reproduction of the PSM paper headlines.
#
# IMPORTANT: the captured artifacts are NOT shipped in this repo (captures/ is
# .gitignored). This script regenerates figures/tables *from* those artifacts
# once they are on disk; from a clean checkout it will abort (exit 3) rather
# than emit empty figures. Regenerate captures/ on the cluster first (see the
# "Where the captured artifacts came from" section below).
#
# When captures/ is present, regenerates from the captured JSON/MP4 artifacts:
#   - Table 2 H3 acceptance verdicts (13/14 PASS claim across LookOut +
#     SLOPER4D + Nymeria, three encoders).
#   - F2 (PSM vs any-of-K coverage at K=8, Gemini-3.1-Pro) table + PDF/SVG.
#   - F3 (H3 sensitivity, 3 encoders, 14-session mean ±1σ) PDF/SVG.
#   - F6 (memory + latency vs session length) PDF/SVG.
#   - Bootstrap 95 % CIs on the four-session Aria detail (mIoU + Hit@5).
#
# What this script DOES NOT do (and cannot, without external resources):
#   - Encoder feature extraction. The CLIP-L/CLIP-bigG/SigLIP-2L feature
#     H5s under $PSM_FEATURE_ROOT (default: ./features/) require a
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

# Resolve python: prefer $PSM_PY env var (operator override), else the
# user-active `python` if it looks like the psm env, else `python` on
# PATH and trust the user to have requirements-paper.txt installed.
if [[ -n "${PSM_PY:-}" ]]; then
  PY="$PSM_PY"
elif command -v python >/dev/null 2>&1; then
  PY_PATH="$(command -v python)"
  case "$PY_PATH" in
    *"/envs/psm/"*) PY="$PY_PATH" ;;
    *)              PY="$PY_PATH" ;;
  esac
else
  printf '[FATAL] No python interpreter on PATH.\n' >&2
  printf '         Activate a Python 3.12 env with requirements-paper.txt installed,\n' >&2
  printf '         or set PSM_PY=/abs/path/to/python.\n' >&2
  exit 2
fi

if [[ ! -x "$PY" ]]; then
  printf '[FATAL] $PSM_PY=%s is not executable.\n' "$PY" >&2
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

# captures/ is NOT shipped in the repo (it is .gitignored — the JSON/MP4
# artifacts are regenerated on the cluster; see the header). If it is absent
# the reproduction cannot do anything, so fail loudly instead of printing a
# green "done" over an empty run. Count how many of the 14 sessions actually
# have artifacts so downstream steps can refuse to fabricate a PASS summary.
missing_core=0
for d in captures benchmarks; do
  if [[ -d "$REPO/$d" ]]; then
    printf '       OK      %s\n' "$d"
  else
    printf '       MISSING %s\n' "$d"
    missing_core=$((missing_core + 1))
  fi
done
if [[ ! -d "$REPO/captures/mllm_baseline" ]]; then
  printf '       MISSING captures/mllm_baseline (F2 step will be skipped)\n'
fi
if [[ "$missing_core" -gt 0 ]]; then
  printf '\n'
  printf '[FATAL] captures/ or benchmarks/ is absent. This repo does NOT ship the\n' >&2
  printf '        captured artifacts (they are .gitignored and regenerated on the\n' >&2
  printf '        cluster — see this script'\''s header for the extraction/eval steps).\n' >&2
  printf '        Nothing to reproduce from a clean checkout; aborting rather than\n' >&2
  printf '        emitting empty figures and a hardcoded PASS claim.\n' >&2
  exit 3
fi
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
n_ran=$((n_pass + n_fail))
if [[ "$n_ran" -eq 0 ]]; then
  printf '       Summary: 0/%d sessions had artifacts on disk — NOTHING was reproduced.\n' \
    "${#V1_SESSIONS[@]}" >&2
  printf '       The paper claim (13/14 PASS) is NOT demonstrated by this run. Regenerate\n' >&2
  printf '       captures/aria_*_h3 / captures/sloper4d_*_h3 on the cluster first.\n' >&2
  exit 4
fi
printf '       Summary: %d PASS / %d FAIL / %d SKIP  (of %d sessions; paper claim: 13/14 PASS)\n' \
  "$n_pass" "$n_fail" "$n_skip" "${#V1_SESSIONS[@]}"
if [[ "$n_skip" -gt 0 ]]; then
  printf '       [WARN] %d session(s) skipped for missing artifacts — this is a PARTIAL\n' "$n_skip"
  printf '              reproduction; the 13/14 claim requires all 14 present.\n'
fi
printf '       (for full per-encoder breakdown re-run h3_acceptance.py without redirect)\n'
printf '\n'

# ----------------------------------------------------------------------
# Step 3/5 — any-of-K coverage aggregation (F2 table)
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
