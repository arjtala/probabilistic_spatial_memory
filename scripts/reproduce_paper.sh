#!/usr/bin/env bash
# reproduce_paper.sh — one-shot reproduction of the PSM paper headlines.
#
# IMPORTANT: the captured artifacts are NOT shipped in this repo (captures/ is
# .gitignored). This script regenerates figures/tables *from* those artifacts
# once they are on disk; from a clean checkout it will abort (exit 3) rather
# than emit empty figures. Regenerate captures/ on the cluster first (see the
# "Where the captured artifacts came from" section below).
#
# When captures/ is present, regenerates the submitted paper artifacts:
#   - Aggregate H3 sensitivity verdicts, using pre-registered CLIP-L (9/14).
#   - Fixed-budget summary, paired session-bootstrap comparisons, and figure.
#   - Coordinate-permutation and grid-boundary paired control summaries.
#   - HDD analytical modeled-memory figure from the RTK-derived JSON.
#
# What this script DOES NOT do (and cannot, without external resources):
#   - Encoder feature extraction. The CLIP-L/CLIP-bigG/SigLIP-2L feature
#     H5s under $PSM_FEATURE_ROOT (default: ./features/) require a
#     cluster GPU pass (~12 GPU-hours wall for the full 14-session set).
#   - H3-resolution sweep rerun. captures/aria_*_h3/, captures/sloper4d_*_h3/
#     are the output of `scripts/eval_hyperparam_sweep.sh` after features
#     and questions are on disk.
#   - Fixed-budget sweep rerun. Use scripts/slurm/fixed_budget_suite.sbatch
#     after downloading and extracting the provider-licensed datasets.
# Use scripts/aria_acceptance.sh or scripts/sloper4d_acceptance.sh to drive
# the full pipeline end-to-end when the prerequisites are met.
#
# Where the captured artifacts came from:
#   - captures/aria_*_h3/         scripts/aria_acceptance.sh (LookOut MPS)
#   - captures/sloper4d_*_h3/     scripts/sloper4d_acceptance.sh (SLOPER4D
#                                   + Nymeria; the shelby_arroyo session
#                                   reuses the SLOPER4D harness)
#   - captures/wearables_fixed_budget/  fixed_budget_suite.sbatch MODE=base
#   - captures/{grid_controls,coordinate_null}/  the same sbatch in grid/null mode
#   - captures/hdd/memory_vs_area.json  scripts/hdd_memory_vs_area.py
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
TOTAL=4

printf '╔════════════════════════════════════════════════════════════╗\n'
printf '║  PSM paper — one-shot reproduction                          ║\n'
printf '╠════════════════════════════════════════════════════════════╣\n'
printf '║  repo:   %-50s║\n' "$REPO"
printf '║  python: %-50s║\n' "$PY"
printf '╚════════════════════════════════════════════════════════════╝\n'
printf '\n'

# ----------------------------------------------------------------------
# Step 1/4 — sanity check (env + captures on disk)
# ----------------------------------------------------------------------
printf '[1/%d] Sanity check — interpreter + captures layout\n' "$TOTAL"
printf '       python: %s\n' "$PY"
deps_missing=0
"$PY" -c "import numpy, h3, h5py, matplotlib" 2>/dev/null || {
  printf '       [WARN] one of numpy/h3/h5py/matplotlib is not importable.\n'
  deps_missing=1
}

# Captures are not shipped in the repo. Require every artifact family used by
# the submitted figures/controls so a partial checkout cannot look successful.
missing_core=0
required_artifacts=(
  "captures/wearables_fixed_budget"
  "captures/grid_controls"
  "captures/coordinate_null"
  "captures/hdd/memory_vs_area.json"
)
for rel in "${required_artifacts[@]}"; do
  if [[ -e "$REPO/$rel" ]]; then
    printf '       OK      %s\n' "$rel"
  else
    printf '       MISSING %s\n' "$rel"
    missing_core=$((missing_core + 1))
  fi
done
if [[ "$missing_core" -gt 0 ]]; then
  printf '\n'
  printf '[FATAL] Required captures are absent. The repository does not redistribute\n' >&2
  printf '        provider-licensed datasets or generated captures. Download each\n' >&2
  printf '        dataset from its provider and run the extraction/evaluation commands\n' >&2
  printf '        listed in README.md and this script before reproducing paper assets.\n' >&2
  exit 3
fi
if [[ "$deps_missing" -ne 0 ]]; then
  printf '[FATAL] Install requirements-paper.txt in the selected interpreter.\n' >&2
  exit 2
fi
printf '\n'

# ----------------------------------------------------------------------
# Step 2/4 — pre-registered H3 acceptance verdicts
# ----------------------------------------------------------------------
# 14 v1 sessions: (sequence_id, captures_dir). Order matches the paper table.
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

printf '[2/%d] H3 acceptance verdicts — pre-registered CLIP-L (14 sessions)\n' "$TOTAL"
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
        --sequence "$seq" \
        --encoder clipL >/dev/null 2>&1; then
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
  printf '       The paper claim (pre-registered CLIP-L: 9/14 PASS) is not demonstrated. Regenerate\n' >&2
  printf '       captures/aria_*_h3 / captures/sloper4d_*_h3 on the cluster first.\n' >&2
  exit 4
fi
printf '       Summary: %d PASS / %d FAIL / %d SKIP  (of %d sessions; expected: 9/14 PASS)\n' \
  "$n_pass" "$n_fail" "$n_skip" "${#V1_SESSIONS[@]}"
if [[ "$n_skip" -gt 0 ]]; then
  printf '[FATAL] %d session(s) skipped; the 9/14 result requires all 14.\n' "$n_skip" >&2
  exit 4
fi
if [[ "$n_pass" -ne 9 || "$n_fail" -ne 5 ]]; then
  printf '[FATAL] CLIP-L verdict mismatch: observed %d/%d PASS, expected 9/14.\n' \
    "$n_pass" "$n_ran" >&2
  exit 5
fi
printf '       (for the post-hoc per-encoder breakdown, run h3_acceptance.py without --encoder)\n'
printf '\n'

# ----------------------------------------------------------------------
# Step 3/4 — fixed-budget table, paired controls, and submitted figure
# ----------------------------------------------------------------------
printf '[3/%d] Fixed-budget allocation — summary, paired CIs, controls, figure\n' "$TOTAL"
BASE="$REPO/captures/wearables_fixed_budget"
GRID="$REPO/captures/grid_controls"
NULL="$REPO/captures/coordinate_null"

"$PY" "$REPO/scripts/summarize_fixed_budget.py" \
  "$BASE" --recursive --strict --quiet \
  --out "$BASE/summary.md"
"$PY" "$REPO/scripts/compare_fixed_budget.py" \
  "$BASE" --recursive \
  --method-a spatial_priority --method-b global_reservoir \
  --budget 128 --h3-resolution 12 \
  --out "$BASE/paired_primary.md"
"$PY" "$REPO/scripts/plot_fixed_budget.py" "$BASE" --recursive

for transform in e4p5 n4p5 rot30; do
  "$PY" "$REPO/scripts/compare_fixed_budget.py" \
    "$BASE" "$GRID" --recursive \
    --method-a spatial_priority --method-b spatial_priority \
    --transform-a base --transform-b "$transform" \
    --budget 128 --h3-resolution 12 \
    --out "$GRID/paired_base_vs_${transform}.md"
done
for transform in perm101 perm202 perm303; do
  "$PY" "$REPO/scripts/compare_fixed_budget.py" \
    "$BASE" "$NULL" --recursive \
    --method-a spatial_priority --method-b spatial_priority \
    --transform-a base --transform-b "$transform" \
    --budget 128 --h3-resolution 12 \
    --out "$NULL/paired_base_vs_${transform}.md"
done
printf '\n'

# ----------------------------------------------------------------------
# Step 4/4 — corrected HDD analytical memory figure
# ----------------------------------------------------------------------
printf '[4/%d] HDD modeled memory — submitted SVG + PDF\n' "$TOTAL"
"$PY" "$REPO/scripts/plot_f5_hdd_memory.py" \
  --json "$REPO/captures/hdd/memory_vs_area.json"
printf '\n'

printf '═════ Done. Generated artifacts:\n'
printf '       captures/wearables_fixed_budget/{summary,paired_primary}.md\n'
printf '       captures/{grid_controls,coordinate_null}/paired_base_vs_*.md\n'
printf '       journal/figures/fixed_budget.{svg,pdf}\n'
printf '       journal/figures/f5_hdd_memory.{svg,pdf}\n'
printf '       H3 CLIP-L 9/14 verdict printed above.\n'
