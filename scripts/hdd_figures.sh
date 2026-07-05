#!/bin/bash
# Run the HDD Option-B figures. Two roots, because they consume different data:
#   - F-HDD-1 (memory-vs-area) is GPS-ONLY -> reads the raw RELEASE dir
#     (general/csv/rtk_pos.csv); it does NOT need the extraction and is already
#     complete, but we regenerate it here for a one-command refresh.
#   - F-HDD-2 (HLL cardinality) + F-HDD-3 (cross-session retrieval) consume the
#     per-drive feature H5s at <FEATURES>/<drive_day>/<drive_id>/<H5NAME>.
#
# Runs on any host that sees /checkpoint + has h3/numpy/h5py/matplotlib (e.g.
# this login host via /opt/conda/bin/python) -- pure analysis, no GPU. F-HDD-2
# --validate additionally needs the built targets/psm + captures/hdd/revisit_density.json.
#
#   bash scripts/hdd_figures.sh              # after the extraction array finishes
#
# Env:
#   RELEASE   raw HDD release root (F-HDD-1)   default below
#   FEATURES  extraction features root (F-HDD-2/3)   default below
#   PYTHON    interpreter with h3>=4.0 (default: /opt/conda/bin/python)
#   H5NAME    per-drive H5 basename (default clip_l_features.h5)

set -euo pipefail

RELEASE="${RELEASE:-/checkpoint/dream/arjangt/video_retrieval/hdd/release_2019_07_08}"
FEATURES="${FEATURES:-/checkpoint/dream/arjangt/video_retrieval/hdd/features}"
PYTHON="${PYTHON:-/opt/conda/bin/python}"
H5NAME="${H5NAME:-clip_l_features.h5}"

if ! "$PYTHON" -c "import h3, numpy, h5py" 2>/dev/null; then
  echo "ERR: $PYTHON lacks h3/numpy/h5py. Use an env with the viz extras "
  echo "     (pip install 'h3>=4.0' numpy h5py matplotlib), or set PYTHON=." >&2
  exit 1
fi

echo "=== F-HDD-1: memory-vs-area (GPS-only, release root) ==="
"$PYTHON" scripts/hdd_memory_vs_area.py "$RELEASE" --plot

n_h5=$(find "$FEATURES" -mindepth 3 -maxdepth 3 -name "$H5NAME" 2>/dev/null | wc -l)
echo "=== features: $n_h5 x $H5NAME under $FEATURES ==="
if [[ "$n_h5" -eq 0 ]]; then
  echo "[hdd-figures] no features yet -- F-HDD-2/3 need the extraction array; "
  echo "              F-HDD-1 above is done. Re-run for F-HDD-2/3 once features land." >&2
  exit 2
fi
if [[ "$n_h5" -lt 100 ]]; then
  echo "[hdd-figures] NOTE: only $n_h5 drives extracted so far (expected ~130);"
  echo "              F-HDD-2/3 will reflect partial coverage. Re-run when complete."
fi

echo "=== F-HDD-2: HLL cardinality accrual + decay (engine cross-checked) ==="
"$PYTHON" scripts/hdd_hll_cardinality.py "$FEATURES" --h5-name "$H5NAME" --validate --plot

echo "=== F-HDD-3: self-supervised cross-session retrieval ==="
"$PYTHON" scripts/hdd_cross_session_retrieval.py "$FEATURES" --h5-name "$H5NAME" --plot

echo
echo "[hdd-figures] done. JSON -> captures/hdd/  |  SVGs -> journal/figures/hdd_*"

