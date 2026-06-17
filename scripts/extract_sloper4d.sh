#!/usr/bin/env bash
# Extract CLIP-L features from SLOPER4D sequences for PSM evaluation.
#
# Reads each sequence's rgb_data/*.mp4 (egocentric video) and
# lidar_data/lidar_trajectory.txt (LiDAR-SLAM global XYZ), projects
# XYZ to WGS84 lat/lng via a fake origin at Xiamen University (where
# the dataset was captured), and writes a v2 features.h5 per sequence.
#
# Usage:
#   bash scripts/extract_sloper4d.sh /path/to/SLOPER4D
#
# Env knobs:
#   ROOT          SLOPER4D dataset root (or pass as $1)
#   OUT_ROOT      where features.h5 files land (default: $ROOT)
#   FPS           sample fps (default: 1)
#   MODEL         clip_l (default)
#   ORIGIN_LAT    fake WGS84 origin latitude  (default: 24.4381, Xiamen Univ)
#   ORIGIN_LNG    fake WGS84 origin longitude (default: 118.0992)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-${1:?Usage: $0 <SLOPER4D_ROOT>}}"
OUT_ROOT="${OUT_ROOT:-$ROOT}"
FPS="${FPS:-1}"
ORIGIN_LAT="${ORIGIN_LAT:-24.4381}"
ORIGIN_LNG="${ORIGIN_LNG:-118.0992}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SLOPER4D → PSM feature extraction                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  root:       $ROOT"
echo "║  out:        $OUT_ROOT"
echo "║  fps:        $FPS"
echo "║  origin:     $ORIGIN_LAT, $ORIGIN_LNG (Xiamen Univ)"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# Step 1: Probe sequences and print summary
echo "━━━ Probing sequences ━━━"
python3 -c "
import json, sys
sys.path.insert(0, '$REPO/extraction')
from psm_extraction.io.sloper4d import probe_sequences
from pathlib import Path

results = probe_sequences(
    Path('$ROOT'),
    origin_lat=$ORIGIN_LAT,
    origin_lng=$ORIGIN_LNG,
)
print(f'Found {len(results)} sequences:')
for r in results:
    if 'error' in r:
        print(f'  {r[\"sequence\"]:30s}  ERROR: {r[\"error\"]}')
    else:
        print(f'  {r[\"sequence\"]:30s}  {r[\"trajectory_m\"]:7.0f}m  {r[\"bbox_extent_m\"]:6.0f}m bbox  {r[\"frames\"]:6d} frames  {r[\"duration_s\"]:6.0f}s')
"
echo

# Step 2: Extract per sequence
echo "━━━ Extracting features ━━━"
python3 "$REPO/scripts/extract_sloper4d_sessions.py" \
  --root "$ROOT" \
  --out-root "$OUT_ROOT" \
  --fps "$FPS" \
  --origin-lat "$ORIGIN_LAT" \
  --origin-lng "$ORIGIN_LNG"

echo
echo "═══ Done. Features at $OUT_ROOT/<sequence>/clip_l_features.h5 ═══"
