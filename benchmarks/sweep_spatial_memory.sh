#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PROFILE=${PROFILE:-local}
OBSERVE_OPS_LIST=${OBSERVE_OPS_LIST:-"50000 200000 1000000"}
GRID_CELLS_LIST=${GRID_CELLS_LIST:-"64 256 1024 4096"}
QUERY_OPS_LIST=${QUERY_OPS_LIST:-"50000 200000 1000000"}

case "$PROFILE" in
  local) TARGET_DIR="targets" ;;
  portable|debug|sanitize) TARGET_DIR="targets/$PROFILE" ;;
  *)
    printf 'Unsupported PROFILE: %s\n' "$PROFILE" >&2
    exit 1
    ;;
esac

BENCH_TARGET="$TARGET_DIR/benchmark_spatial_memory"
BENCH_BIN="./$BENCH_TARGET"

cd "$REPO_ROOT"
make PROFILE="$PROFILE" "$BENCH_TARGET" >/dev/null

printf 'profile,observe_ops,grid_cells,query_ops,scenario,ops,tiles,total,secs,ops_per_sec\n'

for observe_ops in $OBSERVE_OPS_LIST; do
  for grid_cells in $GRID_CELLS_LIST; do
    for query_ops in $QUERY_OPS_LIST; do
      "$BENCH_BIN" "$observe_ops" "$grid_cells" "$query_ops" \
        | awk -v profile="$PROFILE" \
               -v observe_ops="$observe_ops" \
               -v grid_cells="$grid_cells" \
               -v query_ops="$query_ops" '
          $1 == "observe_same_cell" || $1 == "observe_grid" || $1 == "query_grid" || $1 == "query_intervals" || $1 == "query_similar" {
            scenario = $1
            ops = ""
            tiles = ""
            total = ""
            secs = ""
            ops_per_sec = ""
            for (i = 2; i <= NF; i++) {
              split($i, kv, "=")
              if (kv[1] == "ops") ops = kv[2]
              else if (kv[1] == "tiles") tiles = kv[2]
              else if (kv[1] == "total") total = kv[2]
              else if (kv[1] == "secs") secs = kv[2]
              else if (kv[1] == "ops/sec") ops_per_sec = kv[2]
            }
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                   profile, observe_ops, grid_cells, query_ops, scenario,
                   ops, tiles, total, secs, ops_per_sec
          }'
    done
  done
done
