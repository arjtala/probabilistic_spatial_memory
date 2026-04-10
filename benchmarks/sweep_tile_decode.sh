#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PROFILE=${PROFILE:-local}
TOTAL_DECODES_LIST=${TOTAL_DECODES_LIST:-"4000 16000 64000"}
THREAD_COUNT_LIST=${THREAD_COUNT_LIST:-"1 2 4 8"}
SOURCE_MODE_LIST=${SOURCE_MODE_LIST:-"memory disk"}

case "$PROFILE" in
  local) TARGET_DIR="targets" ;;
  portable|debug|sanitize) TARGET_DIR="targets/$PROFILE" ;;
  *)
    printf 'Unsupported PROFILE: %s\n' "$PROFILE" >&2
    exit 1
    ;;
esac

BENCH_TARGET="$TARGET_DIR/benchmark_tile_decode"
BENCH_BIN="./$BENCH_TARGET"

cd "$REPO_ROOT"
make PROFILE="$PROFILE" "$BENCH_TARGET" >/dev/null

printf 'profile,total_decodes,thread_count,source_mode,failures,elapsed_sec,decodes_per_sec,megapixels_per_sec\n'

for total_decodes in $TOTAL_DECODES_LIST; do
  for thread_count in $THREAD_COUNT_LIST; do
    for source_mode in $SOURCE_MODE_LIST; do
      "$BENCH_BIN" "$total_decodes" "$thread_count" "$source_mode" \
        | awk -F ':' -v profile="$PROFILE" \
                     -v total_decodes="$total_decodes" \
                     -v thread_count="$thread_count" '
          function trim(text) {
            sub(/^[[:space:]]+/, "", text)
            sub(/[[:space:]]+$/, "", text)
            return text
          }
          /^[[:space:]]+source_mode:/ { source_mode = trim($2) }
          /^[[:space:]]+failures:/ { failures = trim($2) }
          /^[[:space:]]+elapsed_sec:/ { elapsed_sec = trim($2) }
          /^[[:space:]]+decodes_per_sec:/ { decodes_per_sec = trim($2) }
          /^[[:space:]]+megapixels\/sec:/ { megapixels_per_sec = trim($2) }
          END {
            printf "%s,%s,%s,%s,%s,%s,%s,%s\n",
                   profile, total_decodes, thread_count, source_mode,
                   failures, elapsed_sec, decodes_per_sec, megapixels_per_sec
          }'
    done
  done
done
