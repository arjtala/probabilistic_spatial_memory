#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/tile.h"

Tile *Tile_new(const double lat, const double lng, const int resolution, const size_t capacity, const size_t precision) {
  if (resolution < 0 || resolution > 15) {
    fprintf(stderr, "Tile_new: H3 resolution must be in [0, 15], got %d\n",
            resolution);
    return NULL;
  }
  if (!isfinite(lat) || !isfinite(lng) || lat < -90.0 || lat > 90.0 ||
      lng < -180.0 || lng > 180.0) {
    fprintf(stderr, "Tile_new: invalid lat/lng\n");
    return NULL;
  }
  LatLng loc;
  loc.lat = degsToRads(lat);
  loc.lng = degsToRads(lng);
  H3Index cellId;
  H3Error err = latLngToCell(&loc, resolution, &cellId);
  if (err) {
    fprintf(stderr, "Tile_new: invalid lat/lng or resolution\n");
    return NULL;
  }

  Tile *tile = (Tile *)malloc(sizeof(Tile));
  if (NULL == tile) {
    fprintf(stderr, "Tile_new: out of memory\n");
    return NULL;
  }

  RingBuffer *rb = RingBuffer_new(capacity, precision);
  if (!rb) {
    free(tile);
    return NULL;
  }

  tile->cellId = cellId;
  tile->rb = rb;
  return tile;
}

void Tile_free(Tile *tile) {
  if (!tile) return;
  RingBuffer_free(tile->rb);
  free(tile);
}

// Record an observation in this tile's current time window. Gets the current HLL
//  from the ring buffer and adds the data to it. This is how you say "I saw this thing in this location."
void Tile_add(Tile *tile, const void *data, size_t size) {
  HLL *current = tile ? RingBuffer_current(tile->rb) : NULL;
  if (!current) return;
  HLL_add(current, data, size);
}

// Rotate to the next time window. This is called on a timer (e.g. every 5 minutes). The
// current HLL slot is left as-is, head moves forward, and the new slot is reset. Old observations naturally
// fall off when the buffer wraps.
void Tile_advance(Tile *tile) {
  if (!tile) return;
  RingBuffer_advance(tile->rb);
}

//  Tile_query(tile, n) — "How many distinct things were seen here over the last N time windows?" Merges N slots
//  from the ring buffer, gets the count, frees the merged HLL, and returns the estimate.
double Tile_query(Tile *tile, const size_t n) {
  if (!tile) return 0.0;
  HLL *hll = RingBuffer_merge_window(tile->rb, n);
  if (!hll) return 0.0;
  double count = HLL_count(hll);
  freeHLL(hll);
  return count;
}
