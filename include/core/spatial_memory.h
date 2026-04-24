#ifndef SPATIAL_MEMORY_H
#define SPATIAL_MEMORY_H

#include <stdbool.h>
#include "core/tile_table.h"

// SpatialMemory maps H3 cells to per-tile ring buffers of HLL counters.
// Observations are bucketed by location and queried approximately over recent
// time windows without unbounded memory growth.

#define H3_INDEX_HEX_STRING_LENGTH 17
#define DEFAULT_RESOLUTION 10

typedef struct {
  TileTable *tiles;          // dynamic map of tile pointers keyed by H3Index
  int resolution;            // H3 resolution for all tiles
  size_t capacity;           // ring buffer capacity (shared across tiles)
  size_t precision;          // HLL precision (shared across tiles)
  size_t exemplar_capacity;  // per-tile reservoir capacity (0 = disabled)
} SpatialMemory;

typedef bool (*SpatialMemoryTileVisitor)(H3Index cell_id, Tile *tile,
                                         void *user_data);

// Construct a new SpatialMemory. Pass 0 for exemplar_capacity to disable
// per-tile reservoir sampling (no allocation, no sampling work during
// observe). Larger values create a fixed-size reservoir per tile.
SpatialMemory *SpatialMemory_new(const int resolution, const size_t capacity,
                                 const size_t precision,
                                 const size_t exemplar_capacity);
// Record an observation at timestamp t. Widens the per-slot [t_min, t_max]
// interval and feeds the per-tile exemplar reservoir (when configured).
bool SpatialMemory_observe(SpatialMemory *sm, double t, const double lat,
                           const double lng, const void *data, size_t size);
size_t SpatialMemory_advance_to_timestamp(SpatialMemory *sm, double timestamp,
                                          double *window_anchor,
                                          double time_window_sec);
void SpatialMemory_advance_all(SpatialMemory *sm);
bool SpatialMemory_query(SpatialMemory *sm, const double lat, const double lng,
                         const size_t n, double *out_count);
size_t SpatialMemory_tile_count(SpatialMemory *sm);
bool SpatialMemory_for_each_tile(SpatialMemory *sm,
                                 SpatialMemoryTileVisitor visitor,
                                 void *user_data);

// Result row for SpatialMemory_query_intervals.
typedef struct {
  H3Index cell;
  double t_min;
  double t_max;
  double count;  // HLL cardinality estimate over the full ring-buffer window.
} SpatialMemoryInterval;

// Query the H3 k-ring neighborhood at (lat, lng) for non-empty tiles and
// return their merged-window [t_min, t_max] intervals along with an
// observation-count estimate. Results are sorted by t_max descending (most
// recent first); ties broken by count descending.
//
// Writes up to max_out entries to out. Returns the total number of non-empty
// tiles found in the neighborhood — this may exceed the number actually
// written, and callers can compare against max_out to detect truncation.
// Probe mode (out == NULL or max_out == 0) scans without writing.
size_t SpatialMemory_query_intervals(SpatialMemory *sm, double lat, double lng,
                                     int k_ring, SpatialMemoryInterval *out,
                                     size_t max_out);

void SpatialMemory_free(SpatialMemory *sm);

#endif
