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

// Result row for SpatialMemory_query_similar. Reports the single winning
// exemplar per tile (the one with the highest cosine similarity to the query)
// together with the containing cell's merged-window interval for downstream
// grounding.
typedef struct {
  H3Index cell;
  double similarity;  // cosine similarity in [-1, 1] for the winning exemplar.
  double exemplar_t;  // timestamp of the winning exemplar.
  double t_min;       // tile's merged-window [t_min, t_max] (exemplar_t if
  double t_max;       //  the ring buffer has aged out).
  double count;       // HLL cardinality over the merged window, or 0 if aged out.
} SpatialMemorySimilar;

// Rank tiles by the best cosine similarity of any stored exemplar to the query
// vector. Query is a flat float32 vector of `dim` elements; exemplars whose
// stored byte size does not match `dim * sizeof(float)` are skipped.
//
// k_ring < 0 searches every tile in the SpatialMemory (center_lat / center_lng
// are ignored). k_ring >= 0 restricts the scan to the H3 k-ring neighborhood
// around (center_lat, center_lng).
//
// Results are sorted by similarity descending; ties broken by t_max descending.
// Returns the total number of matched tiles; probe mode (out == NULL or
// max_out == 0) scans without writing.
size_t SpatialMemory_query_similar(SpatialMemory *sm, const float *query,
                                   size_t dim, double center_lat,
                                   double center_lng, int k_ring,
                                   SpatialMemorySimilar *out, size_t max_out);

void SpatialMemory_free(SpatialMemory *sm);

#endif
