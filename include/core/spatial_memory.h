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
  TileTable *tiles;  // dynamic map of tile pointers keyed by H3Index
  int resolution;    // H3 resolution for all tiles
  size_t capacity;   // ring buffer capacity (shared across tiles)
  size_t precision;  // HLL precision (shared across tiles)
} SpatialMemory;

typedef bool (*SpatialMemoryTileVisitor)(H3Index cell_id, Tile *tile,
                                         void *user_data);

SpatialMemory *SpatialMemory_new(const int resolution, const size_t capacity,
                                 const size_t precision);
bool SpatialMemory_observe(SpatialMemory *sm, const double lat, const double lng,
                           const void *data, size_t size);
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
void SpatialMemory_free(SpatialMemory *sm);

#endif
