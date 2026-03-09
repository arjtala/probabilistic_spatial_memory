#ifndef SPATIAL_H
#define SPATIAL_H

#include "core/tile.h"
#include "vendor/probabilistic_data_structures/lib/hash.h"

//  spatial_memory — the top-level engine that manages a collection of tiles.
//  It'll essentially be a hash map from H3Index → Tile*, with an API like:
//  - SpatialMemory_new(resolution, capacity, precision) — create the engine
//  with shared config
//  - SpatialMemory_observe(sm, lat, lng, data, size) — auto-creates tile if
/* //  needed, adds observation */
/* //  - SpatialMemory_advance_all(sm) — rotates all tiles to next time window */
/* //  - SpatialMemory_query(sm, lat, lng, n) — query distinct count at a location */
/* //  - SpatialMemory_free(sm) — cleanup */

#define INITIAL_TILE_CAPACITY 16
#define H3_INDEX_HEX_STRING_LENGTH 17
#define DEFAULT_RESOLUTION 10

typedef struct {
  HashTable *tiles;  // dynamic array of tile pointers
  int resolution;    // H3 resolution for all tiles
  size_t capacity;   // ring buffer capacity (shared across tiles)
  size_t precision;  // HLL precision (shared across tiles)
} SpatialMemory;

SpatialMemory *SpatialMemory_new(const int resolution, const size_t capacity,
                                 const size_t precision);
void SpatialMemory_observe(SpatialMemory *sm, const double lat, const double lng, const void *data,
                           size_t size);
void SpatialMemory_advance_all(SpatialMemory *sm);
double SpatialMemory_query(SpatialMemory *sm, const double lat, const double lng, const size_t n);
size_t SpatialMemory_tile_count(SpatialMemory *sm);
void SpatialMemory_free(SpatialMemory *sm);

#endif
