#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <h3/h3api.h>
#include "core/spatial_memory.h"

static bool spatial_memory_coords_to_cell(const SpatialMemory *sm, double lat,
                                          double lng, H3Index *out_cell_id) {
  if (!sm || !out_cell_id) return false;
  return Tile_coords_to_cell(lat, lng, sm->resolution, out_cell_id,
                             "SpatialMemory");
}

SpatialMemory *SpatialMemory_new(const int resolution, const size_t capacity,
                                 const size_t precision,
                                 const size_t exemplar_capacity) {
  if (resolution < 0 || resolution > 15) {
    fprintf(stderr, "SpatialMemory_new: H3 resolution must be in [0, 15], got %d\n",
            resolution);
    return NULL;
  }
  if (capacity == 0) {
    fprintf(stderr, "SpatialMemory_new: capacity must be greater than 0\n");
    return NULL;
  }
  if (!RingBuffer_precision_is_valid(precision)) {
    fprintf(stderr,
            "SpatialMemory_new: precision %zu is out of range [%zu, %zu]\n",
            precision, RingBuffer_precision_min(), RingBuffer_precision_max());
    return NULL;
  }

  SpatialMemory *sm = (SpatialMemory *)malloc(sizeof(SpatialMemory));
  if (NULL == sm) {
    fprintf(stderr, "Unable to initialize SpatialMemory: out of memory.\n");
    return NULL;
  }
  sm->tiles = TileTable_create();
  if (NULL == sm->tiles) {
    fprintf(stderr, "Unable to initialize tiles: out of memory.\n");
    free(sm);
    return NULL;
  }
  sm->resolution = resolution;
  sm->capacity = capacity;
  sm->precision = precision;
  sm->exemplar_capacity = exemplar_capacity;
  return sm;
}

bool SpatialMemory_observe(SpatialMemory *sm, double t, const double lat,
                           const double lng, const void *data, size_t size) {
  if (!sm || !data || size == 0) {
    return false;
  }
  H3Index cell_id;
  if (!spatial_memory_coords_to_cell(sm, lat, lng, &cell_id)) {
    return false;
  }
  Tile *tile = TileTable_get(sm->tiles, cell_id);
  if (NULL == tile) {
    tile = Tile_new(lat, lng, sm->resolution, sm->capacity, sm->precision,
                    sm->exemplar_capacity);
    if (!tile) {
      return false;
    }
    if (!TileTable_set(sm->tiles, cell_id, tile)) {
      Tile_free(tile);
      return false;
    }
  }
  Tile_observe(tile, t, data, size);
  return true;
}

size_t SpatialMemory_advance_to_timestamp(SpatialMemory *sm, double timestamp,
                                          double *window_anchor,
                                          double time_window_sec) {
  size_t advances = 0;

  if (!sm || !window_anchor || time_window_sec <= 0.0) {
    return 0;
  }
  if (*window_anchor < 0.0) {
    *window_anchor = timestamp;
    return 0;
  }

  while (timestamp - *window_anchor >= time_window_sec) {
    SpatialMemory_advance_all(sm);
    *window_anchor += time_window_sec;
    advances++;
  }
  return advances;
}

void SpatialMemory_advance_all(SpatialMemory *sm) {
  if (!sm) return;
  TileTableIterator it = TileTable_iterator(sm->tiles);
  while (TileTable_next(&it)) {
    Tile_advance(it.value);
  }
}

bool SpatialMemory_query(SpatialMemory *sm, const double lat, const double lng,
                         const size_t n, double *out_count) {
  if (!out_count) {
    return false;
  }
  *out_count = 0.0;
  if (!sm) return false;

  H3Index cell_id;
  if (!spatial_memory_coords_to_cell(sm, lat, lng, &cell_id)) {
    return false;
  }
  Tile *tile = TileTable_get(sm->tiles, cell_id);
  if (NULL == tile) {
    return true;
  }
  *out_count = Tile_query(tile, n);
  return true;
}

size_t SpatialMemory_tile_count(SpatialMemory *sm) {
  if (!sm) return 0;
  return TileTable_size(sm->tiles);
}

bool SpatialMemory_for_each_tile(SpatialMemory *sm,
                                 SpatialMemoryTileVisitor visitor,
                                 void *user_data) {
  if (!sm || !visitor) return false;

  TileTableIterator it = TileTable_iterator(sm->tiles);
  while (TileTable_next(&it)) {
    if (!visitor(it.key, it.value, user_data)) {
      return false;
    }
  }
  return true;
}

static int compare_intervals_desc(const void *a, const void *b) {
  const SpatialMemoryInterval *ia = (const SpatialMemoryInterval *)a;
  const SpatialMemoryInterval *ib = (const SpatialMemoryInterval *)b;
  if (ia->t_max > ib->t_max) return -1;
  if (ia->t_max < ib->t_max) return 1;
  if (ia->count > ib->count) return -1;
  if (ia->count < ib->count) return 1;
  return 0;
}

size_t SpatialMemory_query_intervals(SpatialMemory *sm, double lat, double lng,
                                     int k_ring, SpatialMemoryInterval *out,
                                     size_t max_out) {
  if (!sm || k_ring < 0) return 0;

  H3Index center;
  if (!spatial_memory_coords_to_cell(sm, lat, lng, &center)) {
    return 0;
  }

  int64_t max_cells = 0;
  if (maxGridDiskSize(k_ring, &max_cells) || max_cells <= 0) {
    return 0;
  }

  H3Index *cells = (H3Index *)calloc((size_t)max_cells, sizeof(H3Index));
  if (!cells) return 0;

  if (gridDisk(center, k_ring, cells)) {
    free(cells);
    return 0;
  }

  SpatialMemoryInterval *scratch = (SpatialMemoryInterval *)malloc(
      (size_t)max_cells * sizeof(SpatialMemoryInterval));
  if (!scratch) {
    free(cells);
    return 0;
  }

  size_t nfound = 0;
  for (int64_t i = 0; i < max_cells; ++i) {
    H3Index cell = cells[i];
    if (cell == H3_NULL) continue;

    Tile *tile = TileTable_get(sm->tiles, cell);
    if (!tile) continue;

    // Merge across the entire ring buffer (head plus capacity-1 prior slots).
    RingBufferWindow window =
        RingBuffer_merge_window(tile->rb, sm->capacity - 1);
    if (!window.sketch) continue;  // OOM: skip this cell, keep scanning.
    if (window.is_empty) {
      RingBufferHLL_release(window.sketch);
      continue;
    }

    scratch[nfound].cell = cell;
    scratch[nfound].t_min = window.t_min;
    scratch[nfound].t_max = window.t_max;
    scratch[nfound].count = RingBufferHLL_count(window.sketch);
    nfound++;
    RingBufferHLL_release(window.sketch);
  }

  free(cells);

  qsort(scratch, nfound, sizeof(SpatialMemoryInterval), compare_intervals_desc);

  if (out && max_out > 0) {
    size_t to_copy = (nfound < max_out) ? nfound : max_out;
    memcpy(out, scratch, to_copy * sizeof(SpatialMemoryInterval));
  }

  free(scratch);
  return nfound;
}

void SpatialMemory_free(SpatialMemory *sm) {
  if (!sm) return;
  TileTable_free(sm->tiles);
  free(sm);
}
