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

static int compare_similar_desc(const void *a, const void *b) {
  const SpatialMemorySimilar *sa = (const SpatialMemorySimilar *)a;
  const SpatialMemorySimilar *sb = (const SpatialMemorySimilar *)b;
  if (sa->similarity > sb->similarity) return -1;
  if (sa->similarity < sb->similarity) return 1;
  if (sa->t_max > sb->t_max) return -1;
  if (sa->t_max < sb->t_max) return 1;
  return 0;
}

// Score a single tile by finding its highest-similarity exemplar against the
// query. Populates *out and returns true on success, or false when the tile
// has no usable exemplars (wrong byte size, zero-norm, or empty reservoir).
static bool score_tile_similar(Tile *tile, const float *query, size_t dim,
                               double q_norm, size_t rb_capacity,
                               SpatialMemorySimilar *out) {
  size_t n_ex = Tile_exemplar_count(tile);
  if (n_ex == 0) return false;

  const size_t want_bytes = dim * sizeof(float);
  double best_sim = -2.0;  // any valid cosine in [-1, 1] beats this sentinel.
  double best_t = 0.0;
  bool found = false;

  for (size_t i = 0; i < n_ex; ++i) {
    const TileExemplar *ex = Tile_exemplar_at(tile, i);
    if (!ex || !ex->data || ex->size != want_bytes) continue;
    const float *v = (const float *)ex->data;
    double dot = 0.0, v_norm_sq = 0.0;
    for (size_t d = 0; d < dim; ++d) {
      double qi = (double)query[d];
      double vi = (double)v[d];
      dot += qi * vi;
      v_norm_sq += vi * vi;
    }
    if (v_norm_sq <= 0.0) continue;
    double sim = dot / (q_norm * sqrt(v_norm_sq));
    if (sim > best_sim) {
      best_sim = sim;
      best_t = ex->t;
      found = true;
    }
  }
  if (!found) return false;

  double t_min = best_t;
  double t_max = best_t;
  double count = 0.0;
  RingBufferWindow win = RingBuffer_merge_window(tile->rb, rb_capacity - 1);
  if (win.sketch) {
    if (!win.is_empty) {
      t_min = win.t_min;
      t_max = win.t_max;
      count = RingBufferHLL_count(win.sketch);
    }
    RingBufferHLL_release(win.sketch);
  }

  out->cell = tile->cellId;
  out->similarity = best_sim;
  out->exemplar_t = best_t;
  out->t_min = t_min;
  out->t_max = t_max;
  out->count = count;
  return true;
}

size_t SpatialMemory_query_similar(SpatialMemory *sm, const float *query,
                                   size_t dim, double center_lat,
                                   double center_lng, int k_ring,
                                   SpatialMemorySimilar *out, size_t max_out) {
  if (!sm || !query || dim == 0) return 0;

  double q_norm_sq = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    double qi = (double)query[i];
    q_norm_sq += qi * qi;
  }
  if (q_norm_sq <= 0.0) return 0;
  double q_norm = sqrt(q_norm_sq);

  H3Index *cells = NULL;
  int64_t n_cells = 0;
  const bool use_neighborhood = (k_ring >= 0);
  if (use_neighborhood) {
    H3Index center;
    if (!spatial_memory_coords_to_cell(sm, center_lat, center_lng, &center)) {
      return 0;
    }
    if (maxGridDiskSize(k_ring, &n_cells) || n_cells <= 0) {
      return 0;
    }
    cells = (H3Index *)calloc((size_t)n_cells, sizeof(H3Index));
    if (!cells) return 0;
    if (gridDisk(center, k_ring, cells)) {
      free(cells);
      return 0;
    }
  }

  size_t max_scratch;
  if (use_neighborhood) {
    max_scratch = (size_t)n_cells;
  } else {
    max_scratch = SpatialMemory_tile_count(sm);
  }
  if (max_scratch == 0) {
    free(cells);
    return 0;
  }

  SpatialMemorySimilar *scratch = (SpatialMemorySimilar *)malloc(
      max_scratch * sizeof(SpatialMemorySimilar));
  if (!scratch) {
    free(cells);
    return 0;
  }

  size_t nfound = 0;
  if (use_neighborhood) {
    for (int64_t i = 0; i < n_cells; ++i) {
      H3Index cell = cells[i];
      if (cell == H3_NULL) continue;
      Tile *tile = TileTable_get(sm->tiles, cell);
      if (!tile) continue;
      if (score_tile_similar(tile, query, dim, q_norm, sm->capacity,
                             &scratch[nfound])) {
        nfound++;
      }
    }
  } else {
    TileTableIterator it = TileTable_iterator(sm->tiles);
    while (TileTable_next(&it)) {
      if (score_tile_similar(it.value, query, dim, q_norm, sm->capacity,
                             &scratch[nfound])) {
        nfound++;
      }
    }
  }
  free(cells);

  qsort(scratch, nfound, sizeof(SpatialMemorySimilar), compare_similar_desc);

  if (out && max_out > 0) {
    size_t to_copy = (nfound < max_out) ? nfound : max_out;
    memcpy(out, scratch, to_copy * sizeof(SpatialMemorySimilar));
  }
  free(scratch);
  return nfound;
}

void SpatialMemory_free(SpatialMemory *sm) {
  if (!sm) return;
  TileTable_free(sm->tiles);
  free(sm);
}
