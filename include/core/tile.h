#ifndef TILE_H
#define TILE_H

#include <stdint.h>
#include <stddef.h>
#include "core/ring_buffer.h"
#include <h3/h3api.h>

// Per-tile reservoir-sampled exemplar. Owns a malloc'd copy of the embedding
// bytes so the caller's buffer can be freed after Tile_observe returns.
typedef struct {
  double t;
  void *data;
  size_t size;
} TileExemplar;

typedef struct {
  H3Index cellId;
  RingBuffer *rb;   // time-sliced HLL ring buffer

  // Reservoir-sampled exemplars (Algorithm R). Bounded storage per tile —
  // capacity is set at tile creation and fixed for the life of the tile.
  // When capacity is 0 exemplars is NULL and sampling is fully disabled
  // (zero overhead beyond a couple of branches).
  TileExemplar *exemplars;
  size_t exemplar_capacity;
  size_t exemplar_count;
  uint64_t exemplar_seen;
} Tile;

bool Tile_coords_to_cell(double lat, double lng, int resolution,
                         H3Index *out_cell_id, const char *context);
Tile *Tile_new(const double lat, const double lng, const int resolution,
               const size_t capacity, const size_t precision,
               const size_t exemplar_capacity);
void Tile_free(Tile *tile);
// Record an observation at timestamp t. Updates the current ring-buffer slot's
// HLL + [t_min, t_max] interval and feeds the reservoir sampler. The data
// buffer may be freed by the caller after this call returns (the reservoir
// copies the bytes).
void Tile_observe(Tile *tile, double t, const void *data, size_t size);
void Tile_advance(Tile *tile);
double Tile_query(Tile *tile, const size_t n);

// Read-only accessors for sampled exemplars. Returns 0/NULL when the tile has
// no exemplar storage configured or the index is out of range.
size_t Tile_exemplar_count(const Tile *tile);
const TileExemplar *Tile_exemplar_at(const Tile *tile, size_t idx);

#endif
