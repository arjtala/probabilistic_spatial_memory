#ifndef TILE_H
#define TILE_H

#include <stddef.h>
#include "core/ring_buffer.h"
#include <h3/h3api.h>

typedef struct {
  H3Index cellId;    // or lat/lon pair as an alternative
  RingBuffer *rb;   // time-sliced HLL ring buffer
} Tile;

bool Tile_coords_to_cell(double lat, double lng, int resolution,
                         H3Index *out_cell_id, const char *context);
Tile *Tile_new(const double lat, const double lng, const int resolution, const size_t capacity, const size_t precision);
void Tile_free(Tile *tile);
void Tile_add(Tile *tile, const void *data, size_t size);
void Tile_advance(Tile *tile);
double Tile_query(Tile *tile, const size_t n);

#endif
