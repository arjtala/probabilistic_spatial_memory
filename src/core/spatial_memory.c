#include <stdio.h>
#include <stdlib.h>
#include "core/spatial_memory.h"

SpatialMemory *SpatialMemory_new(const int resolution, const size_t capacity,
                                 const size_t precision) {
  SpatialMemory *sm = (SpatialMemory *)malloc(sizeof(SpatialMemory));
  if (NULL == sm) {
    fprintf(stderr, "Unable to initialize SpatialMemory: out of memory.\n");
    exit(EXIT_FAILURE);
  }
  sm->tiles = HashTable_create((void (*)(void *))Tile_free);
  if (NULL == sm->tiles) {
    fprintf(stderr, "Unable to initialize tiles: out of memory.\n");
    exit(EXIT_FAILURE);
  }
  sm->resolution = resolution;
  sm->capacity = capacity;
  sm->precision = precision;
  return sm;
}

void SpatialMemory_observe(SpatialMemory *sm, const double lat, const double lng, const void *data,
                           size_t size) {
  LatLng loc;
  loc.lat = degsToRads(lat);
  loc.lng = degsToRads(lng);
  H3Index cellId;
  H3Error err = latLngToCell(&loc, sm->resolution, &cellId);
  if (err) {
    fprintf(stderr, "Failed to create H3Index : invalid lat/lng or resolution\n");
    exit(EXIT_FAILURE);
  }
  char hexString[H3_INDEX_HEX_STRING_LENGTH];
  h3ToString(cellId, hexString, sizeof(hexString));
  Tile *tile = HashTable_get(sm->tiles, hexString);
  if (NULL == tile) {
    tile = Tile_new(lat, lng, sm->resolution, sm->capacity, sm->precision);
    HashTable_set(sm->tiles, hexString, tile);
  }
  Tile_add(tile, data, size);
}

void SpatialMemory_advance_all(SpatialMemory *sm) {
  HashTableIterator it = HashTable_iterator(sm->tiles);
  while (HashTable_next(&it)) {
    Tile_advance(it.value);
  }
}

double SpatialMemory_query(SpatialMemory *sm, const double lat, const double lng, const size_t n) {
  double count = 0.0;
  LatLng loc;
  loc.lat = degsToRads(lat);
  loc.lng = degsToRads(lng);
  H3Index cellId;
  H3Error err = latLngToCell(&loc, sm->resolution, &cellId);
  if (err) {
    fprintf(stderr, "Failed to create H3Index : invalid lat/lng or resolution\n");
    exit(EXIT_FAILURE);
  }
  char hexString[H3_INDEX_HEX_STRING_LENGTH];
  h3ToString(cellId, hexString, sizeof(hexString));
  Tile *tile = HashTable_get(sm->tiles, hexString);
  if (NULL == tile) {
    return count;
  }
  count = Tile_query(tile, n);
  return count;
}

size_t SpatialMemory_tile_count(SpatialMemory *sm) {
  return HashTable_size(sm->tiles);
}

void SpatialMemory_free(SpatialMemory *sm) {
  HashTable_free(sm->tiles);
  free(sm);
}
