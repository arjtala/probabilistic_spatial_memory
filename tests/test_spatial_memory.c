#include "hash.h"
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "spatial_memory.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#define CAPACITY 3
#define PRECISION 4
#define RESOLUTION 10
#define LAT 51.380083
#define LNG -0.311041
#define CELLID_RS10 0x8a194aca6907fffULL

void test_sm_new(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  ASSERT(RESOLUTION == sm->resolution, RESOLUTION, sm->resolution);
  ASSERT(CAPACITY == sm->capacity, CAPACITY, (int)sm->capacity);
  ASSERT(PRECISION == sm->precision, PRECISION, (int)sm->precision);
  ASSERT(NULL != sm->tiles, 1, NULL != sm->tiles);
  SpatialMemory_free(sm);
}

void test_sm_observe(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  const char *pb = "peanut butter";
  SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));

  LatLng loc;
  loc.lat = degsToRads(LAT);
  loc.lng = degsToRads(LNG);
  H3Index cellId;
  H3Error err = latLngToCell(&loc, RESOLUTION, &cellId);
  if (err) {
    fprintf(stderr, "Failed to create H3Index : invalid lat/lng or resolution\n");
    exit(EXIT_FAILURE);
  }
  char hexString[H3_INDEX_HEX_STRING_LENGTH];
  h3ToString(cellId, hexString, sizeof(hexString));
  Tile *tile = HashTable_get(sm->tiles, hexString);
  ASSERT(NULL != tile, 1, NULL != tile);
  ASSERT(1 == HashTable_size(sm->tiles), 1, 1 == HashTable_size(sm->tiles));
  SpatialMemory_free(sm);
}

void test_sm_query(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  const char *pb = "peanut butter";
  SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  int count = (int)SpatialMemory_query(sm, LAT, LNG, 1);
  ASSERT(count >= 1, 1, count);
  SpatialMemory_free(sm);
}

void test_sm_advance_all(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  const char *pb = "peanut butter";
  LatLng loc;
  loc.lat = degsToRads(LAT);
  loc.lng = degsToRads(LNG);
  H3Index cellId;
  H3Error err = latLngToCell(&loc, RESOLUTION, &cellId);
  if (err) {
    fprintf(stderr, "Failed to create H3Index : invalid lat/lng or resolution\n");
    exit(EXIT_FAILURE);
  }
  char hexString[H3_INDEX_HEX_STRING_LENGTH];
  h3ToString(cellId, hexString, sizeof(hexString));

  SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  SpatialMemory_advance_all(sm);
  Tile *tile = HashTable_get(sm->tiles, hexString);
  int count = (int)Tile_query(tile, 0);
  ASSERT(0 == count, 0, count);
  SpatialMemory_free(sm);
}

void test_sm_multi_tile(void) {
  const char *pb = "peanut butter";
  const char *sushi = "sushi";
  const double tokyo_lat = 35.68;
  const double tokyo_lng = 139.68;
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  SpatialMemory_observe(sm, tokyo_lat, tokyo_lng, sushi, strlen(sushi));
  int count_lon = (int)SpatialMemory_query(sm, LAT, LNG, 0);
  int count_tok = (int)SpatialMemory_query(sm, tokyo_lat, tokyo_lng, 0);
  int num_hash_tables = (int)HashTable_size(sm->tiles);
  ASSERT(count_tok >= 1, count_tok, 1);
  ASSERT(count_lon >= 1, count_lon, 1);
  ASSERT(2 == num_hash_tables, 2, num_hash_tables);
  SpatialMemory_free(sm);
}

int main(void) {
  RUN_TEST(test_sm_new);
  RUN_TEST(test_sm_observe);
  RUN_TEST(test_sm_query);
  RUN_TEST(test_sm_advance_all);
  RUN_TEST(test_sm_multi_tile);

  return 0;
}
