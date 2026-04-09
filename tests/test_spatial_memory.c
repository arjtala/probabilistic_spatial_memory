#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "core/spatial_memory.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#define CAPACITY 3
#define PRECISION 4
#define RESOLUTION 10
#define LAT 51.380083
#define LNG -0.311041
#define CELLID_RS10 0x8a194aca6907fffULL

void test_sm_new(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  ASSERT(NULL != sm, 1, NULL != sm);
  ASSERT(RESOLUTION == sm->resolution, RESOLUTION, sm->resolution);
  ASSERT(CAPACITY == sm->capacity, CAPACITY, (int)sm->capacity);
  ASSERT(PRECISION == sm->precision, PRECISION, (int)sm->precision);
  ASSERT(NULL != sm->tiles, 1, NULL != sm->tiles);
  SpatialMemory_free(sm);
}

void test_sm_new_invalid_resolution(void) {
  SpatialMemory *sm = SpatialMemory_new(16, CAPACITY, PRECISION);
  ASSERT(NULL == sm, 1, NULL == sm);
}

void test_sm_new_invalid_precision(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY,
                                        RingBuffer_precision_max() + 1);
  ASSERT(NULL == sm, 1, NULL == sm);
}

void test_sm_observe(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  const char *pb = "peanut butter";
  bool ok = SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);

  LatLng loc;
  loc.lat = degsToRads(LAT);
  loc.lng = degsToRads(LNG);
  H3Index cellId;
  H3Error err = latLngToCell(&loc, RESOLUTION, &cellId);
  if (err) {
    fprintf(stderr, "Failed to create H3Index : invalid lat/lng or resolution\n");
    exit(EXIT_FAILURE);
  }
  Tile *tile = TileTable_get(sm->tiles, cellId);
  ASSERT(NULL != tile, 1, NULL != tile);
  ASSERT(1 == TileTable_size(sm->tiles), 1, 1 == TileTable_size(sm->tiles));
  SpatialMemory_free(sm);
}

void test_sm_query(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  const char *pb = "peanut butter";
  double query_count = 0.0;
  bool ok = SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 1, &query_count);
  ASSERT(ok, 1, ok);
  int count = (int)query_count;
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

  bool ok = SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  SpatialMemory_advance_all(sm);
  Tile *tile = TileTable_get(sm->tiles, cellId);
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
  double count_lon = 0.0;
  double count_tok = 0.0;
  bool ok = SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_observe(sm, tokyo_lat, tokyo_lng, sushi, strlen(sushi));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 0, &count_lon);
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, tokyo_lat, tokyo_lng, 0, &count_tok);
  ASSERT(ok, 1, ok);
  int num_tiles = (int)TileTable_size(sm->tiles);
  ASSERT(count_tok >= 1, 1, (int)count_tok);
  ASSERT(count_lon >= 1, 1, (int)count_lon);
  ASSERT(2 == num_tiles, 2, num_tiles);
  SpatialMemory_free(sm);
}

void test_sm_invalid_observe_does_not_crash(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  const char *pb = "peanut butter";
  bool ok = SpatialMemory_observe(sm, 100.0, LNG, pb, strlen(pb));
  ASSERT(!ok, 0, ok);
  ASSERT(0 == (int)SpatialMemory_tile_count(sm), 0, (int)SpatialMemory_tile_count(sm));
  SpatialMemory_free(sm);
}

void test_sm_invalid_query_returns_false(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  double count = 123.0;
  bool ok = SpatialMemory_query(sm, 100.0, LNG, 0, &count);
  ASSERT(!ok, 0, ok);
  ASSERT(0 == (int)count, 0, (int)count);
  SpatialMemory_free(sm);
}

void test_sm_same_cell_deduplicates_observations(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION);
  const char *pb = "peanut butter";
  double first_count = 0.0;
  double second_count = 0.0;

  bool ok = SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 0, &first_count);
  ASSERT(ok, 1, ok);

  ok = SpatialMemory_observe(sm, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 0, &second_count);
  ASSERT(ok, 1, ok);

  ASSERT(1 == (int)SpatialMemory_tile_count(sm), 1, (int)SpatialMemory_tile_count(sm));
  ASSERT((int)first_count == (int)second_count, (int)first_count, (int)second_count);
  SpatialMemory_free(sm);
}

int main(void) {
  RUN_TEST(test_sm_new);
  RUN_TEST(test_sm_new_invalid_resolution);
  RUN_TEST(test_sm_new_invalid_precision);
  RUN_TEST(test_sm_observe);
  RUN_TEST(test_sm_query);
  RUN_TEST(test_sm_advance_all);
  RUN_TEST(test_sm_multi_tile);
  RUN_TEST(test_sm_invalid_observe_does_not_crash);
  RUN_TEST(test_sm_invalid_query_returns_false);
  RUN_TEST(test_sm_same_cell_deduplicates_observations);

  return 0;
}
