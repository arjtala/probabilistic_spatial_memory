#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "core/tile.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#define CAPACITY 3
#define PRECISION 4
#define RESOLUTION 10
#define LAT 51.380083
#define LNG -0.311041
#define CELLID_RS10 0x8a194aca6907fffULL

void test_tile_new(void) {
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION);
  ASSERT(NULL != tile, 1, NULL != tile);
  ASSERT(CELLID_RS10 == tile->cellId, 1, CELLID_RS10 == tile->cellId);
  ASSERT(CAPACITY == tile->rb->capacity, CAPACITY, (int)tile->rb->capacity);
  ASSERT(PRECISION == tile->rb->precision, PRECISION, (int)tile->rb->precision);
  ASSERT(0 == tile->rb->head, 0, (int)tile->rb->head);
  Tile_free(tile);
}

void test_tile_new_invalid_input(void) {
  Tile *bad_resolution = Tile_new(LAT, LNG, 16, CAPACITY, PRECISION);
  Tile *bad_lat = Tile_new(100.0, LNG, RESOLUTION, CAPACITY, PRECISION);
  Tile *bad_precision = Tile_new(LAT, LNG, RESOLUTION, CAPACITY,
                                 RingBuffer_precision_min() - 1);
  ASSERT(NULL == bad_resolution, 1, NULL == bad_resolution);
  ASSERT(NULL == bad_lat, 1, NULL == bad_lat);
  ASSERT(NULL == bad_precision, 1, NULL == bad_precision);
}

void test_tile_coords_to_cell(void) {
  H3Index cell_id = 0;
  bool ok = Tile_coords_to_cell(LAT, LNG, RESOLUTION, &cell_id, "test");
  ASSERT(ok, 1, ok);
  ASSERT(cell_id == CELLID_RS10, 1, cell_id == CELLID_RS10);
  ok = Tile_coords_to_cell(100.0, LNG, RESOLUTION, &cell_id, "test");
  ASSERT(!ok, 0, ok);
}

void test_tile_add(void) {
  const char *pb = "peanut butter";
  Tile *tile = Tile_new(0.0, 0.0, RESOLUTION, CAPACITY, PRECISION);
  Tile_add(tile, pb, strlen(pb));
  RingBufferHLL *current = RingBuffer_current(tile->rb);
  int count = (int)RingBufferHLL_count(current);
  ASSERT(count >= 1, count, 1);
  RingBufferHLL_release(current);
  Tile_free(tile);
}

void test_tile_advance(void) {
  const char *pb = "peanut butter";
  Tile *tile = Tile_new(0.0, 0.0, RESOLUTION, CAPACITY, PRECISION);
  Tile_add(tile, pb, strlen(pb));
  Tile_advance(tile);
  RingBufferHLL *current = RingBuffer_current(tile->rb);
  int count = (int)RingBufferHLL_count(current);
  ASSERT(0 == count, 0, count);
  RingBufferHLL_release(current);
  Tile_free(tile);
}

void test_tile_query(void) {
  const char *pb = "peanut butter";
  const char *j = "jelly";
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION);
  Tile_add(tile, pb, strlen(pb));
  Tile_advance(tile);
  Tile_add(tile, j, strlen(j));
  int count = (int)Tile_query(tile, 1);
  ASSERT(2 <= count, 2, count);
  Tile_free(tile);
}

void test_tile_same_cell(void) {
  const double lat2 = 51.379778;
  const double lng2 = -0.311022;
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION);
  Tile *tile2 = Tile_new(lat2, lng2, RESOLUTION, CAPACITY, PRECISION);
  ASSERT(tile2->cellId == tile->cellId, 1, tile2->cellId == tile->cellId);
  Tile_free(tile);
  Tile_free(tile2);
}

int main(void) {
  RUN_TEST(test_tile_new);
  RUN_TEST(test_tile_new_invalid_input);
  RUN_TEST(test_tile_coords_to_cell);
  RUN_TEST(test_tile_add);
  RUN_TEST(test_tile_advance);
  RUN_TEST(test_tile_query);
  RUN_TEST(test_tile_same_cell);

  return 0;
}
