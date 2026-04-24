#define _POSIX_C_SOURCE 200809L
#include <float.h>
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
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, 0);
  ASSERT(NULL != tile, 1, NULL != tile);
  ASSERT(CELLID_RS10 == tile->cellId, 1, CELLID_RS10 == tile->cellId);
  ASSERT(CAPACITY == tile->rb->capacity, CAPACITY, (int)tile->rb->capacity);
  ASSERT(PRECISION == tile->rb->precision, PRECISION, (int)tile->rb->precision);
  ASSERT(0 == tile->rb->head, 0, (int)tile->rb->head);
  ASSERT(0 == (int)tile->exemplar_capacity, 0, (int)tile->exemplar_capacity);
  ASSERT(0 == (int)tile->exemplar_count, 0, (int)tile->exemplar_count);
  ASSERT(NULL == tile->exemplars, 1, NULL == tile->exemplars);
  Tile_free(tile);
}

void test_tile_new_invalid_input(void) {
  Tile *bad_resolution = Tile_new(LAT, LNG, 16, CAPACITY, PRECISION, 0);
  Tile *bad_lat = Tile_new(100.0, LNG, RESOLUTION, CAPACITY, PRECISION, 0);
  Tile *bad_precision = Tile_new(LAT, LNG, RESOLUTION, CAPACITY,
                                 RingBuffer_precision_min() - 1, 0);
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

void test_tile_observe(void) {
  const char *pb = "peanut butter";
  Tile *tile = Tile_new(0.0, 0.0, RESOLUTION, CAPACITY, PRECISION, 0);
  Tile_observe(tile, 0.0, pb, strlen(pb));
  RingBufferHLL *current = RingBuffer_current(tile->rb);
  int count = (int)RingBufferHLL_count(current);
  ASSERT(count >= 1, count, 1);
  RingBufferHLL_release(current);
  Tile_free(tile);
}

void test_tile_advance(void) {
  const char *pb = "peanut butter";
  Tile *tile = Tile_new(0.0, 0.0, RESOLUTION, CAPACITY, PRECISION, 0);
  Tile_observe(tile, 0.0, pb, strlen(pb));
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
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, 0);
  Tile_observe(tile, 0.0, pb, strlen(pb));
  Tile_advance(tile);
  Tile_observe(tile, 1.0, j, strlen(j));
  int count = (int)Tile_query(tile, 1);
  ASSERT(2 <= count, 2, count);
  Tile_free(tile);
}

void test_tile_same_cell(void) {
  const double lat2 = 51.379778;
  const double lng2 = -0.311022;
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, 0);
  Tile *tile2 = Tile_new(lat2, lng2, RESOLUTION, CAPACITY, PRECISION, 0);
  ASSERT(tile2->cellId == tile->cellId, 1, tile2->cellId == tile->cellId);
  Tile_free(tile);
  Tile_free(tile2);
}

void test_tile_exemplars_disabled(void) {
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  for (int i = 0; i < 100; ++i) {
    Tile_observe(tile, (double)i, pb, strlen(pb));
  }
  ASSERT(0 == (int)Tile_exemplar_count(tile), 0, (int)Tile_exemplar_count(tile));
  ASSERT(NULL == Tile_exemplar_at(tile, 0), 1, NULL == Tile_exemplar_at(tile, 0));
  Tile_free(tile);
}

void test_tile_exemplars_under_capacity_retains_all(void) {
  const size_t cap = 4;
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, cap);
  ASSERT(NULL != tile, 1, NULL != tile);
  ASSERT(NULL != tile->exemplars, 1, NULL != tile->exemplars);

  const char *items[] = {"a", "bb", "ccc"};
  size_t n = sizeof(items) / sizeof(items[0]);
  for (size_t i = 0; i < n; ++i) {
    Tile_observe(tile, (double)i, items[i], strlen(items[i]));
  }
  ASSERT(n == Tile_exemplar_count(tile), (int)n, (int)Tile_exemplar_count(tile));
  for (size_t i = 0; i < n; ++i) {
    const TileExemplar *ex = Tile_exemplar_at(tile, i);
    ASSERT(NULL != ex, 1, NULL != ex);
    ASSERT((double)i == ex->t, (int)i, (int)ex->t);
    ASSERT(strlen(items[i]) == ex->size, (int)strlen(items[i]), (int)ex->size);
    ASSERT(0 == memcmp(ex->data, items[i], ex->size), 1,
           0 == memcmp(ex->data, items[i], ex->size));
  }
  Tile_free(tile);
}

void test_tile_exemplars_over_capacity_bounds_count(void) {
  const size_t cap = 5;
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, cap);
  ASSERT(NULL != tile, 1, NULL != tile);

  char buf[16];
  const size_t total = 200;
  for (size_t i = 0; i < total; ++i) {
    snprintf(buf, sizeof(buf), "obs-%zu", i);
    Tile_observe(tile, (double)i, buf, strlen(buf));
  }
  ASSERT(cap == Tile_exemplar_count(tile), (int)cap,
         (int)Tile_exemplar_count(tile));
  ASSERT(total == tile->exemplar_seen, (int)total, (int)tile->exemplar_seen);

  for (size_t i = 0; i < cap; ++i) {
    const TileExemplar *ex = Tile_exemplar_at(tile, i);
    ASSERT(NULL != ex, 1, NULL != ex);
    ASSERT(ex->size > 0, 1, ex->size > 0);
    ASSERT(NULL != ex->data, 1, NULL != ex->data);
  }
  Tile_free(tile);
}

void test_tile_exemplars_statistical_sanity(void) {
  // Over T observations into a capacity-K reservoir, the first-observation
  // retention probability is K/T. Repeat many trials and check the empirical
  // retention rate is within a loose band (pure sanity, not a statistical
  // test). We identify the first observation by giving it a unique payload.
  const size_t cap = 10;
  const size_t total = 10000;
  const size_t trials = 400;
  const char *first_tag = "first";
  const char *other_tag = "other";

  size_t retained = 0;
  for (size_t trial = 0; trial < trials; ++trial) {
    Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, cap);
    if (!tile) {
      fprintf(stderr, "failed to create tile for exemplar stats\n");
      exit(EXIT_FAILURE);
    }

    Tile_observe(tile, 0.0, first_tag, strlen(first_tag));
    for (size_t i = 1; i < total; ++i) {
      Tile_observe(tile, (double)i, other_tag, strlen(other_tag));
    }

    bool found = false;
    for (size_t i = 0; i < Tile_exemplar_count(tile); ++i) {
      const TileExemplar *ex = Tile_exemplar_at(tile, i);
      if (ex && ex->size == strlen(first_tag) &&
          memcmp(ex->data, first_tag, ex->size) == 0) {
        found = true;
        break;
      }
    }
    if (found) retained++;
    Tile_free(tile);
  }
  double rate = (double)retained / (double)trials;
  // Expected rate is K/T = 10/10000 = 0.001 per trial; with 400 trials the
  // expected retention count is ~0.4. This is too low for a tight bound, so
  // we simply demand the empirical count stays well under the conservative
  // ceiling and does not match an obvious bug where every observation is
  // retained. This asserts the reservoir actually samples (not "retain all").
  ASSERT(rate < 0.05, 1, rate < 0.05);
}

void test_tile_exemplars_freed_on_tile_free(void) {
  // Exercise the malloc/free pattern under sanitize. Observing twice as many
  // items as capacity guarantees at least one eviction path runs.
  const size_t cap = 4;
  Tile *tile = Tile_new(LAT, LNG, RESOLUTION, CAPACITY, PRECISION, cap);
  char buf[16];
  for (size_t i = 0; i < 64; ++i) {
    snprintf(buf, sizeof(buf), "x-%zu", i);
    Tile_observe(tile, (double)i, buf, strlen(buf));
  }
  Tile_free(tile);
}

int main(void) {
  RUN_TEST(test_tile_new);
  RUN_TEST(test_tile_new_invalid_input);
  RUN_TEST(test_tile_coords_to_cell);
  RUN_TEST(test_tile_observe);
  RUN_TEST(test_tile_advance);
  RUN_TEST(test_tile_query);
  RUN_TEST(test_tile_same_cell);
  RUN_TEST(test_tile_exemplars_disabled);
  RUN_TEST(test_tile_exemplars_under_capacity_retains_all);
  RUN_TEST(test_tile_exemplars_over_capacity_bounds_count);
  RUN_TEST(test_tile_exemplars_statistical_sanity);
  RUN_TEST(test_tile_exemplars_freed_on_tile_free);

  return 0;
}
