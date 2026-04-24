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

typedef struct {
  size_t count;
  bool saw_expected_cell;
} TileVisitState;

typedef struct {
  H3Index target_cell;
  Tile *found_tile;
} TileLookupState;

static bool count_tiles(H3Index cell_id, Tile *tile, void *user_data) {
  TileVisitState *state = (TileVisitState *)user_data;
  if (!state || !tile) return false;
  state->count++;
  if (cell_id == CELLID_RS10 && tile->cellId == CELLID_RS10) {
    state->saw_expected_cell = true;
  }
  return true;
}

static bool lookup_tile(H3Index cell_id, Tile *tile, void *user_data) {
  TileLookupState *state = (TileLookupState *)user_data;
  if (!state || !tile) return false;
  if (cell_id == state->target_cell) {
    state->found_tile = tile;
  }
  return true;
}

static Tile *find_tile_for_cell(SpatialMemory *sm, H3Index cell_id) {
  TileLookupState state = {.target_cell = cell_id, .found_tile = NULL};
  if (!SpatialMemory_for_each_tile(sm, lookup_tile, &state)) {
    return NULL;
  }
  return state.found_tile;
}

void test_sm_new(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  ASSERT(NULL != sm, 1, NULL != sm);
  ASSERT(RESOLUTION == sm->resolution, RESOLUTION, sm->resolution);
  ASSERT(CAPACITY == sm->capacity, CAPACITY, (int)sm->capacity);
  ASSERT(PRECISION == sm->precision, PRECISION, (int)sm->precision);
  ASSERT(0 == (int)sm->exemplar_capacity, 0, (int)sm->exemplar_capacity);
  ASSERT(0 == (int)SpatialMemory_tile_count(sm), 0,
         (int)SpatialMemory_tile_count(sm));
  SpatialMemory_free(sm);
}

void test_sm_new_invalid_resolution(void) {
  SpatialMemory *sm = SpatialMemory_new(16, CAPACITY, PRECISION, 0);
  ASSERT(NULL == sm, 1, NULL == sm);
}

void test_sm_new_invalid_precision(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY,
                                        RingBuffer_precision_max() + 1, 0);
  ASSERT(NULL == sm, 1, NULL == sm);
}

void test_sm_observe(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  bool ok = SpatialMemory_observe(sm, 0.0, LAT, LNG, pb, strlen(pb));
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
  Tile *tile = find_tile_for_cell(sm, cellId);
  ASSERT(NULL != tile, 1, NULL != tile);
  ASSERT(1 == (int)SpatialMemory_tile_count(sm), 1,
         (int)SpatialMemory_tile_count(sm));
  SpatialMemory_free(sm);
}

void test_sm_query(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  double query_count = 0.0;
  bool ok = SpatialMemory_observe(sm, 0.0, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 1, &query_count);
  ASSERT(ok, 1, ok);
  int count = (int)query_count;
  ASSERT(count >= 1, 1, count);
  SpatialMemory_free(sm);
}

void test_sm_advance_all(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
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

  bool ok = SpatialMemory_observe(sm, 0.0, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  SpatialMemory_advance_all(sm);
  Tile *tile = find_tile_for_cell(sm, cellId);
  int count = (int)Tile_query(tile, 0);
  ASSERT(0 == count, 0, count);
  SpatialMemory_free(sm);
}

void test_sm_multi_tile(void) {
  const char *pb = "peanut butter";
  const char *sushi = "sushi";
  const double tokyo_lat = 35.68;
  const double tokyo_lng = 139.68;
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  double count_lon = 0.0;
  double count_tok = 0.0;
  bool ok = SpatialMemory_observe(sm, 0.0, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_observe(sm, 1.0, tokyo_lat, tokyo_lng, sushi, strlen(sushi));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 0, &count_lon);
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, tokyo_lat, tokyo_lng, 0, &count_tok);
  ASSERT(ok, 1, ok);
  int num_tiles = (int)SpatialMemory_tile_count(sm);
  ASSERT(count_tok >= 1, 1, (int)count_tok);
  ASSERT(count_lon >= 1, 1, (int)count_lon);
  ASSERT(2 == num_tiles, 2, num_tiles);
  SpatialMemory_free(sm);
}

void test_sm_invalid_observe_does_not_crash(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  bool ok = SpatialMemory_observe(sm, 0.0, 100.0, LNG, pb, strlen(pb));
  ASSERT(!ok, 0, ok);
  ASSERT(0 == (int)SpatialMemory_tile_count(sm), 0, (int)SpatialMemory_tile_count(sm));
  SpatialMemory_free(sm);
}

void test_sm_invalid_query_returns_false(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  double count = 123.0;
  bool ok = SpatialMemory_query(sm, 100.0, LNG, 0, &count);
  ASSERT(!ok, 0, ok);
  ASSERT(0 == (int)count, 0, (int)count);
  SpatialMemory_free(sm);
}

void test_sm_same_cell_deduplicates_observations(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  double first_count = 0.0;
  double second_count = 0.0;

  bool ok = SpatialMemory_observe(sm, 0.0, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 0, &first_count);
  ASSERT(ok, 1, ok);

  ok = SpatialMemory_observe(sm, 1.0, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, LAT, LNG, 0, &second_count);
  ASSERT(ok, 1, ok);

  ASSERT(1 == (int)SpatialMemory_tile_count(sm), 1, (int)SpatialMemory_tile_count(sm));
  ASSERT((int)first_count == (int)second_count, (int)first_count, (int)second_count);
  SpatialMemory_free(sm);
}

void test_sm_for_each_tile_enumerates_entries(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  TileVisitState state = {0};

  bool ok = SpatialMemory_observe(sm, 0.0, LAT, LNG, pb, strlen(pb));
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_for_each_tile(sm, count_tiles, &state);
  ASSERT(ok, 1, ok);
  ASSERT(1 == (int)state.count, 1, (int)state.count);
  ASSERT(state.saw_expected_cell, 1, state.saw_expected_cell);
  SpatialMemory_free(sm);
}

void test_sm_exemplar_capacity_propagates_to_tile(void) {
  const size_t cap = 4;
  SpatialMemory *sm =
      SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, cap);
  ASSERT(NULL != sm, 1, NULL != sm);
  ASSERT(cap == sm->exemplar_capacity, (int)cap, (int)sm->exemplar_capacity);

  const char *items[] = {"alpha", "bravo", "charlie"};
  for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); ++i) {
    bool ok = SpatialMemory_observe(sm, (double)i, LAT, LNG, items[i],
                                    strlen(items[i]));
    ASSERT(ok, 1, ok);
  }

  LatLng loc;
  loc.lat = degsToRads(LAT);
  loc.lng = degsToRads(LNG);
  H3Index cellId;
  H3Error err = latLngToCell(&loc, RESOLUTION, &cellId);
  if (err) {
    fprintf(stderr, "Failed to create H3Index : invalid lat/lng or resolution\n");
    exit(EXIT_FAILURE);
  }

  Tile *tile = find_tile_for_cell(sm, cellId);
  ASSERT(NULL != tile, 1, NULL != tile);
  ASSERT(3 == (int)Tile_exemplar_count(tile), 3,
         (int)Tile_exemplar_count(tile));
  SpatialMemory_free(sm);
}

void test_sm_query_intervals_empty(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  SpatialMemoryInterval out[4];
  size_t n = SpatialMemory_query_intervals(sm, LAT, LNG, 0, out, 4);
  ASSERT(0 == (int)n, 0, (int)n);
  SpatialMemory_free(sm);
}

void test_sm_query_intervals_single_hit(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  SpatialMemory_observe(sm, 123.5, LAT, LNG, pb, strlen(pb));

  SpatialMemoryInterval out[4];
  size_t n = SpatialMemory_query_intervals(sm, LAT, LNG, 0, out, 4);
  ASSERT(1 == (int)n, 1, (int)n);
  ASSERT(CELLID_RS10 == out[0].cell, 1, CELLID_RS10 == out[0].cell);
  ASSERT(out[0].t_min == 123.5, 1, out[0].t_min == 123.5);
  ASSERT(out[0].t_max == 123.5, 1, out[0].t_max == 123.5);
  ASSERT(out[0].count >= 1.0, 1, out[0].count >= 1.0);
  SpatialMemory_free(sm);
}

void test_sm_query_intervals_sorted_by_tmax_desc(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  // Offsets of 0.01 deg (~1.1 km at LAT=51) are ~17 hops apart at res 10, so
  // k_ring=30 is needed to cover all three from the middle point.
  SpatialMemory_observe(sm, 100.0, LAT, LNG, pb, strlen(pb));
  SpatialMemory_observe(sm, 200.0, LAT + 0.01, LNG, pb, strlen(pb));
  SpatialMemory_observe(sm, 150.0, LAT + 0.02, LNG, pb, strlen(pb));

  SpatialMemoryInterval out[16];
  size_t n = SpatialMemory_query_intervals(sm, LAT + 0.01, LNG, 30, out, 16);
  ASSERT(n >= 3, 1, n >= 3);
  ASSERT(out[0].t_max == 200.0, 1, out[0].t_max == 200.0);
  ASSERT(out[1].t_max == 150.0, 1, out[1].t_max == 150.0);
  ASSERT(out[2].t_max == 100.0, 1, out[2].t_max == 100.0);
  SpatialMemory_free(sm);
}

void test_sm_query_intervals_truncation_returns_total(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  SpatialMemory_observe(sm, 100.0, LAT, LNG, pb, strlen(pb));
  SpatialMemory_observe(sm, 200.0, LAT + 0.01, LNG, pb, strlen(pb));
  SpatialMemory_observe(sm, 150.0, LAT + 0.02, LNG, pb, strlen(pb));

  SpatialMemoryInterval out[1];
  size_t n = SpatialMemory_query_intervals(sm, LAT + 0.01, LNG, 30, out, 1);
  ASSERT(n >= 3, 1, n >= 3);  // total found, not number written.
  ASSERT(out[0].t_max == 200.0, 1, out[0].t_max == 200.0);
  SpatialMemory_free(sm);
}

void test_sm_query_intervals_probe_mode(void) {
  SpatialMemory *sm = SpatialMemory_new(RESOLUTION, CAPACITY, PRECISION, 0);
  const char *pb = "peanut butter";
  SpatialMemory_observe(sm, 100.0, LAT, LNG, pb, strlen(pb));
  SpatialMemory_observe(sm, 200.0, LAT + 0.01, LNG, pb, strlen(pb));

  size_t n_null = SpatialMemory_query_intervals(sm, LAT, LNG, 30, NULL, 10);
  size_t n_zero_max;
  SpatialMemoryInterval out[1];
  n_zero_max = SpatialMemory_query_intervals(sm, LAT, LNG, 30, out, 0);
  ASSERT(n_null == n_zero_max, 1, n_null == n_zero_max);
  ASSERT(n_null >= 2, 1, n_null >= 2);
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
  RUN_TEST(test_sm_for_each_tile_enumerates_entries);
  RUN_TEST(test_sm_exemplar_capacity_propagates_to_tile);
  RUN_TEST(test_sm_query_intervals_empty);
  RUN_TEST(test_sm_query_intervals_single_hit);
  RUN_TEST(test_sm_query_intervals_sorted_by_tmax_desc);
  RUN_TEST(test_sm_query_intervals_truncation_returns_total);
  RUN_TEST(test_sm_query_intervals_probe_mode);

  return 0;
}
