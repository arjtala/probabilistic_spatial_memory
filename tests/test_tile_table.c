#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/tile_table.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#define CAPACITY 3
#define PRECISION 4
#define RESOLUTION 10

static Tile *make_tile(double lat, double lng) {
  Tile *tile = Tile_new(lat, lng, RESOLUTION, CAPACITY, PRECISION, 0);
  if (!tile) {
    fprintf(stderr, "Failed to create tile for test\n");
    exit(EXIT_FAILURE);
  }
  return tile;
}

void test_tile_table_set_and_get(void) {
  TileTable *table = TileTable_create();
  Tile *tile = make_tile(51.380083, -0.311041);

  ASSERT(TileTable_set(table, tile->cellId, tile), 1, 1);
  ASSERT(TileTable_get(table, tile->cellId) == tile, 1,
         TileTable_get(table, tile->cellId) == tile);
  ASSERT(1 == (int)TileTable_size(table), 1, (int)TileTable_size(table));

  TileTable_free(table);
}

void test_tile_table_iterator_visits_entries(void) {
  TileTable *table = TileTable_create();
  Tile *tile_a = make_tile(51.380083, -0.311041);
  Tile *tile_b = make_tile(35.680000, 139.680000);
  bool saw_a = false;
  bool saw_b = false;

  ASSERT(TileTable_set(table, tile_a->cellId, tile_a), 1, 1);
  ASSERT(TileTable_set(table, tile_b->cellId, tile_b), 1, 1);

  TileTableIterator it = TileTable_iterator(table);
  while (TileTable_next(&it)) {
    if (it.key == tile_a->cellId && it.value == tile_a) saw_a = true;
    if (it.key == tile_b->cellId && it.value == tile_b) saw_b = true;
  }

  ASSERT(saw_a, 1, saw_a);
  ASSERT(saw_b, 1, saw_b);
  TileTable_free(table);
}

void test_tile_table_expands_for_many_entries(void) {
  TileTable *table = TileTable_create();

  for (int i = 0; i < 24; ++i) {
    double lat = -40.0 + (double)i;
    double lng = 80.0 + (double)i;
    Tile *tile = make_tile(lat, lng);
    ASSERT(TileTable_set(table, tile->cellId, tile), 1, 1);
    ASSERT(TileTable_get(table, tile->cellId) == tile, 1,
           TileTable_get(table, tile->cellId) == tile);
  }

  ASSERT(24 == (int)TileTable_size(table), 24, (int)TileTable_size(table));
  TileTable_free(table);
}

int main(void) {
  RUN_TEST(test_tile_table_set_and_get);
  RUN_TEST(test_tile_table_iterator_visits_entries);
  RUN_TEST(test_tile_table_expands_for_many_entries);
  return 0;
}
