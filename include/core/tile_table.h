#ifndef TILE_TABLE_H
#define TILE_TABLE_H

#include <stdbool.h>
#include <stddef.h>
#include "core/tile.h"

#define INITIAL_TILE_TABLE_CAPACITY 16

typedef struct TileTable TileTable;
typedef struct {
  H3Index key;
  Tile *value;
  TileTable *_table;
  size_t _index;
} TileTableIterator;

TileTable *TileTable_create(void);
void TileTable_free(TileTable *table);
Tile *TileTable_get(TileTable *table, H3Index key);
bool TileTable_set(TileTable *table, H3Index key, Tile *tile);
size_t TileTable_size(TileTable *table);
TileTableIterator TileTable_iterator(TileTable *table);
bool TileTable_next(TileTableIterator *it);

#endif
