#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "core/tile_table.h"

typedef struct {
  H3Index key;
  Tile *value;
  bool occupied;
} TileTableEntry;

struct TileTable {
  TileTableEntry *entries;
  size_t capacity;
  size_t length;
};

static uint64_t tile_table_hash(H3Index key) {
  uint64_t x = (uint64_t)key + 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static size_t tile_table_find_slot(const TileTableEntry *entries, size_t capacity,
                                   H3Index key, bool *found) {
  size_t index = (size_t)(tile_table_hash(key) & (uint64_t)(capacity - 1));

  while (entries[index].occupied) {
    if (entries[index].key == key) {
      if (found) *found = true;
      return index;
    }
    index = (index + 1) & (capacity - 1);
  }

  if (found) *found = false;
  return index;
}

static void tile_table_insert_entry(TileTableEntry *entries, size_t capacity,
                                    H3Index key, Tile *value) {
  bool found = false;
  size_t index = tile_table_find_slot(entries, capacity, key, &found);

  entries[index].key = key;
  entries[index].value = value;
  entries[index].occupied = true;
}

static bool tile_table_expand(TileTable *table) {
  TileTableEntry *new_entries;
  size_t new_capacity;

  if (!table) return false;
  if (table->capacity > SIZE_MAX / 2) return false;

  new_capacity = table->capacity * 2;
  new_entries = calloc(new_capacity, sizeof(*new_entries));
  if (!new_entries) return false;

  for (size_t i = 0; i < table->capacity; ++i) {
    if (!table->entries[i].occupied) continue;
    tile_table_insert_entry(new_entries, new_capacity, table->entries[i].key,
                            table->entries[i].value);
  }

  free(table->entries);
  table->entries = new_entries;
  table->capacity = new_capacity;
  return true;
}

TileTable *TileTable_create(void) {
  TileTable *table = calloc(1, sizeof(*table));

  if (!table) return NULL;

  table->capacity = INITIAL_TILE_TABLE_CAPACITY;
  table->entries = calloc(table->capacity, sizeof(*table->entries));
  if (!table->entries) {
    free(table);
    return NULL;
  }

  return table;
}

void TileTable_free(TileTable *table) {
  if (!table) return;

  for (size_t i = 0; i < table->capacity; ++i) {
    if (table->entries[i].occupied) {
      Tile_free(table->entries[i].value);
    }
  }

  free(table->entries);
  free(table);
}

Tile *TileTable_get(TileTable *table, H3Index key) {
  bool found = false;
  size_t index;

  if (!table || table->capacity == 0) return NULL;

  index = tile_table_find_slot(table->entries, table->capacity, key, &found);
  if (!found) return NULL;
  return table->entries[index].value;
}

bool TileTable_set(TileTable *table, H3Index key, Tile *tile) {
  bool found = false;
  size_t index;

  if (!table || !tile) return false;
  if (table->length >= table->capacity / 2 && !tile_table_expand(table)) {
    return false;
  }

  index = tile_table_find_slot(table->entries, table->capacity, key, &found);
  if (!found) {
    table->length++;
  }

  table->entries[index].key = key;
  table->entries[index].value = tile;
  table->entries[index].occupied = true;
  return true;
}

size_t TileTable_size(TileTable *table) {
  return table ? table->length : 0;
}

TileTableIterator TileTable_iterator(TileTable *table) {
  TileTableIterator it;

  memset(&it, 0, sizeof(it));
  it._table = table;
  return it;
}

bool TileTable_next(TileTableIterator *it) {
  TileTable *table;

  if (!it || !it->_table) return false;
  table = it->_table;

  while (it->_index < table->capacity) {
    size_t index = it->_index++;
    if (!table->entries[index].occupied) continue;
    it->key = table->entries[index].key;
    it->value = table->entries[index].value;
    return true;
  }

  return false;
}
