#ifndef VIZ_TILE_DISK_CACHE_H
#define VIZ_TILE_DISK_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

#define TILE_MAP_CACHE_PATH_CAP PATH_MAX
#define TILE_MAP_CACHE_NAMESPACE_CAP 96
#define TILE_MAP_DEFAULT_DISK_CACHE_MAX_BYTES (256u * 1024u * 1024u)

typedef struct {
  bool enabled;
  int hits;
  int writes;
  int prunes;
  size_t bytes;
  size_t max_bytes;
  char root[TILE_MAP_CACHE_PATH_CAP];
  char cache_namespace[TILE_MAP_CACHE_NAMESPACE_CAP];
} TileDiskCache;

void TileDiskCache_init(TileDiskCache *cache, const char *style_name,
                        const char *url_template);
bool TileDiskCache_use_root(TileDiskCache *cache, const char *root_path);
void TileDiskCache_configure(TileDiskCache *cache, bool enabled,
                             size_t max_bytes);
void TileDiskCache_refresh_usage(TileDiskCache *cache);
bool TileDiskCache_read(TileDiskCache *cache, int x, int y, int z,
                        uint8_t **out_data, size_t *out_size);
bool TileDiskCache_write(TileDiskCache *cache, int x, int y, int z,
                         const uint8_t *data, size_t size);

#endif
