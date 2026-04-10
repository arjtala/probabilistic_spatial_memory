#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include "viz/tile_disk_cache.h"

typedef struct {
  char path[TILE_MAP_CACHE_PATH_CAP];
  size_t size_bytes;
  time_t modified_at;
} DiskCacheEntry;

typedef struct {
  DiskCacheEntry *entries;
  size_t count;
  size_t capacity;
  size_t total_bytes;
} DiskCacheInventory;

static uint32_t hash_text32(const char *text) {
  uint32_t hash = 2166136261u;

  if (!text) return hash;
  while (*text) {
    hash ^= (uint8_t)*text++;
    hash *= 16777619u;
  }
  return hash;
}

static void build_cache_namespace(TileDiskCache *cache, const char *style_name,
                                  const char *url_template) {
  char sanitized_style[48];
  size_t out_len = 0;
  const char *style = style_name && style_name[0] ? style_name : "default";

  if (!cache) return;
  for (size_t i = 0; style[i] != '\0' && out_len + 1 < sizeof(sanitized_style);
       i++) {
    unsigned char ch = (unsigned char)style[i];
    sanitized_style[out_len++] = isalnum(ch) ? (char)ch : '_';
  }
  if (out_len == 0) {
    sanitized_style[out_len++] = 'x';
  }
  sanitized_style[out_len] = '\0';

  snprintf(cache->cache_namespace, sizeof(cache->cache_namespace), "%s-%08x",
           sanitized_style, hash_text32(url_template));
}

static bool build_cache_namespace_root(const TileDiskCache *cache, char *out,
                                       size_t out_size) {
  if (!cache || !out || out_size == 0 || cache->root[0] == '\0' ||
      cache->cache_namespace[0] == '\0') {
    return false;
  }
  return snprintf(out, out_size, "%s/%s", cache->root,
                  cache->cache_namespace) < (int)out_size;
}

static bool ensure_directory_recursive(const char *path) {
  char tmp[TILE_MAP_CACHE_PATH_CAP];
  size_t len;

  if (!path || path[0] == '\0') return false;
  if (snprintf(tmp, sizeof(tmp), "%s", path) >= (int)sizeof(tmp)) return false;

  len = strlen(tmp);
  if (len == 0) return false;
  if (len > 1 && tmp[len - 1] == '/') {
    tmp[len - 1] = '\0';
  }

  for (char *cursor = tmp + 1; *cursor; cursor++) {
    if (*cursor != '/') continue;
    *cursor = '\0';
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
      *cursor = '/';
      return false;
    }
    *cursor = '/';
  }

  if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
    return false;
  }
  return true;
}

static bool try_enable_disk_cache(TileDiskCache *cache, const char *root_path) {
  char namespace_root[TILE_MAP_CACHE_PATH_CAP];

  if (!cache || !root_path || root_path[0] == '\0') return false;
  if (!ensure_directory_recursive(root_path)) return false;
  if (snprintf(namespace_root, sizeof(namespace_root), "%s/%s", root_path,
               cache->cache_namespace) >= (int)sizeof(namespace_root)) {
    return false;
  }
  if (!ensure_directory_recursive(namespace_root)) return false;
  if (snprintf(cache->root, sizeof(cache->root), "%s", root_path) >=
      (int)sizeof(cache->root)) {
    cache->root[0] = '\0';
    return false;
  }
  cache->enabled = true;
  return true;
}

static bool inventory_append_entry(DiskCacheInventory *inventory,
                                   const char *path, size_t size_bytes,
                                   time_t modified_at) {
  DiskCacheEntry *resized;

  if (!inventory || !path) return false;
  if (inventory->count == inventory->capacity) {
    size_t new_capacity = inventory->capacity ? inventory->capacity * 2 : 32;
    resized = realloc(inventory->entries,
                      new_capacity * sizeof(*inventory->entries));
    if (!resized) return false;
    inventory->entries = resized;
    inventory->capacity = new_capacity;
  }

  if (snprintf(inventory->entries[inventory->count].path,
               sizeof(inventory->entries[inventory->count].path), "%s",
               path) >= (int)sizeof(inventory->entries[inventory->count].path)) {
    return false;
  }
  inventory->entries[inventory->count].size_bytes = size_bytes;
  inventory->entries[inventory->count].modified_at = modified_at;
  inventory->count++;
  inventory->total_bytes += size_bytes;
  return true;
}

static void free_inventory(DiskCacheInventory *inventory) {
  if (!inventory) return;
  free(inventory->entries);
  inventory->entries = NULL;
  inventory->count = 0;
  inventory->capacity = 0;
  inventory->total_bytes = 0;
}

static bool collect_inventory_recursive(const char *dir_path,
                                        DiskCacheInventory *inventory) {
  DIR *dir;
  struct dirent *entry;

  if (!dir_path || !inventory) return false;

  dir = opendir(dir_path);
  if (!dir) {
    return errno == ENOENT;
  }

  while ((entry = readdir(dir)) != NULL) {
    char child_path[TILE_MAP_CACHE_PATH_CAP];
    struct stat st;

    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    if (snprintf(child_path, sizeof(child_path), "%s/%s", dir_path,
                 entry->d_name) >= (int)sizeof(child_path)) {
      closedir(dir);
      return false;
    }
    if (stat(child_path, &st) != 0) {
      continue;
    }
    if (S_ISDIR(st.st_mode)) {
      if (!collect_inventory_recursive(child_path, inventory)) {
        closedir(dir);
        return false;
      }
      continue;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0) {
      continue;
    }
    if (!inventory_append_entry(inventory, child_path, (size_t)st.st_size,
                                st.st_mtime)) {
      closedir(dir);
      return false;
    }
  }

  closedir(dir);
  return true;
}

static int compare_cache_entries_by_age(const void *lhs, const void *rhs) {
  const DiskCacheEntry *a = (const DiskCacheEntry *)lhs;
  const DiskCacheEntry *b = (const DiskCacheEntry *)rhs;

  if (a->modified_at < b->modified_at) return -1;
  if (a->modified_at > b->modified_at) return 1;
  return strcmp(a->path, b->path);
}

static void remove_empty_parent_dirs(const TileDiskCache *cache,
                                     const char *file_path) {
  char namespace_root[TILE_MAP_CACHE_PATH_CAP];
  char work[TILE_MAP_CACHE_PATH_CAP];
  char *slash;

  if (!cache || !file_path) return;
  if (!build_cache_namespace_root(cache, namespace_root,
                                  sizeof(namespace_root))) {
    return;
  }
  if (snprintf(work, sizeof(work), "%s", file_path) >= (int)sizeof(work)) {
    return;
  }

  slash = strrchr(work, '/');
  while (slash) {
    *slash = '\0';
    if (strcmp(work, namespace_root) == 0) break;
    if (rmdir(work) != 0) break;
    slash = strrchr(work, '/');
  }
}

void TileDiskCache_refresh_usage(TileDiskCache *cache) {
  char namespace_root[TILE_MAP_CACHE_PATH_CAP];
  DiskCacheInventory inventory = {0};

  if (!cache) return;
  cache->bytes = 0;
  if (!build_cache_namespace_root(cache, namespace_root, sizeof(namespace_root))) {
    return;
  }
  if (collect_inventory_recursive(namespace_root, &inventory)) {
    cache->bytes = inventory.total_bytes;
  }
  free_inventory(&inventory);
}

static bool prune_disk_cache_to_fit(TileDiskCache *cache, size_t incoming_bytes) {
  char namespace_root[TILE_MAP_CACHE_PATH_CAP];
  DiskCacheInventory inventory = {0};
  size_t bytes_after_prune;
  size_t max_bytes;

  if (!cache || !cache->enabled) return false;

  max_bytes = cache->max_bytes;
  if (max_bytes == 0) {
    TileDiskCache_refresh_usage(cache);
    return true;
  }
  if (incoming_bytes > max_bytes) {
    TileDiskCache_refresh_usage(cache);
    return false;
  }
  if (!build_cache_namespace_root(cache, namespace_root, sizeof(namespace_root))) {
    cache->bytes = 0;
    return false;
  }
  if (!collect_inventory_recursive(namespace_root, &inventory)) {
    cache->bytes = 0;
    return false;
  }

  bytes_after_prune = inventory.total_bytes;
  if (bytes_after_prune + incoming_bytes > max_bytes && inventory.count > 0) {
    qsort(inventory.entries, inventory.count, sizeof(*inventory.entries),
          compare_cache_entries_by_age);

    for (size_t i = 0; i < inventory.count &&
                       bytes_after_prune + incoming_bytes > max_bytes; i++) {
      const DiskCacheEntry *entry = &inventory.entries[i];
      if (unlink(entry->path) != 0) continue;
      if (bytes_after_prune >= entry->size_bytes) {
        bytes_after_prune -= entry->size_bytes;
      } else {
        bytes_after_prune = 0;
      }
      cache->prunes++;
      remove_empty_parent_dirs(cache, entry->path);
    }
  }

  cache->bytes = bytes_after_prune;
  free_inventory(&inventory);
  return bytes_after_prune + incoming_bytes <= max_bytes;
}

static bool build_tile_cache_dir(const TileDiskCache *cache, int x, int z,
                                 char *out, size_t out_size) {
  if (!cache || !cache->enabled || !out || out_size == 0) return false;
  return snprintf(out, out_size, "%s/%s/%d/%d", cache->root,
                  cache->cache_namespace, z, x) < (int)out_size;
}

static bool build_tile_cache_path(const TileDiskCache *cache, int x, int y,
                                  int z, char *out, size_t out_size) {
  if (!cache || !cache->enabled || !out || out_size == 0) return false;
  return snprintf(out, out_size, "%s/%s/%d/%d/%d.tile", cache->root,
                  cache->cache_namespace, z, x, y) < (int)out_size;
}

static bool read_file_into_buffer(const char *path, uint8_t **out_data,
                                  size_t *out_size) {
  struct stat st;
  FILE *file = NULL;
  uint8_t *data = NULL;
  size_t file_size;
  size_t bytes_read;

  if (!path || !out_data || !out_size) return false;
  *out_data = NULL;
  *out_size = 0;

  if (stat(path, &st) != 0 || st.st_size <= 0) return false;

  file_size = (size_t)st.st_size;
  file = fopen(path, "rb");
  if (!file) return false;

  data = malloc(file_size);
  if (!data) {
    fclose(file);
    return false;
  }

  bytes_read = fread(data, 1, file_size, file);
  if (bytes_read != file_size || ferror(file)) {
    fclose(file);
    free(data);
    return false;
  }

  fclose(file);
  *out_data = data;
  *out_size = file_size;
  return true;
}

static bool write_buffer_to_file(const char *path, const uint8_t *data,
                                 size_t size) {
  char tmp_path[TILE_MAP_CACHE_PATH_CAP];
  FILE *file;
  size_t written;
  int close_rc;
  bool wrote_ok = false;

  if (!path || !data || size == 0) return false;
  if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld", path,
               (long)getpid()) >= (int)sizeof(tmp_path)) {
    return false;
  }

  file = fopen(tmp_path, "wb");
  if (!file) return false;

  written = fwrite(data, 1, size, file);
  close_rc = fclose(file);
  wrote_ok = (written == size && close_rc == 0);
  if (!wrote_ok) {
    unlink(tmp_path);
    return false;
  }

  if (rename(tmp_path, path) != 0) {
    unlink(tmp_path);
    return false;
  }
  return true;
}

void TileDiskCache_init(TileDiskCache *cache, const char *style_name,
                        const char *url_template) {
  char candidate[TILE_MAP_CACHE_PATH_CAP];
  const char *home;
  const char *xdg_cache_home;

  if (!cache) return;

  memset(cache, 0, sizeof(*cache));
  cache->max_bytes = TILE_MAP_DEFAULT_DISK_CACHE_MAX_BYTES;
  build_cache_namespace(cache, style_name, url_template);

  home = getenv("HOME");
  xdg_cache_home = getenv("XDG_CACHE_HOME");

#ifdef __APPLE__
  if (home && home[0] != '\0' &&
      snprintf(candidate, sizeof(candidate),
               "%s/Library/Caches/psm-viz/tiles", home) <
          (int)sizeof(candidate) &&
      try_enable_disk_cache(cache, candidate)) {
    TileDiskCache_refresh_usage(cache);
    prune_disk_cache_to_fit(cache, 0);
    return;
  }
#endif

  if (xdg_cache_home && xdg_cache_home[0] != '\0' &&
      snprintf(candidate, sizeof(candidate), "%s/psm-viz/tiles",
               xdg_cache_home) < (int)sizeof(candidate) &&
      try_enable_disk_cache(cache, candidate)) {
    TileDiskCache_refresh_usage(cache);
    prune_disk_cache_to_fit(cache, 0);
    return;
  }

  if (home && home[0] != '\0' &&
      snprintf(candidate, sizeof(candidate), "%s/.cache/psm-viz/tiles", home) <
          (int)sizeof(candidate) &&
      try_enable_disk_cache(cache, candidate)) {
    TileDiskCache_refresh_usage(cache);
    prune_disk_cache_to_fit(cache, 0);
    return;
  }
}

bool TileDiskCache_use_root(TileDiskCache *cache, const char *root_path) {
  if (!cache) return false;
  cache->enabled = false;
  cache->root[0] = '\0';
  cache->bytes = 0;
  return try_enable_disk_cache(cache, root_path);
}

void TileDiskCache_configure(TileDiskCache *cache, bool enabled,
                             size_t max_bytes) {
  if (!cache) return;

  cache->max_bytes = max_bytes;
  if (!enabled) {
    cache->enabled = false;
    TileDiskCache_refresh_usage(cache);
    return;
  }

  if (cache->root[0] == '\0') {
    cache->enabled = false;
    cache->bytes = 0;
    return;
  }

  cache->enabled = true;
  TileDiskCache_refresh_usage(cache);
  prune_disk_cache_to_fit(cache, 0);
}

bool TileDiskCache_read(TileDiskCache *cache, int x, int y, int z,
                        uint8_t **out_data, size_t *out_size) {
  char file_path[TILE_MAP_CACHE_PATH_CAP];

  if (!cache || !cache->enabled) return false;
  if (!build_tile_cache_path(cache, x, y, z, file_path, sizeof(file_path))) {
    return false;
  }
  if (!read_file_into_buffer(file_path, out_data, out_size)) return false;
  cache->hits++;
  return true;
}

bool TileDiskCache_write(TileDiskCache *cache, int x, int y, int z,
                         const uint8_t *data, size_t size) {
  char dir_path[TILE_MAP_CACHE_PATH_CAP];
  char file_path[TILE_MAP_CACHE_PATH_CAP];
  struct stat st;
  size_t existing_size = 0;
  size_t required_bytes = size;

  if (!cache || !cache->enabled || !data || size == 0) return false;
  if (!build_tile_cache_dir(cache, x, z, dir_path, sizeof(dir_path))) {
    return false;
  }
  if (!ensure_directory_recursive(dir_path)) return false;
  if (!build_tile_cache_path(cache, x, y, z, file_path, sizeof(file_path))) {
    return false;
  }
  if (stat(file_path, &st) == 0 && st.st_size > 0) {
    existing_size = (size_t)st.st_size;
    if (existing_size < required_bytes) {
      required_bytes -= existing_size;
    } else {
      required_bytes = 0;
    }
  }
  if (!prune_disk_cache_to_fit(cache, required_bytes)) return false;
  if (!write_buffer_to_file(file_path, data, size)) return false;
  cache->writes++;
  if (cache->bytes >= existing_size) {
    cache->bytes -= existing_size;
  } else {
    cache->bytes = 0;
  }
  cache->bytes += size;
  return true;
}
