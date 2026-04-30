#ifndef VIZ_TILE_MAP_H
#define VIZ_TILE_MAP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <curl/curl.h>
#include "viz/gl_platform.h"
#include "viz/map_view.h"
#include "viz/tile_limits.h"
#include "viz/tile_disk_cache.h"

#define TILE_CACHE_SIZE 64
#define MAX_TILE_UPLOADS_PER_FRAME 1
#define TILE_MAP_STYLE_CAP 64
#define TILE_MAP_URL_CAP 1024
#define TILE_MAP_API_KEY_CAP 256
typedef struct {
  int x, y, z;
  GLuint texture;
  int texture_w;
  int texture_h;
  int last_used_frame;
  bool valid;
} CachedTile;

// In-flight async download
typedef struct {
  uint8_t *data;
  size_t size;
} MemBuffer;

typedef struct {
  CURL *easy;
  MemBuffer buf;
  uint8_t *decoded_pixels;
  int decoded_w;
  int decoded_h;
  int x, y, z;
  bool active;
  bool ready;
  bool decoding;
  bool decoded;
} PendingDownload;

typedef struct {
  int cache_tiles;
  int active_downloads;
  int ready_downloads;
  int decoding_downloads;
  int decoded_downloads;
  int uploads_last_frame;
  int disk_cache_hits;
  int disk_cache_writes;
  int disk_cache_prunes;
  bool disk_cache_enabled;
  size_t disk_cache_bytes;
  size_t disk_cache_max_bytes;
} TileMapStats;

typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint program;
  GLint u_projection;
  GLint u_texture;
  CachedTile cache[TILE_CACHE_SIZE];
  int frame_counter;
  // Async download state
  CURLM *multi;
  PendingDownload pending[TILE_MAP_MAX_PENDING_DOWNLOADS];
  int running_transfers;
  int last_upload_count;
  pthread_mutex_t pending_mutex;
  pthread_cond_t pending_cond;
  pthread_t decode_thread;
  bool shutdown_worker;
  TileDiskCache disk_cache;
  char style_name[TILE_MAP_STYLE_CAP];
  char url_template[TILE_MAP_URL_CAP];
  char api_key[TILE_MAP_API_KEY_CAP];
} TileMap;

TileMap *TileMap_new(GLuint program, const char *style_name,
                     const char *url_template, const char *api_key);
void TileMap_configure_disk_cache(TileMap *tm, bool enabled, size_t max_bytes);
void TileMap_draw(TileMap *tm, double center_lat, double center_lng,
                  double zoom_degrees, MapProjectionMode projection_mode,
                  int viewport_w, int viewport_h,
                  int upload_budget);
void TileMap_get_stats(TileMap *tm, TileMapStats *out_stats);
void TileMap_free(TileMap *tm);

#endif
