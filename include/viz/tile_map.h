#ifndef VIZ_TILE_MAP_H
#define VIZ_TILE_MAP_H

#include <OpenGL/gl3.h>
#include <stdbool.h>
#include <stdint.h>
#include <curl/curl.h>

#define TILE_CACHE_SIZE 64
#define MAX_PENDING_DOWNLOADS 8
#define TILE_MAP_STYLE_CAP 64
#define TILE_MAP_URL_CAP 1024
#define TILE_MAP_API_KEY_CAP 256

typedef struct {
  int x, y, z;
  GLuint texture;
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
  int x, y, z;
  bool active;
} PendingDownload;

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
  PendingDownload pending[MAX_PENDING_DOWNLOADS];
  int running_transfers;
  char style_name[TILE_MAP_STYLE_CAP];
  char url_template[TILE_MAP_URL_CAP];
  char api_key[TILE_MAP_API_KEY_CAP];
} TileMap;

TileMap *TileMap_new(GLuint program, const char *style_name,
                     const char *url_template, const char *api_key);
void TileMap_draw(TileMap *tm, double center_lat, double center_lng,
                  double zoom_degrees, int viewport_w, int viewport_h);
void TileMap_free(TileMap *tm);

#endif
