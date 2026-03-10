#ifndef VIZ_TILE_MAP_H
#define VIZ_TILE_MAP_H

#include <OpenGL/gl3.h>
#include <stdbool.h>
#include <curl/curl.h>

#define TILE_CACHE_SIZE 64
#define MAX_PENDING_DOWNLOADS 8

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
} TileMap;

TileMap *TileMap_new(GLuint program);
void TileMap_draw(TileMap *tm, double center_lat, double center_lng,
                  double zoom_degrees, int viewport_w, int viewport_h);
void TileMap_free(TileMap *tm);

#endif
