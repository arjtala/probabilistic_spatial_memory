#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <OpenGL/gl3.h>
#include "stb/stb_image.h"
#include "viz/tile_map.h"
#include "viz/viz_math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
  size_t total = size * nmemb;
  MemBuffer *buf = (MemBuffer *)userdata;
  uint8_t *tmp = realloc(buf->data, buf->size + total);
  if (!tmp) return 0;
  buf->data = tmp;
  memcpy(buf->data + buf->size, ptr, total);
  buf->size += total;
  return total;
}

// Returns the geographic bounds of an OSM tile: (west, south, east, north) in degrees
static void tile_bounds(int tx, int ty, int z,
                        double *west, double *south, double *east, double *north) {
  double n = pow(2.0, z);
  *west = (double)tx / n * 360.0 - 180.0;
  *east = (double)(tx + 1) / n * 360.0 - 180.0;
  double lat_n = atan(sinh(M_PI * (1.0 - 2.0 * (double)ty / n)));
  double lat_s = atan(sinh(M_PI * (1.0 - 2.0 * (double)(ty + 1) / n)));
  *north = lat_n * 180.0 / M_PI;
  *south = lat_s * 180.0 / M_PI;
}

// ---- Tile cache operations ----
static CachedTile *cache_find(TileMap *tm, int x, int y, int z) {
  for (int i = 0; i < TILE_CACHE_SIZE; i++) {
    CachedTile *ct = &tm->cache[i];
    if (ct->valid && ct->x == x && ct->y == y && ct->z == z) {
      ct->last_used_frame = tm->frame_counter;
      return ct;
    }
  }
  return NULL;
}

static CachedTile *cache_evict(TileMap *tm) {
  int oldest_frame = tm->frame_counter + 1;
  int oldest_idx = 0;

  for (int i = 0; i < TILE_CACHE_SIZE; i++) {
    if (!tm->cache[i].valid) return &tm->cache[i];
    if (tm->cache[i].last_used_frame < oldest_frame) {
      oldest_frame = tm->cache[i].last_used_frame;
      oldest_idx = i;
    }
  }

  CachedTile *ct = &tm->cache[oldest_idx];
  if (ct->valid && ct->texture) {
    glDeleteTextures(1, &ct->texture);
  }
  ct->valid = false;
  return ct;
}

// ---- Async download helpers ----
static bool is_pending(TileMap *tm, int x, int y, int z) {
  for (int i = 0; i < MAX_PENDING_DOWNLOADS; i++) {
    PendingDownload *pd = &tm->pending[i];
    if (pd->active && pd->x == x && pd->y == y && pd->z == z) return true;
  }
  return false;
}

static PendingDownload *find_free_slot(TileMap *tm) {
  for (int i = 0; i < MAX_PENDING_DOWNLOADS; i++) {
    if (!tm->pending[i].active) return &tm->pending[i];
  }
  return NULL;
}

static void start_download(TileMap *tm, int x, int y, int z) {
  if (is_pending(tm, x, y, z)) return;
  PendingDownload *pd = find_free_slot(tm);
  if (!pd) return;  // all slots busy

  char url[256];
  snprintf(url, sizeof(url),
           "https://tile.openstreetmap.org/%d/%d/%d.png", z, x, y);

  pd->easy = curl_easy_init();
  if (!pd->easy) return;

  pd->buf.data = NULL;
  pd->buf.size = 0;
  pd->x = x;
  pd->y = y;
  pd->z = z;
  pd->active = true;

  curl_easy_setopt(pd->easy, CURLOPT_URL, url);
  curl_easy_setopt(pd->easy, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(pd->easy, CURLOPT_WRITEDATA, &pd->buf);
  curl_easy_setopt(pd->easy, CURLOPT_USERAGENT, "psm-viz/1.0");
  curl_easy_setopt(pd->easy, CURLOPT_TIMEOUT, 10L);
  curl_easy_setopt(pd->easy, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(pd->easy, CURLOPT_PRIVATE, pd);

  curl_multi_add_handle(tm->multi, pd->easy);
}

static void poll_downloads(TileMap *tm) {
  curl_multi_perform(tm->multi, &tm->running_transfers);

  CURLMsg *msg;
  int msgs_left;
  while ((msg = curl_multi_info_read(tm->multi, &msgs_left))) {
    if (msg->msg != CURLMSG_DONE) continue;

    CURL *easy = msg->easy_handle;
    PendingDownload *pd = NULL;
    curl_easy_getinfo(easy, CURLINFO_PRIVATE, &pd);
    if (!pd) continue;

    // Process completed download
    if (msg->data.result == CURLE_OK && pd->buf.size > 0) {
      int w, h, channels;
      uint8_t *img = stbi_load_from_memory(pd->buf.data, (int)pd->buf.size,
                                            &w, &h, &channels, 4);
      if (img) {
        CachedTile *ct = cache_find(tm, pd->x, pd->y, pd->z);
        if (!ct) {
          ct = cache_evict(tm);
        }
        if (ct->valid && ct->texture) {
          glDeleteTextures(1, &ct->texture);
        }

        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, img);
        glBindTexture(GL_TEXTURE_2D, 0);
        stbi_image_free(img);

        ct->x = pd->x;
        ct->y = pd->y;
        ct->z = pd->z;
        ct->texture = tex;
        ct->last_used_frame = tm->frame_counter;
        ct->valid = true;
      }
    }

    // Cleanup
    curl_multi_remove_handle(tm->multi, easy);
    curl_easy_cleanup(easy);
    free(pd->buf.data);
    pd->buf.data = NULL;
    pd->buf.size = 0;
    pd->easy = NULL;
    pd->active = false;
  }
}

// ---- Public API ----
TileMap *TileMap_new(GLuint program) {
  TileMap *tm = calloc(1, sizeof(TileMap));
  if (!tm) return NULL;

  tm->program = program;
  tm->u_projection = glGetUniformLocation(program, "u_projection");
  tm->u_texture = glGetUniformLocation(program, "u_texture");
  tm->frame_counter = 0;
  tm->running_transfers = 0;

  glGenVertexArrays(1, &tm->vao);
  glGenBuffers(1, &tm->vbo);

  glBindVertexArray(tm->vao);
  glBindBuffer(GL_ARRAY_BUFFER, tm->vbo);

  // Layout: vec2 position, vec2 texcoord = 4 floats per vertex
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);

  curl_global_init(CURL_GLOBAL_DEFAULT);
  tm->multi = curl_multi_init();

  return tm;
}

void TileMap_draw(TileMap *tm, double center_lat, double center_lng,
                  double zoom_degrees, int viewport_w, int viewport_h) {
  if (center_lat == 0.0 && center_lng == 0.0) return;  // no data yet

  tm->frame_counter++;

  // Progress any in-flight downloads (non-blocking)
  poll_downloads(tm);

  int z = osm_zoom_from_degrees(zoom_degrees);
  int cx, cy;
  latlon_to_tile(center_lat, center_lng, z, &cx, &cy);

  int n_tiles = (int)pow(2.0, z);
  double cos_center = cos(center_lat * M_PI / 180.0);
  double aspect = (double)viewport_h / (double)viewport_w;
  double half_w = zoom_degrees;
  double half_h = zoom_degrees * aspect;

  // Build same ortho projection as hex_renderer
  float proj[16] = {0};
  proj[0] = (float)(1.0 / half_w);
  proj[5] = (float)(1.0 / half_h);
  proj[10] = -1.0f;
  proj[15] = 1.0f;

  glUseProgram(tm->program);
  glUniformMatrix4fv(tm->u_projection, 1, GL_FALSE, proj);

  // Dynamic grid radius: enough tiles to cover the viewport
  double tile_degrees = 360.0 / pow(2.0, z);
  int tiles_needed = (int)ceil(2.0 * zoom_degrees / tile_degrees) + 1;
  int radius = (tiles_needed / 2) + 1;
  if (radius < 2) radius = 2;
  if (radius > 5) radius = 5;
  for (int dy = -radius; dy <= radius; dy++) {
    for (int dx = -radius; dx <= radius; dx++) {
      int tx = cx + dx;
      int ty = cy + dy;
      if (tx < 0 || tx >= n_tiles || ty < 0 || ty >= n_tiles) continue;

      CachedTile *ct = cache_find(tm, tx, ty, z);
      if (!ct) {
        // Kick off async download, skip rendering this tile for now
        start_download(tm, tx, ty, z);
        continue;
      }

      // Compute tile geographic bounds, project to same coord space as hex_renderer
      double west, south, east, north;
      tile_bounds(tx, ty, z, &west, &south, &east, &north);

      float x0 = (float)((west - center_lng) * cos_center);
      float x1 = (float)((east - center_lng) * cos_center);
      float y0 = (float)(south - center_lat);
      float y1 = (float)(north - center_lat);

      // Quad: 2 triangles, each vertex = pos(2) + texcoord(2)
      float quad[] = {
          x0, y0, 0.0f, 1.0f,  // bottom-left (texcoord flipped: V=1 at bottom)
          x1, y0, 1.0f, 1.0f,  // bottom-right
          x0, y1, 0.0f, 0.0f,  // top-left
          x0, y1, 0.0f, 0.0f,  // top-left
          x1, y0, 1.0f, 1.0f,  // bottom-right
          x1, y1, 1.0f, 0.0f,  // top-right
      };

      glBindBuffer(GL_ARRAY_BUFFER, tm->vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_DYNAMIC_DRAW);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, ct->texture);
      glUniform1i(tm->u_texture, 0);

      glBindVertexArray(tm->vao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
    }
  }
}

void TileMap_free(TileMap *tm) {
  if (!tm) return;
  // Cleanup pending downloads
  for (int i = 0; i < MAX_PENDING_DOWNLOADS; i++) {
    PendingDownload *pd = &tm->pending[i];
    if (pd->active && pd->easy) {
      curl_multi_remove_handle(tm->multi, pd->easy);
      curl_easy_cleanup(pd->easy);
      free(pd->buf.data);
    }
  }
  if (tm->multi) curl_multi_cleanup(tm->multi);

  for (int i = 0; i < TILE_CACHE_SIZE; i++) {
    if (tm->cache[i].valid && tm->cache[i].texture) {
      glDeleteTextures(1, &tm->cache[i].texture);
    }
  }
  glDeleteBuffers(1, &tm->vbo);
  glDeleteVertexArrays(1, &tm->vao);
  curl_global_cleanup();
  free(tm);
}
