#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stb/stb_image.h"
#include "viz/gl_platform.h"
#include "viz/tile_disk_cache.h"
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

static bool queue_tile_from_disk(TileMap *tm, PendingDownload *pd,
                                 int x, int y, int z) {
  MemBuffer disk_buf = {0};

  if (!tm || !pd) return false;
  if (!TileDiskCache_read(&tm->disk_cache, x, y, z,
                          &disk_buf.data, &disk_buf.size)) {
    return false;
  }

  pthread_mutex_lock(&tm->pending_mutex);
  if (!pd->active || pd->easy || pd->ready || pd->decoding || pd->decoded ||
      pd->x != x || pd->y != y || pd->z != z) {
    pthread_mutex_unlock(&tm->pending_mutex);
    free(disk_buf.data);
    return false;
  }
  pd->buf = disk_buf;
  pd->active = false;
  pd->ready = true;
  pthread_cond_signal(&tm->pending_cond);
  pthread_mutex_unlock(&tm->pending_mutex);
  return true;
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
  ct->valid = false;
  return ct;
}

// ---- Async download helpers ----
static bool slot_is_busy(const PendingDownload *pd) {
  return pd && (pd->active || pd->ready || pd->decoding || pd->decoded);
}

static bool is_pending_locked(const TileMap *tm, int x, int y, int z) {
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    const PendingDownload *pd = &tm->pending[i];
    if (slot_is_busy(pd) &&
        pd->x == x && pd->y == y && pd->z == z) {
      return true;
    }
  }
  return false;
}

static PendingDownload *find_free_slot_locked(TileMap *tm) {
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    if (!slot_is_busy(&tm->pending[i])) return &tm->pending[i];
  }
  return NULL;
}

static void clear_pending_download(PendingDownload *pd) {
  if (!pd) return;
  free(pd->buf.data);
  free(pd->decoded_pixels);
  pd->buf.data = NULL;
  pd->buf.size = 0;
  pd->decoded_pixels = NULL;
  pd->decoded_w = 0;
  pd->decoded_h = 0;
  pd->easy = NULL;
  pd->x = 0;
  pd->y = 0;
  pd->z = 0;
  pd->active = false;
  pd->ready = false;
  pd->decoding = false;
  pd->decoded = false;
}

static bool has_decode_work(const TileMap *tm) {
  if (!tm) return false;
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    const PendingDownload *pd = &tm->pending[i];
    if (pd->ready && !pd->decoding && !pd->decoded) {
      return true;
    }
  }
  return false;
}

static void *tile_decode_worker(void *userdata) {
  TileMap *tm = (TileMap *)userdata;

  for (;;) {
    PendingDownload *pd = NULL;
    uint8_t *compressed = NULL;
    size_t compressed_size = 0;
    int slot = -1;

    pthread_mutex_lock(&tm->pending_mutex);
    while (!tm->shutdown_worker && !has_decode_work(tm)) {
      pthread_cond_wait(&tm->pending_cond, &tm->pending_mutex);
    }
    if (tm->shutdown_worker) {
      pthread_mutex_unlock(&tm->pending_mutex);
      return NULL;
    }

    for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
      PendingDownload *candidate = &tm->pending[i];
      if (candidate->ready && !candidate->decoding && !candidate->decoded) {
        pd = candidate;
        slot = i;
        pd->decoding = true;
        pd->ready = false;
        compressed = pd->buf.data;
        compressed_size = pd->buf.size;
        pd->buf.data = NULL;
        pd->buf.size = 0;
        break;
      }
    }
    pthread_mutex_unlock(&tm->pending_mutex);

    if (!pd || !compressed || compressed_size == 0) {
      free(compressed);
      pthread_mutex_lock(&tm->pending_mutex);
      if (slot >= 0) {
        clear_pending_download(&tm->pending[slot]);
      }
      pthread_mutex_unlock(&tm->pending_mutex);
      continue;
    }

    int w = 0, h = 0, channels = 0;
    uint8_t *decoded = stbi_load_from_memory(compressed, (int)compressed_size,
                                             &w, &h, &channels, 4);
    free(compressed);

    pthread_mutex_lock(&tm->pending_mutex);
    pd = &tm->pending[slot];
    pd->decoding = false;
    if (decoded) {
      pd->decoded_pixels = decoded;
      pd->decoded_w = w;
      pd->decoded_h = h;
      pd->decoded = true;
    } else {
      clear_pending_download(pd);
    }
    pthread_mutex_unlock(&tm->pending_mutex);
  }
}

static bool append_text(char *dst, size_t dst_size, size_t *dst_len,
                        const char *text) {
  size_t text_len = strlen(text);
  if (*dst_len + text_len + 1 > dst_size) return false;
  memcpy(dst + *dst_len, text, text_len);
  *dst_len += text_len;
  dst[*dst_len] = '\0';
  return true;
}

static bool append_int(char *dst, size_t dst_size, size_t *dst_len, int value) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%d", value);
  return append_text(dst, dst_size, dst_len, buf);
}

static bool format_tile_url(const TileMap *tm, int x, int y, int z,
                            char *out, size_t out_size) {
  size_t out_len = 0;
  const char *cursor;
  const char *subdomains = "abcd";
  char subdomain[2] = {subdomains[(x + y + z) & 3], '\0'};

  if (!tm || !out || out_size == 0 || tm->url_template[0] == '\0') return false;
  cursor = tm->url_template;
  out[0] = '\0';

  while (*cursor) {
    if (strncmp(cursor, "{x}", 3) == 0) {
      if (!append_int(out, out_size, &out_len, x)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{y}", 3) == 0) {
      if (!append_int(out, out_size, &out_len, y)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{z}", 3) == 0) {
      if (!append_int(out, out_size, &out_len, z)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{s}", 3) == 0) {
      if (!append_text(out, out_size, &out_len, subdomain)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{api_key}", 9) == 0) {
      if (!append_text(out, out_size, &out_len, tm->api_key)) return false;
      cursor += 9;
    } else {
      char ch[2] = {*cursor, '\0'};
      if (!append_text(out, out_size, &out_len, ch)) return false;
      cursor++;
    }
  }

  return true;
}

static void start_download(TileMap *tm, int x, int y, int z) {
  char url[TILE_MAP_URL_CAP];
  CURL *easy = NULL;
  PendingDownload *pd = NULL;
  bool slot_reserved = false;

  pthread_mutex_lock(&tm->pending_mutex);
  if (is_pending_locked(tm, x, y, z)) {
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  pd = find_free_slot_locked(tm);
  if (!pd) {
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  clear_pending_download(pd);
  pd->x = x;
  pd->y = y;
  pd->z = z;
  pd->active = true;
  slot_reserved = true;
  pthread_mutex_unlock(&tm->pending_mutex);

  if (slot_reserved && queue_tile_from_disk(tm, pd, x, y, z)) {
    return;
  }

  if (!format_tile_url(tm, x, y, z, url, sizeof(url))) {
    pthread_mutex_lock(&tm->pending_mutex);
    clear_pending_download(pd);
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  easy = curl_easy_init();
  if (!easy) {
    pthread_mutex_lock(&tm->pending_mutex);
    clear_pending_download(pd);
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  pthread_mutex_lock(&tm->pending_mutex);
  if (!pd->active || pd->easy || pd->ready || pd->decoding || pd->decoded ||
      pd->x != x || pd->y != y || pd->z != z) {
    pthread_mutex_unlock(&tm->pending_mutex);
    curl_easy_cleanup(easy);
    return;
  }
  pd->easy = easy;
  pthread_mutex_unlock(&tm->pending_mutex);

  curl_easy_setopt(easy, CURLOPT_URL, url);
  curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(easy, CURLOPT_WRITEDATA, &pd->buf);
  curl_easy_setopt(easy, CURLOPT_USERAGENT, "psm-viz/1.0");
  curl_easy_setopt(easy, CURLOPT_TIMEOUT, 10L);
  curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(easy, CURLOPT_PRIVATE, pd);

  if (curl_multi_add_handle(tm->multi, easy) != CURLM_OK) {
    pthread_mutex_lock(&tm->pending_mutex);
    if (pd->easy == easy) {
      clear_pending_download(pd);
    }
    pthread_mutex_unlock(&tm->pending_mutex);
    curl_easy_cleanup(easy);
  }
}

static void poll_downloads(TileMap *tm) {
  curl_multi_perform(tm->multi, &tm->running_transfers);

  CURLMsg *msg;
  int msgs_left;
  while ((msg = curl_multi_info_read(tm->multi, &msgs_left))) {
    if (msg->msg != CURLMSG_DONE) continue;

    CURL *easy = msg->easy_handle;
    PendingDownload *pd = NULL;
    bool ready;
    curl_easy_getinfo(easy, CURLINFO_PRIVATE, &pd);
    if (!pd) continue;

    // Cleanup
    curl_multi_remove_handle(tm->multi, easy);
    curl_easy_cleanup(easy);
    ready = (msg->data.result == CURLE_OK && pd->buf.size > 0);
    if (ready) {
      TileDiskCache_write(&tm->disk_cache, pd->x, pd->y, pd->z,
                          pd->buf.data, pd->buf.size);
    }
    pthread_mutex_lock(&tm->pending_mutex);
    pd->easy = NULL;
    pd->active = false;
    pd->ready = ready;
    if (pd->ready) {
      pthread_cond_signal(&tm->pending_cond);
    } else {
      clear_pending_download(pd);
    }
    pthread_mutex_unlock(&tm->pending_mutex);
  }
}

static void upload_cached_tile_texture(CachedTile *ct, const uint8_t *img,
                                       int w, int h) {
  if (!ct || !img || w <= 0 || h <= 0) return;

  if (!ct->texture) {
    glGenTextures(1, &ct->texture);
    glBindTexture(GL_TEXTURE_2D, ct->texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glBindTexture(GL_TEXTURE_2D, ct->texture);
  }

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if (ct->texture_w != w || ct->texture_h != h) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, img);
    ct->texture_w = w;
    ct->texture_h = h;
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                    img);
  }
  glBindTexture(GL_TEXTURE_2D, 0);
}

static void process_ready_downloads(TileMap *tm, int budget) {
  if (!tm || budget <= 0) return;

  int processed = 0;
  tm->last_upload_count = 0;
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS && processed < budget; i++) {
    uint8_t *decoded = NULL;
    int w = 0;
    int h = 0;
    int x = 0;
    int y = 0;
    int z = 0;

    pthread_mutex_lock(&tm->pending_mutex);
    PendingDownload *pd = &tm->pending[i];
    if (!pd->decoded || !pd->decoded_pixels) {
      pthread_mutex_unlock(&tm->pending_mutex);
      continue;
    }
    decoded = pd->decoded_pixels;
    pd->decoded_pixels = NULL;
    w = pd->decoded_w;
    h = pd->decoded_h;
    x = pd->x;
    y = pd->y;
    z = pd->z;
    clear_pending_download(pd);
    pthread_mutex_unlock(&tm->pending_mutex);

    CachedTile *ct = cache_find(tm, x, y, z);
    if (!ct) {
      ct = cache_evict(tm);
    }
    upload_cached_tile_texture(ct, decoded, w, h);
    ct->x = x;
    ct->y = y;
    ct->z = z;
    ct->last_used_frame = tm->frame_counter;
    ct->valid = true;
    stbi_image_free(decoded);
    processed++;
  }
  tm->last_upload_count = processed;
}

// ---- Public API ----
TileMap *TileMap_new(GLuint program, const char *style_name,
                     const char *url_template, const char *api_key) {
  TileMap *tm = calloc(1, sizeof(TileMap));
  if (!tm) return NULL;

  if (!style_name || !url_template ||
      snprintf(tm->style_name, sizeof(tm->style_name), "%s", style_name) >=
          (int)sizeof(tm->style_name) ||
      snprintf(tm->url_template, sizeof(tm->url_template), "%s", url_template) >=
          (int)sizeof(tm->url_template) ||
      (api_key &&
       snprintf(tm->api_key, sizeof(tm->api_key), "%s", api_key) >=
           (int)sizeof(tm->api_key))) {
    free(tm);
    return NULL;
  }

  tm->program = program;
  tm->u_projection = glGetUniformLocation(program, "u_projection");
  tm->u_texture = glGetUniformLocation(program, "u_texture");
  tm->frame_counter = 0;
  tm->running_transfers = 0;
  if (pthread_mutex_init(&tm->pending_mutex, NULL) != 0) {
    free(tm);
    return NULL;
  }
  if (pthread_cond_init(&tm->pending_cond, NULL) != 0) {
    pthread_mutex_destroy(&tm->pending_mutex);
    free(tm);
    return NULL;
  }
  TileDiskCache_init(&tm->disk_cache, tm->style_name, tm->url_template);

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
  if (!tm->multi) {
    pthread_mutex_destroy(&tm->pending_mutex);
    pthread_cond_destroy(&tm->pending_cond);
    glDeleteBuffers(1, &tm->vbo);
    glDeleteVertexArrays(1, &tm->vao);
    curl_global_cleanup();
    free(tm);
    return NULL;
  }

  if (pthread_create(&tm->decode_thread, NULL, tile_decode_worker, tm) != 0) {
    curl_multi_cleanup(tm->multi);
    pthread_mutex_destroy(&tm->pending_mutex);
    pthread_cond_destroy(&tm->pending_cond);
    glDeleteBuffers(1, &tm->vbo);
    glDeleteVertexArrays(1, &tm->vao);
    curl_global_cleanup();
    free(tm);
    return NULL;
  }

  return tm;
}

void TileMap_configure_disk_cache(TileMap *tm, bool enabled, size_t max_bytes) {
  if (!tm) return;
  TileDiskCache_configure(&tm->disk_cache, enabled, max_bytes);
}

void TileMap_draw(TileMap *tm, double center_lat, double center_lng,
                  double zoom_degrees, int viewport_w, int viewport_h,
                  int upload_budget) {
  if (center_lat == 0.0 && center_lng == 0.0) {
    tm->last_upload_count = 0;
    return;  // no data yet
  }

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

  float proj[16];
  build_ortho_projection(proj, half_w, half_h, 0.0, 0.0);

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

  // Amortize PNG decode and texture upload work to keep map rendering responsive.
  if (upload_budget < 1) upload_budget = 1;
  if (upload_budget > TILE_MAP_MAX_PENDING_DOWNLOADS) {
    upload_budget = TILE_MAP_MAX_PENDING_DOWNLOADS;
  }
  process_ready_downloads(tm, upload_budget);
}

void TileMap_get_stats(TileMap *tm, TileMapStats *out_stats) {
  if (!tm || !out_stats) return;

  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->uploads_last_frame = tm->last_upload_count;
  out_stats->disk_cache_enabled = tm->disk_cache.enabled;
  out_stats->disk_cache_hits = tm->disk_cache.hits;
  out_stats->disk_cache_writes = tm->disk_cache.writes;
  out_stats->disk_cache_prunes = tm->disk_cache.prunes;
  out_stats->disk_cache_bytes = tm->disk_cache.bytes;
  out_stats->disk_cache_max_bytes = tm->disk_cache.max_bytes;

  for (int i = 0; i < TILE_CACHE_SIZE; i++) {
    if (tm->cache[i].valid) {
      out_stats->cache_tiles++;
    }
  }

  pthread_mutex_lock(&tm->pending_mutex);
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    const PendingDownload *pd = &tm->pending[i];
    if (pd->active) out_stats->active_downloads++;
    if (pd->ready) out_stats->ready_downloads++;
    if (pd->decoding) out_stats->decoding_downloads++;
    if (pd->decoded) out_stats->decoded_downloads++;
  }
  pthread_mutex_unlock(&tm->pending_mutex);
}

void TileMap_free(TileMap *tm) {
  if (!tm) return;
  pthread_mutex_lock(&tm->pending_mutex);
  tm->shutdown_worker = true;
  pthread_cond_signal(&tm->pending_cond);
  pthread_mutex_unlock(&tm->pending_mutex);
  pthread_join(tm->decode_thread, NULL);

  // Cleanup pending downloads
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    PendingDownload *pd = &tm->pending[i];
    if (pd->active && pd->easy) {
      curl_multi_remove_handle(tm->multi, pd->easy);
      curl_easy_cleanup(pd->easy);
    }
    clear_pending_download(pd);
  }
  if (tm->multi) curl_multi_cleanup(tm->multi);
  pthread_mutex_destroy(&tm->pending_mutex);
  pthread_cond_destroy(&tm->pending_cond);

  for (int i = 0; i < TILE_CACHE_SIZE; i++) {
    if (tm->cache[i].texture) {
      glDeleteTextures(1, &tm->cache[i].texture);
    }
  }
  glDeleteBuffers(1, &tm->vbo);
  glDeleteVertexArrays(1, &tm->vao);
  curl_global_cleanup();
  free(tm);
}
