#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stb/stb_image.h"
#include "viz/gl_platform.h"
#include "viz/tile_map.h"
#include "viz/tile_pipeline.h"
#include "viz/viz_math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
  ct->valid = false;
  return ct;
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
  TileDecodedImage image;

  if (!tm || budget <= 0) return;

  int processed = 0;
  tm->last_upload_count = 0;
  while (processed < budget && TilePipeline_take_decoded(tm, &image)) {
    CachedTile *ct = cache_find(tm, image.x, image.y, image.z);
    if (!ct) {
      ct = cache_evict(tm);
    }
    upload_cached_tile_texture(ct, image.pixels, image.width, image.height);
    ct->x = image.x;
    ct->y = image.y;
    ct->z = image.z;
    ct->last_used_frame = tm->frame_counter;
    ct->valid = true;
    stbi_image_free(image.pixels);
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

  if (!TilePipeline_init(tm)) {
    glDeleteBuffers(1, &tm->vbo);
    glDeleteVertexArrays(1, &tm->vao);
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
                  double zoom_degrees, MapProjectionMode projection_mode,
                  int viewport_w, int viewport_h,
                  int upload_budget) {
  if (center_lat == 0.0 && center_lng == 0.0) {
    tm->last_upload_count = 0;
    return;  // no data yet
  }

  tm->frame_counter++;

  // Progress any in-flight downloads (non-blocking)
  TilePipeline_poll_downloads(tm);

  int z = osm_zoom_from_degrees(zoom_degrees);
  int cx, cy;
  latlon_to_tile(center_lat, center_lng, z, &cx, &cy);

  int n_tiles = (int)pow(2.0, z);
  double cos_center = cos(center_lat * M_PI / 180.0);
  double aspect = (double)viewport_h / (double)viewport_w;
  double half_w = zoom_degrees;
  double half_h = zoom_degrees * aspect;
  double basis_xx, basis_xy, basis_yx, basis_yy;
  double visible_half_span = half_w;

  float proj[16];
  build_map_projection(proj, projection_mode, half_w, half_h, 0.0, 0.0);

  glUseProgram(tm->program);
  glUniformMatrix4fv(tm->u_projection, 1, GL_FALSE, proj);

  // Dynamic grid radius: enough tiles to cover the viewport
  double tile_degrees = 360.0 / pow(2.0, z);
  VizMap_projection_basis(projection_mode, &basis_xx, &basis_xy, &basis_yx,
                          &basis_yy);
  (void)basis_xx;
  (void)basis_yx;
  (void)basis_yy;
  if (projection_mode == MAP_PROJECTION_ISOMETRIC) {
    visible_half_span = fmax(half_h, half_w + fabs(basis_xy) * half_h);
  }
  int tiles_needed = (int)ceil(2.0 * visible_half_span / tile_degrees) + 1;
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
        TilePipeline_start_download(tm, tx, ty, z);
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

  TilePipeline_accumulate_stats(tm, out_stats);
}

void TileMap_free(TileMap *tm) {
  if (!tm) return;
  TilePipeline_shutdown(tm);

  for (int i = 0; i < TILE_CACHE_SIZE; i++) {
    if (tm->cache[i].texture) {
      glDeleteTextures(1, &tm->cache[i].texture);
    }
  }
  glDeleteBuffers(1, &tm->vbo);
  glDeleteVertexArrays(1, &tm->vao);
  free(tm);
}
