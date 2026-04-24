#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <h3/h3api.h>
#include "viz/hex_renderer.h"
#include "viz/viz_math.h"

typedef struct {
  HexRenderer *hr;
  SpatialMemory *sm;
  double max_value;
  double sum_lat;
  double sum_lng;
  size_t n_tiles;
} HexRendererStats;

typedef struct {
  HexRenderer *hr;
  SpatialMemory *sm;
  float *verts;
  size_t vi;
  double max_value;
  double cos_center;
} HexRendererBuild;

static double tile_heatmap_value(HexHeatmapMode mode, Tile *tile,
                                 SpatialMemory *sm, double *out_total,
                                 double *out_current) {
  double total = 0.0;
  double current = 0.0;

  if (!tile || !sm) return 0.0;

  total = Tile_query(tile, sm->capacity - 1);
  current = Tile_query(tile, 0);
  if (out_total) *out_total = total;
  if (out_current) *out_current = current;

  switch (mode) {
  case HEX_HEATMAP_MODE_CURRENT:
    return current;
  case HEX_HEATMAP_MODE_RECENCY:
    return (total > 0.0) ? current / total : 0.0;
  case HEX_HEATMAP_MODE_TOTAL:
  default:
    return total;
  }
}

static bool collect_hex_renderer_stats(H3Index cell_id, Tile *tile,
                                       void *user_data) {
  HexRendererStats *stats = (HexRendererStats *)user_data;
  (void)cell_id;
  if (!stats || !tile || !stats->sm) return false;

  double value = tile_heatmap_value(stats->hr->heatmap_mode, tile, stats->sm,
                                    NULL, NULL);
  if (value <= 0.0) return true;
  if (value > stats->max_value) stats->max_value = value;

  LatLng center;
  cellToLatLng(tile->cellId, &center);
  stats->sum_lat += radsToDegs(center.lat);
  stats->sum_lng += radsToDegs(center.lng);
  stats->n_tiles++;
  return true;
}

static bool append_hex_renderer_tile(H3Index cell_id, Tile *tile,
                                     void *user_data) {
  HexRendererBuild *build = (HexRendererBuild *)user_data;
  (void)cell_id;
  if (!build || !tile || !build->sm || !build->verts) return false;

  double total = 0.0;
  double current = 0.0;
  double value = tile_heatmap_value(build->hr->heatmap_mode, tile, build->sm,
                                    &total, &current);
  if (value <= 0.0) return true;
  double t = value / build->max_value;

  float r, g, b, a;
  count_to_color(t, &r, &g, &b, &a);

  if (build->hr->heatmap_mode == HEX_HEATMAP_MODE_TOTAL &&
      current < 1.0 && total > 0.0) {
    a *= 0.4f + 0.6f * (float)(current / total);
  }

  CellBoundary boundary;
  cellToBoundary(tile->cellId, &boundary);

  LatLng cell_center;
  cellToLatLng(tile->cellId, &cell_center);
  float cx = (float)((radsToDegs(cell_center.lng) - build->hr->center_lng) *
                     build->cos_center);
  float cy = (float)(radsToDegs(cell_center.lat) - build->hr->center_lat);

  for (int i = 0; i < boundary.numVerts; i++) {
    int next = (i + 1) % boundary.numVerts;

    float bx0 = (float)((radsToDegs(boundary.verts[i].lng) -
                         build->hr->center_lng) * build->cos_center);
    float by0 = (float)(radsToDegs(boundary.verts[i].lat) -
                        build->hr->center_lat);
    float bx1 = (float)((radsToDegs(boundary.verts[next].lng) -
                         build->hr->center_lng) * build->cos_center);
    float by1 = (float)(radsToDegs(boundary.verts[next].lat) -
                        build->hr->center_lat);

    build->verts[build->vi * 6 + 0] = cx;
    build->verts[build->vi * 6 + 1] = cy;
    build->verts[build->vi * 6 + 2] = r;
    build->verts[build->vi * 6 + 3] = g;
    build->verts[build->vi * 6 + 4] = b;
    build->verts[build->vi * 6 + 5] = a;
    build->vi++;

    build->verts[build->vi * 6 + 0] = bx0;
    build->verts[build->vi * 6 + 1] = by0;
    build->verts[build->vi * 6 + 2] = r;
    build->verts[build->vi * 6 + 3] = g;
    build->verts[build->vi * 6 + 4] = b;
    build->verts[build->vi * 6 + 5] = a;
    build->vi++;

    build->verts[build->vi * 6 + 0] = bx1;
    build->verts[build->vi * 6 + 1] = by1;
    build->verts[build->vi * 6 + 2] = r;
    build->verts[build->vi * 6 + 3] = g;
    build->verts[build->vi * 6 + 4] = b;
    build->verts[build->vi * 6 + 5] = a;
    build->vi++;
  }

  return true;
}

HexRenderer *HexRenderer_new(GLuint program) {
  HexRenderer *hr = calloc(1, sizeof(HexRenderer));
  if (!hr) return NULL;

  hr->program = program;
  hr->u_projection = glGetUniformLocation(program, "u_projection");
  hr->zoom = 0.005;  // ~0.005 degrees ≈ 500m at equator
  hr->vertex_count = 0;
  hr->verts = NULL;
  hr->verts_capacity = 0;
  // Sentinel guaranteed to miss on first draw; avoid NAN for -ffast-math safety.
  hr->cached_cos_lat = -999.0;
  hr->cached_cos_lat_key = -999.0;

  glGenVertexArrays(1, &hr->vao);
  glGenBuffers(1, &hr->vbo);

  glBindVertexArray(hr->vao);
  glBindBuffer(GL_ARRAY_BUFFER, hr->vbo);

  // Layout: vec2 position, vec4 color = 6 floats per vertex
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
  return hr;
}

bool HexRenderer_parse_heatmap_mode(const char *text, HexHeatmapMode *out_mode) {
  if (!text || !out_mode) return false;
  if (strcmp(text, "total") == 0) {
    *out_mode = HEX_HEATMAP_MODE_TOTAL;
    return true;
  }
  if (strcmp(text, "current") == 0) {
    *out_mode = HEX_HEATMAP_MODE_CURRENT;
    return true;
  }
  if (strcmp(text, "recency") == 0) {
    *out_mode = HEX_HEATMAP_MODE_RECENCY;
    return true;
  }
  return false;
}

const char *HexRenderer_heatmap_mode_name(HexHeatmapMode mode) {
  switch (mode) {
  case HEX_HEATMAP_MODE_CURRENT:
    return "current";
  case HEX_HEATMAP_MODE_RECENCY:
    return "recency";
  case HEX_HEATMAP_MODE_TOTAL:
  default:
    return "total";
  }
}

HexHeatmapMode HexRenderer_next_heatmap_mode(HexHeatmapMode mode) {
  int next = ((int)mode + 1) % (int)HEX_HEATMAP_MODE_COUNT;
  if (next < 0) next = 0;
  return (HexHeatmapMode)next;
}

void HexRenderer_set_heatmap_mode(HexRenderer *hr, HexHeatmapMode mode) {
  if (!hr) return;
  if (mode < 0 || mode >= HEX_HEATMAP_MODE_COUNT) {
    mode = HEX_HEATMAP_MODE_TOTAL;
  }
  hr->heatmap_mode = mode;
}

void HexRenderer_update(HexRenderer *hr, SpatialMemory *sm) {
  if (!hr || !sm) return;
  size_t tile_count = SpatialMemory_tile_count(sm);
  if (tile_count == 0) {
    hr->vertex_count = 0;
    return;
  }

  // First pass: find max count and compute center for projection
  HexRendererStats stats = {.hr = hr, .sm = sm, .max_value = 0.0};
  if (!SpatialMemory_for_each_tile(sm, collect_hex_renderer_stats, &stats)) {
    hr->vertex_count = 0;
    return;
  }
  if (stats.n_tiles == 0) {
    hr->vertex_count = 0;
    return;
  }
  if (stats.max_value <= 0.0) {
    stats.max_value = 1.0;
  }

  if (stats.n_tiles > 0) {
    hr->center_lat = stats.sum_lat / (double)stats.n_tiles;
    hr->center_lng = stats.sum_lng / (double)stats.n_tiles;
  }

  // Each hex has at most MAX_CELL_BNDRY_VERTS boundary vertices
  // Triangle fan: center + N boundary verts -> N triangles -> 3*N vertices
  // Max 10 boundary verts per hex -> 30 verts per hex
  size_t max_verts = tile_count * 30;
  size_t required = max_verts * 6;
  if (hr->verts_capacity < required) {
    float *new_verts = realloc(hr->verts, required * sizeof(float));
    if (!new_verts) {
      hr->vertex_count = 0;
      return;
    }
    hr->verts = new_verts;
    hr->verts_capacity = required;
  }
  HexRendererBuild build = {
      .hr = hr,
      .sm = sm,
      .verts = hr->verts,
      .vi = 0,
      .max_value = stats.max_value,
      .cos_center = cos(hr->center_lat * M_PI / 180.0),
  };
  if (!SpatialMemory_for_each_tile(sm, append_hex_renderer_tile, &build)) {
    hr->vertex_count = 0;
    return;
  }

  hr->vertex_count = build.vi;

  glBindBuffer(GL_ARRAY_BUFFER, hr->vbo);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(build.vi * 6 * sizeof(float)),
               hr->verts, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void HexRenderer_draw(HexRenderer *hr, int viewport_w, int viewport_h,
                      double map_center_lat, double map_center_lng) {
  if (hr->vertex_count == 0) return;

  glUseProgram(hr->program);

  // Build orthographic projection matrix
  // Map [center - zoom, center + zoom] to [-1, 1]
  double aspect = (double)viewport_h / (double)viewport_w;
  double half_w = hr->zoom;
  double half_h = hr->zoom * aspect;

  // Hex vertices are baked relative to hr->center; translate to map_center.
  // Cache cos(center_lat * pi/180) keyed by center_lat to avoid recomputing
  // every draw (called on every frame).
  if (hr->center_lat != hr->cached_cos_lat_key) {
    hr->cached_cos_lat = cos(hr->center_lat * M_PI / 180.0);
    hr->cached_cos_lat_key = hr->center_lat;
  }
  double offset_x = (map_center_lng - hr->center_lng) * hr->cached_cos_lat;
  double offset_y = map_center_lat - hr->center_lat;

  float proj[16];
  build_ortho_projection(proj, half_w, half_h, offset_x, offset_y);

  glUniformMatrix4fv(hr->u_projection, 1, GL_FALSE, proj);

  glBindVertexArray(hr->vao);
  glDrawArrays(GL_TRIANGLES, 0, (GLsizei)hr->vertex_count);
  glBindVertexArray(0);
}

void HexRenderer_free(HexRenderer *hr) {
  if (!hr) return;
  glDeleteBuffers(1, &hr->vbo);
  glDeleteVertexArrays(1, &hr->vao);
  free(hr->verts);
  free(hr);
}
