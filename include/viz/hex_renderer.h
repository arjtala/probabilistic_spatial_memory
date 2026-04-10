#ifndef VIZ_HEX_RENDERER_H
#define VIZ_HEX_RENDERER_H

#include "viz/gl_platform.h"
#include "core/spatial_memory.h"

typedef enum {
  HEX_HEATMAP_MODE_TOTAL = 0,
  HEX_HEATMAP_MODE_CURRENT = 1,
  HEX_HEATMAP_MODE_RECENCY = 2,
  HEX_HEATMAP_MODE_COUNT
} HexHeatmapMode;

typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint program;
  GLint u_projection;
  size_t vertex_count;
  // Viewport bounds (lat/lng in degrees) for projection
  double center_lat;
  double center_lng;
  double zoom;  // degrees of longitude visible in half-width
  double pan_offset_lat;
  double pan_offset_lng;
  HexHeatmapMode heatmap_mode;
} HexRenderer;

HexRenderer *HexRenderer_new(GLuint program);
bool HexRenderer_parse_heatmap_mode(const char *text, HexHeatmapMode *out_mode);
const char *HexRenderer_heatmap_mode_name(HexHeatmapMode mode);
HexHeatmapMode HexRenderer_next_heatmap_mode(HexHeatmapMode mode);
void HexRenderer_set_heatmap_mode(HexRenderer *hr, HexHeatmapMode mode);
void HexRenderer_update(HexRenderer *hr, SpatialMemory *sm);
void HexRenderer_draw(HexRenderer *hr, int viewport_w, int viewport_h,
                      double map_center_lat, double map_center_lng);
void HexRenderer_free(HexRenderer *hr);

#endif
