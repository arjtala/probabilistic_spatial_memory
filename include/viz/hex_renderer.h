#ifndef VIZ_HEX_RENDERER_H
#define VIZ_HEX_RENDERER_H

#include <OpenGL/gl3.h>
#include "core/spatial_memory.h"

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
} HexRenderer;

HexRenderer *HexRenderer_new(GLuint program);
void HexRenderer_update(HexRenderer *hr, SpatialMemory *sm);
void HexRenderer_draw(HexRenderer *hr, int viewport_w, int viewport_h,
                      double map_center_lat, double map_center_lng);
void HexRenderer_free(HexRenderer *hr);

#endif
