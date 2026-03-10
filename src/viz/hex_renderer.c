#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <h3/h3api.h>
#include "viz/hex_renderer.h"

// Viridis colormap: dark purple (0) -> teal (0.5) -> yellow (1.0)
// Polynomial approximation (Matt Zucker / matplotlib)
static void count_to_color(double t, float *r, float *g, float *b, float *a) {
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;

  // Horner form: c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))))
  double cr = -5.4355 + t * 4.7764;
  cr = 6.2283 + t * cr;
  cr = -4.6342 + t * cr;
  cr = -0.3309 + t * cr;
  cr = 0.1051 + t * cr;
  cr = 0.2777 + t * cr;

  double cg = 4.6459 + t * (-13.7451);
  cg = 14.1799 + t * cg;
  cg = -5.7991 + t * cg;
  cg = 0.2148 + t * cg;
  cg = 1.4046 + t * cg;
  cg = 0.0054 + t * cg;

  double cb = 26.3124 + t * (-65.3530);
  cb = 56.6906 + t * cb;
  cb = -19.3324 + t * cb;
  cb = 0.0951 + t * cb;
  cb = 1.3846 + t * cb;
  cb = 0.3341 + t * cb;

  // Clamp to [0,1]
  if (cr < 0.0) cr = 0.0; if (cr > 1.0) cr = 1.0;
  if (cg < 0.0) cg = 0.0; if (cg > 1.0) cg = 1.0;
  if (cb < 0.0) cb = 0.0; if (cb > 1.0) cb = 1.0;

  *r = (float)cr;
  *g = (float)cg;
  *b = (float)cb;
  *a = 0.7f + 0.3f * (float)t;
}

HexRenderer *HexRenderer_new(GLuint program) {
  HexRenderer *hr = calloc(1, sizeof(HexRenderer));
  if (!hr) return NULL;

  hr->program = program;
  hr->u_projection = glGetUniformLocation(program, "u_projection");
  hr->zoom = 0.005;  // ~0.005 degrees ≈ 500m at equator
  hr->vertex_count = 0;

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

void HexRenderer_update(HexRenderer *hr, SpatialMemory *sm) {
  size_t tile_count = SpatialMemory_tile_count(sm);
  if (tile_count == 0) {
    hr->vertex_count = 0;
    return;
  }

  // First pass: find max count and compute center for projection
  double max_count = 1.0;
  double sum_lat = 0.0, sum_lng = 0.0;
  size_t n_tiles = 0;

  HashTableIterator it = HashTable_iterator(sm->tiles);
  while (HashTable_next(&it)) {
    Tile *tile = (Tile *)it.value;
    double count = Tile_query(tile, sm->capacity - 1);
    if (count > max_count) max_count = count;

    LatLng center;
    cellToLatLng(tile->cellId, &center);
    sum_lat += radsToDegs(center.lat);
    sum_lng += radsToDegs(center.lng);
    n_tiles++;
  }

  if (n_tiles > 0) {
    hr->center_lat = sum_lat / (double)n_tiles;
    hr->center_lng = sum_lng / (double)n_tiles;
  }

  // Each hex has at most MAX_CELL_BNDRY_VERTS boundary vertices
  // Triangle fan: center + N boundary verts -> N triangles -> 3*N vertices
  // Max 10 boundary verts per hex -> 30 verts per hex
  size_t max_verts = tile_count * 30;
  float *verts = malloc(max_verts * 6 * sizeof(float));
  if (!verts) return;
  size_t vi = 0;

  double cos_center = cos(hr->center_lat * M_PI / 180.0);

  it = HashTable_iterator(sm->tiles);
  while (HashTable_next(&it)) {
    Tile *tile = (Tile *)it.value;
    double count = Tile_query(tile, sm->capacity - 1);
    double t = count / max_count;

    float r, g, b, a;
    count_to_color(t, &r, &g, &b, &a);

    // Fade alpha based on recency: check current window vs total
    double current = Tile_query(tile, 0);
    if (current < 1.0 && count > 0.0) {
      a *= 0.4f + 0.6f * (float)(current / count);
    }

    CellBoundary boundary;
    cellToBoundary(tile->cellId, &boundary);

    // Compute hex center in projected coordinates
    LatLng cell_center;
    cellToLatLng(tile->cellId, &cell_center);
    float cx = (float)((radsToDegs(cell_center.lng) - hr->center_lng) * cos_center);
    float cy = (float)(radsToDegs(cell_center.lat) - hr->center_lat);

    // Emit triangle fan as individual triangles
    for (int i = 0; i < boundary.numVerts; i++) {
      int next = (i + 1) % boundary.numVerts;

      float bx0 = (float)((radsToDegs(boundary.verts[i].lng) - hr->center_lng) * cos_center);
      float by0 = (float)(radsToDegs(boundary.verts[i].lat) - hr->center_lat);
      float bx1 = (float)((radsToDegs(boundary.verts[next].lng) - hr->center_lng) * cos_center);
      float by1 = (float)(radsToDegs(boundary.verts[next].lat) - hr->center_lat);

      // Triangle: center, v[i], v[i+1]
      // Vertex 0: center
      verts[vi * 6 + 0] = cx;
      verts[vi * 6 + 1] = cy;
      verts[vi * 6 + 2] = r;
      verts[vi * 6 + 3] = g;
      verts[vi * 6 + 4] = b;
      verts[vi * 6 + 5] = a;
      vi++;

      // Vertex 1: boundary[i]
      verts[vi * 6 + 0] = bx0;
      verts[vi * 6 + 1] = by0;
      verts[vi * 6 + 2] = r;
      verts[vi * 6 + 3] = g;
      verts[vi * 6 + 4] = b;
      verts[vi * 6 + 5] = a;
      vi++;

      // Vertex 2: boundary[i+1]
      verts[vi * 6 + 0] = bx1;
      verts[vi * 6 + 1] = by1;
      verts[vi * 6 + 2] = r;
      verts[vi * 6 + 3] = g;
      verts[vi * 6 + 4] = b;
      verts[vi * 6 + 5] = a;
      vi++;
    }
  }

  hr->vertex_count = vi;

  glBindBuffer(GL_ARRAY_BUFFER, hr->vbo);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(vi * 6 * sizeof(float)), verts, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  free(verts);
}

void HexRenderer_draw(HexRenderer *hr, int viewport_w, int viewport_h) {
  if (hr->vertex_count == 0) return;

  glUseProgram(hr->program);

  // Build orthographic projection matrix
  // Map [center - zoom, center + zoom] to [-1, 1]
  double aspect = (double)viewport_h / (double)viewport_w;
  double half_w = hr->zoom;
  double half_h = hr->zoom * aspect;

  // Column-major orthographic matrix
  float proj[16] = {0};
  proj[0] = (float)(1.0 / half_w);
  proj[5] = (float)(1.0 / half_h);
  proj[10] = -1.0f;
  proj[15] = 1.0f;

  glUniformMatrix4fv(hr->u_projection, 1, GL_FALSE, proj);

  glBindVertexArray(hr->vao);
  glDrawArrays(GL_TRIANGLES, 0, (GLsizei)hr->vertex_count);
  glBindVertexArray(0);
}

void HexRenderer_free(HexRenderer *hr) {
  if (!hr) return;
  glDeleteBuffers(1, &hr->vbo);
  glDeleteVertexArrays(1, &hr->vao);
  free(hr);
}
