#ifndef VIZ_UI_OVERLAY_H
#define VIZ_UI_OVERLAY_H

#include <stdbool.h>
#include <stddef.h>

#define UI_OVERLAY_VERTEX_FLOATS 6

typedef struct {
  float r;
  float g;
  float b;
  float a;
} UiOverlayColor;

typedef struct {
  float *data;
  size_t vertex_count;
  size_t vertex_capacity;
  int viewport_w;
  int viewport_h;
} UiOverlayMesh;

/* Vertices are emitted as interleaved {x, y, r, g, b, a} floats. */
void UiOverlay_init(UiOverlayMesh *mesh);
void UiOverlay_reset(UiOverlayMesh *mesh, int viewport_w, int viewport_h);
void UiOverlay_free(UiOverlayMesh *mesh);

const float *UiOverlay_vertices(const UiOverlayMesh *mesh);
size_t UiOverlay_vertex_count(const UiOverlayMesh *mesh);

/* Text layout uses top-left pixel coordinates and a fixed built-in bitmap font. */
float UiOverlay_text_height(float scale);
float UiOverlay_text_width(const char *text, float scale, float tracking);

bool UiOverlay_add_rect(UiOverlayMesh *mesh, float x, float y, float w, float h,
                        UiOverlayColor color);
bool UiOverlay_add_hgradient(UiOverlayMesh *mesh, float x, float y, float w,
                             float h, UiOverlayColor left,
                             UiOverlayColor right);
bool UiOverlay_add_text(UiOverlayMesh *mesh, float x, float y, float scale,
                        float tracking, UiOverlayColor color,
                        const char *text);

#endif
