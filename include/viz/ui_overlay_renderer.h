#ifndef VIZ_UI_OVERLAY_RENDERER_H
#define VIZ_UI_OVERLAY_RENDERER_H

#include "viz/gl_platform.h"
#include "viz/ui_overlay.h"

typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint program;
  GLint u_projection;
} UiOverlayRenderer;

UiOverlayRenderer UiOverlayRenderer_create(GLuint program);
void UiOverlayRenderer_draw(UiOverlayRenderer *renderer,
                            const UiOverlayMesh *mesh);
void UiOverlayRenderer_free(UiOverlayRenderer *renderer);

#endif
