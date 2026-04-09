#ifndef VIZ_ATTENTION_OVERLAY_H
#define VIZ_ATTENTION_OVERLAY_H

#include <stdbool.h>
#include <stddef.h>
#include "viz/gl_platform.h"

typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint texture;
  GLuint program;
  GLint u_opacity;
  GLint u_colormap;
  bool has_data;
  size_t attn_size;
} AttentionOverlay;

AttentionOverlay AttentionOverlay_create(GLuint program);
void AttentionOverlay_update_aspect(AttentionOverlay *ao, int video_w,
                                    int video_h, int viewport_w,
                                    int viewport_h);
void AttentionOverlay_upload(AttentionOverlay *ao, float *raw_map, size_t size);
void AttentionOverlay_draw(AttentionOverlay *ao, int colormap);
void AttentionOverlay_free(AttentionOverlay *ao);

#endif
