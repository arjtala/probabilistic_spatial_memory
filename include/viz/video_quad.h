#ifndef VIZ_VIDEO_QUAD_H
#define VIZ_VIDEO_QUAD_H

#include <stdint.h>
#include "viz/gl_platform.h"

typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint texture;
  GLuint program;
  int texture_w;
  int texture_h;
} VideoQuad;

VideoQuad VideoQuad_create(GLuint program);
void VideoQuad_update_aspect(VideoQuad *vq, int video_w, int video_h,
                             int viewport_w, int viewport_h);
void VideoQuad_upload(VideoQuad *vq, uint8_t *rgb, int w, int h);
void VideoQuad_draw(VideoQuad *vq);
void VideoQuad_free(VideoQuad *vq);

#endif
