#ifndef VIZ_PROGRESS_BAR_H
#define VIZ_PROGRESS_BAR_H

#include "viz/gl_platform.h"

typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint program;
  GLint u_projection;
} ProgressBar;

ProgressBar ProgressBar_create(GLuint program);
void ProgressBar_draw(ProgressBar *pb, double progress);
void ProgressBar_draw_pause_icon(ProgressBar *pb);
void ProgressBar_free(ProgressBar *pb);

#endif
