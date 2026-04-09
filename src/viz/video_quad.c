#include "viz/video_quad.h"
#include "viz/viz_math.h"

static void update_textured_quad_aspect(GLuint vbo, int video_w, int video_h,
                                        int viewport_w, int viewport_h) {
  float quad[16];
  if (!compute_aspect_quad(quad, video_w, video_h, viewport_w, viewport_h)) return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(quad), quad);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

VideoQuad VideoQuad_create(GLuint program) {
  VideoQuad vq = {0};
  vq.program = program;

  float quad[] = {
      -1.0f, -1.0f, 0.0f, 1.0f,
       1.0f, -1.0f, 1.0f, 1.0f,
      -1.0f,  1.0f, 0.0f, 0.0f,
       1.0f,  1.0f, 1.0f, 0.0f,
  };

  glGenVertexArrays(1, &vq.vao);
  glGenBuffers(1, &vq.vbo);
  glBindVertexArray(vq.vao);
  glBindBuffer(GL_ARRAY_BUFFER, vq.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);

  glGenTextures(1, &vq.texture);
  glBindTexture(GL_TEXTURE_2D, vq.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);

  return vq;
}

void VideoQuad_update_aspect(VideoQuad *vq, int video_w, int video_h,
                             int viewport_w, int viewport_h) {
  if (!vq) return;
  update_textured_quad_aspect(vq->vbo, video_w, video_h, viewport_w, viewport_h);
}

void VideoQuad_upload(VideoQuad *vq, uint8_t *rgb, int w, int h) {
  if (!vq || !rgb) return;
  glBindTexture(GL_TEXTURE_2D, vq->texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if (vq->texture_w != w || vq->texture_h != h) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 rgb);
    vq->texture_w = w;
    vq->texture_h = h;
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE,
                    rgb);
  }
  glBindTexture(GL_TEXTURE_2D, 0);
}

void VideoQuad_draw(VideoQuad *vq) {
  if (!vq) return;
  glUseProgram(vq->program);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, vq->texture);
  glBindVertexArray(vq->vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
}

void VideoQuad_free(VideoQuad *vq) {
  if (!vq) return;
  glDeleteTextures(1, &vq->texture);
  glDeleteBuffers(1, &vq->vbo);
  glDeleteVertexArrays(1, &vq->vao);
}
