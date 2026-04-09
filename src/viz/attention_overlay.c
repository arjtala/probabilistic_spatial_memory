#include "viz/attention_overlay.h"
#include "viz/viz_math.h"

static void update_textured_quad_aspect(GLuint vbo, int video_w, int video_h,
                                        int viewport_w, int viewport_h) {
  float quad[16];
  if (!compute_aspect_quad(quad, video_w, video_h, viewport_w, viewport_h)) return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(quad), quad);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

AttentionOverlay AttentionOverlay_create(GLuint program) {
  AttentionOverlay ao;
  ao.program = program;
  ao.u_opacity = glGetUniformLocation(program, "u_opacity");
  ao.u_colormap = glGetUniformLocation(program, "u_colormap");
  ao.has_data = false;
  ao.attn_size = 0;

  float quad[] = {
      -1.0f, -1.0f, 0.0f, 1.0f,
       1.0f, -1.0f, 1.0f, 1.0f,
      -1.0f,  1.0f, 0.0f, 0.0f,
       1.0f,  1.0f, 1.0f, 0.0f,
  };

  glGenVertexArrays(1, &ao.vao);
  glGenBuffers(1, &ao.vbo);
  glBindVertexArray(ao.vao);
  glBindBuffer(GL_ARRAY_BUFFER, ao.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);

  glGenTextures(1, &ao.texture);
  glBindTexture(GL_TEXTURE_2D, ao.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);

  return ao;
}

void AttentionOverlay_update_aspect(AttentionOverlay *ao, int video_w,
                                    int video_h, int viewport_w,
                                    int viewport_h) {
  if (!ao) return;
  update_textured_quad_aspect(ao->vbo, video_w, video_h, viewport_w, viewport_h);
}

void AttentionOverlay_upload(AttentionOverlay *ao, float *raw_map, size_t size) {
  size_t n;
  float min_val, max_val, range;

  if (!ao || !raw_map || size == 0) return;
  n = size * size;

  min_val = raw_map[0];
  max_val = raw_map[0];
  for (size_t i = 1; i < n; i++) {
    if (raw_map[i] < min_val) min_val = raw_map[i];
    if (raw_map[i] > max_val) max_val = raw_map[i];
  }

  range = max_val - min_val;
  if (range > 1e-8f) {
    for (size_t i = 0; i < n; i++) {
      raw_map[i] = (raw_map[i] - min_val) / range;
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      raw_map[i] = 0.0f;
    }
  }

  glBindTexture(GL_TEXTURE_2D, ao->texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, (GLsizei)size, (GLsizei)size, 0,
               GL_RED, GL_FLOAT, raw_map);
  glBindTexture(GL_TEXTURE_2D, 0);

  ao->has_data = true;
  ao->attn_size = size;
}

void AttentionOverlay_draw(AttentionOverlay *ao, int colormap) {
  if (!ao || !ao->has_data) return;

  glUseProgram(ao->program);
  glUniform1f(ao->u_opacity, 0.5f);
  glUniform1i(ao->u_colormap, colormap);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, ao->texture);
  glBindVertexArray(ao->vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
}

void AttentionOverlay_free(AttentionOverlay *ao) {
  if (!ao) return;
  glDeleteTextures(1, &ao->texture);
  glDeleteBuffers(1, &ao->vbo);
  glDeleteVertexArrays(1, &ao->vao);
}
