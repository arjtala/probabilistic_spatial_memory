#include <string.h>
#include "viz/progress_bar.h"
#include "viz/viz_math.h"

ProgressBar ProgressBar_create(GLuint program) {
  ProgressBar pb;
  pb.program = program;
  pb.u_projection = glGetUniformLocation(program, "u_projection");

  glGenVertexArrays(1, &pb.vao);
  glGenBuffers(1, &pb.vbo);

  glBindVertexArray(pb.vao);
  glBindBuffer(GL_ARRAY_BUFFER, pb.vbo);
  glBufferData(GL_ARRAY_BUFFER, 12 * 6 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
  return pb;
}

void ProgressBar_draw(ProgressBar *pb, double progress) {
  float identity[16];
  float fill_x;
  float y0 = -1.0f, y1 = -0.95f;
  float bg_r = 0.15f, bg_g = 0.15f, bg_b = 0.15f, bg_a = 0.6f;
  float fl_r = 0.2f, fl_g = 0.5f, fl_b = 1.0f, fl_a = 0.9f;
  float verts[12 * 6];

  if (!pb) return;
  if (progress < 0.0) progress = 0.0;
  if (progress > 1.0) progress = 1.0;

  fill_x = -1.0f + 2.0f * (float)progress;
  memcpy(verts, (float[12 * 6]){
      -1.0f, y0, bg_r, bg_g, bg_b, bg_a,
       1.0f, y0, bg_r, bg_g, bg_b, bg_a,
      -1.0f, y1, bg_r, bg_g, bg_b, bg_a,
      -1.0f, y1, bg_r, bg_g, bg_b, bg_a,
       1.0f, y0, bg_r, bg_g, bg_b, bg_a,
       1.0f, y1, bg_r, bg_g, bg_b, bg_a,
      -1.0f,  y0, fl_r, fl_g, fl_b, fl_a,
      fill_x, y0, fl_r, fl_g, fl_b, fl_a,
      -1.0f,  y1, fl_r, fl_g, fl_b, fl_a,
      -1.0f,  y1, fl_r, fl_g, fl_b, fl_a,
      fill_x, y0, fl_r, fl_g, fl_b, fl_a,
      fill_x, y1, fl_r, fl_g, fl_b, fl_a,
  }, sizeof(verts));

  build_identity_matrix(identity);

  glUseProgram(pb->program);
  glUniformMatrix4fv(pb->u_projection, 1, GL_FALSE, identity);

  glBindBuffer(GL_ARRAY_BUFFER, pb->vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);

  glBindVertexArray(pb->vao);
  glDrawArrays(GL_TRIANGLES, 0, 12);
  glBindVertexArray(0);
}

void ProgressBar_draw_pause_icon(ProgressBar *pb) {
  float identity[16];
  float bar_w = 0.04f, bar_h = 0.12f, gap = 0.035f;
  float cx = 0.0f, cy = 0.0f;
  float r = 1.0f, g = 1.0f, b = 1.0f, a = 0.7f;
  float verts[12 * 6];

  if (!pb) return;

  memcpy(verts, (float[12 * 6]){
      cx - gap - bar_w, cy - bar_h, r, g, b, a,
      cx - gap,         cy - bar_h, r, g, b, a,
      cx - gap - bar_w, cy + bar_h, r, g, b, a,
      cx - gap - bar_w, cy + bar_h, r, g, b, a,
      cx - gap,         cy - bar_h, r, g, b, a,
      cx - gap,         cy + bar_h, r, g, b, a,
      cx + gap,         cy - bar_h, r, g, b, a,
      cx + gap + bar_w, cy - bar_h, r, g, b, a,
      cx + gap,         cy + bar_h, r, g, b, a,
      cx + gap,         cy + bar_h, r, g, b, a,
      cx + gap + bar_w, cy - bar_h, r, g, b, a,
      cx + gap + bar_w, cy + bar_h, r, g, b, a,
  }, sizeof(verts));

  build_identity_matrix(identity);

  glUseProgram(pb->program);
  glUniformMatrix4fv(pb->u_projection, 1, GL_FALSE, identity);

  glBindBuffer(GL_ARRAY_BUFFER, pb->vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);

  glBindVertexArray(pb->vao);
  glDrawArrays(GL_TRIANGLES, 0, 12);
  glBindVertexArray(0);
}

void ProgressBar_draw_start_overlay(ProgressBar *pb) {
  float identity[16];
  float verts[30 * 6];
  float cy = 0.06f;
  float panel_x0 = -0.34f, panel_x1 = 0.34f;
  float panel_y0 = -0.24f, panel_y1 = 0.30f;
  float glow_x0 = -0.30f, glow_x1 = 0.30f;
  float glow_y0 = -0.18f, glow_y1 = 0.24f;
  float tri_left = -0.08f, tri_right = 0.12f, tri_half_h = 0.12f;
  float hint_bar_w = 0.032f, hint_bar_h = 0.07f, hint_gap = 0.024f;

  if (!pb) return;

  memcpy(verts, (float[30 * 6]){
      panel_x0, panel_y0, 0.02f, 0.04f, 0.08f, 0.60f,
      panel_x1, panel_y0, 0.02f, 0.04f, 0.08f, 0.60f,
      panel_x0, panel_y1, 0.02f, 0.04f, 0.08f, 0.60f,
      panel_x0, panel_y1, 0.02f, 0.04f, 0.08f, 0.60f,
      panel_x1, panel_y0, 0.02f, 0.04f, 0.08f, 0.60f,
      panel_x1, panel_y1, 0.02f, 0.04f, 0.08f, 0.60f,

      glow_x0, glow_y0, 0.16f, 0.30f, 0.55f, 0.16f,
      glow_x1, glow_y0, 0.16f, 0.30f, 0.55f, 0.16f,
      glow_x0, glow_y1, 0.16f, 0.30f, 0.55f, 0.16f,
      glow_x0, glow_y1, 0.16f, 0.30f, 0.55f, 0.16f,
      glow_x1, glow_y0, 0.16f, 0.30f, 0.55f, 0.16f,
      glow_x1, glow_y1, 0.16f, 0.30f, 0.55f, 0.16f,

      tri_left, cy - tri_half_h, 1.00f, 1.00f, 1.00f, 0.92f,
      tri_left, cy + tri_half_h, 1.00f, 1.00f, 1.00f, 0.92f,
      tri_right, cy,             1.00f, 1.00f, 1.00f, 0.92f,

      -hint_gap - hint_bar_w, -0.12f - hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
      -hint_gap,               -0.12f - hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
      -hint_gap - hint_bar_w, -0.12f + hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
      -hint_gap - hint_bar_w, -0.12f + hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
      -hint_gap,               -0.12f - hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
      -hint_gap,               -0.12f + hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,

       hint_gap,               -0.12f - hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
       hint_gap + hint_bar_w, -0.12f - hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
       hint_gap,               -0.12f + hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
       hint_gap,               -0.12f + hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
       hint_gap + hint_bar_w, -0.12f - hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
       hint_gap + hint_bar_w, -0.12f + hint_bar_h, 1.00f, 1.00f, 1.00f, 0.78f,
  }, sizeof(verts));

  build_identity_matrix(identity);

  glUseProgram(pb->program);
  glUniformMatrix4fv(pb->u_projection, 1, GL_FALSE, identity);

  glBindBuffer(GL_ARRAY_BUFFER, pb->vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);

  glBindVertexArray(pb->vao);
  glDrawArrays(GL_TRIANGLES, 0, 30);
  glBindVertexArray(0);
}

void ProgressBar_free(ProgressBar *pb) {
  if (!pb) return;
  glDeleteBuffers(1, &pb->vbo);
  glDeleteVertexArrays(1, &pb->vao);
}
