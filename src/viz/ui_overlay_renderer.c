#include "viz/ui_overlay_renderer.h"
#include "viz/viz_math.h"

UiOverlayRenderer UiOverlayRenderer_create(GLuint program) {
  UiOverlayRenderer renderer;

  renderer.program = program;
  renderer.u_projection = glGetUniformLocation(program, "u_projection");
  renderer.vao = 0;
  renderer.vbo = 0;

  glGenVertexArrays(1, &renderer.vao);
  glGenBuffers(1, &renderer.vbo);

  glBindVertexArray(renderer.vao);
  glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo);
  glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return renderer;
}

void UiOverlayRenderer_draw(UiOverlayRenderer *renderer,
                            const UiOverlayMesh *mesh) {
  float projection[16];
  const float *vertices;
  size_t vertex_count;

  if (!renderer || !mesh || mesh->viewport_w <= 0 || mesh->viewport_h <= 0) {
    return;
  }

  vertices = UiOverlay_vertices(mesh);
  vertex_count = UiOverlay_vertex_count(mesh);
  if (!vertices || vertex_count == 0) return;

  build_ortho_projection(projection, (double)mesh->viewport_w / 2.0,
                         (double)mesh->viewport_h / 2.0,
                         (double)mesh->viewport_w / 2.0,
                         (double)mesh->viewport_h / 2.0);

  glUseProgram(renderer->program);
  glUniformMatrix4fv(renderer->u_projection, 1, GL_FALSE, projection);
  glBindBuffer(GL_ARRAY_BUFFER, renderer->vbo);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(vertex_count * 6 * sizeof(float)),
               vertices, GL_DYNAMIC_DRAW);
  glBindVertexArray(renderer->vao);
  glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertex_count);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void UiOverlayRenderer_free(UiOverlayRenderer *renderer) {
  if (!renderer) return;
  if (renderer->vbo) glDeleteBuffers(1, &renderer->vbo);
  if (renderer->vao) glDeleteVertexArrays(1, &renderer->vao);
  renderer->vao = 0;
  renderer->vbo = 0;
}
