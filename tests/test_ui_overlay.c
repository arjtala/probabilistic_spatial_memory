#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>

#include "viz/ui_overlay.h"

static void fail(const char *message) {
  fprintf(stderr, "FAIL: %s\n", message);
  exit(EXIT_FAILURE);
}

static void assert_true(int condition, const char *message) {
  if (!condition) {
    fail(message);
  }
}

static void assert_close(double expected, double actual, double tolerance,
                         const char *message) {
  double diff = expected - actual;
  if (diff < 0.0) diff = -diff;
  if (diff > tolerance) {
    fprintf(stderr, "expected %.6f but got %.6f\n", expected, actual);
    fail(message);
  }
}

static void assert_has_vertex(const UiOverlayMesh *mesh, double expected_x,
                              double expected_y, double tolerance,
                              const char *message) {
  const float *verts = UiOverlay_vertices(mesh);
  size_t vertex_count = UiOverlay_vertex_count(mesh);

  for (size_t i = 0; i < vertex_count; i++) {
    size_t base = i * UI_OVERLAY_VERTEX_FLOATS;
    double dx = expected_x - verts[base];
    double dy = expected_y - verts[base + 1];
    if (dx < 0.0) dx = -dx;
    if (dy < 0.0) dy = -dy;
    if (dx <= tolerance && dy <= tolerance) {
      return;
    }
  }

  fail(message);
}

static void test_rect_emits_two_triangles_in_pixel_space(void) {
  UiOverlayMesh mesh;
  UiOverlayColor color = {0.2f, 0.3f, 0.4f, 0.5f};
  const float *verts;

  UiOverlay_init(&mesh);
  UiOverlay_reset(&mesh, 200, 100);
  assert_true(UiOverlay_add_rect(&mesh, 10.0f, 20.0f, 30.0f, 15.0f, color),
              "rectangle emission failed");
  assert_true(UiOverlay_vertex_count(&mesh) == 6,
              "rectangle should emit two triangles");

  verts = UiOverlay_vertices(&mesh);
  assert_close(10.0, verts[0], 1e-6, "first vertex x mismatch");
  assert_close(80.0, verts[1], 1e-6, "first vertex y mismatch");
  assert_close(40.0, verts[UI_OVERLAY_VERTEX_FLOATS + 0], 1e-6,
               "second vertex x mismatch");
  assert_close(80.0, verts[UI_OVERLAY_VERTEX_FLOATS + 1], 1e-6,
               "second vertex y mismatch");
  assert_close(10.0, verts[2 * UI_OVERLAY_VERTEX_FLOATS + 0], 1e-6,
               "third vertex x mismatch");
  assert_close(65.0, verts[2 * UI_OVERLAY_VERTEX_FLOATS + 1], 1e-6,
               "third vertex y mismatch");

  UiOverlay_free(&mesh);
}

static void test_gradient_uses_distinct_left_and_right_colors(void) {
  UiOverlayMesh mesh;
  UiOverlayColor left = {0.0f, 0.1f, 0.2f, 0.3f};
  UiOverlayColor right = {0.9f, 0.8f, 0.7f, 0.6f};
  const float *verts;

  UiOverlay_init(&mesh);
  UiOverlay_reset(&mesh, 100, 100);
  assert_true(
      UiOverlay_add_hgradient(&mesh, 0.0f, 0.0f, 20.0f, 10.0f, left, right),
      "gradient emission failed");

  verts = UiOverlay_vertices(&mesh);
  assert_close(0.0, verts[2], 1e-6, "left vertex red mismatch");
  assert_close(0.1, verts[3], 1e-6, "left vertex green mismatch");
  assert_close(0.9, verts[UI_OVERLAY_VERTEX_FLOATS + 2], 1e-6,
               "right vertex red mismatch");
  assert_close(0.8, verts[UI_OVERLAY_VERTEX_FLOATS + 3], 1e-6,
               "right vertex green mismatch");

  UiOverlay_free(&mesh);
}

static void test_text_metrics_and_font_coverage(void) {
  UiOverlayMesh mesh;

  assert_true((int)UiOverlay_text_width("AB", 2.0f, 0.0f) == 22,
              "unexpected text width");
  assert_true((int)UiOverlay_text_height(3.0f) == 21,
              "unexpected text height");

  UiOverlay_init(&mesh);
  UiOverlay_reset(&mesh, 120, 80);
  assert_true(UiOverlay_add_text(&mesh, 5.0f, 6.0f, 1.0f, 0.0f,
                                 (UiOverlayColor){1.0f, 1.0f, 1.0f, 1.0f},
                                 "aA9:/?"),
              "text emission failed");
  assert_true(UiOverlay_vertex_count(&mesh) > 0,
              "text should emit vertices");
  assert_true((UiOverlay_vertex_count(&mesh) % 6) == 0,
              "text vertices should be triangle-aligned");

  UiOverlay_free(&mesh);
}

static void test_multiline_text_advances_rows(void) {
  UiOverlayMesh mesh;

  UiOverlay_init(&mesh);
  UiOverlay_reset(&mesh, 100, 100);
  assert_true(UiOverlay_add_text(&mesh, 0.0f, 0.0f, 1.0f, 0.0f,
                                 (UiOverlayColor){1.0f, 0.0f, 0.0f, 1.0f},
                                 "I\nI"),
              "multiline text emission failed");

  assert_has_vertex(&mesh, 1.0, 100.0, 1e-6, "missing first-line vertex");
  assert_has_vertex(&mesh, 1.0, 91.0, 1e-6,
                    "missing second-line vertex at advanced row");

  UiOverlay_free(&mesh);
}

int main(void) {
  test_rect_emits_two_triangles_in_pixel_space();
  test_gradient_uses_distinct_left_and_right_colors();
  test_text_metrics_and_font_coverage();
  test_multiline_text_advances_rows();
  printf("ui_overlay tests passed\n");
  return 0;
}
