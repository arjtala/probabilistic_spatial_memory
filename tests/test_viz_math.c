#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "viz/viz_math.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void assert_close(double expected, double actual, double tolerance) {
  printf("%.4f ~= %.4f\n", expected, actual);
  if (fabs(expected - actual) > tolerance) {
    fprintf(stderr, "!!! Assertion failed: expected %.6f but got %.6f\n",
            expected, actual);
    exit(EXIT_FAILURE);
  }
}

static void assert_in_unit_interval(float value) {
  printf("[0,1] contains %.4f\n", value);
  if (value < 0.0f || value > 1.0f) {
    fprintf(stderr, "!!! Assertion failed: %.6f is outside [0, 1]\n", value);
    exit(EXIT_FAILURE);
  }
}

static void test_count_to_color(void) {
  float r, g, b, a;

  // Negative t clamps to 0 -> black, fully unsaturated.
  count_to_color(-1.0, &r, &g, &b, &a);
  assert_close(0.0, r, 1e-4);
  assert_close(0.0, g, 1e-4);
  assert_close(0.0, b, 1e-4);
  assert_close(0.7, a, 1e-4);

  // t=0.5 sits halfway through segment 3 (green->cyan): G=1, B=0.5.
  count_to_color(0.5, &r, &g, &b, &a);
  assert_close(0.0, r, 1e-4);
  assert_close(1.0, g, 1e-4);
  assert_close(0.5, b, 1e-4);
  assert_close(0.85, a, 1e-4);
  assert_in_unit_interval(r);
  assert_in_unit_interval(g);
  assert_in_unit_interval(b);

  // t>1 clamps to 1 -> white at the top of the cube tour.
  count_to_color(2.0, &r, &g, &b, &a);
  assert_close(1.0, r, 1e-4);
  assert_close(1.0, g, 1e-4);
  assert_close(1.0, b, 1e-4);
  assert_close(1.0, a, 1e-4);
}

static void test_classify_motion(void) {
  float stationary[3] = {0.0f, 0.0f, 9.81f};
  float walking[3] = {0.0f, 0.0f, 11.0f};
  float running[3] = {0.0f, 0.0f, 14.5f};

  ASSERT(classify_motion(stationary) == MOTION_STATIONARY,
         MOTION_STATIONARY, classify_motion(stationary));
  ASSERT(classify_motion(walking) == MOTION_WALKING,
         MOTION_WALKING, classify_motion(walking));
  ASSERT(classify_motion(running) == MOTION_RUNNING,
         MOTION_RUNNING, classify_motion(running));
}

static void test_normalize_angle(void) {
  assert_close(-(M_PI - 0.1), normalize_angle((float)M_PI + 0.1f), 1e-4);
  assert_close(M_PI - 0.1, normalize_angle(-(float)M_PI - 0.1f), 1e-4);
  assert_close(0.5, normalize_angle(0.5f), 1e-6);
}

static void test_estimate_speed(void) {
  assert_close(0.0, estimate_speed(MOTION_STATIONARY, 0.0f), 1e-6);
  assert_close(1.2, estimate_speed(MOTION_WALKING, 0.5f), 1e-6);
  assert_close(2.5, estimate_speed(MOTION_WALKING, 3.0f), 1e-6);
  assert_close(2.5, estimate_speed(MOTION_RUNNING, 3.0f), 1e-6);
  assert_close(6.0, estimate_speed(MOTION_RUNNING, 10.0f), 1e-6);
  assert_close(6.0, estimate_speed(MOTION_RUNNING, 99.0f), 1e-6);
}

static void test_osm_zoom_from_degrees(void) {
  ASSERT(osm_zoom_from_degrees(500.0) == 0, 0, osm_zoom_from_degrees(500.0));
  ASSERT(osm_zoom_from_degrees(5.0) == 6, 6, osm_zoom_from_degrees(5.0));
  ASSERT(osm_zoom_from_degrees(0.000001) == 19, 19,
         osm_zoom_from_degrees(0.000001));
}

static void test_latlon_to_tile(void) {
  int tx = -1, ty = -1;
  latlon_to_tile(0.0, 0.0, 1, &tx, &ty);
  ASSERT(tx == 1, 1, tx);
  ASSERT(ty == 1, 1, ty);

  latlon_to_tile(37.484, -122.148, 10, &tx, &ty);
  ASSERT(tx == 164, 164, tx);
  ASSERT(ty == 396, 396, ty);
}

static void test_compute_aspect_quad(void) {
  float quad[16];

  ASSERT(!compute_aspect_quad(NULL, 320, 240, 800, 600), 1, 1);
  ASSERT(compute_aspect_quad(quad, 200, 100, 100, 100), 1, 1);
  assert_close(-1.0, quad[0], 1e-6);
  assert_close(-0.5, quad[1], 1e-6);
  assert_close(1.0, quad[12], 1e-6);
  assert_close(0.5, quad[13], 1e-6);
}

static void test_build_identity_matrix(void) {
  float matrix[16];
  build_identity_matrix(matrix);

  assert_close(1.0, matrix[0], 1e-6);
  assert_close(1.0, matrix[5], 1e-6);
  assert_close(1.0, matrix[10], 1e-6);
  assert_close(1.0, matrix[15], 1e-6);
  assert_close(0.0, matrix[1], 1e-6);
  assert_close(0.0, matrix[12], 1e-6);
}

static void test_build_ortho_projection(void) {
  float matrix[16];
  build_ortho_projection(matrix, 2.0, 1.0, 1.0, -0.5);

  assert_close(0.5, matrix[0], 1e-6);
  assert_close(1.0, matrix[5], 1e-6);
  assert_close(-1.0, matrix[10], 1e-6);
  assert_close(-0.5, matrix[12], 1e-6);
  assert_close(0.5, matrix[13], 1e-6);
  assert_close(1.0, matrix[15], 1e-6);
}

int main(void) {
  RUN_TEST(test_count_to_color);
  RUN_TEST(test_classify_motion);
  RUN_TEST(test_normalize_angle);
  RUN_TEST(test_estimate_speed);
  RUN_TEST(test_osm_zoom_from_degrees);
  RUN_TEST(test_latlon_to_tile);
  RUN_TEST(test_compute_aspect_quad);
  RUN_TEST(test_build_identity_matrix);
  RUN_TEST(test_build_ortho_projection);
  return 0;
}
