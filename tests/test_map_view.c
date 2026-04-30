#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "viz/map_view.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void assert_close(double expected, double actual, double tolerance) {
  printf("%.9f ~= %.9f\n", expected, actual);
  if (fabs(expected - actual) > tolerance) {
    fprintf(stderr, "!!! Assertion failed: expected %.12f but got %.12f\n",
            expected, actual);
    exit(EXIT_FAILURE);
  }
}

static double safe_cos_lat(double lat) {
  double value = cos(lat * M_PI / 180.0);

  if (fabs(value) < 1e-6) {
    return value < 0.0 ? -1e-6 : 1e-6;
  }
  return value;
}

static void projection_basis(MapProjectionMode mode, double *m00, double *m01,
                             double *m10, double *m11) {
  switch (mode) {
  case MAP_PROJECTION_ISOMETRIC:
    *m00 = 1.0;
    *m01 = 0.55;
    *m10 = 0.0;
    *m11 = 0.62;
    break;
  case MAP_PROJECTION_ORTHOGRAPHIC:
  default:
    *m00 = 1.0;
    *m01 = 0.0;
    *m10 = 0.0;
    *m11 = 1.0;
    break;
  }
}

static void screen_point_to_geo(MapProjectionMode mode,
                                double center_lat, double center_lng,
                                double zoom, int viewport_w, int viewport_h,
                                double map_x, double map_y, double *out_lat,
                                double *out_lng) {
  double half_w = zoom;
  double half_h = zoom * (double)viewport_h / (double)viewport_w;
  double m00, m01, m10, m11;
  double det, nx, ny, px, py, wx, wy;

  projection_basis(mode, &m00, &m01, &m10, &m11);
  det = m00 * m11 - m01 * m10;
  nx = (map_x / (double)viewport_w) * 2.0 - 1.0;
  ny = 1.0 - (map_y / (double)viewport_h) * 2.0;
  px = nx * (fabs(m00) * half_w + fabs(m01) * half_h);
  py = ny * (fabs(m10) * half_w + fabs(m11) * half_h);
  wx = (m11 * px - m01 * py) / det;
  wy = (-m10 * px + m00 * py) / det;
  *out_lat = center_lat + wy;
  *out_lng = center_lng + wx / safe_cos_lat(center_lat);
}

static void test_clamp_zoom_limits(void) {
  assert_close(0.0001, VizMap_clamp_zoom(0.0), 1e-12);
  assert_close(1.0, VizMap_clamp_zoom(2.0), 1e-12);
  assert_close(0.25, VizMap_clamp_zoom(0.25), 1e-12);
}

static void test_zoom_about_cursor_preserves_focus_point(void) {
  const double center_lat = 37.484;
  const double center_lng = -122.148;
  const double zoom = 0.3;
  const int viewport_w = 800;
  const int viewport_h = 600;
  const double map_x = 620.0;
  const double map_y = 140.0;
  double focus_lat_before, focus_lng_before;
  double new_center_lat, new_center_lng, new_zoom;
  double focus_lat_after, focus_lng_after;

  screen_point_to_geo(MAP_PROJECTION_ORTHOGRAPHIC,
                      center_lat, center_lng, zoom, viewport_w, viewport_h,
                      map_x, map_y, &focus_lat_before, &focus_lng_before);
  ASSERT(VizMap_zoom_about_viewport_point(center_lat, center_lng, zoom,
                                          MAP_PROJECTION_ORTHOGRAPHIC, 0.8,
                                          viewport_w, viewport_h, map_x, map_y,
                                          &new_center_lat, &new_center_lng,
                                          &new_zoom),
         1, 1);

  screen_point_to_geo(MAP_PROJECTION_ORTHOGRAPHIC,
                      new_center_lat, new_center_lng, new_zoom, viewport_w,
                      viewport_h, map_x, map_y, &focus_lat_after,
                      &focus_lng_after);
  assert_close(focus_lat_before, focus_lat_after, 1e-9);
  assert_close(focus_lng_before, focus_lng_after, 1e-9);
}

static void test_zoom_about_center_keeps_center_fixed(void) {
  double new_center_lat, new_center_lng, new_zoom;

  ASSERT(VizMap_zoom_about_viewport_point(10.0, 20.0, 0.5,
                                          MAP_PROJECTION_ORTHOGRAPHIC,
                                          1.25, 1000, 500,
                                          500.0, 250.0, &new_center_lat,
                                          &new_center_lng, &new_zoom),
         1, 1);
  assert_close(10.0, new_center_lat, 1e-12);
  assert_close(20.0, new_center_lng, 1e-12);
  assert_close(0.625, new_zoom, 1e-12);
}

static void test_pan_center_translates_by_view_scale(void) {
  double new_center_lat, new_center_lng;
  double expected_lat;
  double expected_lng;

  ASSERT(VizMap_pan_center(10.0, 20.0, 0.5,
                           MAP_PROJECTION_ORTHOGRAPHIC,
                           800, 600, 40.0, -30.0,
                           &new_center_lat, &new_center_lng),
         1, 1);

  expected_lat = 10.0 + (-30.0 * (2.0 * 0.5 * (600.0 / 800.0) / 600.0));
  expected_lng = 20.0 + (-40.0 * (2.0 * 0.5 / 800.0)) / safe_cos_lat(10.0);
  assert_close(expected_lat, new_center_lat, 1e-12);
  assert_close(expected_lng, new_center_lng, 1e-12);
}

static void test_follow_step_is_smooth_and_bounded(void) {
  double out_lat, out_lng;
  double alpha = 1.0 - exp(-8.0 * 0.25);

  VizMap_step_follow(10.0, 20.0, 14.0, 30.0, 8.0, 0.25, &out_lat, &out_lng);
  assert_close(10.0 + (14.0 - 10.0) * alpha, out_lat, 1e-12);
  assert_close(20.0 + (30.0 - 20.0) * alpha, out_lng, 1e-12);
  ASSERT(out_lat > 10.0 && out_lat < 14.0, 1, out_lat > 10.0 && out_lat < 14.0);
  ASSERT(out_lng > 20.0 && out_lng < 30.0, 1, out_lng > 20.0 && out_lng < 30.0);
}

static void test_isometric_zoom_about_cursor_preserves_focus_point(void) {
  const double center_lat = 37.484;
  const double center_lng = -122.148;
  const double zoom = 0.3;
  const int viewport_w = 800;
  const int viewport_h = 600;
  const double map_x = 620.0;
  const double map_y = 140.0;
  double focus_lat_before, focus_lng_before;
  double new_center_lat, new_center_lng, new_zoom;
  double focus_lat_after, focus_lng_after;

  screen_point_to_geo(MAP_PROJECTION_ISOMETRIC,
                      center_lat, center_lng, zoom, viewport_w, viewport_h,
                      map_x, map_y, &focus_lat_before, &focus_lng_before);
  ASSERT(VizMap_zoom_about_viewport_point(center_lat, center_lng, zoom,
                                          MAP_PROJECTION_ISOMETRIC, 0.8,
                                          viewport_w, viewport_h, map_x, map_y,
                                          &new_center_lat, &new_center_lng,
                                          &new_zoom),
         1, 1);
  screen_point_to_geo(MAP_PROJECTION_ISOMETRIC,
                      new_center_lat, new_center_lng, new_zoom, viewport_w,
                      viewport_h, map_x, map_y, &focus_lat_after,
                      &focus_lng_after);
  assert_close(focus_lat_before, focus_lat_after, 1e-9);
  assert_close(focus_lng_before, focus_lng_after, 1e-9);
}

static void test_invalid_viewport_is_rejected(void) {
  double out_lat = 0.0, out_lng = 0.0, out_zoom = 0.0;

  ASSERT(!VizMap_zoom_about_viewport_point(0.0, 0.0, 0.2,
                                           MAP_PROJECTION_ORTHOGRAPHIC,
                                           0.8, 0, 500, 10.0, 10.0, &out_lat,
                                           &out_lng, &out_zoom),
         0, 0);
  ASSERT(!VizMap_pan_center(0.0, 0.0, 0.2, MAP_PROJECTION_ORTHOGRAPHIC, 500,
                            0, 1.0, 1.0, &out_lat, &out_lng),
         0, 0);
}

int main(void) {
  RUN_TEST(test_clamp_zoom_limits);
  RUN_TEST(test_zoom_about_cursor_preserves_focus_point);
  RUN_TEST(test_zoom_about_center_keeps_center_fixed);
  RUN_TEST(test_pan_center_translates_by_view_scale);
  RUN_TEST(test_isometric_zoom_about_cursor_preserves_focus_point);
  RUN_TEST(test_follow_step_is_smooth_and_bounded);
  RUN_TEST(test_invalid_viewport_is_rejected);
  return 0;
}
