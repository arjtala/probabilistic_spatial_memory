#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "viz/imu_processor.h"
#include "viz/viz_math.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define R_EARTH 6371000.0   // meters, matches src/viz/imu_processor.c

static void assert_close(double expected, double actual, double tolerance) {
  printf("%.8f ~= %.8f\n", expected, actual);
  if (fabs(expected - actual) > tolerance) {
    fprintf(stderr, "!!! Assertion failed: expected %.10f but got %.10f\n",
            expected, actual);
    exit(EXIT_FAILURE);
  }
}

// Heading integrates gyro projected onto the (up-pointing) gravity estimate.
// With gravity along +Z, yaw_rate = -gyro[2]; a constant gyro[2] = -0.5 rad/s
// over two dt=1.0s steps must accumulate to +1.0 rad. The very first update
// only seeds state (dt=0), so it does not contribute to the heading.
static void test_heading_integration(void) {
  ImuProcessor *proc = ImuProcessor_new(0.5f);
  float accel[3] = {0.0f, 0.0f, 9.81f};
  float gyro[3] = {0.0f, 0.0f, -0.5f};

  ImuPointMeta m0 = ImuProcessor_update(proc, accel, gyro, 0.0, 0.0, 0.0);
  assert_close(0.0, m0.heading_rad, 1e-6);

  ImuPointMeta m1 = ImuProcessor_update(proc, accel, gyro, 1.0, 0.0, 0.0);
  assert_close(0.5, m1.heading_rad, 1e-4);

  ImuPointMeta m2 = ImuProcessor_update(proc, accel, gyro, 2.0, 0.0, 0.0);
  assert_close(1.0, m2.heading_rad, 1e-4);

  ImuProcessor_free(proc);
}

// Dead-reckoning displacement against a hand-computed value. With alpha=1.0
// the complementary blend is pure dead-reckoning (no GPS pull), heading stays
// 0 (gyro=0), so motion is due north only. accel magnitude 10.31 gives
// deviation 0.5 -> MOTION_WALKING -> speed 1.2 m/s. Over dt=1.0s:
//   dlat = 1.2 * 1.0 * cos(0) / R_EARTH * (180/pi)
//   dlng = 0  (sin(0) == 0)
static void test_dead_reckoning_displacement(void) {
  ImuProcessor *proc = ImuProcessor_new(1.0f);
  float accel[3] = {0.0f, 0.0f, 10.31f};
  float gyro[3] = {0.0f, 0.0f, 0.0f};

  // First update seeds the dead-reckoned position at the GPS origin.
  ImuProcessor_update(proc, accel, gyro, 0.0, 0.0, 0.0);

  // Second update integrates one step of northward walking.
  ImuProcessor_update(proc, accel, gyro, 1.0, 0.0, 0.0);

  double lat = 0.0, lng = 0.0;
  ImuProcessor_get_blended_position(proc, &lat, &lng);

  double expected_lat = 1.2 * 1.0 * 1.0 / R_EARTH * (180.0 / M_PI);
  assert_close(expected_lat, lat, 1e-8);
  assert_close(0.0, lng, 1e-8);

  ImuProcessor_free(proc);
}

// The dt gate accepts 0 < dt <= 5.0 and rejects dt > 5.0. On the low side of
// the boundary (dt == 5.0) heading integrates; just past it (dt > 5.0) the
// step is dropped and dead-reckoning snaps back to the raw GPS fix.
static void test_dt_gating_boundary(void) {
  float accel[3] = {0.0f, 0.0f, 9.81f};
  float gyro[3] = {0.0f, 0.0f, -0.5f};

  // dt == 5.0 is inside the gate: heading integrates by yaw_rate * dt.
  ImuProcessor *inside = ImuProcessor_new(1.0f);
  ImuProcessor_update(inside, accel, gyro, 0.0, 0.0, 0.0);
  ImuPointMeta in_meta = ImuProcessor_update(inside, accel, gyro, 5.0, 0.0, 0.0);
  assert_close(2.5, in_meta.heading_rad, 1e-4);  // 0.5 rad/s * 5.0 s
  ImuProcessor_free(inside);

  // dt > 5.0 is outside the gate: heading is frozen and the dead-reckoned
  // position is reset to the raw GPS coordinate handed in on this update.
  ImuProcessor *outside = ImuProcessor_new(1.0f);
  ImuProcessor_update(outside, accel, gyro, 0.0, 0.0, 0.0);
  ImuPointMeta out_meta =
      ImuProcessor_update(outside, accel, gyro, 6.0, 12.0, 34.0);
  assert_close(0.0, out_meta.heading_rad, 1e-6);

  double lat = 0.0, lng = 0.0;
  ImuProcessor_get_blended_position(outside, &lat, &lng);
  assert_close(12.0, lat, 1e-9);
  assert_close(34.0, lng, 1e-9);
  ImuProcessor_free(outside);
}

int main(void) {
  RUN_TEST(test_heading_integration);
  RUN_TEST(test_dead_reckoning_displacement);
  RUN_TEST(test_dt_gating_boundary);
  return 0;
}
