#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "viz/gps_trace.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

static void assert_close(double expected, double actual, double tolerance) {
  printf("%.6f ~= %.6f\n", expected, actual);
  if (actual < expected - tolerance || actual > expected + tolerance) {
    fprintf(stderr, "!!! Assertion failed: expected %.9f but got %.9f\n",
            expected, actual);
    exit(EXIT_FAILURE);
  }
}

static GpsTrace make_trace_with_capacity(size_t capacity) {
  GpsTrace gt;

  memset(&gt, 0, sizeof(gt));
  gt.capacity = capacity;
  gt.imu_meta_capacity = capacity;
  if (capacity == 0) return gt;

  gt.lats = calloc(capacity, sizeof(double));
  gt.lngs = calloc(capacity, sizeof(double));
  gt.imu_meta = calloc(capacity, sizeof(ImuPointMeta));
  if (!gt.lats || !gt.lngs || !gt.imu_meta) {
    fprintf(stderr, "Allocation failed while building test trace\n");
    free(gt.lats);
    free(gt.lngs);
    free(gt.imu_meta);
    exit(EXIT_FAILURE);
  }
  return gt;
}

static void free_trace_arrays(GpsTrace *gt) {
  if (!gt) return;
  free(gt->lats);
  free(gt->lngs);
  free(gt->imu_meta);
}

static void test_push_grows_and_preserves_existing_points(void) {
  GpsTrace gt = make_trace_with_capacity(1);
  ImuPointMeta imu = {
      .motion = MOTION_RUNNING,
      .heading_rad = 1.25f,
      .pitch_rad = -0.5f,
      .has_imu = true,
  };

  GpsTrace_push(&gt, 12.5, 34.5, NULL);
  ASSERT(gt.count == 1, 1, (int)gt.count);
  ASSERT(gt.capacity == 1, 1, (int)gt.capacity);
  ASSERT(!gt.has_any_imu, 0, gt.has_any_imu ? 1 : 0);
  ASSERT(gt.imu_meta[0].has_imu == false, 0, gt.imu_meta[0].has_imu ? 1 : 0);

  GpsTrace_push(&gt, 13.5, 35.5, &imu);

  ASSERT(gt.count == 2, 2, (int)gt.count);
  ASSERT(gt.capacity >= 2, 1, gt.capacity >= 2 ? 1 : 0);
  ASSERT(gt.imu_meta_capacity == gt.capacity, (int)gt.capacity,
         (int)gt.imu_meta_capacity);
  ASSERT(gt.has_any_imu, 1, gt.has_any_imu ? 1 : 0);
  assert_close(12.5, gt.lats[0], 1e-9);
  assert_close(34.5, gt.lngs[0], 1e-9);
  assert_close(13.5, gt.lats[1], 1e-9);
  assert_close(35.5, gt.lngs[1], 1e-9);
  ASSERT(gt.imu_meta[1].has_imu, 1, gt.imu_meta[1].has_imu ? 1 : 0);
  ASSERT(gt.imu_meta[1].motion == MOTION_RUNNING, MOTION_RUNNING,
         gt.imu_meta[1].motion);
  ASSERT(gt.dirty, 1, gt.dirty ? 1 : 0);

  free_trace_arrays(&gt);
}

static void test_push_bootstraps_zero_capacity_trace(void) {
  GpsTrace gt = make_trace_with_capacity(0);

  GpsTrace_push(&gt, 1.0, 2.0, NULL);

  ASSERT(gt.count == 1, 1, (int)gt.count);
  ASSERT(gt.capacity >= 1, 1, gt.capacity >= 1 ? 1 : 0);
  ASSERT(gt.imu_meta_capacity == gt.capacity, (int)gt.capacity,
         (int)gt.imu_meta_capacity);
  assert_close(1.0, gt.lats[0], 1e-9);
  assert_close(2.0, gt.lngs[0], 1e-9);

  free_trace_arrays(&gt);
}

static void test_push_rejects_unrepresentable_growth(void) {
  GpsTrace gt;

  memset(&gt, 0, sizeof(gt));
  gt.count = SIZE_MAX / 2 + 1;
  gt.capacity = gt.count;
  gt.imu_meta_capacity = gt.count;

  GpsTrace_push(&gt, 1.0, 2.0, NULL);

  ASSERT(gt.count == SIZE_MAX / 2 + 1, 1, 1);
  ASSERT(gt.capacity == SIZE_MAX / 2 + 1, 1, 1);
  ASSERT(gt.imu_meta_capacity == SIZE_MAX / 2 + 1, 1, 1);
  ASSERT(!gt.dirty, 0, gt.dirty ? 1 : 0);
}

int main(void) {
  RUN_TEST(test_push_grows_and_preserves_existing_points);
  RUN_TEST(test_push_bootstraps_zero_capacity_trace);
  RUN_TEST(test_push_rejects_unrepresentable_growth);
  return 0;
}
