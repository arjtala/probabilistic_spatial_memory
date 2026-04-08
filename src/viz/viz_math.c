#include <math.h>
#include <string.h>
#include "viz/viz_math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GRAVITY 9.81f

void count_to_color(double t, float *r, float *g, float *b, float *a) {
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;

  // Horner form: c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))))
  double cr = -5.4355 + t * 4.7764;
  cr = 6.2283 + t * cr;
  cr = -4.6342 + t * cr;
  cr = -0.3309 + t * cr;
  cr = 0.1051 + t * cr;
  cr = 0.2777 + t * cr;

  double cg = 4.6459 + t * (-13.7451);
  cg = 14.1799 + t * cg;
  cg = -5.7991 + t * cg;
  cg = 0.2148 + t * cg;
  cg = 1.4046 + t * cg;
  cg = 0.0054 + t * cg;

  double cb = 26.3124 + t * (-65.3530);
  cb = 56.6906 + t * cb;
  cb = -19.3324 + t * cb;
  cb = 0.0951 + t * cb;
  cb = 1.3846 + t * cb;
  cb = 0.3341 + t * cb;

  if (cr < 0.0) cr = 0.0;
  if (cr > 1.0) cr = 1.0;
  if (cg < 0.0) cg = 0.0;
  if (cg > 1.0) cg = 1.0;
  if (cb < 0.0) cb = 0.0;
  if (cb > 1.0) cb = 1.0;

  *r = (float)cr;
  *g = (float)cg;
  *b = (float)cb;
  *a = 0.7f + 0.3f * (float)t;
}

MotionState classify_motion(const float accel[3]) {
  float mag = sqrtf(accel[0] * accel[0] + accel[1] * accel[1] +
                    accel[2] * accel[2]);
  float deviation = fabsf(mag - GRAVITY);
  if (deviation < 0.5f) return MOTION_STATIONARY;
  if (deviation < 3.0f) return MOTION_WALKING;
  return MOTION_RUNNING;
}

float normalize_angle(float angle) {
  while (angle > (float)M_PI) angle -= 2.0f * (float)M_PI;
  while (angle < -(float)M_PI) angle += 2.0f * (float)M_PI;
  return angle;
}

float estimate_speed(MotionState motion, float deviation) {
  switch (motion) {
  case MOTION_STATIONARY:
    return 0.0f;
  case MOTION_WALKING: {
    float t = (deviation - 0.5f) / 2.5f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return 1.2f + t * 1.3f;
  }
  case MOTION_RUNNING: {
    float t = (deviation - 3.0f) / 7.0f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return 2.5f + t * 3.5f;
  }
  }
  return 0.0f;
}

int osm_zoom_from_degrees(double zoom_degrees) {
  if (zoom_degrees <= 0.0) zoom_degrees = 0.001;
  double z = ceil(log2(360.0 / (2.0 * zoom_degrees)));
  if (z < 0) z = 0;
  if (z > 19) z = 19;
  return (int)z;
}

void latlon_to_tile(double lat, double lng, int z, int *tx, int *ty) {
  double n = pow(2.0, z);
  *tx = (int)floor((lng + 180.0) / 360.0 * n);
  double lat_rad = lat * M_PI / 180.0;
  *ty = (int)floor((1.0 - log(tan(lat_rad) + 1.0 / cos(lat_rad)) / M_PI) /
                   2.0 * n);
}

bool compute_aspect_quad(float quad[16], int video_w, int video_h,
                         int viewport_w, int viewport_h) {
  if (!quad || video_w <= 0 || video_h <= 0 ||
      viewport_w <= 0 || viewport_h <= 0) {
    return false;
  }

  double video_aspect = (double)video_w / (double)video_h;
  double viewport_aspect = (double)viewport_w / (double)viewport_h;

  float sx = 1.0f, sy = 1.0f;
  if (video_aspect > viewport_aspect) {
    sy = (float)(viewport_aspect / video_aspect);
  } else {
    sx = (float)(video_aspect / viewport_aspect);
  }

  const float textured_quad[16] = {
      -sx, -sy, 0.0f, 1.0f,
       sx, -sy, 1.0f, 1.0f,
      -sx,  sy, 0.0f, 0.0f,
       sx,  sy, 1.0f, 0.0f,
  };
  memcpy(quad, textured_quad, sizeof(textured_quad));
  return true;
}

void build_identity_matrix(float matrix[16]) {
  if (!matrix) return;
  for (int i = 0; i < 16; i++) matrix[i] = 0.0f;
  matrix[0] = 1.0f;
  matrix[5] = 1.0f;
  matrix[10] = 1.0f;
  matrix[15] = 1.0f;
}

void build_ortho_projection(float matrix[16], double half_w, double half_h,
                            double offset_x, double offset_y) {
  build_identity_matrix(matrix);
  matrix[0] = (float)(1.0 / half_w);
  matrix[5] = (float)(1.0 / half_h);
  matrix[10] = -1.0f;
  matrix[12] = (float)(-offset_x / half_w);
  matrix[13] = (float)(-offset_y / half_h);
}
