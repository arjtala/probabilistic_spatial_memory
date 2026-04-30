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

  // RGB-cube tour: black -> red -> yellow -> green -> cyan -> blue ->
  // magenta -> white. Seven equal segments along the cube edges.
  double s = t * 7.0;
  int seg = (int)s;
  if (seg > 6) seg = 6;
  double f = s - (double)seg;

  double cr = 0.0, cg = 0.0, cb = 0.0;
  switch (seg) {
  case 0: cr = f;             cg = 0.0;           cb = 0.0;           break;
  case 1: cr = 1.0;           cg = f;             cb = 0.0;           break;
  case 2: cr = 1.0 - f;       cg = 1.0;           cb = 0.0;           break;
  case 3: cr = 0.0;           cg = 1.0;           cb = f;             break;
  case 4: cr = 0.0;           cg = 1.0 - f;       cb = 1.0;           break;
  case 5: cr = f;             cg = 0.0;           cb = 1.0;           break;
  case 6: cr = 1.0;           cg = f;             cb = 1.0;           break;
  }

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

void build_map_projection(float matrix[16], MapProjectionMode projection_mode,
                          double half_w, double half_h, double offset_x,
                          double offset_y) {
  double m00, m01, m10, m11;
  double proj_half_w, proj_half_h;

  build_identity_matrix(matrix);
  VizMap_projection_basis(projection_mode, &m00, &m01, &m10, &m11);
  VizMap_projected_half_extents(projection_mode, half_w, half_h, &proj_half_w,
                                &proj_half_h);
  if (proj_half_w <= 0.0) proj_half_w = half_w;
  if (proj_half_h <= 0.0) proj_half_h = half_h;

  matrix[0] = (float)(m00 / proj_half_w);
  matrix[4] = (float)(m01 / proj_half_w);
  matrix[1] = (float)(m10 / proj_half_h);
  matrix[5] = (float)(m11 / proj_half_h);
  matrix[10] = -1.0f;
  matrix[12] = (float)(-(m00 * offset_x + m01 * offset_y) / proj_half_w);
  matrix[13] = (float)(-(m10 * offset_x + m11 * offset_y) / proj_half_h);
}
