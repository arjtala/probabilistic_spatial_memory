#ifndef VIZ_VIZ_MATH_H
#define VIZ_VIZ_MATH_H

#include "viz/map_view.h"
#include "viz/imu_processor.h"

void count_to_color(double t, float *r, float *g, float *b, float *a);
MotionState classify_motion(const float accel[3]);
float normalize_angle(float angle);
float estimate_speed(MotionState motion, float deviation);
int osm_zoom_from_degrees(double zoom_degrees);
void latlon_to_tile(double lat, double lng, int z, int *tx, int *ty);
bool compute_aspect_quad(float quad[16], int video_w, int video_h,
                         int viewport_w, int viewport_h);
void build_identity_matrix(float matrix[16]);
void build_ortho_projection(float matrix[16], double half_w, double half_h,
                            double offset_x, double offset_y);
void build_map_projection(float matrix[16], MapProjectionMode projection_mode,
                          double half_w, double half_h, double offset_x,
                          double offset_y);

#endif
