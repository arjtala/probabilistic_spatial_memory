#ifndef VIZ_VIZ_MATH_H
#define VIZ_VIZ_MATH_H

#include "viz/imu_processor.h"

void count_to_color(double t, float *r, float *g, float *b, float *a);
MotionState classify_motion(const float accel[3]);
float normalize_angle(float angle);
float estimate_speed(MotionState motion, float deviation);
int osm_zoom_from_degrees(double zoom_degrees);
void latlon_to_tile(double lat, double lng, int z, int *tx, int *ty);

#endif
