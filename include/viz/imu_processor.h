#ifndef VIZ_IMU_PROCESSOR_H
#define VIZ_IMU_PROCESSOR_H

#include <stdbool.h>

typedef enum {
  MOTION_STATIONARY = 0,
  MOTION_WALKING = 1,
  MOTION_RUNNING = 2
} MotionState;

typedef struct {
  MotionState motion;
  float heading_rad;    // 0=north, CW positive
  float pitch_rad;      // camera pitch: +up, -down (from gravity)
  bool has_imu;
} ImuPointMeta;

typedef struct {
  float heading_rad;
  double prev_timestamp;
  bool initialized;
  double dr_lat, dr_lng;     // dead-reckoned position
  double gps_lat, gps_lng;   // last raw GPS
  float alpha;               // complementary filter weight (0.3)
  // Gravity vector estimation (running exponential average of accel)
  float gravity[3];          // unit vector in gravity direction
  bool gravity_valid;        // true once we have a stable estimate
} ImuProcessor;

ImuProcessor *ImuProcessor_new(float alpha);
ImuPointMeta ImuProcessor_update(ImuProcessor *proc, const float accel[3],
                                 const float gyro[3], double timestamp,
                                 double gps_lat, double gps_lng);
void ImuProcessor_get_blended_position(const ImuProcessor *proc,
                                       double *lat, double *lng);
void ImuProcessor_reset(ImuProcessor *proc);
void ImuProcessor_free(ImuProcessor *proc);

#endif
