#include <math.h>
#include <stdlib.h>
#include "viz/imu_processor.h"
#include "viz/viz_math.h"

#define GRAVITY 9.81f
#define R_EARTH 6371000.0   // meters
#define DEG_TO_RAD (M_PI / 180.0)
// Exponential smoothing factor for gravity estimate (low = more smoothing)
#define GRAVITY_ALPHA 0.02f

ImuProcessor *ImuProcessor_new(float alpha) {
  ImuProcessor *proc = calloc(1, sizeof(ImuProcessor));
  if (!proc) return NULL;
  proc->alpha = alpha;
  proc->heading_rad = 0.0f;
  proc->prev_timestamp = -1.0;
  proc->initialized = false;
  proc->gravity_valid = false;
  proc->gravity[0] = proc->gravity[1] = proc->gravity[2] = 0.0f;
  return proc;
}

// Update gravity estimate with exponential moving average of accel readings.
// Returns the yaw rate: gyro projected onto gravity direction.
static float compute_yaw_rate(ImuProcessor *proc, const float accel[3],
                              const float gyro[3]) {
  if (!proc->gravity_valid) {
    // Seed with first accel reading
    proc->gravity[0] = accel[0];
    proc->gravity[1] = accel[1];
    proc->gravity[2] = accel[2];
    proc->gravity_valid = true;
  } else {
    // Exponential moving average — heavily smoothed to reject dynamic accel
    proc->gravity[0] += GRAVITY_ALPHA * (accel[0] - proc->gravity[0]);
    proc->gravity[1] += GRAVITY_ALPHA * (accel[1] - proc->gravity[1]);
    proc->gravity[2] += GRAVITY_ALPHA * (accel[2] - proc->gravity[2]);
  }

  // Normalize gravity vector
  float gx = proc->gravity[0];
  float gy = proc->gravity[1];
  float gz = proc->gravity[2];
  float gmag = sqrtf(gx * gx + gy * gy + gz * gz);
  if (gmag < 1.0f) return 0.0f;  // degenerate, skip

  float inv_gmag = 1.0f / gmag;
  float ux = gx * inv_gmag;
  float uy = gy * inv_gmag;
  float uz = gz * inv_gmag;

  // Yaw rate around the vertical (down) axis: negate because the accel-based
  // gravity estimate points UP, but CW-from-above (turning right) should be
  // positive to match the bearing convention (0=N, +90=E).
  return -(gyro[0] * ux + gyro[1] * uy + gyro[2] * uz);
}

ImuPointMeta ImuProcessor_update(ImuProcessor *proc, const float accel[3],
                                 const float gyro[3], double timestamp,
                                 double gps_lat, double gps_lng) {
  ImuPointMeta meta;
  meta.has_imu = true;

  // Motion classification
  meta.motion = classify_motion(accel);

  // Compute dt
  double dt = 0.0;
  if (proc->prev_timestamp >= 0.0) {
    dt = timestamp - proc->prev_timestamp;
  }
  proc->prev_timestamp = timestamp;

  // Heading integration: project gyro onto gravity vector for yaw
  float yaw_rate = compute_yaw_rate(proc, accel, gyro);
  if (dt > 0.0 && dt <= 5.0) {
    proc->heading_rad += yaw_rate * (float)dt;
    proc->heading_rad = normalize_angle(proc->heading_rad);
  }
  meta.heading_rad = proc->heading_rad;

  // Pitch from gravity: camera optical axis is (0,0,-1) in sensor frame
  // dot(camera, gravity_unit) = -uz  →  pitch = asin(-uz)
  if (proc->gravity_valid) {
    float gx = proc->gravity[0], gy = proc->gravity[1], gz = proc->gravity[2];
    float gmag = sqrtf(gx * gx + gy * gy + gz * gz);
    if (gmag > 1.0f) {
      float uz = gz / gmag;
      meta.pitch_rad = asinf(fmaxf(-1.0f, fminf(1.0f, -uz)));
    } else {
      meta.pitch_rad = 0.0f;
    }
  } else {
    meta.pitch_rad = 0.0f;
  }

  // Dead reckoning
  if (!proc->initialized) {
    proc->dr_lat = gps_lat;
    proc->dr_lng = gps_lng;
    proc->gps_lat = gps_lat;
    proc->gps_lng = gps_lng;
    proc->initialized = true;
  } else if (dt > 0.0 && dt <= 5.0) {
    float mag = sqrtf(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
    float deviation = fabsf(mag - GRAVITY);
    float speed = estimate_speed(meta.motion, deviation);

    double dlat = speed * dt * cos((double)proc->heading_rad) / R_EARTH;
    double dlng = speed * dt * sin((double)proc->heading_rad) /
                  (R_EARTH * cos(proc->dr_lat * DEG_TO_RAD));
    dlat *= (180.0 / M_PI);
    dlng *= (180.0 / M_PI);

    proc->dr_lat += dlat;
    proc->dr_lng += dlng;

    proc->gps_lat = gps_lat;
    proc->gps_lng = gps_lng;

    double blended_lat = proc->alpha * proc->dr_lat + (1.0 - proc->alpha) * gps_lat;
    double blended_lng = proc->alpha * proc->dr_lng + (1.0 - proc->alpha) * gps_lng;

    proc->dr_lat = blended_lat;
    proc->dr_lng = blended_lng;
  } else {
    proc->dr_lat = gps_lat;
    proc->dr_lng = gps_lng;
    proc->gps_lat = gps_lat;
    proc->gps_lng = gps_lng;
  }

  return meta;
}

void ImuProcessor_get_blended_position(const ImuProcessor *proc,
                                       double *lat, double *lng) {
  *lat = proc->dr_lat;
  *lng = proc->dr_lng;
}

void ImuProcessor_reset(ImuProcessor *proc) {
  if (!proc) return;
  proc->heading_rad = 0.0f;
  proc->prev_timestamp = -1.0;
  proc->initialized = false;
  proc->dr_lat = 0.0;
  proc->dr_lng = 0.0;
  proc->gps_lat = 0.0;
  proc->gps_lng = 0.0;
  proc->gravity_valid = false;
  proc->gravity[0] = proc->gravity[1] = proc->gravity[2] = 0.0f;
}

void ImuProcessor_free(ImuProcessor *proc) {
  free(proc);
}
