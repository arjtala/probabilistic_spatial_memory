#ifndef VIZ_GPS_TRACE_H
#define VIZ_GPS_TRACE_H

#include <stdbool.h>
#include <stddef.h>
#include "viz/gl_platform.h"
#include "viz/imu_processor.h"

typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint program;
  GLint u_projection;
  double *lats;
  double *lngs;
  size_t count;
  size_t capacity;
  size_t vertex_count;
  // Own center — running average of all accumulated points
  double center_lat;
  double center_lng;
  // Dirty tracking — skip re-upload when nothing changed
  bool dirty;
  size_t last_upload_count;
  double last_center_lat;
  double last_center_lng;
  // Reusable vertex buffer to avoid malloc/free per frame
  float *verts;
  size_t verts_capacity;  // in number of vertices
  // IMU metadata parallel array
  ImuPointMeta *imu_meta;
  size_t imu_meta_capacity;
  bool has_any_imu;
} GpsTrace;

GpsTrace *GpsTrace_new(GLuint program);
void GpsTrace_push(GpsTrace *gt, double lat, double lng, const ImuPointMeta *imu);
void GpsTrace_clear(GpsTrace *gt);
void GpsTrace_upload(GpsTrace *gt, double proj_center_lat, double proj_center_lng);
void GpsTrace_draw(GpsTrace *gt, int viewport_w, int viewport_h, double zoom);
void GpsTrace_free(GpsTrace *gt);

#endif
