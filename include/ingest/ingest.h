#ifndef INGEST_H
#define INGEST_H

#include <stdbool.h>
#include <hdf5.h>
#include "core/spatial_memory.h"

#define DINO "dino"
#define JEPA "jepa"
#define TIMESTAMPS "timestamps"
#define LAT "lat"
#define LNG "lng"
#define EMBEDDINGS "embeddings"
#define ATTENTION_MAPS "attention_maps"
#define PREDICTION_MAPS "prediction_maps"
#define ACCEL "accel"
#define GYRO  "gyro"

typedef struct {
  hid_t dataset_ts;
  hid_t dataset_lat;
  hid_t dataset_lng;
  hid_t dataset_emb;
  hid_t dataset_attn;
  size_t n_records;
  size_t emb_dimension;
  size_t attn_size;
  size_t cursor;
  float *embedding_buf;
  float *attn_buf;
  hid_t dataset_accel;   // -1 if missing
  hid_t dataset_gyro;    // -1 if missing
  float accel_buf[3];
  float gyro_buf[3];
  bool has_imu;          // true if BOTH datasets opened
} IngestReader;

typedef struct {
  double timestamp;
  double lat;
  double lng;
  float *embedding;
  size_t embedding_dim;
  float *attention_map;
  size_t attn_size;
  float accel[3];
  float gyro[3];
  bool has_imu;
} IngestRecord;

typedef enum {
  INGEST_READ_ERROR = -1,
  INGEST_READ_EOF = 0,
  INGEST_READ_OK = 1,
} IngestReadStatus;

IngestReader *IngestReader_open(hid_t file, const char *group);
void IngestReader_close(IngestReader *reader);
IngestReadStatus IngestReader_next(IngestReader *reader,
                                   IngestRecord *record);
bool IngestReader_run(IngestReader *reader, SpatialMemory *sm,
                      const double time_window_sec);

// ---- High-rate IMU/GPS reader ----

typedef struct {
  double timestamp;
  float accel[3];
  float gyro[3];
} ImuRecord;

typedef struct {
  // IMU datasets (from "imu/" group) — row-by-row reading
  hid_t imu_dataset_ts;
  hid_t imu_dataset_accel;
  hid_t imu_dataset_gyro;
  size_t imu_n_records;
  size_t imu_cursor;
  float imu_accel_buf[3];
  float imu_gyro_buf[3];
  bool has_imu;

  // GPS arrays (from "gps/" group) — loaded fully into memory
  double *gps_ts;
  double *gps_lat;
  double *gps_lng;
  size_t gps_n_records;
  size_t gps_cursor;   // tracks interpolation position (advances forward only)
  bool has_gps;

  double imu_first_ts; // timestamp of first IMU sample (for timing alignment)
} ImuGpsReader;

ImuGpsReader *ImuGpsReader_open(hid_t file);
IngestReadStatus ImuGpsReader_next_imu(ImuGpsReader *r, ImuRecord *rec);
bool ImuGpsReader_interpolate_gps(ImuGpsReader *r, double timestamp,
                                  double *lat, double *lng);
void ImuGpsReader_close(ImuGpsReader *r);

#endif
