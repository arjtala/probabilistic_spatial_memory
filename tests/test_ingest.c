#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <hdf5.h>
#include "core/spatial_memory.h"
#include "ingest/ingest.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#define TEST_RESOLUTION 10
#define TEST_CAPACITY 4
#define TEST_PRECISION 10
#define TEST_LAT 51.380083
#define TEST_LNG -0.311041

static void write_doubles_1d(hid_t group, const char *name, const double *data,
                             hsize_t n) {
  hid_t space = H5Screate_simple(1, &n, NULL);
  hid_t dataset = H5Dcreate(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0 ||
      H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               data) < 0) {
    fprintf(stderr, "Failed to write dataset '%s'\n", name);
    exit(EXIT_FAILURE);
  }
  H5Dclose(dataset);
  H5Sclose(space);
}

static void write_floats_2d(hid_t group, const char *name, const float *data,
                            hsize_t rows, hsize_t cols) {
  hsize_t dims[2] = {rows, cols};
  hid_t space = H5Screate_simple(2, dims, NULL);
  hid_t dataset = H5Dcreate(group, name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0 ||
      H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               data) < 0) {
    fprintf(stderr, "Failed to write dataset '%s'\n", name);
    exit(EXIT_FAILURE);
  }
  H5Dclose(dataset);
  H5Sclose(space);
}

static void write_floats_3d(hid_t group, const char *name, const float *data,
                            hsize_t depth, hsize_t rows, hsize_t cols) {
  hsize_t dims[3] = {depth, rows, cols};
  hid_t space = H5Screate_simple(3, dims, NULL);
  hid_t dataset = H5Dcreate(group, name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0 ||
      H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               data) < 0) {
    fprintf(stderr, "Failed to write dataset '%s'\n", name);
    exit(EXIT_FAILURE);
  }
  H5Dclose(dataset);
  H5Sclose(space);
}

static char *create_temp_path(void) {
  char template[] = "/tmp/psm_ingest_XXXXXX";
  int fd = mkstemp(template);
  if (fd < 0) {
    perror("mkstemp");
    exit(EXIT_FAILURE);
  }
  close(fd);
  return strdup(template);
}

static void create_valid_dino_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t group = H5Gcreate(file, DINO, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double timestamps[] = {100.0, 104.0, 111.0};
  double lat[] = {1.0, 2.0, 3.0};
  double lng[] = {4.0, 5.0, 6.0};
  float embeddings[] = {
      1.0f, 2.0f,
      3.0f, 4.0f,
      5.0f, 6.0f,
  };
  float attention_maps[] = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 9.0f, 10.0f, 11.0f,
  };
  float accel[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  float gyro[] = {
      9.0f, 8.0f, 7.0f,
      6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f,
  };

  write_doubles_1d(group, TIMESTAMPS, timestamps, 3);
  write_doubles_1d(group, LAT, lat, 3);
  write_doubles_1d(group, LNG, lng, 3);
  write_floats_2d(group, EMBEDDINGS, embeddings, 3, 2);
  write_floats_3d(group, ATTENTION_MAPS, attention_maps, 3, 2, 2);
  write_floats_2d(group, ACCEL, accel, 3, 3);
  write_floats_2d(group, GYRO, gyro, 3, 3);

  H5Gclose(group);
  H5Fclose(file);
}

static void create_mismatched_dino_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t group = H5Gcreate(file, DINO, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double timestamps[] = {1.0, 2.0};
  double lat[] = {1.0};
  double lng[] = {3.0, 4.0};
  float embeddings[] = {
      1.0f, 2.0f,
      3.0f, 4.0f,
  };

  write_doubles_1d(group, TIMESTAMPS, timestamps, 2);
  write_doubles_1d(group, LAT, lat, 1);
  write_doubles_1d(group, LNG, lng, 2);
  write_floats_2d(group, EMBEDDINGS, embeddings, 2, 2);

  H5Gclose(group);
  H5Fclose(file);
}

static void create_gap_dino_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t group = H5Gcreate(file, DINO, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double timestamps[] = {0.0, 11.0};
  double lat[] = {TEST_LAT, TEST_LAT};
  double lng[] = {TEST_LNG, TEST_LNG};
  float embeddings[] = {
      1.0f, 2.0f,
      3.0f, 4.0f,
  };

  write_doubles_1d(group, TIMESTAMPS, timestamps, 2);
  write_doubles_1d(group, LAT, lat, 2);
  write_doubles_1d(group, LNG, lng, 2);
  write_floats_2d(group, EMBEDDINGS, embeddings, 2, 2);

  H5Gclose(group);
  H5Fclose(file);
}

static void create_valid_imu_gps_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t imu_group = H5Gcreate(file, "imu", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t gps_group = H5Gcreate(file, "gps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double imu_ts[] = {10.0, 11.0, 12.0};
  float accel[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  float gyro[] = {
      9.0f, 8.0f, 7.0f,
      6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f,
  };
  double gps_ts[] = {10.0, 12.0};
  double gps_lat[] = {1.0, 3.0};
  double gps_lng[] = {4.0, 6.0};

  write_doubles_1d(imu_group, TIMESTAMPS, imu_ts, 3);
  write_floats_2d(imu_group, ACCEL, accel, 3, 3);
  write_floats_2d(imu_group, GYRO, gyro, 3, 3);
  write_doubles_1d(gps_group, TIMESTAMPS, gps_ts, 2);
  write_doubles_1d(gps_group, LAT, gps_lat, 2);
  write_doubles_1d(gps_group, LNG, gps_lng, 2);

  H5Gclose(imu_group);
  H5Gclose(gps_group);
  H5Fclose(file);
}

static void create_imu_only_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t imu_group = H5Gcreate(file, "imu", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double imu_ts[] = {10.0, 11.0};
  float accel[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  float gyro[] = {
      6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f,
  };

  write_doubles_1d(imu_group, TIMESTAMPS, imu_ts, 2);
  write_floats_2d(imu_group, ACCEL, accel, 2, 3);
  write_floats_2d(imu_group, GYRO, gyro, 2, 3);

  H5Gclose(imu_group);
  H5Fclose(file);
}

static void create_imu_with_empty_gps_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t imu_group = H5Gcreate(file, "imu", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t gps_group = H5Gcreate(file, "gps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double imu_ts[] = {10.0, 11.0};
  float accel[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  float gyro[] = {
      6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f,
  };
  double unused = 0.0;

  write_doubles_1d(imu_group, TIMESTAMPS, imu_ts, 2);
  write_floats_2d(imu_group, ACCEL, accel, 2, 3);
  write_floats_2d(imu_group, GYRO, gyro, 2, 3);
  write_doubles_1d(gps_group, TIMESTAMPS, &unused, 0);
  write_doubles_1d(gps_group, LAT, &unused, 0);
  write_doubles_1d(gps_group, LNG, &unused, 0);

  H5Gclose(imu_group);
  H5Gclose(gps_group);
  H5Fclose(file);
}

static void create_mismatched_imu_gps_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t gps_group = H5Gcreate(file, "gps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double gps_ts[] = {10.0, 12.0};
  double gps_lat[] = {1.0};
  double gps_lng[] = {4.0, 6.0};

  write_doubles_1d(gps_group, TIMESTAMPS, gps_ts, 2);
  write_doubles_1d(gps_group, LAT, gps_lat, 1);
  write_doubles_1d(gps_group, LNG, gps_lng, 2);

  H5Gclose(gps_group);
  H5Fclose(file);
}

void test_ingest_open_and_next(void) {
  char *path = create_temp_path();
  create_valid_dino_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  IngestReader *reader = IngestReader_open(file, DINO);
  ASSERT(NULL != reader, 1, NULL != reader);
  ASSERT(3 == (int)reader->n_records, 3, (int)reader->n_records);
  ASSERT(2 == (int)reader->emb_dimension, 2, (int)reader->emb_dimension);
  ASSERT(2 == (int)reader->attn_size, 2, (int)reader->attn_size);
  ASSERT(1 == reader->has_imu, 1, reader->has_imu);

  IngestRecord record;
  IngestReadStatus status = IngestReader_next(reader, &record);
  bool ok = (status == INGEST_READ_OK);
  ASSERT(ok, 1, ok);
  ASSERT(100 == (int)record.timestamp, 100, (int)record.timestamp);
  ASSERT(1 == (int)record.lat, 1, (int)record.lat);
  ASSERT(4 == (int)record.lng, 4, (int)record.lng);
  ASSERT(2 == (int)record.embedding[1], 2, (int)record.embedding[1]);
  ASSERT(2 == (int)record.attention_map[2], 2, (int)record.attention_map[2]);
  ASSERT(1 == record.has_imu, 1, record.has_imu);
  ASSERT(2 == (int)record.accel[1], 2, (int)record.accel[1]);
  ASSERT(8 == (int)record.gyro[1], 8, (int)record.gyro[1]);
  status = IngestReader_next(reader, &record);
  ok = (status == INGEST_READ_OK);
  ASSERT(ok, 1, ok);
  status = IngestReader_next(reader, &record);
  ok = (status == INGEST_READ_OK);
  ASSERT(ok, 1, ok);
  status = IngestReader_next(reader, &record);
  ASSERT(INGEST_READ_EOF == status, INGEST_READ_EOF, status);

  IngestReader_close(reader);
  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_ingest_rejects_mismatched_lengths(void) {
  char *path = create_temp_path();
  create_mismatched_dino_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  IngestReader *reader = IngestReader_open(file, DINO);
  ASSERT(NULL == reader, 1, NULL == reader);

  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_ingest_run_preserves_empty_windows(void) {
  char *path = create_temp_path();
  create_gap_dino_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  IngestReader *reader = IngestReader_open(file, DINO);
  SpatialMemory *sm = SpatialMemory_new(TEST_RESOLUTION, TEST_CAPACITY,
                                        TEST_PRECISION);

  ASSERT(NULL != reader, 1, NULL != reader);
  ASSERT(NULL != sm, 1, NULL != sm);

  ASSERT(IngestReader_run(reader, sm, 5.0), 1, 1);

  double current_only = 0.0;
  double current_plus_prev = 0.0;
  double current_plus_two_prev = 0.0;
  bool ok = SpatialMemory_query(sm, TEST_LAT, TEST_LNG, 0, &current_only);
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, TEST_LAT, TEST_LNG, 1, &current_plus_prev);
  ASSERT(ok, 1, ok);
  ok = SpatialMemory_query(sm, TEST_LAT, TEST_LNG, 2, &current_plus_two_prev);
  ASSERT(ok, 1, ok);
  ASSERT(1 <= current_only, 1, (int)current_only);
  ASSERT(current_plus_prev == current_only, (int)current_only,
         (int)current_plus_prev);
  ASSERT(current_plus_two_prev > current_plus_prev, 1,
         current_plus_two_prev > current_plus_prev);

  SpatialMemory_free(sm);
  IngestReader_close(reader);
  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_imu_gps_open_and_interpolate(void) {
  char *path = create_temp_path();
  create_valid_imu_gps_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  ImuGpsReader *reader = ImuGpsReader_open(file);
  ASSERT(NULL != reader, 1, NULL != reader);
  ASSERT(1 == reader->has_imu, 1, reader->has_imu);
  ASSERT(1 == reader->has_gps, 1, reader->has_gps);
  ASSERT(3 == (int)reader->imu_n_records, 3, (int)reader->imu_n_records);
  ASSERT(2 == (int)reader->gps_n_records, 2, (int)reader->gps_n_records);
  ASSERT(10 == (int)reader->imu_first_ts, 10, (int)reader->imu_first_ts);

  ImuRecord rec;
  IngestReadStatus status = ImuGpsReader_next_imu(reader, &rec);
  bool ok = (status == INGEST_READ_OK);
  ASSERT(ok, 1, ok);
  ASSERT(10 == (int)rec.timestamp, 10, (int)rec.timestamp);
  ASSERT(2 == (int)rec.accel[1], 2, (int)rec.accel[1]);
  ASSERT(8 == (int)rec.gyro[1], 8, (int)rec.gyro[1]);

  double lat = 0.0;
  double lng = 0.0;
  ASSERT(ImuGpsReader_interpolate_gps(reader, 11.0, &lat, &lng), 1, 1);
  ASSERT(2 == (int)lat, 2, (int)lat);
  ASSERT(5 == (int)lng, 5, (int)lng);

  ASSERT(ImuGpsReader_interpolate_gps(reader, 9.0, &lat, &lng), 1, 1);
  ASSERT(1 == (int)lat, 1, (int)lat);
  ASSERT(4 == (int)lng, 4, (int)lng);

  ASSERT(ImuGpsReader_interpolate_gps(reader, 13.0, &lat, &lng), 1, 1);
  ASSERT(3 == (int)lat, 3, (int)lat);
  ASSERT(6 == (int)lng, 6, (int)lng);

  status = ImuGpsReader_next_imu(reader, &rec);
  ASSERT(INGEST_READ_OK == status, INGEST_READ_OK, status);
  status = ImuGpsReader_next_imu(reader, &rec);
  ASSERT(INGEST_READ_OK == status, INGEST_READ_OK, status);
  status = ImuGpsReader_next_imu(reader, &rec);
  ASSERT(INGEST_READ_EOF == status, INGEST_READ_EOF, status);

  ImuGpsReader_close(reader);
  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_imu_gps_rejects_mismatched_lengths(void) {
  char *path = create_temp_path();
  create_mismatched_imu_gps_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  ImuGpsReader *reader = ImuGpsReader_open(file);
  ASSERT(NULL == reader, 1, NULL == reader);

  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_imu_gps_open_without_gps_data(void) {
  char *path = create_temp_path();
  create_imu_only_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  ImuGpsReader *reader = ImuGpsReader_open(file);
  ASSERT(NULL != reader, 1, NULL != reader);
  ASSERT(1 == reader->has_imu, 1, reader->has_imu);
  ASSERT(0 == reader->has_gps, 0, reader->has_gps);
  ASSERT(0 == (int)reader->gps_n_records, 0, (int)reader->gps_n_records);

  ImuRecord rec;
  IngestReadStatus status = ImuGpsReader_next_imu(reader, &rec);
  ASSERT(INGEST_READ_OK == status, INGEST_READ_OK, status);
  ASSERT(10 == (int)rec.timestamp, 10, (int)rec.timestamp);

  double lat = 123.0;
  double lng = 456.0;
  bool ok = ImuGpsReader_interpolate_gps(reader, 10.5, &lat, &lng);
  ASSERT(!ok, 0, ok);
  ASSERT(0 == (int)lat, 0, (int)lat);
  ASSERT(0 == (int)lng, 0, (int)lng);

  ImuGpsReader_close(reader);
  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_imu_gps_open_with_empty_gps_data(void) {
  char *path = create_temp_path();
  create_imu_with_empty_gps_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  ImuGpsReader *reader = ImuGpsReader_open(file);
  ASSERT(NULL != reader, 1, NULL != reader);
  ASSERT(1 == reader->has_imu, 1, reader->has_imu);
  ASSERT(0 == reader->has_gps, 0, reader->has_gps);
  ASSERT(0 == (int)reader->gps_n_records, 0, (int)reader->gps_n_records);

  double lat = -1.0;
  double lng = -1.0;
  bool ok = ImuGpsReader_interpolate_gps(reader, 10.5, &lat, &lng);
  ASSERT(!ok, 0, ok);
  ASSERT(0 == (int)lat, 0, (int)lat);
  ASSERT(0 == (int)lng, 0, (int)lng);

  ImuGpsReader_close(reader);
  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_ingest_reader_invalid_args(void) {
  IngestRecord record;
  IngestReadStatus status = INGEST_READ_OK;
  bool ok = true;
  IngestReader *reader = IngestReader_open(-1, DINO);
  ASSERT(NULL == reader, 1, NULL == reader);
  reader = IngestReader_open(-1, NULL);
  ASSERT(NULL == reader, 1, NULL == reader);
  status = IngestReader_next(NULL, &record);
  ASSERT(INGEST_READ_ERROR == status, INGEST_READ_ERROR, status);
  status = IngestReader_next((IngestReader *)1, NULL);
  ASSERT(INGEST_READ_ERROR == status, INGEST_READ_ERROR, status);
  ok = IngestReader_run(NULL, NULL, 1.0);
  ASSERT(!ok, 0, ok);
}

void test_imu_gps_invalid_args(void) {
  double lat = 123.0;
  double lng = 456.0;
  IngestReadStatus status = INGEST_READ_OK;
  bool ok = true;
  ImuGpsReader *reader = ImuGpsReader_open(-1);
  ASSERT(NULL == reader, 1, NULL == reader);
  status = ImuGpsReader_next_imu(NULL, NULL);
  ASSERT(INGEST_READ_ERROR == status, INGEST_READ_ERROR, status);
  ok = ImuGpsReader_interpolate_gps(NULL, 0.0, &lat, &lng);
  ASSERT(!ok, 0, ok);
  ASSERT(0 == (int)lat, 0, (int)lat);
  ASSERT(0 == (int)lng, 0, (int)lng);
}

int main(void) {
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  RUN_TEST(test_ingest_open_and_next);
  RUN_TEST(test_ingest_rejects_mismatched_lengths);
  RUN_TEST(test_ingest_run_preserves_empty_windows);
  RUN_TEST(test_ingest_reader_invalid_args);
  RUN_TEST(test_imu_gps_open_and_interpolate);
  RUN_TEST(test_imu_gps_rejects_mismatched_lengths);
  RUN_TEST(test_imu_gps_open_without_gps_data);
  RUN_TEST(test_imu_gps_open_with_empty_gps_data);
  RUN_TEST(test_imu_gps_invalid_args);

  return 0;
}
