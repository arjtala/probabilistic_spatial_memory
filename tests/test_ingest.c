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
  bool ok = IngestReader_next(reader, &record);
  ASSERT(ok, 1, ok);
  ASSERT(100 == (int)record.timestamp, 100, (int)record.timestamp);
  ASSERT(1 == (int)record.lat, 1, (int)record.lat);
  ASSERT(4 == (int)record.lng, 4, (int)record.lng);
  ASSERT(2 == (int)record.embedding[1], 2, (int)record.embedding[1]);
  ASSERT(2 == (int)record.attention_map[2], 2, (int)record.attention_map[2]);
  ASSERT(1 == record.has_imu, 1, record.has_imu);
  ASSERT(2 == (int)record.accel[1], 2, (int)record.accel[1]);
  ASSERT(8 == (int)record.gyro[1], 8, (int)record.gyro[1]);

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

  IngestReader_run(reader, sm, 5.0);

  int current_only = (int)SpatialMemory_query(sm, TEST_LAT, TEST_LNG, 0);
  int current_plus_prev = (int)SpatialMemory_query(sm, TEST_LAT, TEST_LNG, 1);
  int current_plus_two_prev = (int)SpatialMemory_query(sm, TEST_LAT, TEST_LNG, 2);
  ASSERT(1 <= current_only, 1, current_only);
  ASSERT(current_plus_prev == current_only, current_only, current_plus_prev);
  ASSERT(current_plus_two_prev > current_plus_prev, 1,
         current_plus_two_prev > current_plus_prev);

  SpatialMemory_free(sm);
  IngestReader_close(reader);
  H5Fclose(file);
  unlink(path);
  free(path);
}

int main(void) {
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  RUN_TEST(test_ingest_open_and_next);
  RUN_TEST(test_ingest_rejects_mismatched_lengths);
  RUN_TEST(test_ingest_run_preserves_empty_windows);

  return 0;
}
