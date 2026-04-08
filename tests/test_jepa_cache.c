#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <hdf5.h>
#include "ingest/ingest.h"
#include "viz/jepa_cache.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

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
  char template[] = "/tmp/psm_jepa_XXXXXX";
  int fd = mkstemp(template);
  if (fd < 0) {
    perror("mkstemp");
    exit(EXIT_FAILURE);
  }
  close(fd);
  return strdup(template);
}

static void fill_map(float *dst, float value) {
  for (size_t i = 0; i < JEPA_MAP_DIM * JEPA_MAP_DIM; ++i) {
    dst[i] = value;
  }
}

static void create_valid_jepa_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t group = H5Gcreate(file, JEPA, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double timestamps[] = {10.0, 20.0};
  float prediction_maps[2 * JEPA_MAP_DIM * JEPA_MAP_DIM];
  fill_map(prediction_maps, 1.0f);
  fill_map(prediction_maps + JEPA_MAP_DIM * JEPA_MAP_DIM, 2.0f);

  write_doubles_1d(group, TIMESTAMPS, timestamps, 2);
  write_floats_3d(group, PREDICTION_MAPS, prediction_maps, 2, JEPA_MAP_DIM,
                  JEPA_MAP_DIM);

  H5Gclose(group);
  H5Fclose(file);
}

static void create_invalid_jepa_file(const char *path) {
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t group = H5Gcreate(file, JEPA, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  double timestamps[] = {10.0};
  float prediction_maps[2 * JEPA_MAP_DIM * JEPA_MAP_DIM];
  fill_map(prediction_maps, 1.0f);
  fill_map(prediction_maps + JEPA_MAP_DIM * JEPA_MAP_DIM, 2.0f);

  write_doubles_1d(group, TIMESTAMPS, timestamps, 1);
  write_floats_3d(group, PREDICTION_MAPS, prediction_maps, 2, JEPA_MAP_DIM,
                  JEPA_MAP_DIM);

  H5Gclose(group);
  H5Fclose(file);
}

void test_jepa_cache_lookup(void) {
  char *path = create_temp_path();
  create_valid_jepa_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  JepaCache *cache = JepaCache_load(file);
  ASSERT(NULL != cache, 1, NULL != cache);
  ASSERT(2 == (int)cache->n_records, 2, (int)cache->n_records);

  float *map = NULL;
  bool ok = JepaCache_lookup(cache, 15.0, &map);
  ASSERT(ok, 1, ok);
  ASSERT(1 == (int)map[0], 1, (int)map[0]);
  ok = JepaCache_lookup(cache, 25.0, &map);
  ASSERT(ok, 1, ok);
  ASSERT(2 == (int)map[0], 2, (int)map[0]);

  JepaCache_reset(cache);
  ok = JepaCache_lookup(cache, 5.0, &map);
  ASSERT(ok, 1, ok);
  ASSERT(1 == (int)map[0], 1, (int)map[0]);

  JepaCache_free(cache);
  H5Fclose(file);
  unlink(path);
  free(path);
}

void test_jepa_cache_rejects_mismatched_lengths(void) {
  char *path = create_temp_path();
  create_invalid_jepa_file(path);

  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  JepaCache *cache = JepaCache_load(file);
  ASSERT(NULL == cache, 1, NULL == cache);

  H5Fclose(file);
  unlink(path);
  free(path);
}

int main(void) {
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  RUN_TEST(test_jepa_cache_lookup);
  RUN_TEST(test_jepa_cache_rejects_mismatched_lengths);

  return 0;
}
