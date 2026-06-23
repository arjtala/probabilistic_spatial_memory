#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hdf5.h>

#include "core/spatial_memory.h"
#include "core/tile.h"
#include "ingest/ingest.h"

static double monotonic_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_double_asc(const void *a, const void *b) {
  const double da = *(const double *)a;
  const double db = *(const double *)b;
  if (da < db) return -1;
  if (da > db) return 1;
  return 0;
}

static bool parse_size_arg(const char *text, const char *name, size_t *out) {
  char *end = NULL;
  unsigned long long value;

  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0ULL) {
    fprintf(stderr, "Invalid %s: '%s'\n", name, text);
    return false;
  }
  *out = (size_t)value;
  return true;
}

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s <features.h5> [group] [query_ops]\n"
          "\n"
          "Benchmarks in-process PSM query_similar latency after ingesting\n"
          "a real HDF5 feature bank at the paper deployment config:\n"
          "h3=12, C=60, p=10, R=128, top=5.\n",
          argv0);
}

static bool ingest_and_capture_query(IngestReader *reader, SpatialMemory *sm,
                                     double time_window_sec, float **query_out,
                                     size_t *dim_out, size_t *records_out) {
  IngestRecord record;
  double window_anchor = -1.0;
  float *query = NULL;
  size_t dim = 0;
  size_t records = 0;

  while (true) {
    IngestReadStatus status = IngestReader_next(reader, &record);
    if (status == INGEST_READ_EOF) break;
    if (status == INGEST_READ_ERROR) {
      free(query);
      return false;
    }

    if (!query) {
      dim = record.embedding_dim;
      query = (float *)malloc(dim * sizeof(float));
      if (!query) return false;
      memcpy(query, record.embedding, dim * sizeof(float));
    }

    SpatialMemory_advance_to_timestamp(sm, record.timestamp, &window_anchor,
                                       time_window_sec);
    if (!SpatialMemory_observe(sm, record.timestamp, record.lat, record.lng,
                               record.embedding,
                               record.embedding_dim * sizeof(float))) {
      fprintf(stderr, "warn: skipped invalid observation at %.3f\n",
              record.timestamp);
    }
    records++;
  }

  if (!query || dim == 0 || records == 0) {
    free(query);
    return false;
  }

  *query_out = query;
  *dim_out = dim;
  *records_out = records;
  return true;
}

static void run_query_bench(SpatialMemory *sm, const float *query, size_t dim,
                            size_t query_ops, size_t top,
                            size_t per_cell_cap) {
  SpatialMemorySimilar *out =
      (SpatialMemorySimilar *)calloc(top, sizeof(SpatialMemorySimilar));
  if (!out) {
    fprintf(stderr, "failed to allocate query output\n");
    exit(EXIT_FAILURE);
  }
  double *samples_us = (double *)calloc(query_ops, sizeof(double));
  if (!samples_us) {
    fprintf(stderr, "failed to allocate latency samples\n");
    free(out);
    exit(EXIT_FAILURE);
  }

  // Warm cache/branch predictors without counting setup effects.
  for (size_t i = 0; i < 100; ++i) {
    (void)SpatialMemory_query_similar(sm, query, dim, 0.0, 0.0, -1, out, top,
                                      per_cell_cap);
  }

  volatile double similarity_sum = 0.0;
  volatile size_t candidates_sum = 0;
  double start = monotonic_seconds();
  for (size_t i = 0; i < query_ops; ++i) {
    double query_start = monotonic_seconds();
    size_t n = SpatialMemory_query_similar(sm, query, dim, 0.0, 0.0, -1, out,
                                           top, per_cell_cap);
    samples_us[i] = (monotonic_seconds() - query_start) * 1e6;
    candidates_sum += n;
    if (top > 0) similarity_sum += out[0].similarity;
  }
  double elapsed = monotonic_seconds() - start;
  double mean_us = query_ops > 0 ? (elapsed / (double)query_ops) * 1e6 : 0.0;
  qsort(samples_us, query_ops, sizeof(double), compare_double_asc);
  double median_us = 0.0;
  double p95_us = 0.0;
  if (query_ops > 0) {
    size_t p95_idx = ((query_ops * 95) + 99) / 100;
    if (p95_idx == 0) p95_idx = 1;
    p95_idx -= 1;
    if (p95_idx >= query_ops) p95_idx = query_ops - 1;
    p95_us = samples_us[p95_idx];
    if (query_ops % 2 == 0) {
      median_us =
          0.5 * (samples_us[(query_ops / 2) - 1] + samples_us[query_ops / 2]);
    } else {
      median_us = samples_us[query_ops / 2];
    }
  }

  printf("query_similar_actual ops=%zu dim=%zu top=%zu per_cell_cap=%zu "
         "secs=%.6f mean_us=%.3f median_us=%.3f p95_us=%.3f "
         "candidates_per_query=%.1f checksum=%.3f\n",
         query_ops, dim, top, per_cell_cap, elapsed, mean_us, median_us, p95_us,
         query_ops ? (double)candidates_sum / (double)query_ops : 0.0,
         similarity_sum);
  free(samples_us);
  free(out);
}

int main(int argc, char **argv) {
  const char *features = NULL;
  const char *group = "clip";
  size_t query_ops = 1000;
  const int h3_resolution = 12;
  const size_t capacity = 60;
  const size_t precision = 10;
  const size_t exemplars = 128;
  const double time_window_sec = 30.0;
  const size_t top = 5;

  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }
  features = argv[1];
  if (argc > 2) group = argv[2];
  if (argc > 3 && !parse_size_arg(argv[3], "query_ops", &query_ops)) {
    return 1;
  }

  Tile_set_random_seed(42);
  SpatialMemory *sm = SpatialMemory_new(h3_resolution, capacity, precision,
                                        exemplars, EXEMPLAR_CODEC_RAW);
  if (!sm) {
    fprintf(stderr, "failed to create SpatialMemory\n");
    return 1;
  }

  hid_t file = H5Fopen(features, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    fprintf(stderr, "failed to open %s\n", features);
    SpatialMemory_free(sm);
    return 1;
  }

  IngestReader *reader = IngestReader_open(file, group);
  if (!reader) {
    fprintf(stderr, "failed to open group %s\n", group);
    H5Fclose(file);
    SpatialMemory_free(sm);
    return 1;
  }

  float *query = NULL;
  size_t dim = 0;
  size_t records = 0;
  double ingest_start = monotonic_seconds();
  bool ok = ingest_and_capture_query(reader, sm, time_window_sec, &query, &dim,
                                     &records);
  double ingest_elapsed = monotonic_seconds() - ingest_start;
  IngestReader_close(reader);
  H5Fclose(file);

  if (!ok) {
    fprintf(stderr, "failed during ingest\n");
    SpatialMemory_free(sm);
    return 1;
  }

  printf("nymeria_psm_query file=%s group=%s records=%zu dim=%zu tiles=%zu "
         "h3=%d C=%zu p=%zu R=%zu ingest_secs=%.6f\n",
         features, group, records, dim, SpatialMemory_tile_count(sm),
         h3_resolution, capacity, precision, exemplars, ingest_elapsed);

  run_query_bench(sm, query, dim, query_ops, top, 1);
  run_query_bench(sm, query, dim, query_ops, top, top);

  free(query);
  SpatialMemory_free(sm);
  return 0;
}
