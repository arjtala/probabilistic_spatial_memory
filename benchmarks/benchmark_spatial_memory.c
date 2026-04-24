#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core/spatial_memory.h"

typedef struct {
  double lat;
  double lng;
} Coord;

static double monotonic_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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

static void fill_grid(Coord *coords, size_t count) {
  size_t side = (size_t)ceil(sqrt((double)count));
  size_t idx = 0;
  const double base_lat = 37.7749;
  const double base_lng = -122.4194;
  const double step = 0.0025;

  for (size_t y = 0; y < side && idx < count; y++) {
    double y_off = (double)y - (double)(side - 1) / 2.0;
    for (size_t x = 0; x < side && idx < count; x++) {
      double x_off = (double)x - (double)(side - 1) / 2.0;
      coords[idx].lat = base_lat + y_off * step;
      coords[idx].lng = base_lng + x_off * step;
      idx++;
    }
  }
}

static void fail_benchmark(const char *message) {
  fprintf(stderr, "%s\n", message);
  exit(EXIT_FAILURE);
}

static void populate_grid_memory(SpatialMemory *sm, const Coord *coords,
                                 size_t observe_ops, size_t grid_cells) {
  for (size_t i = 0; i < observe_ops; i++) {
    uint64_t payload = (uint64_t)i;
    const Coord *coord = &coords[i % grid_cells];
    // Real ingest drives SpatialMemory_observe with an HDF5 timestamp; the
    // benchmark has no such stream, so use the loop counter as a monotonic
    // stand-in. The precise value is irrelevant for throughput measurements.
    if (!SpatialMemory_observe(sm, (double)i, coord->lat, coord->lng,
                               &payload, sizeof(payload))) {
      fail_benchmark("SpatialMemory_observe failed during benchmark setup");
    }
  }
}

static void run_same_cell_observe(size_t observe_ops) {
  SpatialMemory *sm = SpatialMemory_new(DEFAULT_RESOLUTION,
                                        DEFAULT_CAPACITY,
                                        DEFAULT_PRECISION,
                                        0);
  if (!sm) fail_benchmark("Failed to create SpatialMemory for same-cell benchmark");

  const double lat = 37.7749;
  const double lng = -122.4194;
  double start = monotonic_seconds();
  for (size_t i = 0; i < observe_ops; i++) {
    uint64_t payload = (uint64_t)i;
    if (!SpatialMemory_observe(sm, (double)i, lat, lng, &payload,
                               sizeof(payload))) {
      fail_benchmark("SpatialMemory_observe failed in same-cell benchmark");
    }
  }
  double elapsed = monotonic_seconds() - start;

  printf("observe_same_cell  ops=%zu  tiles=%zu  secs=%.3f  ops/sec=%.0f\n",
         observe_ops, SpatialMemory_tile_count(sm), elapsed,
         elapsed > 0.0 ? (double)observe_ops / elapsed : 0.0);
  SpatialMemory_free(sm);
}

static void run_grid_observe(const Coord *coords, size_t observe_ops,
                             size_t grid_cells) {
  SpatialMemory *sm = SpatialMemory_new(DEFAULT_RESOLUTION,
                                        DEFAULT_CAPACITY,
                                        DEFAULT_PRECISION,
                                        0);
  if (!sm) fail_benchmark("Failed to create SpatialMemory for grid benchmark");

  double start = monotonic_seconds();
  populate_grid_memory(sm, coords, observe_ops, grid_cells);
  double elapsed = monotonic_seconds() - start;

  printf("observe_grid       ops=%zu  tiles=%zu  secs=%.3f  ops/sec=%.0f\n",
         observe_ops, SpatialMemory_tile_count(sm), elapsed,
         elapsed > 0.0 ? (double)observe_ops / elapsed : 0.0);
  SpatialMemory_free(sm);
}

static void run_query_intervals(const Coord *coords, size_t observe_ops,
                                size_t grid_cells, size_t query_ops,
                                int k_ring, size_t top) {
  SpatialMemory *sm = SpatialMemory_new(DEFAULT_RESOLUTION,
                                        DEFAULT_CAPACITY,
                                        DEFAULT_PRECISION,
                                        0);
  if (!sm) {
    fail_benchmark("Failed to create SpatialMemory for query_intervals benchmark");
  }

  populate_grid_memory(sm, coords, observe_ops, grid_cells);

  SpatialMemoryInterval *out = NULL;
  if (top > 0) {
    out = (SpatialMemoryInterval *)malloc(top * sizeof(SpatialMemoryInterval));
    if (!out) {
      SpatialMemory_free(sm);
      fail_benchmark("Failed to allocate query_intervals results buffer");
    }
  }

  double min_lat = coords[0].lat, max_lat = coords[0].lat;
  double min_lng = coords[0].lng, max_lng = coords[0].lng;
  for (size_t i = 1; i < grid_cells; ++i) {
    if (coords[i].lat < min_lat) min_lat = coords[i].lat;
    if (coords[i].lat > max_lat) max_lat = coords[i].lat;
    if (coords[i].lng < min_lng) min_lng = coords[i].lng;
    if (coords[i].lng > max_lng) max_lng = coords[i].lng;
  }

  size_t hits = 0;
  double start = monotonic_seconds();
  for (size_t i = 0; i < query_ops; ++i) {
    // Deterministic pseudo-random lat/lng within the populated bounding box
    // (mul-constant hash, no external RNG — keeps bench reproducible).
    double u = (double)((i * 2654435761u) & 0xFFFFu) / 65535.0;
    double v = (double)((i * 40503u) & 0xFFFFu) / 65535.0;
    double lat = min_lat + u * (max_lat - min_lat);
    double lng = min_lng + v * (max_lng - min_lng);
    size_t n = SpatialMemory_query_intervals(sm, lat, lng, k_ring, out, top);
    if (n > 0) hits++;
  }
  double elapsed = monotonic_seconds() - start;

  double mean_us = query_ops > 0 ? (elapsed / (double)query_ops) * 1e6 : 0.0;
  printf("query_intervals    ops=%zu  k_ring=%d  top=%zu  hits=%zu  secs=%.3f  ops/sec=%.0f  mean_us=%.3f\n",
         query_ops, k_ring, top, hits, elapsed,
         elapsed > 0.0 ? (double)query_ops / elapsed : 0.0, mean_us);
  free(out);
  SpatialMemory_free(sm);
}

static void run_query_similar(const Coord *coords, size_t observe_ops,
                              size_t grid_cells, size_t query_ops,
                              size_t dim, size_t exemplar_capacity) {
  SpatialMemory *sm = SpatialMemory_new(DEFAULT_RESOLUTION,
                                        DEFAULT_CAPACITY,
                                        DEFAULT_PRECISION,
                                        exemplar_capacity);
  if (!sm) fail_benchmark("Failed to create SpatialMemory for query_similar benchmark");

  // Per-observation float vector — deterministically seeded by index so the
  // reservoir gets a spread of values rather than identical bytes.
  float *vec = (float *)malloc(dim * sizeof(float));
  if (!vec) {
    SpatialMemory_free(sm);
    fail_benchmark("Failed to allocate embedding vector for query_similar setup");
  }
  for (size_t i = 0; i < observe_ops; ++i) {
    const Coord *coord = &coords[i % grid_cells];
    for (size_t d = 0; d < dim; ++d) {
      vec[d] = (float)((i * 2654435761u + d * 40503u) & 0xFFFFu) / 65535.0f;
    }
    if (!SpatialMemory_observe(sm, (double)i, coord->lat, coord->lng, vec,
                               dim * sizeof(float))) {
      free(vec);
      SpatialMemory_free(sm);
      fail_benchmark("SpatialMemory_observe failed during query_similar setup");
    }
  }

  float *query = (float *)malloc(dim * sizeof(float));
  if (!query) {
    free(vec);
    SpatialMemory_free(sm);
    fail_benchmark("Failed to allocate query vector");
  }
  for (size_t d = 0; d < dim; ++d) query[d] = (float)d / (float)dim;

  const size_t TOP = 10;
  SpatialMemorySimilar *out = (SpatialMemorySimilar *)malloc(
      TOP * sizeof(SpatialMemorySimilar));
  if (!out) {
    free(query);
    free(vec);
    SpatialMemory_free(sm);
    fail_benchmark("Failed to allocate similar result buffer");
  }

  size_t hits = 0;
  double start = monotonic_seconds();
  for (size_t i = 0; i < query_ops; ++i) {
    size_t n = SpatialMemory_query_similar(sm, query, dim, 0.0, 0.0, -1, out,
                                           TOP);
    if (n > 0) hits++;
  }
  double elapsed = monotonic_seconds() - start;

  double mean_us = query_ops > 0 ? (elapsed / (double)query_ops) * 1e6 : 0.0;
  printf("query_similar      ops=%zu  dim=%zu  exemplars=%zu  tiles=%zu  hits=%zu  secs=%.3f  ops/sec=%.0f  mean_us=%.3f\n",
         query_ops, dim, exemplar_capacity, SpatialMemory_tile_count(sm),
         hits, elapsed,
         elapsed > 0.0 ? (double)query_ops / elapsed : 0.0, mean_us);

  free(out);
  free(query);
  free(vec);
  SpatialMemory_free(sm);
}

static void run_grid_query(const Coord *coords, size_t observe_ops,
                           size_t query_ops, size_t grid_cells) {
  SpatialMemory *sm = SpatialMemory_new(DEFAULT_RESOLUTION,
                                        DEFAULT_CAPACITY,
                                        DEFAULT_PRECISION,
                                        0);
  if (!sm) fail_benchmark("Failed to create SpatialMemory for query benchmark");

  populate_grid_memory(sm, coords, observe_ops, grid_cells);

  double total = 0.0;
  double start = monotonic_seconds();
  for (size_t i = 0; i < query_ops; i++) {
    const Coord *coord = &coords[i % grid_cells];
    double count = 0.0;
    if (!SpatialMemory_query(sm, coord->lat, coord->lng,
                             DEFAULT_CAPACITY - 1, &count)) {
      fail_benchmark("SpatialMemory_query failed in grid benchmark");
    }
    total += count;
  }
  double elapsed = monotonic_seconds() - start;

  printf("query_grid         ops=%zu  total=%.0f  secs=%.3f  ops/sec=%.0f\n",
         query_ops, total, elapsed,
         elapsed > 0.0 ? (double)query_ops / elapsed : 0.0);
  SpatialMemory_free(sm);
}

int main(int argc, char *argv[]) {
  size_t observe_ops = 200000;
  size_t query_ops = 200000;
  size_t grid_cells = 1024;

  if (argc > 1 && !parse_size_arg(argv[1], "observe_ops", &observe_ops)) {
    return 1;
  }
  if (argc > 2 && !parse_size_arg(argv[2], "grid_cells", &grid_cells)) {
    return 1;
  }
  if (argc > 3 && !parse_size_arg(argv[3], "query_ops", &query_ops)) {
    return 1;
  }

  Coord *coords = malloc(grid_cells * sizeof(*coords));
  if (!coords) {
    fail_benchmark("Failed to allocate benchmark coordinate grid");
  }
  fill_grid(coords, grid_cells);

  printf("SpatialMemory benchmark\n");
  printf("  observe_ops: %zu\n", observe_ops);
  printf("  grid_cells:  %zu\n", grid_cells);
  printf("  query_ops:   %zu\n", query_ops);

  run_same_cell_observe(observe_ops);
  run_grid_observe(coords, observe_ops, grid_cells);
  run_grid_query(coords, observe_ops, query_ops, grid_cells);
  // Location-trace query latency — representative k_ring=2, top=5.
  run_query_intervals(coords, observe_ops, grid_cells, query_ops, 2, 5);
  // Semantic retrieval latency — modest dim=128, 4 exemplars per tile.
  run_query_similar(coords, observe_ops, grid_cells, query_ops, 128, 4);

  free(coords);
  return 0;
}
