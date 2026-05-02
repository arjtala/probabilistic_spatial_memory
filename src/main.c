#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <h3/h3api.h>
#include "core/tile.h"
#include "ingest/ingest.h"

static void cell_center_degrees(H3Index cell, double *out_lat, double *out_lng) {
  LatLng center;
  cellToLatLng(cell, &center);
  *out_lat = radsToDegs(center.lat);
  *out_lng = radsToDegs(center.lng);
}

#ifndef PSM_VERSION
#define PSM_VERSION "unknown"
#endif

// Synthetic getopt values for long-only flags introduced by --last-seen mode.
enum {
  LAST_SEEN_OPT_VAL = 0x1000,
  K_RING_OPT_VAL,
  TOP_OPT_VAL,
  SEARCH_OPT_VAL,
  CENTER_OPT_VAL,
  EXEMPLARS_OPT_VAL,
  SEED_OPT_VAL,
};

typedef enum {
  OUTPUT_TEXT = 0,
  OUTPUT_JSON = 1,
} OutputFormat;

typedef struct {
  const char *filepath;
  const char *group;
  double time_window_sec;
  int h3_resolution;
  size_t capacity;
  size_t precision;
  OutputFormat output_format;
  bool last_seen_mode;
  double last_seen_lat;
  double last_seen_lng;
  int last_seen_k_ring;
  size_t last_seen_top;
  bool search_mode;
  const char *search_path;
  bool has_center;
  double center_lat;
  double center_lng;
  size_t exemplar_capacity;
  bool has_seed;
  uint64_t seed;
} CliOptions;

typedef struct {
  size_t capacity;
  OutputFormat output_format;
  bool first;
} TilePrinterState;

static bool parse_positive_double(const char *text, const char *name,
                                  double *out_value) {
  char *end = NULL;
  double value;

  errno = 0;
  value = strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0') {
    fprintf(stderr, "Invalid %s: '%s'\n", name, text);
    return false;
  }
  if (value <= 0.0) {
    fprintf(stderr, "%s must be greater than 0, got '%s'\n", name, text);
    return false;
  }
  *out_value = value;
  return true;
}

static bool parse_int_in_range(const char *text, const char *name, int min_value,
                               int max_value, int *out_value) {
  char *end = NULL;
  long value;

  errno = 0;
  value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    fprintf(stderr, "Invalid %s: '%s'\n", name, text);
    return false;
  }
  if (value < min_value || value > max_value) {
    fprintf(stderr, "%s must be in [%d, %d], got '%s'\n",
            name, min_value, max_value, text);
    return false;
  }
  *out_value = (int)value;
  return true;
}

static bool parse_lat_lng_pair(const char *text, double *out_lat,
                               double *out_lng) {
  if (!text || !out_lat || !out_lng) return false;
  const char *comma = strchr(text, ',');
  if (!comma || comma == text || *(comma + 1) == '\0') {
    fprintf(stderr, "Invalid lat,lng pair: '%s' (expected LAT,LNG)\n", text);
    return false;
  }
  char *end = NULL;
  errno = 0;
  double lat = strtod(text, &end);
  if (errno != 0 || end != comma) {
    fprintf(stderr, "Invalid latitude in '%s'\n", text);
    return false;
  }
  errno = 0;
  double lng = strtod(comma + 1, &end);
  if (errno != 0 || end == comma + 1 || *end != '\0') {
    fprintf(stderr, "Invalid longitude in '%s'\n", text);
    return false;
  }
  *out_lat = lat;
  *out_lng = lng;
  return true;
}

static bool parse_size_t_in_range(const char *text, const char *name,
                                  size_t min_value, size_t max_value,
                                  size_t *out_value) {
  char *end = NULL;
  unsigned long long value;

  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    fprintf(stderr, "Invalid %s: '%s'\n", name, text);
    return false;
  }
  if (value < min_value || value > max_value || value > (unsigned long long)SIZE_MAX) {
    fprintf(stderr, "%s must be in [%zu, %zu], got '%s'\n",
            name, min_value, max_value, text);
    return false;
  }
  *out_value = (size_t)value;
  return true;
}

static void print_json_string(FILE *stream, const char *text) {
  if (!stream) return;
  fputc('"', stream);
  if (text) {
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor; cursor++) {
      switch (*cursor) {
      case '\\':
        fputs("\\\\", stream);
        break;
      case '"':
        fputs("\\\"", stream);
        break;
      case '\b':
        fputs("\\b", stream);
        break;
      case '\f':
        fputs("\\f", stream);
        break;
      case '\n':
        fputs("\\n", stream);
        break;
      case '\r':
        fputs("\\r", stream);
        break;
      case '\t':
        fputs("\\t", stream);
        break;
      default:
        if (*cursor < 0x20) {
          fprintf(stream, "\\u%04x", (unsigned int)*cursor);
        } else {
          fputc((int)*cursor, stream);
        }
        break;
      }
    }
  }
  fputc('"', stream);
}

static void print_usage(const char *program) {
  fprintf(stderr, "Usage:\n");
  fprintf(stderr,
          "  %s -f <file.h5> [-g <group>] [-t <sec>] [-r <res>] [-C <capacity>] [-p <precision>] [-j]\n",
          program);
  fprintf(stderr,
          "  %s <file.h5> [group] [time_window_sec] [h3_resolution] [capacity] [precision]\n",
          program);
  fprintf(stderr, "\nOptions:\n");
  fprintf(stderr, "  -f, --file <path>       HDF5 feature file\n");
  fprintf(stderr, "  -g, --group <name>      Group to ingest (default: %s)\n", DINO);
  fprintf(stderr, "  -t, --time-window <sec> Time window in seconds (default: %.1f)\n", 5.0);
  fprintf(stderr, "  -r, --resolution <res>  H3 resolution 0-15 (default: %d)\n",
          DEFAULT_RESOLUTION);
  fprintf(stderr, "  -C, --capacity <count>  Ring buffer capacity (default: %d)\n",
          DEFAULT_CAPACITY);
  fprintf(stderr, "  -p, --precision <bits>  HLL precision (default: %d, valid: %zu-%zu)\n",
          DEFAULT_PRECISION, RingBuffer_precision_min(),
          RingBuffer_precision_max());
  fprintf(stderr, "  -j, --json              Emit machine-readable JSON summary\n");
  fprintf(stderr, "  -h, --help              Print this help\n");
  fprintf(stderr, "  -v, --version           Print psm version and exit\n");
  fprintf(stderr, "  --last-seen LAT,LNG     Query last-seen intervals around (lat,lng)\n");
  fprintf(stderr, "  --k-ring N              H3 neighborhood radius (default: 0)\n");
  fprintf(stderr, "  --top N                 Max results to print (default: 5)\n");
  fprintf(stderr, "  --search <path>         Retrieve by embedding similarity (binary float32 LE vector)\n");
  fprintf(stderr, "  --center LAT,LNG        Optional geographic center for --search\n");
  fprintf(stderr, "  --exemplars N           Per-tile reservoir size (auto-set to 8 with --search)\n");
  fprintf(stderr, "  --seed N                Reservoir-sampler seed (uint64, decimal or 0x...) for reproducible runs\n");
  fprintf(stderr, "\nRetention window = capacity * time-window (default: 12 * 5.0s = 60s).\n");
  fprintf(stderr, "Observations older than this age out of each tile's ring buffer.\n");
  fprintf(stderr, "For multi-minute sessions, widen with e.g. -C 30 -t 60 (30-min window).\n");
}

static void cli_options_init(CliOptions *options) {
  if (!options) return;
  options->filepath = NULL;
  options->group = DINO;
  options->time_window_sec = 5.0;
  options->h3_resolution = DEFAULT_RESOLUTION;
  options->capacity = DEFAULT_CAPACITY;
  options->precision = DEFAULT_PRECISION;
  options->output_format = OUTPUT_TEXT;
  options->last_seen_mode = false;
  options->last_seen_lat = 0.0;
  options->last_seen_lng = 0.0;
  options->last_seen_k_ring = 0;
  options->last_seen_top = 5;
  options->search_mode = false;
  options->search_path = NULL;
  options->has_center = false;
  options->center_lat = 0.0;
  options->center_lng = 0.0;
  options->exemplar_capacity = 0;
  options->has_seed = false;
  options->seed = 0;
}

static bool load_float_vector_file(const char *path, float **out_data,
                                   size_t *out_dim) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open %s: %s\n", path, strerror(errno));
    return false;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fprintf(stderr, "Failed to seek in %s\n", path);
    fclose(f);
    return false;
  }
  long fsz = ftell(f);
  if (fsz <= 0 || (size_t)fsz % sizeof(float) != 0) {
    fprintf(stderr,
            "Invalid size for %s (%ld bytes; must be a nonzero multiple of %zu)\n",
            path, fsz, sizeof(float));
    fclose(f);
    return false;
  }
  rewind(f);
  size_t dim = (size_t)fsz / sizeof(float);
  float *data = (float *)malloc((size_t)fsz);
  if (!data) {
    fprintf(stderr, "Failed to allocate query vector (%ld bytes)\n", fsz);
    fclose(f);
    return false;
  }
  if (fread(data, 1, (size_t)fsz, f) != (size_t)fsz) {
    fprintf(stderr, "Failed to read query vector from %s\n", path);
    free(data);
    fclose(f);
    return false;
  }
  fclose(f);
  *out_data = data;
  *out_dim = dim;
  return true;
}

static bool apply_positional_args(CliOptions *options, int argc, char *argv[],
                                  int start_index) {
  int index = start_index;

  if (!options) return false;
  if (!options->filepath && index < argc) {
    options->filepath = argv[index++];
  }
  if (index < argc) {
    options->group = argv[index++];
  }
  if (index < argc &&
      !parse_positive_double(argv[index++], "time window",
                             &options->time_window_sec)) {
    return false;
  }
  if (index < argc &&
      !parse_int_in_range(argv[index++], "H3 resolution", 0, 15,
                          &options->h3_resolution)) {
    return false;
  }
  if (index < argc &&
      !parse_size_t_in_range(argv[index++], "capacity", 1, SIZE_MAX,
                             &options->capacity)) {
    return false;
  }
  if (index < argc &&
      !parse_size_t_in_range(argv[index++], "precision",
                             RingBuffer_precision_min(),
                             RingBuffer_precision_max(),
                             &options->precision)) {
    return false;
  }
  if (index < argc) {
    fprintf(stderr, "Unexpected extra positional argument: '%s'\n", argv[index]);
    return false;
  }
  return true;
}

static bool parse_cli_options(int argc, char *argv[], CliOptions *options) {
  int opt;
  static const struct option long_options[] = {
      {"file", required_argument, NULL, 'f'},
      {"group", required_argument, NULL, 'g'},
      {"time-window", required_argument, NULL, 't'},
      {"resolution", required_argument, NULL, 'r'},
      {"capacity", required_argument, NULL, 'C'},
      {"precision", required_argument, NULL, 'p'},
      {"json", no_argument, NULL, 'j'},
      {"help", no_argument, NULL, 'h'},
      {"version", no_argument, NULL, 'v'},
      {"last-seen", required_argument, NULL, LAST_SEEN_OPT_VAL},
      {"k-ring", required_argument, NULL, K_RING_OPT_VAL},
      {"top", required_argument, NULL, TOP_OPT_VAL},
      {"search", required_argument, NULL, SEARCH_OPT_VAL},
      {"center", required_argument, NULL, CENTER_OPT_VAL},
      {"exemplars", required_argument, NULL, EXEMPLARS_OPT_VAL},
      {"seed", required_argument, NULL, SEED_OPT_VAL},
      {0, 0, 0, 0},
  };

  if (!options) return false;
  cli_options_init(options);

  while ((opt = getopt_long(argc, argv, "f:g:t:r:C:p:jhv", long_options, NULL)) != -1) {
    switch (opt) {
    case 'f':
      options->filepath = optarg;
      break;
    case 'g':
      options->group = optarg;
      break;
    case 't':
      if (!parse_positive_double(optarg, "time window",
                                 &options->time_window_sec)) {
        return false;
      }
      break;
    case 'r':
      if (!parse_int_in_range(optarg, "H3 resolution", 0, 15,
                              &options->h3_resolution)) {
        return false;
      }
      break;
    case 'C':
      if (!parse_size_t_in_range(optarg, "capacity", 1, SIZE_MAX,
                                 &options->capacity)) {
        return false;
      }
      break;
    case 'p':
      if (!parse_size_t_in_range(optarg, "precision",
                                 RingBuffer_precision_min(),
                                 RingBuffer_precision_max(),
                                 &options->precision)) {
        return false;
      }
      break;
    case 'j':
      options->output_format = OUTPUT_JSON;
      break;
    case 'h':
      print_usage(argv[0]);
      exit(0);
    case 'v':
      printf("psm version %s\n", PSM_VERSION);
      exit(0);
    case LAST_SEEN_OPT_VAL:
      if (!parse_lat_lng_pair(optarg, &options->last_seen_lat,
                              &options->last_seen_lng)) {
        return false;
      }
      options->last_seen_mode = true;
      break;
    case K_RING_OPT_VAL:
      if (!parse_int_in_range(optarg, "k-ring", 0, 127,
                              &options->last_seen_k_ring)) {
        return false;
      }
      break;
    case TOP_OPT_VAL:
      if (!parse_size_t_in_range(optarg, "top", 1, SIZE_MAX,
                                 &options->last_seen_top)) {
        return false;
      }
      break;
    case SEARCH_OPT_VAL:
      options->search_path = optarg;
      options->search_mode = true;
      break;
    case CENTER_OPT_VAL:
      if (!parse_lat_lng_pair(optarg, &options->center_lat,
                              &options->center_lng)) {
        return false;
      }
      options->has_center = true;
      break;
    case EXEMPLARS_OPT_VAL:
      if (!parse_size_t_in_range(optarg, "exemplars", 0, SIZE_MAX,
                                 &options->exemplar_capacity)) {
        return false;
      }
      break;
    case SEED_OPT_VAL: {
      char *end = NULL;
      errno = 0;
      unsigned long long parsed = strtoull(optarg, &end, 0);
      if (errno != 0 || end == optarg || *end != '\0') {
        fprintf(stderr, "Invalid seed: '%s'\n", optarg);
        return false;
      }
      options->seed = (uint64_t)parsed;
      options->has_seed = true;
      break;
    }
    default:
      print_usage(argv[0]);
      return false;
    }
  }

  if (!apply_positional_args(options, argc, argv, optind)) {
    return false;
  }
  if (options->last_seen_mode && options->search_mode) {
    fprintf(stderr, "--last-seen and --search are mutually exclusive\n");
    return false;
  }
  // Auto-enable a modest exemplar reservoir when similarity search is requested
  // and the user hasn't overridden --exemplars explicitly.
  if (options->search_mode && options->exemplar_capacity == 0) {
    options->exemplar_capacity = 8;
  }
  if (!options->filepath) {
    print_usage(argv[0]);
    return false;
  }
  if (!options->group || options->group[0] == '\0') {
    fprintf(stderr, "Group must not be empty\n");
    return false;
  }
  return true;
}

static void print_last_seen_text(const SpatialMemoryInterval *results, size_t n,
                                 size_t total_found, const CliOptions *opts,
                                 const IngestReader *reader,
                                 size_t tile_count) {
  printf("File: %s\n", opts->filepath);
  printf("Group: %s\n", opts->group);
  printf("Time window: %.3f sec\n", opts->time_window_sec);
  printf("H3 resolution: %d\n", opts->h3_resolution);
  printf("Capacity: %zu\n", opts->capacity);
  printf("Precision: %zu\n", opts->precision);
  printf("Records: %zu\n", reader->n_records);
  printf("Embedding dim: %zu\n", reader->emb_dimension);
  printf("Tiles created: %zu\n", tile_count);
  printf("Query: lat=%.6f lng=%.6f k_ring=%d top=%zu\n",
         opts->last_seen_lat, opts->last_seen_lng,
         opts->last_seen_k_ring, opts->last_seen_top);
  printf("Last seen (%zu shown of %zu in neighborhood):\n", n, total_found);
  for (size_t i = 0; i < n; ++i) {
    char cell_str[H3_INDEX_HEX_STRING_LENGTH];
    double lat, lng;
    h3ToString(results[i].cell, cell_str, sizeof(cell_str));
    cell_center_degrees(results[i].cell, &lat, &lng);
    printf("  Cell %s  (%.6f,%.6f)  t=[%.3f, %.3f]  count=%.3f\n",
           cell_str, lat, lng,
           results[i].t_min, results[i].t_max, results[i].count);
  }
}

static void print_last_seen_json(const SpatialMemoryInterval *results,
                                 size_t n, const CliOptions *opts,
                                 const IngestReader *reader,
                                 size_t tile_count) {
  fputs("{\n", stdout);
  fputs("  \"schema_version\": 1,\n", stdout);
  fputs("  \"mode\": \"last_seen\",\n", stdout);
  fputs("  \"group\": ", stdout);
  print_json_string(stdout, opts->group);
  fputs(",\n", stdout);
  printf("  \"time_window_sec\": %.3f,\n", opts->time_window_sec);
  printf("  \"h3_resolution\": %d,\n", opts->h3_resolution);
  printf("  \"capacity\": %zu,\n", opts->capacity);
  printf("  \"precision\": %zu,\n", opts->precision);
  printf("  \"record_count\": %zu,\n", reader->n_records);
  printf("  \"embedding_dim\": %zu,\n", reader->emb_dimension);
  printf("  \"tile_count\": %zu,\n", tile_count);
  printf("  \"query\": {\"lat\": %.6f, \"lng\": %.6f, \"k_ring\": %d, \"top\": %zu},\n",
         opts->last_seen_lat, opts->last_seen_lng,
         opts->last_seen_k_ring, opts->last_seen_top);
  fputs("  \"results\": [\n", stdout);
  for (size_t i = 0; i < n; ++i) {
    char cell_str[H3_INDEX_HEX_STRING_LENGTH];
    double lat, lng;
    h3ToString(results[i].cell, cell_str, sizeof(cell_str));
    cell_center_degrees(results[i].cell, &lat, &lng);
    printf("    {\"cell\":\"%s\",\"lat\":%.6f,\"lng\":%.6f,"
           "\"t_min\":%.3f,\"t_max\":%.3f,\"count\":%.3f}%s\n",
           cell_str, lat, lng,
           results[i].t_min, results[i].t_max, results[i].count,
           (i + 1 == n) ? "" : ",");
  }
  fputs("  ]\n}\n", stdout);
}

static bool run_last_seen_output(SpatialMemory *sm, const CliOptions *opts,
                                 const IngestReader *reader) {
  size_t tile_count = SpatialMemory_tile_count(sm);
  size_t top = opts->last_seen_top;
  SpatialMemoryInterval *results = NULL;
  if (top > 0) {
    results = (SpatialMemoryInterval *)calloc(top, sizeof(SpatialMemoryInterval));
    if (!results) {
      fprintf(stderr, "Failed to allocate last-seen results buffer\n");
      return false;
    }
  }
  size_t total_found = SpatialMemory_query_intervals(
      sm, opts->last_seen_lat, opts->last_seen_lng, opts->last_seen_k_ring,
      results, top);
  size_t written = (total_found < top) ? total_found : top;

  if (opts->output_format == OUTPUT_JSON) {
    print_last_seen_json(results, written, opts, reader, tile_count);
  } else {
    print_last_seen_text(results, written, total_found, opts, reader,
                         tile_count);
  }
  free(results);
  return true;
}

static void print_similar_text(const SpatialMemorySimilar *results, size_t n,
                               size_t total_found, size_t dim,
                               const CliOptions *opts,
                               const IngestReader *reader, size_t tile_count) {
  printf("File: %s\n", opts->filepath);
  printf("Group: %s\n", opts->group);
  printf("Time window: %.3f sec\n", opts->time_window_sec);
  printf("H3 resolution: %d\n", opts->h3_resolution);
  printf("Capacity: %zu\n", opts->capacity);
  printf("Precision: %zu\n", opts->precision);
  printf("Records: %zu\n", reader->n_records);
  printf("Embedding dim: %zu\n", reader->emb_dimension);
  printf("Tiles created: %zu\n", tile_count);
  printf("Query: dim=%zu top=%zu", dim, opts->last_seen_top);
  if (opts->has_center) {
    printf(" center=%.6f,%.6f k_ring=%d", opts->center_lat, opts->center_lng,
           opts->last_seen_k_ring);
  } else {
    printf(" scope=global");
  }
  printf("\n");
  printf("Similar (%zu shown of %zu matched):\n", n, total_found);
  for (size_t i = 0; i < n; ++i) {
    char cell_str[H3_INDEX_HEX_STRING_LENGTH];
    double lat, lng;
    h3ToString(results[i].cell, cell_str, sizeof(cell_str));
    cell_center_degrees(results[i].cell, &lat, &lng);
    printf("  Cell %s  (%.6f,%.6f)  sim=%.4f  exemplar_t=%.3f  t=[%.3f, %.3f]  count=%.3f\n",
           cell_str, lat, lng,
           results[i].similarity, results[i].exemplar_t,
           results[i].t_min, results[i].t_max, results[i].count);
  }
}

static void print_similar_json(const SpatialMemorySimilar *results, size_t n,
                               size_t dim, const CliOptions *opts,
                               const IngestReader *reader, size_t tile_count) {
  fputs("{\n", stdout);
  fputs("  \"schema_version\": 1,\n", stdout);
  fputs("  \"mode\": \"search\",\n", stdout);
  fputs("  \"group\": ", stdout);
  print_json_string(stdout, opts->group);
  fputs(",\n", stdout);
  printf("  \"time_window_sec\": %.3f,\n", opts->time_window_sec);
  printf("  \"h3_resolution\": %d,\n", opts->h3_resolution);
  printf("  \"capacity\": %zu,\n", opts->capacity);
  printf("  \"precision\": %zu,\n", opts->precision);
  printf("  \"record_count\": %zu,\n", reader->n_records);
  printf("  \"embedding_dim\": %zu,\n", reader->emb_dimension);
  printf("  \"tile_count\": %zu,\n", tile_count);
  printf("  \"query\": {\"dim\": %zu, \"top\": %zu", dim, opts->last_seen_top);
  if (opts->has_center) {
    printf(", \"center_lat\": %.6f, \"center_lng\": %.6f, \"k_ring\": %d",
           opts->center_lat, opts->center_lng, opts->last_seen_k_ring);
  }
  fputs("},\n", stdout);
  fputs("  \"results\": [\n", stdout);
  for (size_t i = 0; i < n; ++i) {
    char cell_str[H3_INDEX_HEX_STRING_LENGTH];
    double lat, lng;
    h3ToString(results[i].cell, cell_str, sizeof(cell_str));
    cell_center_degrees(results[i].cell, &lat, &lng);
    printf("    {\"cell\":\"%s\",\"lat\":%.6f,\"lng\":%.6f,"
           "\"similarity\":%.6f,\"exemplar_t\":%.3f,"
           "\"t_min\":%.3f,\"t_max\":%.3f,\"count\":%.3f}%s\n",
           cell_str, lat, lng,
           results[i].similarity, results[i].exemplar_t,
           results[i].t_min, results[i].t_max, results[i].count,
           (i + 1 == n) ? "" : ",");
  }
  fputs("  ]\n}\n", stdout);
}

static bool run_similar_output(SpatialMemory *sm, const CliOptions *opts,
                               const IngestReader *reader) {
  float *query = NULL;
  size_t dim = 0;
  if (!load_float_vector_file(opts->search_path, &query, &dim)) {
    return false;
  }

  size_t tile_count = SpatialMemory_tile_count(sm);
  size_t top = opts->last_seen_top;
  SpatialMemorySimilar *results = NULL;
  if (top > 0) {
    results =
        (SpatialMemorySimilar *)calloc(top, sizeof(SpatialMemorySimilar));
    if (!results) {
      fprintf(stderr, "Failed to allocate similar results buffer\n");
      free(query);
      return false;
    }
  }

  double center_lat = opts->has_center ? opts->center_lat : 0.0;
  double center_lng = opts->has_center ? opts->center_lng : 0.0;
  int k_ring = opts->has_center ? opts->last_seen_k_ring : -1;
  size_t total_found = SpatialMemory_query_similar(
      sm, query, dim, center_lat, center_lng, k_ring, results, top);
  size_t written = (total_found < top) ? total_found : top;

  if (opts->output_format == OUTPUT_JSON) {
    print_similar_json(results, written, dim, opts, reader, tile_count);
  } else {
    print_similar_text(results, written, total_found, dim, opts, reader,
                       tile_count);
  }
  free(results);
  free(query);
  return true;
}

static bool print_tile_summary(H3Index cell_id, Tile *tile, void *user_data) {
  TilePrinterState *state = (TilePrinterState *)user_data;
  char cell_string[H3_INDEX_HEX_STRING_LENGTH];
  double current;
  double total;

  if (!state || !tile || state->capacity == 0) return false;

  current = Tile_query(tile, 0);
  total = Tile_query(tile, state->capacity - 1);
  h3ToString(cell_id, cell_string, sizeof(cell_string));

  if (state->output_format == OUTPUT_JSON) {
    if (!state->first) {
      fputs(",\n", stdout);
    }
    state->first = false;
    printf("    {\"cell\":\"%s\",\"current\":%.3f,\"total\":%.3f}",
           cell_string, current, total);
    return true;
  }

  printf("  Cell %s: current=%.3f total=%.3f\n", cell_string, current, total);
  return true;
}

int main(int argc, char *argv[]) {
  CliOptions options;
  SpatialMemory *sm;
  IngestReader *reader;
  hid_t file;
  size_t tile_count;
  TilePrinterState tile_state;

  if (!parse_cli_options(argc, argv, &options)) {
    return 1;
  }

  // Seed the reservoir sampler before any tile gets created so the very
  // first observe is deterministic. Without --seed the sampler self-seeds
  // entropically on first use (preserving prior behavior).
  if (options.has_seed) {
    Tile_set_random_seed(options.seed);
  }

  sm = SpatialMemory_new(options.h3_resolution, options.capacity,
                         options.precision, options.exemplar_capacity);
  if (!sm) {
    fprintf(stderr, "Failed to initialize spatial memory\n");
    return 1;
  }

  file = H5Fopen(options.filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    fprintf(stderr, "Failed to open HDF5 file: %s\n", options.filepath);
    SpatialMemory_free(sm);
    return 1;
  }

  reader = IngestReader_open(file, options.group);
  if (!reader) {
    fprintf(stderr, "Failed to open group '%s' in %s\n",
            options.group, options.filepath);
    H5Fclose(file);
    SpatialMemory_free(sm);
    return 1;
  }

  if (!IngestReader_run(reader, sm, options.time_window_sec)) {
    fprintf(stderr, "Failed while reading records from group '%s'\n",
            options.group);
    IngestReader_close(reader);
    H5Fclose(file);
    SpatialMemory_free(sm);
    return 1;
  }

  if (options.last_seen_mode) {
    bool ok = run_last_seen_output(sm, &options, reader);
    IngestReader_close(reader);
    H5Fclose(file);
    SpatialMemory_free(sm);
    return ok ? 0 : 1;
  }

  if (options.search_mode) {
    bool ok = run_similar_output(sm, &options, reader);
    IngestReader_close(reader);
    H5Fclose(file);
    SpatialMemory_free(sm);
    return ok ? 0 : 1;
  }

  tile_count = SpatialMemory_tile_count(sm);
  tile_state.capacity = sm->capacity;
  tile_state.output_format = options.output_format;
  tile_state.first = true;

  if (options.output_format == OUTPUT_JSON) {
    fputs("{\n", stdout);
    fputs("  \"schema_version\": 1,\n", stdout);
    fputs("  \"group\": ", stdout);
    print_json_string(stdout, options.group);
    fputs(",\n", stdout);
    printf("  \"time_window_sec\": %.3f,\n", options.time_window_sec);
    printf("  \"h3_resolution\": %d,\n", options.h3_resolution);
    printf("  \"capacity\": %zu,\n", options.capacity);
    printf("  \"precision\": %zu,\n", options.precision);
    printf("  \"record_count\": %zu,\n", reader->n_records);
    printf("  \"embedding_dim\": %zu,\n", reader->emb_dimension);
    printf("  \"tile_count\": %zu,\n", tile_count);
    fputs("  \"tiles\": [\n", stdout);
    if (!SpatialMemory_for_each_tile(sm, print_tile_summary, &tile_state)) {
      fprintf(stderr, "Failed to enumerate spatial memory tiles\n");
      fputs("\n  ]\n}\n", stdout);
      IngestReader_close(reader);
      H5Fclose(file);
      SpatialMemory_free(sm);
      return 1;
    }
    fputs("\n  ]\n}\n", stdout);
  } else {
    printf("File: %s\n", options.filepath);
    printf("Group: %s\n", options.group);
    printf("Time window: %.3f sec\n", options.time_window_sec);
    printf("H3 resolution: %d\n", options.h3_resolution);
    printf("Capacity: %zu\n", options.capacity);
    printf("Precision: %zu\n", options.precision);
    printf("Records: %zu\n", reader->n_records);
    printf("Embedding dim: %zu\n", reader->emb_dimension);
    printf("Tiles created: %zu\n", tile_count);
    printf("Tile summaries:\n");
    if (!SpatialMemory_for_each_tile(sm, print_tile_summary, &tile_state)) {
      fprintf(stderr, "Failed to enumerate spatial memory tiles\n");
      IngestReader_close(reader);
      H5Fclose(file);
      SpatialMemory_free(sm);
      return 1;
    }
  }

  IngestReader_close(reader);
  H5Fclose(file);
  SpatialMemory_free(sm);
  return 0;
}
