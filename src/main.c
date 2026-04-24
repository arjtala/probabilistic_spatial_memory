#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ingest/ingest.h"

#ifndef PSM_VERSION
#define PSM_VERSION "unknown"
#endif

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
    default:
      print_usage(argv[0]);
      return false;
    }
  }

  if (!apply_positional_args(options, argc, argv, optind)) {
    return false;
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

  sm = SpatialMemory_new(options.h3_resolution, options.capacity,
                         options.precision, 0);
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
