#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "ingest/ingest.h"

static bool parse_positive_double(const char *text, const char *name,
                                  double *out_value) {
  char *end = NULL;
  errno = 0;
  double value = strtod(text, &end);
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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <file.h5> <group> [time_window_sec]\n", argv[0]);
    return 1;
  }

  const char *filepath = argv[1];
  const char *group = argv[2];
  double time_window_sec = 5.0;
  if (argc > 3 && !parse_positive_double(argv[3], "time window", &time_window_sec)) {
    return 1;
  }

  SpatialMemory *sm = SpatialMemory_new(DEFAULT_RESOLUTION, DEFAULT_CAPACITY, DEFAULT_PRECISION);
  if (!sm) {
    fprintf(stderr, "Failed to initialize spatial memory\n");
    return 1;
  }
  hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    fprintf(stderr, "Failed to open HDF5 file: %s\n", filepath);
    SpatialMemory_free(sm);
    return 1;
  }
  IngestReader *reader = IngestReader_open(file, group);
  if (!reader) {
    fprintf(stderr, "Failed to open group '%s' in %s\n", group, filepath);
    H5Fclose(file);
    SpatialMemory_free(sm);
    return 1;
  }

  printf("Records: %zu, Embedding dim: %zu\n", reader->n_records, reader->emb_dimension);

  if (!IngestReader_run(reader, sm, time_window_sec)) {
    fprintf(stderr, "Failed while reading records from group '%s'\n", group);
    IngestReader_close(reader);
    H5Fclose(file);
    SpatialMemory_free(sm);
    return 1;
  }

  printf("Tiles created: %zu\n", SpatialMemory_tile_count(sm));
  TileTableIterator it = TileTable_iterator(sm->tiles);
  while (TileTable_next(&it)) {
    Tile *tile = (Tile *)it.value;
    double current = Tile_query(tile, 0);                   // current window only
    double total = Tile_query(tile, DEFAULT_CAPACITY - 1);  // all windows
    char cell_string[H3_INDEX_HEX_STRING_LENGTH];
    h3ToString(it.key, cell_string, sizeof(cell_string));
    printf("  Cell %s: current=%.0f total=%.0f\n", cell_string, current, total);
  }

  IngestReader_close(reader);
  H5Fclose(file);
  SpatialMemory_free(sm);

  return 0;
}
