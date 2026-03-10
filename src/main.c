#include <stdio.h>
#include <stdlib.h>
#include "ingest/ingest.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <file.h5> <group> [time_window_sec]\n", argv[0]);
    return 1;
  }

  const char *filepath = argv[1];
  const char *group = argv[2];
  double time_window_sec = argc > 3 ? atof(argv[3]) : 5.0;

  SpatialMemory *sm = SpatialMemory_new(DEFAULT_RESOLUTION, DEFAULT_CAPACITY, DEFAULT_PRECISION);
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

  IngestReader_run(reader, sm, time_window_sec);

  printf("Tiles created: %zu\n", SpatialMemory_tile_count(sm));
  HashTableIterator it = HashTable_iterator(sm->tiles);
  while (HashTable_next(&it)) {
    Tile *tile = (Tile *)it.value;
    double current = Tile_query(tile, 0);                   // current window only
    double total = Tile_query(tile, DEFAULT_CAPACITY - 1);  // all windows
    printf("  Cell %s: current=%.0f total=%.0f\n", it.key, current, total);
  }

  IngestReader_close(reader);
  H5Fclose(file);
  SpatialMemory_free(sm);

  return 0;
}
