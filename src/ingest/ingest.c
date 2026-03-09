#include <stdlib.h>
#include "ingest/ingest.h"

IngestReader *IngestReader_open(hid_t file, const char *group) {
  IngestReader *reader = (IngestReader *)malloc(sizeof(IngestReader));

  hid_t grp = H5Gopen(file, group, H5P_DEFAULT);
  reader->dataset_ts = H5Dopen(grp, TIMESTAMPS, H5P_DEFAULT);
  reader->dataset_lat = H5Dopen(grp, LAT, H5P_DEFAULT);
  reader->dataset_lng = H5Dopen(grp, LNG, H5P_DEFAULT);
  reader->dataset_emb = H5Dopen(grp, EMBEDDINGS, H5P_DEFAULT);
  H5Gclose(grp);

  hid_t space = H5Dget_space(reader->dataset_ts);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(space, dims, NULL);
  reader->n_records = dims[0];
  H5Sclose(space);

  hid_t emb_space = H5Dget_space(reader->dataset_emb);
  hsize_t emb_dims[2];
  H5Sget_simple_extent_dims(emb_space, emb_dims, NULL);
  reader->emb_dimension = emb_dims[1];
  H5Sclose(emb_space);

  reader->cursor = 0;
  reader->embedding_buf = malloc(reader->emb_dimension * sizeof(float));

  return reader;
}

bool IngestReader_next(IngestReader *reader, IngestRecord *record) {
  if (reader->cursor >= reader->n_records)
    return false;

  hsize_t offset1 = reader->cursor;
  hsize_t count1 = 1;
  hid_t mem1 = H5Screate_simple(1, &count1, NULL);

  // timestamp
  hid_t ts_space = H5Dget_space(reader->dataset_ts);
  H5Sselect_hyperslab(ts_space, H5S_SELECT_SET, &offset1, NULL, &count1, NULL);
  H5Dread(reader->dataset_ts, H5T_NATIVE_DOUBLE, mem1, ts_space, H5P_DEFAULT, &record->timestamp);
  H5Sclose(ts_space);

  // latitude
  hid_t lat_space = H5Dget_space(reader->dataset_lat);
  H5Sselect_hyperslab(lat_space, H5S_SELECT_SET, &offset1, NULL, &count1, NULL);
  H5Dread(reader->dataset_lat, H5T_NATIVE_DOUBLE, mem1, lat_space, H5P_DEFAULT, &record->lat);
  H5Sclose(lat_space);

  // longitude
  hid_t lng_space = H5Dget_space(reader->dataset_lng);
  H5Sselect_hyperslab(lng_space, H5S_SELECT_SET, &offset1, NULL, &count1, NULL);
  H5Dread(reader->dataset_lng, H5T_NATIVE_DOUBLE, mem1, lng_space, H5P_DEFAULT, &record->lng);
  H5Sclose(lng_space);

  // embedding (2D: select one row)
  hsize_t cur = reader->cursor;
  hsize_t emb_dim = reader->emb_dimension;
  hsize_t offset2[2] = {cur, 0};
  hsize_t count2[2] = {1, emb_dim};
  hid_t mem2 = H5Screate_simple(1, &emb_dim, NULL);
  hid_t emb_space = H5Dget_space(reader->dataset_emb);
  H5Sselect_hyperslab(emb_space, H5S_SELECT_SET, offset2, NULL, count2, NULL);
  H5Dread(reader->dataset_emb, H5T_NATIVE_FLOAT, mem2, emb_space, H5P_DEFAULT,
          reader->embedding_buf);

  H5Sclose(emb_space);
  H5Sclose(mem1);
  H5Sclose(mem2);

  record->embedding = reader->embedding_buf;
  record->embedding_dim = reader->emb_dimension;
  reader->cursor++;

  return true;
}

void IngestReader_close(IngestReader *reader) {
  H5Dclose(reader->dataset_ts);
  H5Dclose(reader->dataset_lat);
  H5Dclose(reader->dataset_lng);
  H5Dclose(reader->dataset_emb);
  free(reader->embedding_buf);
  free(reader);
}

void IngestReader_run(IngestReader *reader, SpatialMemory *sm, const double time_window_sec) {
  IngestRecord record;
  double last_adv = -1.0;
  while (IngestReader_next(reader, &record)) {
    if (last_adv < 0.0) {
      last_adv = record.timestamp;
    }
    if (record.timestamp - last_adv >= time_window_sec) {
      SpatialMemory_advance_all(sm);
      last_adv = record.timestamp;
    }
    SpatialMemory_observe(sm, record.lat, record.lng, record.embedding,
                          record.embedding_dim * sizeof(float));
  }
}
