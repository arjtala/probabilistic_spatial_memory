#include <stdio.h>
#include <stdlib.h>
#include "ingest/ingest.h"

IngestReader *IngestReader_open(hid_t file, const char *group) {
  if (file < 0) {
    fprintf(stderr, "IngestReader_open: invalid file handle\n");
    return NULL;
  }

  hid_t grp = H5Gopen(file, group, H5P_DEFAULT);
  if (grp < 0) {
    fprintf(stderr, "IngestReader_open: group '%s' not found\n", group);
    return NULL;
  }

  IngestReader *reader = (IngestReader *)malloc(sizeof(IngestReader));
  if (!reader) {
    H5Gclose(grp);
    return NULL;
  }

  reader->dataset_ts = H5Dopen(grp, TIMESTAMPS, H5P_DEFAULT);
  reader->dataset_lat = H5Dopen(grp, LAT, H5P_DEFAULT);
  reader->dataset_lng = H5Dopen(grp, LNG, H5P_DEFAULT);
  reader->dataset_emb = H5Dopen(grp, EMBEDDINGS, H5P_DEFAULT);

  // Try opening attention_maps (optional — not all groups have it)
  reader->dataset_attn = -1;
  reader->attn_size = 0;
  reader->attn_buf = NULL;

  hid_t attn_ds = H5Dopen(grp, ATTENTION_MAPS, H5P_DEFAULT);
  if (attn_ds >= 0) {
    hid_t attn_space = H5Dget_space(attn_ds);
    int rank = H5Sget_simple_extent_ndims(attn_space);
    if (rank == 3) {
      hsize_t attn_dims[3];
      H5Sget_simple_extent_dims(attn_space, attn_dims, NULL);
      reader->dataset_attn = attn_ds;
      reader->attn_size = attn_dims[1]; // spatial dim (e.g. 14)
      reader->attn_buf = malloc(attn_dims[1] * attn_dims[2] * sizeof(float));
    } else {
      H5Dclose(attn_ds);
    }
    H5Sclose(attn_space);
  }

  H5Gclose(grp);

  if (reader->dataset_ts < 0 || reader->dataset_lat < 0 ||
      reader->dataset_lng < 0 || reader->dataset_emb < 0) {
    fprintf(stderr, "IngestReader_open: missing datasets in group '%s'\n", group);
    if (reader->dataset_attn >= 0) H5Dclose(reader->dataset_attn);
    free(reader->attn_buf);
    free(reader);
    return NULL;
  }

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

  // Read attention map if available
  if (reader->dataset_attn >= 0) {
    hsize_t attn_sz = reader->attn_size;
    hsize_t offset3[3] = {cur, 0, 0};
    hsize_t count3[3] = {1, attn_sz, attn_sz};
    hsize_t mem_dims[2] = {attn_sz, attn_sz};
    hid_t mem_attn = H5Screate_simple(2, mem_dims, NULL);
    hid_t attn_space = H5Dget_space(reader->dataset_attn);
    H5Sselect_hyperslab(attn_space, H5S_SELECT_SET, offset3, NULL, count3, NULL);
    H5Dread(reader->dataset_attn, H5T_NATIVE_FLOAT, mem_attn, attn_space, H5P_DEFAULT,
            reader->attn_buf);
    H5Sclose(attn_space);
    H5Sclose(mem_attn);
    record->attention_map = reader->attn_buf;
    record->attn_size = reader->attn_size;
  } else {
    record->attention_map = NULL;
    record->attn_size = 0;
  }

  reader->cursor++;

  return true;
}

void IngestReader_close(IngestReader *reader) {
  H5Dclose(reader->dataset_ts);
  H5Dclose(reader->dataset_lat);
  H5Dclose(reader->dataset_lng);
  H5Dclose(reader->dataset_emb);
  if (reader->dataset_attn >= 0) H5Dclose(reader->dataset_attn);
  free(reader->embedding_buf);
  free(reader->attn_buf);
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
