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

  // Try opening spatial maps: attention_maps (dino) or prediction_maps (jepa)
  reader->dataset_attn = -1;
  reader->attn_size = 0;
  reader->attn_buf = NULL;

  hid_t attn_ds = H5Dopen(grp, ATTENTION_MAPS, H5P_DEFAULT);
  if (attn_ds < 0)
    attn_ds = H5Dopen(grp, PREDICTION_MAPS, H5P_DEFAULT);
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

  // Try opening IMU datasets (optional, both must be present)
  reader->dataset_accel = -1;
  reader->dataset_gyro = -1;
  reader->has_imu = false;

  hid_t accel_ds = H5Dopen(grp, ACCEL, H5P_DEFAULT);
  hid_t gyro_ds = H5Dopen(grp, GYRO, H5P_DEFAULT);
  if (accel_ds >= 0 && gyro_ds >= 0) {
    // Validate both are rank-2 with second dim == 3
    hid_t accel_space = H5Dget_space(accel_ds);
    hid_t gyro_space = H5Dget_space(gyro_ds);
    int accel_rank = H5Sget_simple_extent_ndims(accel_space);
    int gyro_rank = H5Sget_simple_extent_ndims(gyro_space);
    bool valid = false;
    if (accel_rank == 2 && gyro_rank == 2) {
      hsize_t accel_dims[2], gyro_dims[2];
      H5Sget_simple_extent_dims(accel_space, accel_dims, NULL);
      H5Sget_simple_extent_dims(gyro_space, gyro_dims, NULL);
      if (accel_dims[1] == 3 && gyro_dims[1] == 3) {
        valid = true;
      }
    }
    H5Sclose(accel_space);
    H5Sclose(gyro_space);
    if (valid) {
      reader->dataset_accel = accel_ds;
      reader->dataset_gyro = gyro_ds;
      reader->has_imu = true;
    } else {
      H5Dclose(accel_ds);
      H5Dclose(gyro_ds);
    }
  } else {
    if (accel_ds >= 0) H5Dclose(accel_ds);
    if (gyro_ds >= 0) H5Dclose(gyro_ds);
  }

  H5Gclose(grp);

  if (reader->dataset_ts < 0 || reader->dataset_lat < 0 ||
      reader->dataset_lng < 0 || reader->dataset_emb < 0) {
    fprintf(stderr, "IngestReader_open: missing datasets in group '%s'\n", group);
    if (reader->dataset_attn >= 0) H5Dclose(reader->dataset_attn);
    if (reader->dataset_accel >= 0) H5Dclose(reader->dataset_accel);
    if (reader->dataset_gyro >= 0) H5Dclose(reader->dataset_gyro);
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

  // Read IMU data if available
  if (reader->has_imu) {
    hsize_t imu_offset[2] = {cur, 0};
    hsize_t imu_count[2] = {1, 3};
    hsize_t imu_mem_dim = 3;
    hid_t imu_mem = H5Screate_simple(1, &imu_mem_dim, NULL);

    hid_t accel_space = H5Dget_space(reader->dataset_accel);
    H5Sselect_hyperslab(accel_space, H5S_SELECT_SET, imu_offset, NULL, imu_count, NULL);
    H5Dread(reader->dataset_accel, H5T_NATIVE_FLOAT, imu_mem, accel_space, H5P_DEFAULT,
            reader->accel_buf);
    H5Sclose(accel_space);

    hid_t gyro_space = H5Dget_space(reader->dataset_gyro);
    H5Sselect_hyperslab(gyro_space, H5S_SELECT_SET, imu_offset, NULL, imu_count, NULL);
    H5Dread(reader->dataset_gyro, H5T_NATIVE_FLOAT, imu_mem, gyro_space, H5P_DEFAULT,
            reader->gyro_buf);
    H5Sclose(gyro_space);

    H5Sclose(imu_mem);

    record->accel[0] = reader->accel_buf[0];
    record->accel[1] = reader->accel_buf[1];
    record->accel[2] = reader->accel_buf[2];
    record->gyro[0] = reader->gyro_buf[0];
    record->gyro[1] = reader->gyro_buf[1];
    record->gyro[2] = reader->gyro_buf[2];
    record->has_imu = true;
  } else {
    record->accel[0] = record->accel[1] = record->accel[2] = 0.0f;
    record->gyro[0] = record->gyro[1] = record->gyro[2] = 0.0f;
    record->has_imu = false;
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
  if (reader->dataset_accel >= 0) H5Dclose(reader->dataset_accel);
  if (reader->dataset_gyro >= 0) H5Dclose(reader->dataset_gyro);
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

// ---- High-rate IMU/GPS reader ----

ImuGpsReader *ImuGpsReader_open(hid_t file) {
  if (file < 0) return NULL;

  bool found_imu = false;
  bool found_gps = false;

  hid_t imu_ds_ts = -1, imu_ds_accel = -1, imu_ds_gyro = -1;
  size_t imu_n = 0;

  double *gps_ts = NULL, *gps_lat = NULL, *gps_lng = NULL;
  size_t gps_n = 0;

  // Try opening "imu" group
  hid_t imu_grp = H5Gopen(file, "imu", H5P_DEFAULT);
  if (imu_grp >= 0) {
    imu_ds_ts = H5Dopen(imu_grp, TIMESTAMPS, H5P_DEFAULT);
    imu_ds_accel = H5Dopen(imu_grp, ACCEL, H5P_DEFAULT);
    imu_ds_gyro = H5Dopen(imu_grp, GYRO, H5P_DEFAULT);

    if (imu_ds_ts >= 0 && imu_ds_accel >= 0 && imu_ds_gyro >= 0) {
      // Validate accel/gyro are rank-2 with dim[1]==3
      hid_t accel_space = H5Dget_space(imu_ds_accel);
      hid_t gyro_space = H5Dget_space(imu_ds_gyro);
      int accel_rank = H5Sget_simple_extent_ndims(accel_space);
      int gyro_rank = H5Sget_simple_extent_ndims(gyro_space);
      bool valid = false;
      if (accel_rank == 2 && gyro_rank == 2) {
        hsize_t accel_dims[2], gyro_dims[2];
        H5Sget_simple_extent_dims(accel_space, accel_dims, NULL);
        H5Sget_simple_extent_dims(gyro_space, gyro_dims, NULL);
        if (accel_dims[1] == 3 && gyro_dims[1] == 3) {
          valid = true;
        }
      }
      H5Sclose(accel_space);
      H5Sclose(gyro_space);

      if (valid) {
        hid_t ts_space = H5Dget_space(imu_ds_ts);
        hsize_t ts_dims[1];
        H5Sget_simple_extent_dims(ts_space, ts_dims, NULL);
        H5Sclose(ts_space);
        imu_n = ts_dims[0];
        found_imu = true;
      }
    }

    if (!found_imu) {
      if (imu_ds_ts >= 0) H5Dclose(imu_ds_ts);
      if (imu_ds_accel >= 0) H5Dclose(imu_ds_accel);
      if (imu_ds_gyro >= 0) H5Dclose(imu_ds_gyro);
      imu_ds_ts = imu_ds_accel = imu_ds_gyro = -1;
    }
    H5Gclose(imu_grp);
  }

  // Try opening "gps" group — load fully into memory
  hid_t gps_grp = H5Gopen(file, "gps", H5P_DEFAULT);
  if (gps_grp >= 0) {
    hid_t gps_ds_ts = H5Dopen(gps_grp, TIMESTAMPS, H5P_DEFAULT);
    hid_t gps_ds_lat = H5Dopen(gps_grp, LAT, H5P_DEFAULT);
    hid_t gps_ds_lng = H5Dopen(gps_grp, LNG, H5P_DEFAULT);

    if (gps_ds_ts >= 0 && gps_ds_lat >= 0 && gps_ds_lng >= 0) {
      hid_t ts_space = H5Dget_space(gps_ds_ts);
      hsize_t ts_dims[1];
      H5Sget_simple_extent_dims(ts_space, ts_dims, NULL);
      H5Sclose(ts_space);
      gps_n = ts_dims[0];

      if (gps_n > 0) {
        gps_ts = malloc(gps_n * sizeof(double));
        gps_lat = malloc(gps_n * sizeof(double));
        gps_lng = malloc(gps_n * sizeof(double));

        H5Dread(gps_ds_ts, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gps_ts);
        H5Dread(gps_ds_lat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gps_lat);
        H5Dread(gps_ds_lng, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gps_lng);
        found_gps = true;
      }
    }

    if (gps_ds_ts >= 0) H5Dclose(gps_ds_ts);
    if (gps_ds_lat >= 0) H5Dclose(gps_ds_lat);
    if (gps_ds_lng >= 0) H5Dclose(gps_ds_lng);
    H5Gclose(gps_grp);
  }

  if (!found_imu && !found_gps) return NULL;

  ImuGpsReader *r = calloc(1, sizeof(ImuGpsReader));

  r->imu_dataset_ts = imu_ds_ts;
  r->imu_dataset_accel = imu_ds_accel;
  r->imu_dataset_gyro = imu_ds_gyro;
  r->imu_n_records = imu_n;
  r->imu_cursor = 0;
  r->has_imu = found_imu;

  r->gps_ts = gps_ts;
  r->gps_lat = gps_lat;
  r->gps_lng = gps_lng;
  r->gps_n_records = gps_n;
  r->gps_cursor = 0;
  r->has_gps = found_gps;

  // Read first IMU timestamp for timing alignment
  r->imu_first_ts = 0.0;
  if (found_imu && imu_n > 0) {
    hsize_t off = 0, cnt = 1;
    hid_t mem1 = H5Screate_simple(1, &cnt, NULL);
    hid_t ts_sp = H5Dget_space(r->imu_dataset_ts);
    H5Sselect_hyperslab(ts_sp, H5S_SELECT_SET, &off, NULL, &cnt, NULL);
    H5Dread(r->imu_dataset_ts, H5T_NATIVE_DOUBLE, mem1, ts_sp, H5P_DEFAULT,
            &r->imu_first_ts);
    H5Sclose(ts_sp);
    H5Sclose(mem1);
  }

  return r;
}

bool ImuGpsReader_next_imu(ImuGpsReader *r, ImuRecord *rec) {
  if (!r || !r->has_imu || r->imu_cursor >= r->imu_n_records)
    return false;

  hsize_t offset1 = r->imu_cursor;
  hsize_t count1 = 1;
  hid_t mem1 = H5Screate_simple(1, &count1, NULL);

  // Read timestamp (1D slab)
  hid_t ts_space = H5Dget_space(r->imu_dataset_ts);
  H5Sselect_hyperslab(ts_space, H5S_SELECT_SET, &offset1, NULL, &count1, NULL);
  H5Dread(r->imu_dataset_ts, H5T_NATIVE_DOUBLE, mem1, ts_space, H5P_DEFAULT, &rec->timestamp);
  H5Sclose(ts_space);
  H5Sclose(mem1);

  // Read accel + gyro (2D slab, one row)
  hsize_t offset2[2] = {r->imu_cursor, 0};
  hsize_t count2[2] = {1, 3};
  hsize_t mem_dim = 3;
  hid_t mem2 = H5Screate_simple(1, &mem_dim, NULL);

  hid_t accel_space = H5Dget_space(r->imu_dataset_accel);
  H5Sselect_hyperslab(accel_space, H5S_SELECT_SET, offset2, NULL, count2, NULL);
  H5Dread(r->imu_dataset_accel, H5T_NATIVE_FLOAT, mem2, accel_space, H5P_DEFAULT, rec->accel);
  H5Sclose(accel_space);

  hid_t gyro_space = H5Dget_space(r->imu_dataset_gyro);
  H5Sselect_hyperslab(gyro_space, H5S_SELECT_SET, offset2, NULL, count2, NULL);
  H5Dread(r->imu_dataset_gyro, H5T_NATIVE_FLOAT, mem2, gyro_space, H5P_DEFAULT, rec->gyro);
  H5Sclose(gyro_space);

  H5Sclose(mem2);

  r->imu_cursor++;
  return true;
}

void ImuGpsReader_interpolate_gps(ImuGpsReader *r, double timestamp, double *lat, double *lng) {
  if (!r || !r->has_gps || r->gps_n_records == 0) {
    *lat = 0.0;
    *lng = 0.0;
    return;
  }

  // Before first sample — clamp
  if (timestamp <= r->gps_ts[0]) {
    *lat = r->gps_lat[0];
    *lng = r->gps_lng[0];
    return;
  }

  // After last sample — clamp
  if (timestamp >= r->gps_ts[r->gps_n_records - 1]) {
    *lat = r->gps_lat[r->gps_n_records - 1];
    *lng = r->gps_lng[r->gps_n_records - 1];
    return;
  }

  // Advance cursor forward to bracket the timestamp
  while (r->gps_cursor + 1 < r->gps_n_records - 1 &&
         r->gps_ts[r->gps_cursor + 1] < timestamp) {
    r->gps_cursor++;
  }

  size_t i = r->gps_cursor;
  double dt = r->gps_ts[i + 1] - r->gps_ts[i];
  if (dt < 1e-12) {
    *lat = r->gps_lat[i];
    *lng = r->gps_lng[i];
    return;
  }

  double t = (timestamp - r->gps_ts[i]) / dt;
  *lat = r->gps_lat[i] + t * (r->gps_lat[i + 1] - r->gps_lat[i]);
  *lng = r->gps_lng[i] + t * (r->gps_lng[i + 1] - r->gps_lng[i]);
}

void ImuGpsReader_close(ImuGpsReader *r) {
  if (!r) return;
  if (r->imu_dataset_ts >= 0) H5Dclose(r->imu_dataset_ts);
  if (r->imu_dataset_accel >= 0) H5Dclose(r->imu_dataset_accel);
  if (r->imu_dataset_gyro >= 0) H5Dclose(r->imu_dataset_gyro);
  free(r->gps_ts);
  free(r->gps_lat);
  free(r->gps_lng);
  free(r);
}
