#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ingest/ingest.h"

static void close_dataset(hid_t *dataset) {
  if (dataset && *dataset >= 0) {
    H5Dclose(*dataset);
    *dataset = -1;
  }
}

static bool link_exists(hid_t loc, const char *name) {
  htri_t exists = H5Lexists(loc, name, H5P_DEFAULT);
  return exists > 0;
}

static bool get_dataset_dims(hid_t dataset, int expected_rank, hsize_t *dims,
                             const char *name) {
  hid_t space = H5Dget_space(dataset);
  if (space < 0) {
    fprintf(stderr, "Failed to get dataspace for '%s'\n", name);
    return false;
  }

  int rank = H5Sget_simple_extent_ndims(space);
  if (rank != expected_rank) {
    fprintf(stderr, "Dataset '%s' has rank %d, expected %d\n",
            name, rank, expected_rank);
    H5Sclose(space);
    return false;
  }

  if (H5Sget_simple_extent_dims(space, dims, NULL) < 0) {
    fprintf(stderr, "Failed to read dimensions for '%s'\n", name);
    H5Sclose(space);
    return false;
  }

  H5Sclose(space);
  return true;
}

static bool validate_imu_dataset_shapes(hsize_t ts_len,
                                        const hsize_t accel_dims[2],
                                        const hsize_t gyro_dims[2],
                                        const char *context,
                                        const char *group) {
  if (accel_dims[0] != ts_len || gyro_dims[0] != ts_len ||
      accel_dims[1] != 3 || gyro_dims[1] != 3) {
    if (group) {
      fprintf(stderr, "%s: invalid IMU shape in group '%s'\n", context, group);
    } else {
      fprintf(stderr, "%s: invalid IMU dataset shapes\n", context);
    }
    return false;
  }
  return true;
}

static bool read_double_row(hid_t dataset, size_t row, double *out,
                            const char *name) {
  hsize_t offset = row;
  hsize_t count = 1;
  hid_t mem = H5Screate_simple(1, &count, NULL);
  hid_t space = H5Dget_space(dataset);
  bool ok = false;

  if (mem < 0 || space < 0) {
    fprintf(stderr, "Failed to prepare HDF5 read for '%s'\n", name);
    goto cleanup;
  }
  if (H5Sselect_hyperslab(space, H5S_SELECT_SET, &offset, NULL, &count, NULL) < 0) {
    fprintf(stderr, "Failed to select row %zu in '%s'\n", row, name);
    goto cleanup;
  }
  if (H5Dread(dataset, H5T_NATIVE_DOUBLE, mem, space, H5P_DEFAULT, out) < 0) {
    fprintf(stderr, "Failed to read row %zu from '%s'\n", row, name);
    goto cleanup;
  }

  ok = true;

cleanup:
  if (space >= 0) {
    H5Sclose(space);
  }
  if (mem >= 0) {
    H5Sclose(mem);
  }
  return ok;
}

static bool read_float_row(hid_t dataset, size_t row, size_t width, float *out,
                           const char *name) {
  hsize_t offset[2] = {row, 0};
  hsize_t count[2] = {1, width};
  hsize_t mem_dims = width;
  hid_t mem = H5Screate_simple(1, &mem_dims, NULL);
  hid_t space = H5Dget_space(dataset);
  bool ok = false;

  if (mem < 0 || space < 0) {
    fprintf(stderr, "Failed to prepare HDF5 row read for '%s'\n", name);
    goto cleanup;
  }
  if (H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, count, NULL) < 0) {
    fprintf(stderr, "Failed to select row %zu in '%s'\n", row, name);
    goto cleanup;
  }
  if (H5Dread(dataset, H5T_NATIVE_FLOAT, mem, space, H5P_DEFAULT, out) < 0) {
    fprintf(stderr, "Failed to read row %zu from '%s'\n", row, name);
    goto cleanup;
  }

  ok = true;

cleanup:
  if (space >= 0) {
    H5Sclose(space);
  }
  if (mem >= 0) {
    H5Sclose(mem);
  }
  return ok;
}

static bool read_float_square_map(hid_t dataset, size_t row, size_t dim,
                                  float *out, const char *name) {
  hsize_t offset[3] = {row, 0, 0};
  hsize_t count[3] = {1, dim, dim};
  hsize_t mem_dims[2] = {dim, dim};
  hid_t mem = H5Screate_simple(2, mem_dims, NULL);
  hid_t space = H5Dget_space(dataset);
  bool ok = false;

  if (mem < 0 || space < 0) {
    fprintf(stderr, "Failed to prepare HDF5 map read for '%s'\n", name);
    goto cleanup;
  }
  if (H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, count, NULL) < 0) {
    fprintf(stderr, "Failed to select map %zu in '%s'\n", row, name);
    goto cleanup;
  }
  if (H5Dread(dataset, H5T_NATIVE_FLOAT, mem, space, H5P_DEFAULT, out) < 0) {
    fprintf(stderr, "Failed to read map %zu from '%s'\n", row, name);
    goto cleanup;
  }

  ok = true;

cleanup:
  if (space >= 0) {
    H5Sclose(space);
  }
  if (mem >= 0) {
    H5Sclose(mem);
  }
  return ok;
}

static bool read_full_double_dataset(hid_t dataset, double *out,
                                     const char *name) {
  if (H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out) < 0) {
    fprintf(stderr, "Failed to read dataset '%s'\n", name);
    return false;
  }
  return true;
}

static void cleanup_imu_gps_open_state(hid_t *imu_ds_ts,
                                       hid_t *imu_ds_accel,
                                       hid_t *imu_ds_gyro,
                                       double **gps_ts,
                                       double **gps_lat,
                                       double **gps_lng) {
  close_dataset(imu_ds_ts);
  close_dataset(imu_ds_accel);
  close_dataset(imu_ds_gyro);

  free(*gps_ts);
  free(*gps_lat);
  free(*gps_lng);
  *gps_ts = NULL;
  *gps_lat = NULL;
  *gps_lng = NULL;
}

IngestReader *IngestReader_open(hid_t file, const char *group) {
  if (file < 0 || !group || group[0] == '\0') {
    fprintf(stderr, "IngestReader_open: invalid file handle or group\n");
    return NULL;
  }

  hid_t grp = H5Gopen(file, group, H5P_DEFAULT);
  if (grp < 0) {
    fprintf(stderr, "IngestReader_open: group '%s' not found\n", group);
    return NULL;
  }

  IngestReader *reader = (IngestReader *)calloc(1, sizeof(IngestReader));
  if (!reader) {
    H5Gclose(grp);
    return NULL;
  }
  reader->dataset_ts = -1;
  reader->dataset_lat = -1;
  reader->dataset_lng = -1;
  reader->dataset_emb = -1;
  reader->dataset_attn = -1;
  reader->dataset_accel = -1;
  reader->dataset_gyro = -1;
  reader->attn_buf = NULL;
  reader->embedding_buf = NULL;
  reader->has_imu = false;

  reader->dataset_ts = H5Dopen(grp, TIMESTAMPS, H5P_DEFAULT);
  reader->dataset_lat = H5Dopen(grp, LAT, H5P_DEFAULT);
  reader->dataset_lng = H5Dopen(grp, LNG, H5P_DEFAULT);
  reader->dataset_emb = H5Dopen(grp, EMBEDDINGS, H5P_DEFAULT);
  if (reader->dataset_ts < 0 || reader->dataset_lat < 0 ||
      reader->dataset_lng < 0 || reader->dataset_emb < 0) {
    fprintf(stderr, "IngestReader_open: missing datasets in group '%s'\n", group);
    H5Gclose(grp);
    IngestReader_close(reader);
    return NULL;
  }

  hsize_t ts_dims[1];
  hsize_t lat_dims[1];
  hsize_t lng_dims[1];
  hsize_t emb_dims[2];

  if (!get_dataset_dims(reader->dataset_ts, 1, ts_dims, TIMESTAMPS) ||
      !get_dataset_dims(reader->dataset_lat, 1, lat_dims, LAT) ||
      !get_dataset_dims(reader->dataset_lng, 1, lng_dims, LNG) ||
      !get_dataset_dims(reader->dataset_emb, 2, emb_dims, EMBEDDINGS)) {
    H5Gclose(grp);
    IngestReader_close(reader);
    return NULL;
  }

  if (lat_dims[0] != ts_dims[0] || lng_dims[0] != ts_dims[0] ||
      emb_dims[0] != ts_dims[0]) {
    fprintf(stderr, "IngestReader_open: dataset length mismatch in group '%s'\n", group);
    H5Gclose(grp);
    IngestReader_close(reader);
    return NULL;
  }

  reader->n_records = ts_dims[0];
  reader->emb_dimension = emb_dims[1];
  reader->cursor = 0;
  reader->attn_size = 0;

  reader->embedding_buf = malloc(reader->emb_dimension * sizeof(float));
  if (!reader->embedding_buf && reader->emb_dimension > 0) {
    H5Gclose(grp);
    IngestReader_close(reader);
    return NULL;
  }

  if (link_exists(grp, ATTENTION_MAPS) || link_exists(grp, PREDICTION_MAPS)) {
    const char *attn_name = link_exists(grp, ATTENTION_MAPS)
        ? ATTENTION_MAPS
        : PREDICTION_MAPS;
    hsize_t attn_dims[3];

    reader->dataset_attn = H5Dopen(grp, attn_name, H5P_DEFAULT);
    if (reader->dataset_attn < 0 ||
        !get_dataset_dims(reader->dataset_attn, 3, attn_dims, attn_name)) {
      H5Gclose(grp);
      IngestReader_close(reader);
      return NULL;
    }
    if (attn_dims[0] != ts_dims[0] || attn_dims[1] == 0 ||
        attn_dims[1] != attn_dims[2]) {
      fprintf(stderr, "IngestReader_open: invalid spatial map shape in group '%s'\n",
              group);
      H5Gclose(grp);
      IngestReader_close(reader);
      return NULL;
    }

    reader->attn_size = attn_dims[1];
    reader->attn_buf = malloc(reader->attn_size * reader->attn_size * sizeof(float));
    if (!reader->attn_buf) {
      H5Gclose(grp);
      IngestReader_close(reader);
      return NULL;
    }
  }

  bool has_accel = link_exists(grp, ACCEL);
  bool has_gyro = link_exists(grp, GYRO);
  if (has_accel != has_gyro) {
    fprintf(stderr, "IngestReader_open: incomplete IMU datasets in group '%s'\n", group);
    H5Gclose(grp);
    IngestReader_close(reader);
    return NULL;
  }
  if (has_accel) {
    hsize_t accel_dims[2];
    hsize_t gyro_dims[2];

    reader->dataset_accel = H5Dopen(grp, ACCEL, H5P_DEFAULT);
    reader->dataset_gyro = H5Dopen(grp, GYRO, H5P_DEFAULT);
    if (reader->dataset_accel < 0 || reader->dataset_gyro < 0 ||
        !get_dataset_dims(reader->dataset_accel, 2, accel_dims, ACCEL) ||
        !get_dataset_dims(reader->dataset_gyro, 2, gyro_dims, GYRO)) {
      H5Gclose(grp);
      IngestReader_close(reader);
      return NULL;
    }
    if (!validate_imu_dataset_shapes(ts_dims[0], accel_dims, gyro_dims,
                                     "IngestReader_open", group)) {
      H5Gclose(grp);
      IngestReader_close(reader);
      return NULL;
    }
    reader->has_imu = true;
  }

  H5Gclose(grp);
  return reader;
}

IngestReadStatus IngestReader_next(IngestReader *reader, IngestRecord *record) {
  if (record) memset(record, 0, sizeof(*record));
  if (!reader || !record) return INGEST_READ_ERROR;
  if (reader->cursor >= reader->n_records)
    return INGEST_READ_EOF;
  size_t cur = reader->cursor;

  if (!read_double_row(reader->dataset_ts, cur, &record->timestamp, TIMESTAMPS) ||
      !read_double_row(reader->dataset_lat, cur, &record->lat, LAT) ||
      !read_double_row(reader->dataset_lng, cur, &record->lng, LNG) ||
      !read_float_row(reader->dataset_emb, cur, reader->emb_dimension,
                      reader->embedding_buf, EMBEDDINGS)) {
    return INGEST_READ_ERROR;
  }

  record->embedding = reader->embedding_buf;
  record->embedding_dim = reader->emb_dimension;

  // Read attention map if available
  if (reader->dataset_attn >= 0) {
    if (!read_float_square_map(reader->dataset_attn, cur, reader->attn_size,
                               reader->attn_buf, "spatial_map")) {
      return INGEST_READ_ERROR;
    }
    record->attention_map = reader->attn_buf;
    record->attn_size = reader->attn_size;
  } else {
    record->attention_map = NULL;
    record->attn_size = 0;
  }

  // Read IMU data if available
  if (reader->has_imu) {
    if (!read_float_row(reader->dataset_accel, cur, 3, reader->accel_buf, ACCEL) ||
        !read_float_row(reader->dataset_gyro, cur, 3, reader->gyro_buf, GYRO)) {
      return INGEST_READ_ERROR;
    }

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

  return INGEST_READ_OK;
}

void IngestReader_close(IngestReader *reader) {
  if (!reader) return;
  close_dataset(&reader->dataset_ts);
  close_dataset(&reader->dataset_lat);
  close_dataset(&reader->dataset_lng);
  close_dataset(&reader->dataset_emb);
  close_dataset(&reader->dataset_attn);
  close_dataset(&reader->dataset_accel);
  close_dataset(&reader->dataset_gyro);
  free(reader->embedding_buf);
  free(reader->attn_buf);
  free(reader);
}

bool IngestReader_run(IngestReader *reader, SpatialMemory *sm,
                      const double time_window_sec) {
  IngestRecord record;
  double window_anchor = -1.0;
  if (!reader || !sm || time_window_sec <= 0.0) return false;

  while (true) {
    IngestReadStatus status = IngestReader_next(reader, &record);
    if (status == INGEST_READ_EOF) return true;
    if (status == INGEST_READ_ERROR) return false;

    SpatialMemory_advance_to_timestamp(sm, record.timestamp, &window_anchor,
                                       time_window_sec);
    if (!SpatialMemory_observe(sm, record.lat, record.lng, record.embedding,
                               record.embedding_dim * sizeof(float))) {
      fprintf(stderr, "IngestReader_run: skipping invalid observation at timestamp %.3f\n",
              record.timestamp);
    }
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
  if (link_exists(file, "imu")) {
    hid_t imu_grp = H5Gopen(file, "imu", H5P_DEFAULT);
    if (imu_grp < 0) {
      cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                 &gps_ts, &gps_lat, &gps_lng);
      return NULL;
    }

    bool has_ts = link_exists(imu_grp, TIMESTAMPS);
    bool has_accel = link_exists(imu_grp, ACCEL);
    bool has_gyro = link_exists(imu_grp, GYRO);

    if (has_ts || has_accel || has_gyro) {
      hsize_t ts_dims[1];
      hsize_t accel_dims[2];
      hsize_t gyro_dims[2];

      if (!(has_ts && has_accel && has_gyro)) {
        fprintf(stderr, "ImuGpsReader_open: incomplete IMU datasets\n");
        H5Gclose(imu_grp);
        cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                   &gps_ts, &gps_lat, &gps_lng);
        return NULL;
      }

      imu_ds_ts = H5Dopen(imu_grp, TIMESTAMPS, H5P_DEFAULT);
      imu_ds_accel = H5Dopen(imu_grp, ACCEL, H5P_DEFAULT);
      imu_ds_gyro = H5Dopen(imu_grp, GYRO, H5P_DEFAULT);
      if (imu_ds_ts < 0 || imu_ds_accel < 0 || imu_ds_gyro < 0 ||
          !get_dataset_dims(imu_ds_ts, 1, ts_dims, "imu/timestamps") ||
          !get_dataset_dims(imu_ds_accel, 2, accel_dims, "imu/accel") ||
          !get_dataset_dims(imu_ds_gyro, 2, gyro_dims, "imu/gyro")) {
        H5Gclose(imu_grp);
        cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                   &gps_ts, &gps_lat, &gps_lng);
        return NULL;
      }
      if (!validate_imu_dataset_shapes(ts_dims[0], accel_dims, gyro_dims,
                                       "ImuGpsReader_open", NULL)) {
        H5Gclose(imu_grp);
        cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                   &gps_ts, &gps_lat, &gps_lng);
        return NULL;
      }

      imu_n = ts_dims[0];
      found_imu = true;
    }

    H5Gclose(imu_grp);
  }

  // Try opening "gps" group — load fully into memory
  if (link_exists(file, "gps")) {
    hid_t gps_grp = H5Gopen(file, "gps", H5P_DEFAULT);
    if (gps_grp < 0) {
      cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                 &gps_ts, &gps_lat, &gps_lng);
      return NULL;
    }

    bool has_ts = link_exists(gps_grp, TIMESTAMPS);
    bool has_lat = link_exists(gps_grp, LAT);
    bool has_lng = link_exists(gps_grp, LNG);

    if (has_ts || has_lat || has_lng) {
      hid_t gps_ds_ts = -1;
      hid_t gps_ds_lat = -1;
      hid_t gps_ds_lng = -1;
      hsize_t ts_dims[1];
      hsize_t lat_dims[1];
      hsize_t lng_dims[1];

      if (!(has_ts && has_lat && has_lng)) {
        fprintf(stderr, "ImuGpsReader_open: incomplete GPS datasets\n");
        H5Gclose(gps_grp);
        cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                   &gps_ts, &gps_lat, &gps_lng);
        return NULL;
      }

      gps_ds_ts = H5Dopen(gps_grp, TIMESTAMPS, H5P_DEFAULT);
      gps_ds_lat = H5Dopen(gps_grp, LAT, H5P_DEFAULT);
      gps_ds_lng = H5Dopen(gps_grp, LNG, H5P_DEFAULT);
      if (gps_ds_ts < 0 || gps_ds_lat < 0 || gps_ds_lng < 0 ||
          !get_dataset_dims(gps_ds_ts, 1, ts_dims, "gps/timestamps") ||
          !get_dataset_dims(gps_ds_lat, 1, lat_dims, "gps/lat") ||
          !get_dataset_dims(gps_ds_lng, 1, lng_dims, "gps/lng")) {
        H5Gclose(gps_grp);
        close_dataset(&gps_ds_ts);
        close_dataset(&gps_ds_lat);
        close_dataset(&gps_ds_lng);
        cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                   &gps_ts, &gps_lat, &gps_lng);
        return NULL;
      }
      if (lat_dims[0] != ts_dims[0] || lng_dims[0] != ts_dims[0]) {
        fprintf(stderr, "ImuGpsReader_open: GPS dataset length mismatch\n");
        H5Gclose(gps_grp);
        close_dataset(&gps_ds_ts);
        close_dataset(&gps_ds_lat);
        close_dataset(&gps_ds_lng);
        cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                   &gps_ts, &gps_lat, &gps_lng);
        return NULL;
      }

      gps_n = ts_dims[0];
      if (gps_n > 0) {
        gps_ts = malloc(gps_n * sizeof(double));
        gps_lat = malloc(gps_n * sizeof(double));
        gps_lng = malloc(gps_n * sizeof(double));

        if (!gps_ts || !gps_lat || !gps_lng ||
            !read_full_double_dataset(gps_ds_ts, gps_ts, "gps/timestamps") ||
            !read_full_double_dataset(gps_ds_lat, gps_lat, "gps/lat") ||
            !read_full_double_dataset(gps_ds_lng, gps_lng, "gps/lng")) {
          H5Gclose(gps_grp);
          close_dataset(&gps_ds_ts);
          close_dataset(&gps_ds_lat);
          close_dataset(&gps_ds_lng);
          cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                                     &gps_ts, &gps_lat, &gps_lng);
          return NULL;
        }
        found_gps = true;
      }

      close_dataset(&gps_ds_ts);
      close_dataset(&gps_ds_lat);
      close_dataset(&gps_ds_lng);
    }

    H5Gclose(gps_grp);
  }

  if (!found_imu && !found_gps) return NULL;

  ImuGpsReader *r = calloc(1, sizeof(ImuGpsReader));
  if (!r) {
    cleanup_imu_gps_open_state(&imu_ds_ts, &imu_ds_accel, &imu_ds_gyro,
                               &gps_ts, &gps_lat, &gps_lng);
    return NULL;
  }
  r->imu_dataset_ts = -1;
  r->imu_dataset_accel = -1;
  r->imu_dataset_gyro = -1;

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
    if (!read_double_row(r->imu_dataset_ts, 0, &r->imu_first_ts, "imu/timestamps")) {
      ImuGpsReader_close(r);
      return NULL;
    }
  }

  return r;
}

IngestReadStatus ImuGpsReader_next_imu(ImuGpsReader *r, ImuRecord *rec) {
  if (rec) memset(rec, 0, sizeof(*rec));
  if (!r || !rec || !r->has_imu)
    return INGEST_READ_ERROR;
  if (r->imu_cursor >= r->imu_n_records)
    return INGEST_READ_EOF;
  size_t row = r->imu_cursor;
  if (!read_double_row(r->imu_dataset_ts, row, &rec->timestamp, "imu/timestamps") ||
      !read_float_row(r->imu_dataset_accel, row, 3, rec->accel, "imu/accel") ||
      !read_float_row(r->imu_dataset_gyro, row, 3, rec->gyro, "imu/gyro")) {
    return INGEST_READ_ERROR;
  }

  r->imu_cursor++;
  return INGEST_READ_OK;
}

bool ImuGpsReader_interpolate_gps(ImuGpsReader *r, double timestamp, double *lat,
                                  double *lng) {
  if (!lat || !lng) return false;
  if (!r || !r->has_gps || r->gps_n_records == 0) {
    *lat = 0.0;
    *lng = 0.0;
    return false;
  }

  // Before first sample — clamp
  if (timestamp <= r->gps_ts[0]) {
    *lat = r->gps_lat[0];
    *lng = r->gps_lng[0];
    return true;
  }

  // After last sample — clamp
  if (timestamp >= r->gps_ts[r->gps_n_records - 1]) {
    *lat = r->gps_lat[r->gps_n_records - 1];
    *lng = r->gps_lng[r->gps_n_records - 1];
    return true;
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
    return true;
  }

  double t = (timestamp - r->gps_ts[i]) / dt;
  *lat = r->gps_lat[i] + t * (r->gps_lat[i + 1] - r->gps_lat[i]);
  *lng = r->gps_lng[i] + t * (r->gps_lng[i + 1] - r->gps_lng[i]);
  return true;
}

void ImuGpsReader_close(ImuGpsReader *r) {
  if (!r) return;
  close_dataset(&r->imu_dataset_ts);
  close_dataset(&r->imu_dataset_accel);
  close_dataset(&r->imu_dataset_gyro);
  free(r->gps_ts);
  free(r->gps_lat);
  free(r->gps_lng);
  free(r);
}
