#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "viz/jepa_cache.h"
#include "ingest/ingest.h"

static void close_dataset(hid_t *dataset) {
    if (dataset && *dataset >= 0) {
        H5Dclose(*dataset);
        *dataset = -1;
    }
}

static bool get_dataset_dims(hid_t dataset, int expected_rank, hsize_t *dims,
                             const char *name) {
    hid_t space = H5Dget_space(dataset);
    if (space < 0) {
        fprintf(stderr, "JepaCache_load: failed to get dataspace for '%s'\n", name);
        return false;
    }

    int rank = H5Sget_simple_extent_ndims(space);
    if (rank != expected_rank) {
        fprintf(stderr, "JepaCache_load: dataset '%s' has rank %d, expected %d\n",
                name, rank, expected_rank);
        H5Sclose(space);
        return false;
    }
    if (H5Sget_simple_extent_dims(space, dims, NULL) < 0) {
        fprintf(stderr, "JepaCache_load: failed to read dimensions for '%s'\n", name);
        H5Sclose(space);
        return false;
    }

    H5Sclose(space);
    return true;
}

JepaCache *JepaCache_load(hid_t file) {
    if (file < 0) return NULL;

    hid_t grp = H5Gopen(file, JEPA, H5P_DEFAULT);
    if (grp < 0) return NULL;

    hid_t ds_ts = -1;
    hid_t ds_pm = -1;
    JepaCache *jc = NULL;

    ds_ts = H5Dopen(grp, TIMESTAMPS, H5P_DEFAULT);
    ds_pm = H5Dopen(grp, PREDICTION_MAPS, H5P_DEFAULT);
    if (ds_ts < 0 || ds_pm < 0) {
        close_dataset(&ds_ts);
        close_dataset(&ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    hsize_t ts_dims[1];
    hsize_t pm_dims[3];
    if (!get_dataset_dims(ds_ts, 1, ts_dims, TIMESTAMPS) ||
        !get_dataset_dims(ds_pm, 3, pm_dims, PREDICTION_MAPS)) {
        close_dataset(&ds_ts);
        close_dataset(&ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    if (pm_dims[0] != ts_dims[0] ||
        pm_dims[1] != JEPA_MAP_DIM || pm_dims[2] != JEPA_MAP_DIM) {
        fprintf(stderr, "JepaCache_load: expected (%zu, %d, %d), got (%llu, %llu, %llu)\n",
                (size_t)ts_dims[0], JEPA_MAP_DIM, JEPA_MAP_DIM,
                (unsigned long long)pm_dims[0],
                (unsigned long long)pm_dims[1],
                (unsigned long long)pm_dims[2]);
        close_dataset(&ds_ts);
        close_dataset(&ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    size_t n = (size_t)ts_dims[0];
    if (n == 0) {
        close_dataset(&ds_ts);
        close_dataset(&ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    // Bulk-read timestamps and prediction maps
    double *timestamps = malloc(n * sizeof(double));
    float *prediction_maps = malloc(n * JEPA_MAP_DIM * JEPA_MAP_DIM * sizeof(float));
    float *scratch = malloc(JEPA_MAP_DIM * JEPA_MAP_DIM * sizeof(float));

    if (!timestamps || !prediction_maps || !scratch) {
        free(timestamps);
        free(prediction_maps);
        free(scratch);
        close_dataset(&ds_ts);
        close_dataset(&ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    if (H5Dread(ds_ts, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, timestamps) < 0 ||
        H5Dread(ds_pm, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                prediction_maps) < 0) {
        free(timestamps);
        free(prediction_maps);
        free(scratch);
        close_dataset(&ds_ts);
        close_dataset(&ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    close_dataset(&ds_ts);
    close_dataset(&ds_pm);
    H5Gclose(grp);

    jc = malloc(sizeof(JepaCache));
    if (!jc) {
        free(timestamps);
        free(prediction_maps);
        free(scratch);
        return NULL;
    }
    jc->timestamps = timestamps;
    jc->prediction_maps = prediction_maps;
    jc->n_records = n;
    jc->cursor = 0;
    jc->scratch = scratch;

    printf("JepaCache: loaded %zu prediction maps (%dx%d)\n",
           n, JEPA_MAP_DIM, JEPA_MAP_DIM);

    return jc;
}

bool JepaCache_lookup(JepaCache *jc, double timestamp, float **out_map) {
    if (!jc || jc->n_records == 0) return false;

    size_t n_pixels = JEPA_MAP_DIM * JEPA_MAP_DIM;

    // Before first sample — clamp
    if (timestamp <= jc->timestamps[0]) {
        memcpy(jc->scratch, jc->prediction_maps, n_pixels * sizeof(float));
        *out_map = jc->scratch;
        return true;
    }

    // After last sample — clamp
    if (timestamp >= jc->timestamps[jc->n_records - 1]) {
        memcpy(jc->scratch,
               jc->prediction_maps + (jc->n_records - 1) * n_pixels,
               n_pixels * sizeof(float));
        *out_map = jc->scratch;
        return true;
    }

    // Advance cursor to the most recent map at or before timestamp
    while (jc->cursor + 1 < jc->n_records &&
           jc->timestamps[jc->cursor + 1] <= timestamp) {
        jc->cursor++;
    }

    memcpy(jc->scratch, jc->prediction_maps + jc->cursor * n_pixels,
           n_pixels * sizeof(float));
    *out_map = jc->scratch;
    return true;
}

void JepaCache_reset(JepaCache *jc) {
    if (jc) jc->cursor = 0;
}

void JepaCache_free(JepaCache *jc) {
    if (!jc) return;
    free(jc->timestamps);
    free(jc->prediction_maps);
    free(jc->scratch);
    free(jc);
}
