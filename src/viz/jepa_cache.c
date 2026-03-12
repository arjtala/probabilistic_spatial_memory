#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "viz/jepa_cache.h"
#include "ingest/ingest.h"

JepaCache *JepaCache_load(hid_t file) {
    if (file < 0) return NULL;

    hid_t grp = H5Gopen(file, JEPA, H5P_DEFAULT);
    if (grp < 0) return NULL;

    hid_t ds_ts = H5Dopen(grp, TIMESTAMPS, H5P_DEFAULT);
    hid_t ds_pm = H5Dopen(grp, PREDICTION_MAPS, H5P_DEFAULT);

    if (ds_ts < 0 || ds_pm < 0) {
        if (ds_ts >= 0) H5Dclose(ds_ts);
        if (ds_pm >= 0) H5Dclose(ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    // Validate prediction_maps is rank-3 with dims (N, 16, 16)
    hid_t pm_space = H5Dget_space(ds_pm);
    int rank = H5Sget_simple_extent_ndims(pm_space);
    if (rank != 3) {
        H5Sclose(pm_space);
        H5Dclose(ds_ts);
        H5Dclose(ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    hsize_t pm_dims[3];
    H5Sget_simple_extent_dims(pm_space, pm_dims, NULL);
    H5Sclose(pm_space);

    if (pm_dims[1] != JEPA_MAP_DIM || pm_dims[2] != JEPA_MAP_DIM) {
        fprintf(stderr, "JepaCache_load: expected (%zu, %d, %d), got (%llu, %llu, %llu)\n",
                (size_t)pm_dims[0], JEPA_MAP_DIM, JEPA_MAP_DIM,
                (unsigned long long)pm_dims[0],
                (unsigned long long)pm_dims[1],
                (unsigned long long)pm_dims[2]);
        H5Dclose(ds_ts);
        H5Dclose(ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    size_t n = (size_t)pm_dims[0];
    if (n == 0) {
        H5Dclose(ds_ts);
        H5Dclose(ds_pm);
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
        H5Dclose(ds_ts);
        H5Dclose(ds_pm);
        H5Gclose(grp);
        return NULL;
    }

    H5Dread(ds_ts, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, timestamps);
    H5Dread(ds_pm, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, prediction_maps);

    H5Dclose(ds_ts);
    H5Dclose(ds_pm);
    H5Gclose(grp);

    JepaCache *jc = malloc(sizeof(JepaCache));
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
