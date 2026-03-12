#ifndef VIZ_JEPA_CACHE_H
#define VIZ_JEPA_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <hdf5.h>

#define JEPA_MAP_DIM 16

typedef struct {
    double *timestamps;       // (n_records,) bulk-loaded
    float  *prediction_maps;  // (n_records * 16 * 16) bulk-loaded
    size_t  n_records;
    size_t  cursor;           // forward-only bracket position
    float  *scratch;          // (16*16) interpolation scratch buffer
} JepaCache;

JepaCache *JepaCache_load(hid_t file);
bool JepaCache_lookup(JepaCache *jc, double timestamp, float **out_map);
void JepaCache_reset(JepaCache *jc);
void JepaCache_free(JepaCache *jc);

#endif
