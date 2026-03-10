#ifndef INGEST_H
#define INGEST_H

#include <stdbool.h>
#include <hdf5.h>
#include "core/spatial_memory.h"

#define DINO "dino"
#define JEPA "jepa"
#define TIMESTAMPS "timestamps"
#define LAT "lat"
#define LNG "lng"
#define EMBEDDINGS "embeddings"
#define ATTENTION_MAPS "attention_maps"

typedef struct {
  hid_t dataset_ts;
  hid_t dataset_lat;
  hid_t dataset_lng;
  hid_t dataset_emb;
  hid_t dataset_attn;
  size_t n_records;
  size_t emb_dimension;
  size_t attn_size;
  size_t cursor;
  float *embedding_buf;
  float *attn_buf;
} IngestReader;

typedef struct {
  double timestamp;
  double lat;
  double lng;
  float *embedding;
  size_t embedding_dim;
  float *attention_map;
  size_t attn_size;
} IngestRecord;

IngestReader *IngestReader_open(hid_t file, const char *group);
void IngestReader_close(IngestReader *reader);
bool IngestReader_next(IngestReader *reader, IngestRecord *record);
void IngestReader_run(IngestReader *reader, SpatialMemory *sm, const double time_window_sec);

#endif
