#ifndef VIZ_CONFIG_H
#define VIZ_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <limits.h>
#include "core/spatial_memory.h"
#include "ingest/ingest.h"

#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

#define VIZ_CONFIG_GROUP_CAP 64
#define VIZ_CONFIG_STYLE_CAP 64
#define VIZ_CONFIG_API_KEY_CAP 256
#define VIZ_CONFIG_URL_CAP 1024

typedef struct {
  bool has_session_dir;
  char session_dir[PATH_MAX];

  bool has_video_path;
  char video_path[PATH_MAX];

  bool has_features_path;
  char features_path[PATH_MAX];

  char group[VIZ_CONFIG_GROUP_CAP];
  double time_window_sec;
  int h3_resolution;
  double scrub_sensitivity_sec;
  double map_follow_smoothing;
  int tile_uploads_per_frame;

  char tile_style[VIZ_CONFIG_STYLE_CAP];

  bool has_tile_api_key;
  char tile_api_key[VIZ_CONFIG_API_KEY_CAP];

  bool has_tile_url_template;
  char tile_url_template[VIZ_CONFIG_URL_CAP];
} VizConfig;

typedef struct {
  char style_name[VIZ_CONFIG_STYLE_CAP];
  char url_template[VIZ_CONFIG_URL_CAP];
  char api_key[VIZ_CONFIG_API_KEY_CAP];
  bool requires_api_key;
} VizTileSource;

void VizConfig_init(VizConfig *config);
bool VizConfig_load_file(VizConfig *config, const char *path);

bool VizConfig_set_text(char *dst, size_t dst_size, const char *value,
                        const char *field_name);
bool VizConfig_set_optional_text(char *dst, size_t dst_size, bool *has_value,
                                 const char *value, const char *field_name);

bool VizConfig_parse_positive_double(const char *text, const char *name,
                                     double *out_value);
bool VizConfig_parse_int_in_range(const char *text, const char *name,
                                  int min_value, int max_value,
                                  int *out_value);

bool VizConfig_resolve_tile_source(const VizConfig *config, VizTileSource *out);
void VizConfig_print_tile_styles(FILE *stream);

#endif
