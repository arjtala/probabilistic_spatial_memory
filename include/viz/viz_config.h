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
#define VIZ_CONFIG_HEATMAP_MODE_CAP 32

#define VIZ_CONFIG_DEFAULT_VIDEO_DECODE_BUDGET 6
#define VIZ_CONFIG_DEFAULT_INGEST_RECORD_BUDGET 128
#define VIZ_CONFIG_DEFAULT_IMU_SAMPLE_BUDGET 512
#define VIZ_CONFIG_DEFAULT_GPS_POINT_BUDGET 64
#define VIZ_CONFIG_DEFAULT_TILE_DISK_CACHE_MAX_MB 512

#define VIZ_CONFIG_MAX_VIDEO_DECODE_BUDGET 64
#define VIZ_CONFIG_MAX_INGEST_RECORD_BUDGET 4096
#define VIZ_CONFIG_MAX_IMU_SAMPLE_BUDGET 16384
#define VIZ_CONFIG_MAX_GPS_POINT_BUDGET 4096
#define VIZ_CONFIG_MAX_TILE_DISK_CACHE_MAX_MB 16384

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
  bool start_paused;
  bool debug_hud_enabled;
  double scrub_sensitivity_sec;
  double map_follow_smoothing;
  int video_decode_budget;
  int ingest_record_budget;
  int imu_sample_budget;
  int gps_point_budget;
  int tile_uploads_per_frame;
  bool tile_disk_cache_enabled;
  int tile_disk_cache_max_mb;
  char heatmap_mode[VIZ_CONFIG_HEATMAP_MODE_CAP];
  // 0.0 disables hex extrusion; >0 enables cabinet-projection 3D mode.
  // Sane range is 0.10–0.50 (fraction of viewport half-height for max cell).
  double hex_extrude_scale;

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
bool VizConfig_parse_bool(const char *text, const char *name, bool *out_value);
bool VizConfig_parse_int_in_range(const char *text, const char *name,
                                  int min_value, int max_value,
                                  int *out_value);

bool VizConfig_resolve_tile_source(const VizConfig *config, VizTileSource *out);
void VizConfig_print_tile_styles(FILE *stream);

#endif
