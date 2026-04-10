#ifndef VIZ_APP_H
#define VIZ_APP_H

#include <stdbool.h>
#include "core/spatial_memory.h"
#include "ingest/ingest.h"
#include "viz/gps_trace.h"
#include "viz/hex_renderer.h"
#include "viz/imu_processor.h"
#include "viz/jepa_cache.h"
#include "viz/tile_map.h"
#include "viz/video_decoder.h"
#include "viz/video_quad.h"
#include "viz/viz_runtime.h"

typedef struct {
  bool paused;
  bool awaiting_initial_play;
  double playback_speed;

  bool map_view_initialized;
  double map_view_center_lat;
  double map_view_center_lng;

  bool dragging;
  double drag_last_x;
  double drag_last_y;

  VideoDecoder *dec;
  VideoQuad *video_quad;
  HexRenderer *hex_renderer;
  TileMap *tile_map;
  GpsTrace *gps_trace;
  ImuProcessor *imu_proc;
  IngestReader *reader;
  ImuGpsReader *imu_gps;
  JepaCache *jepa_cache;
  SpatialMemory *sm;

  double first_ts;
  double window_anchor;
  double time_window_sec;
  int h3_resolution;

  double video_start_time;
  double video_pts_offset;
  bool video_done;

  bool seek_pending;
  double pending_seek_target;
  double scrub_sensitivity_sec;
  double map_follow_smoothing;
  int tile_uploads_per_frame;
  int video_decode_budget;
  int ingest_record_budget;
  int imu_sample_budget;
  int gps_point_budget;
  VizRuntimeBudgetState budget_state;
  bool debug_hud_enabled;
  double next_debug_title_update;
  VizRuntimeFrameStats frame_stats;
} VizApp;

#endif
