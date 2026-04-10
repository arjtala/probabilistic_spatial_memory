#ifndef VIZ_DEBUG_HUD_H
#define VIZ_DEBUG_HUD_H

#include <stdbool.h>
#include <stddef.h>
#include "viz/gl_platform.h"
#include "viz/tile_map.h"
#include "viz/viz_runtime.h"

typedef struct {
  bool paused;
  bool video_done;
  double playback_speed;
  double current_pts;
  double duration;
  int ingest_record_budget;
  int imu_sample_budget;
  int gps_point_budget;
  int tile_uploads_per_frame;
  TileMapStats tile_stats;
  VizRuntimeFrameStats frame_stats;
  VizRuntimeBudgetState budget_state;
} VizDebugHudSnapshot;

void VizDebugHud_reset_window_title(GLFWwindow *window);
bool VizDebugHud_build_title(char *dst, size_t dst_size,
                             const VizDebugHudSnapshot *snapshot);

#endif
