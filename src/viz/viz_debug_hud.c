#include <stdio.h>
#include <string.h>
#include "viz/viz_debug_hud.h"

void VizDebugHud_reset_window_title(GLFWwindow *window) {
  if (!window) return;
  glfwSetWindowTitle(window, "psm-viz");
}

bool VizDebugHud_build_title(char *dst, size_t dst_size,
                             const VizDebugHudSnapshot *snapshot) {
  char video_budget[32];
  char ingest_budget[32];
  char imu_budget[32];
  char gps_budget[32];
  char tile_budget[32];
  char disk_stats[96];
  const char *state;
  const char *video_backlog = "";
  const char *ingest_backlog = "";
  const char *imu_backlog = "";
  const char *gps_backlog = "";

  if (!dst || dst_size == 0 || !snapshot) return false;

  state = snapshot->paused ? "paused" : (snapshot->video_done ? "done" : "play");
  if (snapshot->frame_stats.video_backlog) video_backlog = "*";
  if (snapshot->frame_stats.ingest_backlog) ingest_backlog = "*";
  if (snapshot->frame_stats.imu_backlog) imu_backlog = "*";
  if (snapshot->frame_stats.gps_backlog) gps_backlog = "*";

  VizRuntime_format_budget_label(video_budget, sizeof(video_budget),
                                 snapshot->frame_stats.decode_budget,
                                 snapshot->frame_stats.decode_base_budget);
  VizRuntime_format_budget_label(
      ingest_budget, sizeof(ingest_budget),
      snapshot->budget_state.effective_ingest_record_budget,
      snapshot->ingest_record_budget);
  VizRuntime_format_budget_label(imu_budget, sizeof(imu_budget),
                                 snapshot->budget_state.effective_imu_sample_budget,
                                 snapshot->imu_sample_budget);
  VizRuntime_format_budget_label(gps_budget, sizeof(gps_budget),
                                 snapshot->budget_state.effective_gps_point_budget,
                                 snapshot->gps_point_budget);
  VizRuntime_format_budget_label(tile_budget, sizeof(tile_budget),
                                 snapshot->budget_state.effective_tile_upload_budget,
                                 snapshot->tile_uploads_per_frame);
  if (snapshot->tile_stats.disk_cache_enabled) {
    snprintf(disk_stats, sizeof(disk_stats), "disk h%d w%d p%d m%llu/%llu",
             snapshot->tile_stats.disk_cache_hits,
             snapshot->tile_stats.disk_cache_writes,
             snapshot->tile_stats.disk_cache_prunes,
             VizRuntime_bytes_to_mib_ceil(snapshot->tile_stats.disk_cache_bytes),
             VizRuntime_bytes_to_mib_ceil(snapshot->tile_stats.disk_cache_max_bytes));
  } else {
    snprintf(disk_stats, sizeof(disk_stats), "disk off");
  }

  return snprintf(
             dst, dst_size,
             "psm-viz | %s %.2fx | pts %.2f/%.2f | v %d/%s%s | in %d/%s%s | "
             "imu %d/%s%s | gps %d/%s%s | tiles act%d rdy%d dec%d pix%d up%d/%s c%d | %s",
             state, snapshot->playback_speed, snapshot->current_pts,
             snapshot->duration, snapshot->frame_stats.decode_steps,
             video_budget, video_backlog, snapshot->frame_stats.drained_records,
             ingest_budget, ingest_backlog, snapshot->frame_stats.drained_imu,
             imu_budget, imu_backlog, snapshot->frame_stats.drained_gps,
             gps_budget, gps_backlog, snapshot->tile_stats.active_downloads,
             snapshot->tile_stats.ready_downloads,
             snapshot->tile_stats.decoding_downloads,
             snapshot->tile_stats.decoded_downloads,
             snapshot->tile_stats.uploads_last_frame, tile_budget,
             snapshot->tile_stats.cache_tiles, disk_stats) < (int)dst_size;
}
