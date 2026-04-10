#include <math.h>
#include <stdio.h>
#include <string.h>
#include "viz/tile_limits.h"
#include "viz/viz_config.h"
#include "viz/viz_runtime.h"

#define VIDEO_DECODE_BUDGET_SCALE_CAP 4
#define ADAPTIVE_VIDEO_BUDGET_SCALE_CAP 8
#define ADAPTIVE_INGEST_BUDGET_SCALE_CAP 8
#define ADAPTIVE_IMU_BUDGET_SCALE_CAP 4
#define ADAPTIVE_GPS_BUDGET_SCALE_CAP 8
#define ADAPTIVE_TILE_UPLOAD_BUDGET_CAP 4

static int clamp_int(int value, int min_value, int max_value) {
  if (value < min_value) return min_value;
  if (value > max_value) return max_value;
  return value;
}

static int scaled_video_decode_base_budget(
    const VizRuntimeBudgetConfig *config) {
  int base = VIZ_CONFIG_DEFAULT_VIDEO_DECODE_BUDGET;
  double scale = 1.0;
  int max_budget;
  double scaled;
  int budget;

  if (config && config->video_decode_budget > 0) {
    base = config->video_decode_budget;
  }
  if (config && config->playback_speed > 1.0) {
    scale = config->playback_speed;
  }

  scaled = (double)base * scale;
  budget = (int)ceil(scaled);
  max_budget = base * VIDEO_DECODE_BUDGET_SCALE_CAP;
  if (budget < base) return base;
  if (budget > max_budget) return max_budget;
  return budget;
}

static int adaptive_budget_cap(int base, int scale_cap, int hard_max) {
  long scaled_cap;

  if (base < 1) base = 1;
  if (scale_cap < 1) scale_cap = 1;
  scaled_cap = (long)base * (long)scale_cap;
  if (scaled_cap < base) scaled_cap = base;
  if (scaled_cap > hard_max) scaled_cap = hard_max;
  return (int)scaled_cap;
}

static int ramp_adaptive_budget_up(int current, int base, int cap) {
  int step;

  current = clamp_int(current, base, cap);
  step = current / 2;
  if (step < base) step = base;
  return clamp_int(current + step, base, cap);
}

static int ramp_adaptive_budget_down(int current, int base) {
  int over = current - base;
  int step;

  if (over <= 0) return base;
  step = over / 4;
  if (step < 1) step = 1;
  return clamp_int(current - step, base, current);
}

static int next_adaptive_budget(int current, int base, int cap, bool pressured,
                                int demand_hint) {
  int next;
  int hinted;

  current = clamp_int(current, base, cap);
  if (!pressured) return ramp_adaptive_budget_down(current, base);

  next = ramp_adaptive_budget_up(current, base, cap);
  if (demand_hint <= 0) return next;

  hinted = clamp_int(base + demand_hint, base, cap);
  if (hinted > next) return hinted;
  return next;
}

void VizRuntime_clear_frame_stats(VizRuntimeFrameStats *stats) {
  if (!stats) return;
  memset(stats, 0, sizeof(*stats));
}

void VizRuntime_init_budget_state(VizRuntimeBudgetState *state,
                                  const VizRuntimeBudgetConfig *config) {
  if (!state) return;

  memset(state, 0, sizeof(*state));
  state->effective_video_decode_budget = scaled_video_decode_base_budget(config);
  state->effective_ingest_record_budget =
      config ? config->ingest_record_budget : 0;
  state->effective_imu_sample_budget = config ? config->imu_sample_budget : 0;
  state->effective_gps_point_budget = config ? config->gps_point_budget : 0;
  state->effective_tile_upload_budget = config ? config->tile_uploads_per_frame
                                               : 0;
}

void VizRuntime_prepare_frame_budgets(
    VizRuntimeBudgetState *state,
    const VizRuntimeBudgetConfig *config,
    const VizRuntimeFrameStats *prev_stats,
    const VizRuntimeTileStats *prev_tiles,
    VizRuntimeFrameStats *frame_stats) {
  int video_base;
  int video_cap;
  int ingest_cap;
  int imu_cap;
  int gps_cap;
  int tile_cap;
  int tile_base;
  bool tile_pressured = false;
  int tile_demand_hint = 0;

  if (!state || !config || !frame_stats) return;

  video_base = scaled_video_decode_base_budget(config);
  video_cap = adaptive_budget_cap(video_base, ADAPTIVE_VIDEO_BUDGET_SCALE_CAP,
                                  VIZ_CONFIG_MAX_VIDEO_DECODE_BUDGET);
  ingest_cap = adaptive_budget_cap(config->ingest_record_budget,
                                   ADAPTIVE_INGEST_BUDGET_SCALE_CAP,
                                   VIZ_CONFIG_MAX_INGEST_RECORD_BUDGET);
  imu_cap = adaptive_budget_cap(config->imu_sample_budget,
                                ADAPTIVE_IMU_BUDGET_SCALE_CAP,
                                VIZ_CONFIG_MAX_IMU_SAMPLE_BUDGET);
  gps_cap = adaptive_budget_cap(config->gps_point_budget,
                                ADAPTIVE_GPS_BUDGET_SCALE_CAP,
                                VIZ_CONFIG_MAX_GPS_POINT_BUDGET);
  tile_base = clamp_int(config->tile_uploads_per_frame, 1,
                        TILE_MAP_MAX_PENDING_DOWNLOADS);
  tile_cap = clamp_int(ADAPTIVE_TILE_UPLOAD_BUDGET_CAP, tile_base,
                       TILE_MAP_MAX_PENDING_DOWNLOADS);

  state->effective_video_decode_budget = next_adaptive_budget(
      state->effective_video_decode_budget, video_base, video_cap,
      prev_stats && prev_stats->video_backlog,
      prev_stats ? prev_stats->decode_steps : 0);
  state->effective_ingest_record_budget = next_adaptive_budget(
      state->effective_ingest_record_budget, config->ingest_record_budget,
      ingest_cap, prev_stats && prev_stats->ingest_backlog,
      prev_stats ? prev_stats->drained_records : 0);
  state->effective_imu_sample_budget = next_adaptive_budget(
      state->effective_imu_sample_budget, config->imu_sample_budget, imu_cap,
      prev_stats && prev_stats->imu_backlog,
      prev_stats ? prev_stats->drained_imu : 0);
  state->effective_gps_point_budget = next_adaptive_budget(
      state->effective_gps_point_budget, config->gps_point_budget, gps_cap,
      prev_stats && prev_stats->gps_backlog,
      prev_stats ? prev_stats->drained_gps : 0);

  if (prev_tiles) {
    tile_pressured = prev_tiles->decoded_downloads >
                         state->effective_tile_upload_budget ||
                     prev_tiles->decoded_downloads >= tile_base + 1;
    tile_demand_hint = prev_tiles->decoded_downloads;
  }
  state->effective_tile_upload_budget = next_adaptive_budget(
      state->effective_tile_upload_budget, tile_base, tile_cap,
      tile_pressured, tile_demand_hint);

  frame_stats->decode_base_budget = video_base;
  frame_stats->decode_budget = state->effective_video_decode_budget;
}

void VizRuntime_format_budget_label(char *dst, size_t dst_size, int effective,
                                    int base) {
  if (!dst || dst_size == 0) return;
  if (effective > base) {
    snprintf(dst, dst_size, "%d(%d)", effective, base);
  } else {
    snprintf(dst, dst_size, "%d", effective);
  }
}

unsigned long long VizRuntime_bytes_to_mib_ceil(size_t bytes) {
  const unsigned long long mib = 1024ull * 1024ull;
  unsigned long long value = (unsigned long long)bytes;

  if (value == 0) return 0;
  return (value + mib - 1ull) / mib;
}
