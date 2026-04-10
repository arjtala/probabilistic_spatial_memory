#ifndef VIZ_RUNTIME_H
#define VIZ_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  int decode_steps;
  int decode_base_budget;
  int decode_budget;
  bool video_backlog;
  int drained_records;
  bool ingest_backlog;
  int drained_imu;
  bool imu_backlog;
  int drained_gps;
  bool gps_backlog;
} VizRuntimeFrameStats;

typedef struct {
  double playback_speed;
  int video_decode_budget;
  int ingest_record_budget;
  int imu_sample_budget;
  int gps_point_budget;
  int tile_uploads_per_frame;
} VizRuntimeBudgetConfig;

typedef struct {
  int effective_video_decode_budget;
  int effective_ingest_record_budget;
  int effective_imu_sample_budget;
  int effective_gps_point_budget;
  int effective_tile_upload_budget;
} VizRuntimeBudgetState;

typedef struct {
  int decoded_downloads;
} VizRuntimeTileStats;

void VizRuntime_clear_frame_stats(VizRuntimeFrameStats *stats);
void VizRuntime_init_budget_state(VizRuntimeBudgetState *state,
                                  const VizRuntimeBudgetConfig *config);
void VizRuntime_prepare_frame_budgets(
    VizRuntimeBudgetState *state,
    const VizRuntimeBudgetConfig *config,
    const VizRuntimeFrameStats *prev_stats,
    const VizRuntimeTileStats *prev_tiles,
    VizRuntimeFrameStats *frame_stats);
void VizRuntime_format_budget_label(char *dst, size_t dst_size, int effective,
                                    int base);
unsigned long long VizRuntime_bytes_to_mib_ceil(size_t bytes);

#endif
