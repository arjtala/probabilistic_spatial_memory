#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "viz/viz_config.h"
#include "viz/viz_runtime.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

static void assert_str_eq(const char *expected, const char *actual) {
  printf("'%s' == '%s'\n", expected, actual ? actual : "(null)");
  if (!actual || strcmp(expected, actual) != 0) {
    fprintf(stderr, "!!! Assertion failed: expected '%s' but got '%s'\n",
            expected, actual ? actual : "(null)");
    exit(EXIT_FAILURE);
  }
}

static VizRuntimeBudgetConfig sample_budget_config(void) {
  VizRuntimeBudgetConfig config = {0};

  config.playback_speed = 1.0;
  config.video_decode_budget = 6;
  config.ingest_record_budget = 128;
  config.imu_sample_budget = 512;
  config.gps_point_budget = 64;
  config.tile_uploads_per_frame = 1;
  return config;
}

static void test_init_budget_state_uses_base_budgets(void) {
  VizRuntimeBudgetConfig config = sample_budget_config();
  VizRuntimeBudgetState state;

  VizRuntime_init_budget_state(&state, &config);
  ASSERT(state.effective_video_decode_budget == 6, 6,
         state.effective_video_decode_budget);
  ASSERT(state.effective_ingest_record_budget == 128, 128,
         state.effective_ingest_record_budget);
  ASSERT(state.effective_imu_sample_budget == 512, 512,
         state.effective_imu_sample_budget);
  ASSERT(state.effective_gps_point_budget == 64, 64,
         state.effective_gps_point_budget);
  ASSERT(state.effective_tile_upload_budget == 1, 1,
         state.effective_tile_upload_budget);
}

static void test_video_budget_scales_with_playback_speed(void) {
  VizRuntimeBudgetConfig config = sample_budget_config();
  VizRuntimeBudgetState state;
  VizRuntimeFrameStats frame_stats;

  config.playback_speed = 2.0;
  VizRuntime_init_budget_state(&state, &config);
  VizRuntime_clear_frame_stats(&frame_stats);
  VizRuntime_prepare_frame_budgets(&state, &config, NULL, NULL, &frame_stats);

  ASSERT(frame_stats.decode_base_budget == 12, 12, frame_stats.decode_base_budget);
  ASSERT(frame_stats.decode_budget == 12, 12, frame_stats.decode_budget);
}

static void test_backlog_raises_budgets_and_idle_decay_brings_them_back(void) {
  VizRuntimeBudgetConfig config = sample_budget_config();
  VizRuntimeBudgetState state;
  VizRuntimeFrameStats prev_stats;
  VizRuntimeFrameStats frame_stats;

  VizRuntime_init_budget_state(&state, &config);
  VizRuntime_clear_frame_stats(&prev_stats);
  prev_stats.video_backlog = true;
  prev_stats.decode_steps = 6;
  prev_stats.ingest_backlog = true;
  prev_stats.drained_records = 128;
  prev_stats.imu_backlog = true;
  prev_stats.drained_imu = 512;
  prev_stats.gps_backlog = true;
  prev_stats.drained_gps = 64;

  VizRuntime_clear_frame_stats(&frame_stats);
  VizRuntime_prepare_frame_budgets(&state, &config, &prev_stats, NULL,
                                   &frame_stats);

  ASSERT(state.effective_video_decode_budget > 6, 1,
         state.effective_video_decode_budget > 6 ? 1 : 0);
  ASSERT(state.effective_ingest_record_budget > 128, 1,
         state.effective_ingest_record_budget > 128 ? 1 : 0);
  ASSERT(state.effective_imu_sample_budget > 512, 1,
         state.effective_imu_sample_budget > 512 ? 1 : 0);
  ASSERT(state.effective_gps_point_budget > 64, 1,
         state.effective_gps_point_budget > 64 ? 1 : 0);

  VizRuntime_clear_frame_stats(&prev_stats);
  for (int i = 0; i < 256; i++) {
    VizRuntime_clear_frame_stats(&frame_stats);
    VizRuntime_prepare_frame_budgets(&state, &config, &prev_stats, NULL,
                                     &frame_stats);
    if (state.effective_video_decode_budget == 6 &&
        state.effective_ingest_record_budget == 128 &&
        state.effective_imu_sample_budget == 512 &&
        state.effective_gps_point_budget == 64) {
      break;
    }
  }

  ASSERT(state.effective_video_decode_budget == 6, 6,
         state.effective_video_decode_budget);
  ASSERT(state.effective_ingest_record_budget == 128, 128,
         state.effective_ingest_record_budget);
  ASSERT(state.effective_imu_sample_budget == 512, 512,
         state.effective_imu_sample_budget);
  ASSERT(state.effective_gps_point_budget == 64, 64,
         state.effective_gps_point_budget);
}

static void test_tile_budget_responds_to_decoded_tile_pressure(void) {
  VizRuntimeBudgetConfig config = sample_budget_config();
  VizRuntimeBudgetState state;
  VizRuntimeFrameStats frame_stats;
  VizRuntimeTileStats tile_stats;

  VizRuntime_init_budget_state(&state, &config);
  tile_stats.decoded_downloads = 3;
  VizRuntime_clear_frame_stats(&frame_stats);
  VizRuntime_prepare_frame_budgets(&state, &config, NULL, &tile_stats,
                                   &frame_stats);

  ASSERT(state.effective_tile_upload_budget == 4, 4,
         state.effective_tile_upload_budget);

  tile_stats.decoded_downloads = 0;
  for (int i = 0; i < 8; i++) {
    VizRuntime_clear_frame_stats(&frame_stats);
    VizRuntime_prepare_frame_budgets(&state, &config, NULL, &tile_stats,
                                     &frame_stats);
  }
  ASSERT(state.effective_tile_upload_budget == 1, 1,
         state.effective_tile_upload_budget);
}

static void test_format_budget_label_and_mib_rounding(void) {
  char label[32];

  VizRuntime_format_budget_label(label, sizeof(label), 128, 128);
  assert_str_eq("128", label);

  VizRuntime_format_budget_label(label, sizeof(label), 384, 128);
  assert_str_eq("384(128)", label);

  ASSERT(VizRuntime_bytes_to_mib_ceil(0) == 0, 0,
         (int)VizRuntime_bytes_to_mib_ceil(0));
  ASSERT(VizRuntime_bytes_to_mib_ceil(1) == 1, 1,
         (int)VizRuntime_bytes_to_mib_ceil(1));
  ASSERT(VizRuntime_bytes_to_mib_ceil(1024u * 1024u) == 1, 1,
         (int)VizRuntime_bytes_to_mib_ceil(1024u * 1024u));
  ASSERT(VizRuntime_bytes_to_mib_ceil(1024u * 1024u + 1u) == 2, 2,
         (int)VizRuntime_bytes_to_mib_ceil(1024u * 1024u + 1u));
}

int main(void) {
  RUN_TEST(test_init_budget_state_uses_base_budgets);
  RUN_TEST(test_video_budget_scales_with_playback_speed);
  RUN_TEST(test_backlog_raises_budgets_and_idle_decay_brings_them_back);
  RUN_TEST(test_tile_budget_responds_to_decoded_tile_pressure);
  RUN_TEST(test_format_budget_label_and_mib_rounding);
  return 0;
}
