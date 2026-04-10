#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "viz/viz_debug_hud.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

static void assert_contains(const char *needle, const char *haystack) {
  printf("contains '%s'\n", needle);
  if (!haystack || !strstr(haystack, needle)) {
    fprintf(stderr, "!!! Assertion failed: '%s' missing from '%s'\n", needle,
            haystack ? haystack : "(null)");
    exit(EXIT_FAILURE);
  }
}

static VizDebugHudSnapshot sample_snapshot(void) {
  VizDebugHudSnapshot snapshot;

  memset(&snapshot, 0, sizeof(snapshot));
  snapshot.playback_speed = 1.5;
  snapshot.current_pts = 12.25;
  snapshot.duration = 48.0;
  snapshot.ingest_record_budget = 128;
  snapshot.imu_sample_budget = 512;
  snapshot.gps_point_budget = 64;
  snapshot.tile_uploads_per_frame = 1;
  snapshot.frame_stats.decode_steps = 7;
  snapshot.frame_stats.decode_base_budget = 6;
  snapshot.frame_stats.decode_budget = 12;
  snapshot.frame_stats.video_backlog = true;
  snapshot.frame_stats.drained_records = 90;
  snapshot.frame_stats.drained_imu = 400;
  snapshot.frame_stats.drained_gps = 33;
  snapshot.budget_state.effective_ingest_record_budget = 192;
  snapshot.budget_state.effective_imu_sample_budget = 768;
  snapshot.budget_state.effective_gps_point_budget = 80;
  snapshot.budget_state.effective_tile_upload_budget = 3;
  snapshot.tile_stats.active_downloads = 2;
  snapshot.tile_stats.ready_downloads = 1;
  snapshot.tile_stats.decoding_downloads = 4;
  snapshot.tile_stats.decoded_downloads = 5;
  snapshot.tile_stats.uploads_last_frame = 2;
  snapshot.tile_stats.cache_tiles = 22;
  snapshot.tile_stats.disk_cache_enabled = true;
  snapshot.tile_stats.disk_cache_hits = 11;
  snapshot.tile_stats.disk_cache_writes = 7;
  snapshot.tile_stats.disk_cache_prunes = 3;
  snapshot.tile_stats.disk_cache_bytes = 9u * 1024u * 1024u;
  snapshot.tile_stats.disk_cache_max_bytes = 256u * 1024u * 1024u;
  return snapshot;
}

static void test_build_title_for_playback_state(void) {
  VizDebugHudSnapshot snapshot = sample_snapshot();
  char title[512];

  ASSERT(VizDebugHud_build_title(title, sizeof(title), &snapshot), 1, 1);
  assert_contains("psm-viz | play 1.50x", title);
  assert_contains("pts 12.25/48.00", title);
  assert_contains("v 7/12(6)*", title);
  assert_contains("in 90/192(128)", title);
  assert_contains("imu 400/768(512)", title);
  assert_contains("gps 33/80(64)", title);
  assert_contains("tiles act2 rdy1 dec4 pix5 up2/3(1) c22", title);
  assert_contains("disk h11 w7 p3 m9/256", title);
}

static void test_build_title_for_paused_no_disk_cache(void) {
  VizDebugHudSnapshot snapshot = sample_snapshot();
  char title[512];

  snapshot.paused = true;
  snapshot.video_done = false;
  snapshot.tile_stats.disk_cache_enabled = false;

  ASSERT(VizDebugHud_build_title(title, sizeof(title), &snapshot), 1, 1);
  assert_contains("psm-viz | paused 1.50x", title);
  assert_contains("disk off", title);
}

static void test_rejects_invalid_output_buffer(void) {
  VizDebugHudSnapshot snapshot = sample_snapshot();

  ASSERT(!VizDebugHud_build_title(NULL, 16, &snapshot), 0, 0);
  ASSERT(!VizDebugHud_build_title((char[1]){0}, 0, &snapshot), 0, 0);
  ASSERT(!VizDebugHud_build_title((char[16]){0}, 16, NULL), 0, 0);
}

int main(void) {
  RUN_TEST(test_build_title_for_playback_state);
  RUN_TEST(test_build_title_for_paused_no_disk_cache);
  RUN_TEST(test_rejects_invalid_output_buffer);
  return 0;
}
