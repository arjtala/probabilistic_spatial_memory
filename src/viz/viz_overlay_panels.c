#include <stdio.h>
#include "viz/viz_overlay_panels.h"
#include "viz/viz_math.h"

static const UiOverlayColor PANEL_BG = {0.04f, 0.06f, 0.10f, 0.78f};
static const UiOverlayColor PANEL_EDGE = {0.16f, 0.26f, 0.38f, 0.55f};
static const UiOverlayColor TITLE = {0.95f, 0.98f, 1.00f, 0.98f};
static const UiOverlayColor BODY = {0.84f, 0.90f, 0.96f, 0.92f};
static const UiOverlayColor MUTED = {0.62f, 0.72f, 0.80f, 0.90f};
static const UiOverlayColor ACCENT = {0.42f, 0.74f, 1.00f, 0.95f};
static const float PANEL_TRACKING = 0.5f;

static bool add_panel_frame(UiOverlayMesh *mesh, float x, float y, float w,
                            float h) {
  return UiOverlay_add_rect(mesh, x, y, w, h, PANEL_BG) &&
         UiOverlay_add_rect(mesh, x, y, w, 1.0f, PANEL_EDGE) &&
         UiOverlay_add_rect(mesh, x, y + h - 1.0f, w, 1.0f, PANEL_EDGE) &&
         UiOverlay_add_rect(mesh, x, y, 1.0f, h, PANEL_EDGE) &&
         UiOverlay_add_rect(mesh, x + w - 1.0f, y, 1.0f, h, PANEL_EDGE);
}

static bool add_text_line(UiOverlayMesh *mesh, float x, float y, float scale,
                          UiOverlayColor color, const char *text) {
  return UiOverlay_add_text(mesh, x, y, scale, PANEL_TRACKING, color, text);
}

static bool add_viridis_bar(UiOverlayMesh *mesh, float x, float y, float w,
                            float h) {
  const int segments = 16;
  float segment_w = w / (float)segments;

  for (int i = 0; i < segments; i++) {
    float t0 = (float)i / (float)segments;
    float t1 = (float)(i + 1) / (float)segments;
    float r0, g0, b0, a0;
    float r1, g1, b1, a1;
    UiOverlayColor left, right;

    count_to_color(t0, &r0, &g0, &b0, &a0);
    count_to_color(t1, &r1, &g1, &b1, &a1);
    left = (UiOverlayColor){r0, g0, b0, 0.95f};
    right = (UiOverlayColor){r1, g1, b1, 0.95f};
    if (!UiOverlay_add_hgradient(mesh, x + segment_w * (float)i, y, segment_w,
                                 h, left, right)) {
      return false;
    }
  }

  return UiOverlay_add_rect(mesh, x, y, w, 1.0f, PANEL_EDGE) &&
         UiOverlay_add_rect(mesh, x, y + h - 1.0f, w, 1.0f, PANEL_EDGE) &&
         UiOverlay_add_rect(mesh, x, y, 1.0f, h, PANEL_EDGE) &&
         UiOverlay_add_rect(mesh, x + w - 1.0f, y, 1.0f, h, PANEL_EDGE);
}

static const char *mode_label(HexHeatmapMode mode) {
  switch (mode) {
  case HEX_HEATMAP_MODE_CURRENT:
    return "CURRENT";
  case HEX_HEATMAP_MODE_RECENCY:
    return "RECENCY";
  case HEX_HEATMAP_MODE_TOTAL:
  default:
    return "TOTAL";
  }
}

static const char *mode_detail(HexHeatmapMode mode) {
  switch (mode) {
  case HEX_HEATMAP_MODE_CURRENT:
    return "ACTIVE BUCKET ONLY";
  case HEX_HEATMAP_MODE_RECENCY:
    return "CURRENT / TOTAL";
  case HEX_HEATMAP_MODE_TOTAL:
  default:
    return "FULL WINDOW MEMORY";
  }
}

static bool build_help_panel(UiOverlayMesh *mesh, bool awaiting_initial_play) {
  const float title_scale = 2.25f;
  const float body_scale = 1.25f;
  const float note_scale = 1.10f;
  const float title_step = 32.0f;
  const float intro_step = 26.0f;
  const float line_step = 20.0f;
  const float x = 24.0f;
  const float y = 24.0f;
  const float w = 384.0f;
  const float h = awaiting_initial_play ? 298.0f : 272.0f;
  float line_y = y + 20.0f;

  if (!add_panel_frame(mesh, x, y, w, h)) return false;
  if (!add_text_line(mesh, x + 16.0f, line_y, title_scale, TITLE,
                     awaiting_initial_play ? "READY TO PLAY" : "CONTROLS")) {
    return false;
  }
  line_y += title_step;
  if (awaiting_initial_play &&
      !add_text_line(mesh, x + 16.0f, line_y, body_scale, ACCENT,
                     "PRESS SPACE TO START, ? TO HIDE HELP")) {
    return false;
  }
  if (awaiting_initial_play) line_y += intro_step;

  return add_text_line(mesh, x + 16.0f, line_y + 0.0f, body_scale, BODY,
                       "SPACE      PLAY / PAUSE") &&
         add_text_line(mesh, x + 16.0f, line_y + 1.0f * line_step, body_scale,
                       BODY,
                       "SCROLL VID SCRUB TIMELINE") &&
         add_text_line(mesh, x + 16.0f, line_y + 2.0f * line_step, body_scale,
                       BODY,
                       "SCROLL MAP ZOOM TO CURSOR") &&
         add_text_line(mesh, x + 16.0f, line_y + 3.0f * line_step, body_scale,
                       BODY,
                       "DRAG MAP   PAN VIEW") &&
         add_text_line(mesh, x + 16.0f, line_y + 4.0f * line_step, body_scale,
                       BODY,
                       "C          RECENTER FOLLOW") &&
         add_text_line(mesh, x + 16.0f, line_y + 5.0f * line_step, body_scale,
                       BODY,
                       "M          CYCLE HEATMAP MODE") &&
         add_text_line(mesh, x + 16.0f, line_y + 6.0f * line_step, body_scale,
                       BODY,
                       "E          TOGGLE ISO MAP + EXTRUDE") &&
         add_text_line(mesh, x + 16.0f, line_y + 7.0f * line_step, body_scale,
                       BODY,
                       "L          TOGGLE LEGEND") &&
         add_text_line(mesh, x + 16.0f, line_y + 8.0f * line_step, body_scale,
                       BODY,
                       "H          TOGGLE TITLE HUD") &&
         add_text_line(mesh, x + 16.0f, line_y + 9.0f * line_step, body_scale,
                       BODY,
                       "P          SAVE .PNG SCREENSHOT") &&
         add_text_line(mesh, x + 16.0f, line_y + 10.0f * line_step, body_scale,
                       BODY,
                       "? / F1     TOGGLE HELP") &&
         add_text_line(mesh, x + 16.0f, line_y + 11.0f * line_step, note_scale,
                       MUTED,
                       "FILES GO TO captures/*.PNG");
}

static bool build_legend_panel(UiOverlayMesh *mesh, int window_w, int window_h,
                               int map_split_x, HexHeatmapMode mode,
                               double zoom_degrees, size_t tile_count) {
  const float title_scale = 2.10f;
  const float body_scale = 1.15f;
  char mode_line[64];
  char detail_line[96];
  char zoom_line[64];
  char tiles_line[64];
  const float w = 344.0f;
  const float h = 168.0f;
  const float x = (float)(window_w - 24) - w;
  float y = 24.0f;

  if (x < (float)map_split_x + 16.0f) {
    y = (float)(window_h - 24) - h;
  }

  snprintf(mode_line, sizeof(mode_line), "MODE  %s", mode_label(mode));
  snprintf(detail_line, sizeof(detail_line), "%s", mode_detail(mode));
  snprintf(zoom_line, sizeof(zoom_line), "ZOOM  %.4F DEG", zoom_degrees);
  snprintf(tiles_line, sizeof(tiles_line), "TILES %zu", tile_count);

  return add_panel_frame(mesh, x, y, w, h) &&
         add_text_line(mesh, x + 16.0f, y + 18.0f, title_scale, TITLE,
                       "HEATMAP") &&
         add_text_line(mesh, x + 16.0f, y + 48.0f, body_scale, ACCENT,
                       mode_line) &&
         add_text_line(mesh, x + 16.0f, y + 68.0f, body_scale, BODY,
                       detail_line) &&
         add_text_line(mesh, x + 16.0f, y + 88.0f, body_scale, MUTED,
                       "NORMALIZED TO HOTTEST TILE") &&
         add_viridis_bar(mesh, x + 16.0f, y + 112.0f, w - 32.0f, 14.0f) &&
         add_text_line(mesh, x + 16.0f, y + 132.0f, body_scale, MUTED,
                       "0.0 LOW") &&
         add_text_line(mesh, x + w - 102.0f, y + 132.0f, body_scale, MUTED,
                       "1.0 HIGH") &&
         add_text_line(mesh, x + 16.0f, y + 148.0f, body_scale, BODY,
                       zoom_line) &&
         add_text_line(mesh, x + w - 122.0f, y + 148.0f, body_scale, BODY,
                       tiles_line);
}

static bool build_status_panel(UiOverlayMesh *mesh, int window_w, int window_h,
                               const char *status_text) {
  const float text_scale = 1.25f;
  const float text_w = UiOverlay_text_width(status_text, text_scale,
                                            PANEL_TRACKING);
  const float w = text_w + 28.0f;
  const float h = 34.0f;
  const float x = ((float)window_w - w) * 0.5f;
  const float y = (float)window_h - h - 28.0f;

  if (!status_text || status_text[0] == '\0') return true;
  return add_panel_frame(mesh, x, y, w, h) &&
         add_text_line(mesh, x + 14.0f, y + 10.0f, text_scale, ACCENT,
                       status_text);
}

bool VizOverlayPanels_build(UiOverlayMesh *mesh, int window_w, int window_h,
                            int map_split_x, bool show_help,
                            bool show_legend, bool awaiting_initial_play,
                            HexHeatmapMode heatmap_mode, double zoom_degrees,
                            size_t tile_count, const char *status_text) {
  if (!mesh || window_w <= 0 || window_h <= 0) return false;

  UiOverlay_reset(mesh, window_w, window_h);
  if (show_help &&
      !build_help_panel(mesh, awaiting_initial_play)) {
    return false;
  }
  if (show_legend &&
      !build_legend_panel(mesh, window_w, window_h, map_split_x, heatmap_mode,
                          zoom_degrees, tile_count)) {
    return false;
  }
  if (!build_status_panel(mesh, window_w, window_h, status_text)) {
    return false;
  }
  return true;
}
