#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <getopt.h>
#include <dirent.h>
#include <string.h>
#include <hdf5.h>

#include "viz/gl_platform.h"
#include "viz/attention_overlay.h"
#include "viz/progress_bar.h"
#include "viz/shader.h"
#include "viz/video_quad.h"
#include "viz/video_decoder.h"
#include "viz/hex_renderer.h"
#include "viz/tile_map.h"
#include "viz/gps_trace.h"
#include "viz/imu_processor.h"
#include "viz/jepa_cache.h"
#include "viz/viz_config.h"
#include "viz/viz_math.h"
#include "ingest/ingest.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VIDEO_DECODE_BUDGET_SCALE_CAP 4

typedef struct {
  bool paused;
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
} VizApp;

static VizApp *app_from_window(GLFWwindow *window) {
  return (VizApp *)glfwGetWindowUserPointer(window);
}

static int video_decode_budget(const VizApp *app) {
  int base = VIZ_CONFIG_DEFAULT_VIDEO_DECODE_BUDGET;
  double scale = 1.0;
  int max_budget;

  if (app && app->video_decode_budget > 0) {
    base = app->video_decode_budget;
  }
  if (app && app->playback_speed > 1.0) {
    scale = app->playback_speed;
  }

  double scaled = (double)base * scale;
  int budget = (int)ceil(scaled);
  max_budget = base * VIDEO_DECODE_BUDGET_SCALE_CAP;
  if (budget < base) return base;
  if (budget > max_budget) return max_budget;
  return budget;
}

static double clamp_map_zoom(double zoom) {
  if (zoom < 0.0001) return 0.0001;
  if (zoom > 1.0) return 1.0;
  return zoom;
}

static bool current_auto_map_center(const VizApp *app, double *out_lat,
                                    double *out_lng) {
  if (!app || !out_lat || !out_lng) return false;
  if (app->gps_trace && app->gps_trace->count > 0) {
    size_t last = app->gps_trace->count - 1;
    *out_lat = app->gps_trace->lats[last];
    *out_lng = app->gps_trace->lngs[last];
    return true;
  }
  if (app->hex_renderer && app->hex_renderer->vertex_count > 0) {
    *out_lat = app->hex_renderer->center_lat;
    *out_lng = app->hex_renderer->center_lng;
    return true;
  }
  return false;
}

static bool current_map_target_center(const VizApp *app, double *out_lat,
                                      double *out_lng) {
  if (!current_auto_map_center(app, out_lat, out_lng)) return false;
  if (app->hex_renderer) {
    *out_lat += app->hex_renderer->pan_offset_lat;
    *out_lng += app->hex_renderer->pan_offset_lng;
  }
  return true;
}

static void snap_map_view_to(VizApp *app, double center_lat, double center_lng) {
  if (!app) return;
  app->map_view_center_lat = center_lat;
  app->map_view_center_lng = center_lng;
  app->map_view_initialized = true;
}

static void set_manual_map_center(VizApp *app, double center_lat,
                                  double center_lng) {
  if (!app) return;
  if (app->hex_renderer) {
    double auto_lat, auto_lng;
    if (current_auto_map_center(app, &auto_lat, &auto_lng)) {
      app->hex_renderer->pan_offset_lat = center_lat - auto_lat;
      app->hex_renderer->pan_offset_lng = center_lng - auto_lng;
    }
  }
  snap_map_view_to(app, center_lat, center_lng);
}

static bool current_render_map_center(const VizApp *app, double *out_lat,
                                      double *out_lng) {
  if (!app || !out_lat || !out_lng) return false;
  if (app->map_view_initialized) {
    *out_lat = app->map_view_center_lat;
    *out_lng = app->map_view_center_lng;
    return true;
  }
  return current_map_target_center(app, out_lat, out_lng);
}

static void zoom_map_about_screen_point(VizApp *app, GLFWwindow *window,
                                        double zoom_factor, double cursor_x,
                                        double cursor_y) {
  if (!app || !app->hex_renderer) return;

  int win_w, win_h;
  glfwGetWindowSize(window, &win_w, &win_h);
  int half_w = win_w / 2;
  int viewport_w = win_w - half_w;
  int viewport_h = win_h;
  double map_x = cursor_x - (double)half_w;
  double map_y = cursor_y;

  if (viewport_w <= 0 || viewport_h <= 0 ||
      map_x < 0.0 || map_x > (double)viewport_w ||
      map_y < 0.0 || map_y > (double)viewport_h) {
    return;
  }

  double center_lat, center_lng;
  if (!current_render_map_center(app, &center_lat, &center_lng)) {
    app->hex_renderer->zoom = clamp_map_zoom(app->hex_renderer->zoom * zoom_factor);
    return;
  }

  double old_zoom = app->hex_renderer->zoom;
  double new_zoom = clamp_map_zoom(old_zoom * zoom_factor);
  if (fabs(new_zoom - old_zoom) < 1e-12) return;

  double aspect = (double)viewport_h / (double)viewport_w;
  double nx = (map_x / (double)viewport_w) * 2.0 - 1.0;
  double ny = 1.0 - (map_y / (double)viewport_h) * 2.0;
  double old_half_h = old_zoom * aspect;
  double cos_center = cos(center_lat * M_PI / 180.0);
  if (fabs(cos_center) < 1e-6) {
    cos_center = (cos_center < 0.0) ? -1e-6 : 1e-6;
  }

  double focus_lat = center_lat + ny * old_half_h;
  double focus_lng = center_lng + (nx * old_zoom) / cos_center;
  double new_half_h = new_zoom * aspect;
  double new_center_lat = focus_lat - ny * new_half_h;
  double cos_new_center = cos(new_center_lat * M_PI / 180.0);
  if (fabs(cos_new_center) < 1e-6) {
    cos_new_center = (cos_new_center < 0.0) ? -1e-6 : 1e-6;
  }
  double new_center_lng = focus_lng - (nx * new_zoom) / cos_new_center;

  app->hex_renderer->zoom = new_zoom;
  set_manual_map_center(app, new_center_lat, new_center_lng);
}

static void reset_playback_timing(VizApp *app, double now) {
  if (!app || !app->dec) return;
  app->video_start_time = now;
  app->video_pts_offset = app->dec->current_pts;
  app->video_done = false;
}

static void reset_replay_state_after_backward_seek(VizApp *app,
                                                   SpatialMemory *replacement_sm) {
  if (!app) return;

  if (app->reader) {
    app->reader->cursor = 0;
  }
  if (app->sm && replacement_sm) {
    SpatialMemory_free(app->sm);
    app->sm = replacement_sm;
  }
  app->window_anchor = -1.0;
  if (app->gps_trace) GpsTrace_clear(app->gps_trace);
  if (app->imu_proc) ImuProcessor_reset(app->imu_proc);
  if (app->imu_gps) {
    app->imu_gps->imu_cursor = 0;
    app->imu_gps->gps_cursor = 0;
  }
  if (app->hex_renderer) {
    app->hex_renderer->pan_offset_lat = 0.0;
    app->hex_renderer->pan_offset_lng = 0.0;
    app->hex_renderer->vertex_count = 0;
  }
  app->map_view_initialized = false;
  if (app->jepa_cache) JepaCache_reset(app->jepa_cache);
}

static bool apply_pending_seek(VizApp *app, double now) {
  if (!app || !app->seek_pending || !app->dec || !app->video_quad) return false;

  double target = app->pending_seek_target;
  if (target < 0.0) target = 0.0;
  if (app->dec->duration > 0.0 && target > app->dec->duration) {
    target = app->dec->duration;
  }
  if (fabs(target - app->dec->current_pts) < 1e-6) {
    app->pending_seek_target = app->dec->current_pts;
    app->seek_pending = false;
    return true;
  }

  bool seek_backward = target + 1e-9 < app->dec->current_pts;
  SpatialMemory *replacement_sm = NULL;
  if (seek_backward && app->sm) {
    replacement_sm = SpatialMemory_new(app->h3_resolution, DEFAULT_CAPACITY,
                                       DEFAULT_PRECISION);
    if (!replacement_sm) {
      fprintf(stderr, "Failed to reset spatial memory after seek\n");
      app->seek_pending = false;
      return false;
    }
  }

  if (!VideoDecoder_seek(app->dec, target)) {
    SpatialMemory_free(replacement_sm);
    app->seek_pending = false;
    return false;
  }

  VideoQuad_upload(app->video_quad, app->dec->rgb_buffer, app->dec->width,
                   app->dec->height);
  reset_playback_timing(app, now);

  if (seek_backward) {
    reset_replay_state_after_backward_seek(app, replacement_sm);
  }

  app->pending_seek_target = app->dec->current_pts;
  app->seek_pending = false;
  return true;
}

// ---- Scroll callback (scrub + zoom) ----
static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  VizApp *app = app_from_window(window);
  int win_w, win_h;

  if (!app) return;
  glfwGetWindowSize(window, &win_w, &win_h);

  double cursor_x, cursor_y;
  glfwGetCursorPos(window, &cursor_x, &cursor_y);

  double half_w = (double)win_w / 2.0;

  if (cursor_x < half_w) {
    // Cursor over video: horizontal scroll to scrub
    if (!app->dec || app->dec->duration <= 0.0) return;

    if (fabs(xoffset) < 1e-9) return;
    double seek_delta = xoffset * app->scrub_sensitivity_sec;
    double base_pts = app->seek_pending ? app->pending_seek_target
                                        : app->dec->current_pts;
    double target = base_pts + seek_delta;
    if (target < 0.0) target = 0.0;
    if (target > app->dec->duration) target = app->dec->duration;
    app->pending_seek_target = target;
    app->seek_pending = true;
  } else {
    // Cursor over map: vertical scroll to zoom
    if (!app->hex_renderer) return;
    if (yoffset > 0) {
      zoom_map_about_screen_point(app, window, 0.9, cursor_x, cursor_y);
    } else if (yoffset < 0) {
      zoom_map_about_screen_point(app, window, 1.1, cursor_x, cursor_y);
    }
  }
}

// ---- Keyboard callback ----
static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                          int mods) {
  VizApp *app = app_from_window(window);
  (void)scancode;
  (void)mods;
  if (!app) return;
  if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

  switch (key) {
  case GLFW_KEY_ESCAPE:
  case GLFW_KEY_Q:
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    break;
  case GLFW_KEY_SPACE:
    app->paused = !app->paused;
    break;
  case GLFW_KEY_EQUAL:  // + key
    if (app->hex_renderer) {
      int win_w, win_h;
      glfwGetWindowSize(window, &win_w, &win_h);
      zoom_map_about_screen_point(app, window, 0.8,
                                  (double)(win_w + win_w / 2) / 2.0,
                                  (double)win_h / 2.0);
    }
    break;
  case GLFW_KEY_MINUS:
    if (app->hex_renderer) {
      int win_w, win_h;
      glfwGetWindowSize(window, &win_w, &win_h);
      zoom_map_about_screen_point(app, window, 1.25,
                                  (double)(win_w + win_w / 2) / 2.0,
                                  (double)win_h / 2.0);
    }
    break;
  case GLFW_KEY_C:
    if (app->hex_renderer) {
      app->hex_renderer->pan_offset_lat = 0.0;
      app->hex_renderer->pan_offset_lng = 0.0;
      double center_lat, center_lng;
      if (current_map_target_center(app, &center_lat, &center_lng)) {
        snap_map_view_to(app, center_lat, center_lng);
      } else {
        app->map_view_initialized = false;
      }
    }
    break;
  case GLFW_KEY_RIGHT:
    app->playback_speed *= 2.0;
    if (app->playback_speed > 16.0) app->playback_speed = 16.0;
    printf("Speed: %.1fx\n", app->playback_speed);
    break;
  case GLFW_KEY_LEFT:
    app->playback_speed *= 0.5;
    if (app->playback_speed < 0.25) app->playback_speed = 0.25;
    printf("Speed: %.1fx\n", app->playback_speed);
    break;
  default:
    break;
  }
}

// ---- Mouse button callback (drag to pan map) ----
static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
  VizApp *app = app_from_window(window);
  (void)mods;
  if (!app) return;
  if (button != GLFW_MOUSE_BUTTON_LEFT) return;

  if (action == GLFW_PRESS) {
    int win_w, win_h;
    glfwGetWindowSize(window, &win_w, &win_h);
    double cx, cy;
    glfwGetCursorPos(window, &cx, &cy);
    if (cx > (double)win_w / 2.0) {
      app->dragging = true;
      app->drag_last_x = cx;
      app->drag_last_y = cy;
    }
  } else if (action == GLFW_RELEASE) {
    app->dragging = false;
  }
}

// ---- Cursor position callback (drag to pan map) ----
static void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
  VizApp *app = app_from_window(window);
  if (!app || !app->dragging || !app->hex_renderer) return;

  int win_w, win_h;
  glfwGetWindowSize(window, &win_w, &win_h);
  int viewport_w = win_w - win_w / 2;
  int viewport_h = win_h;

  double dx = xpos - app->drag_last_x;
  double dy = ypos - app->drag_last_y;
  app->drag_last_x = xpos;
  app->drag_last_y = ypos;

  double aspect = (double)viewport_h / (double)viewport_w;
  double dlat = dy * (2.0 * app->hex_renderer->zoom * aspect / (double)viewport_h);
  double center_lat, center_lng;
  if (current_render_map_center(app, &center_lat, &center_lng)) {
    double cos_center = cos(center_lat * M_PI / 180.0);
    if (fabs(cos_center) < 1e-6) {
      cos_center = (cos_center < 0.0) ? -1e-6 : 1e-6;
    }
    double dlng = -dx * (2.0 * app->hex_renderer->zoom / (double)viewport_w) /
                  cos_center;
    set_manual_map_center(app, center_lat + dlat, center_lng + dlng);
  } else {
    double dlng = -dx * (2.0 * app->hex_renderer->zoom / (double)viewport_w);
    app->hex_renderer->pan_offset_lat += dlat;
    app->hex_renderer->pan_offset_lng += dlng;
  }
}

// ---- Directory scanning helper ----
// Returns malloc'd full path, or NULL. Caller must free().
// If 'exact_name' is non-NULL, matches that filename exactly.
// Otherwise matches the first file ending with 'extension'.
static char *find_file_in_dir(const char *dir, const char *extension,
                              const char *exact_name) {
  DIR *d = opendir(dir);
  if (!d) return NULL;

  struct dirent *ent;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.') continue;

    bool match = false;
    if (exact_name) {
      match = (strcmp(ent->d_name, exact_name) == 0);
    } else if (extension) {
      size_t nlen = strlen(ent->d_name);
      size_t elen = strlen(extension);
      if (nlen > elen && strcmp(ent->d_name + nlen - elen, extension) == 0)
        match = true;
    }

    if (match) {
      size_t dlen = strlen(dir);
      bool needs_slash = (dlen > 0 && dir[dlen - 1] != '/');
      size_t path_len = dlen + needs_slash + strlen(ent->d_name) + 1;
      char *path = malloc(path_len);
      snprintf(path, path_len, "%s%s%s", dir, needs_slash ? "/" : "",
               ent->d_name);
      closedir(d);
      return path;
    }
  }
  closedir(d);
  return NULL;
}

// ---- Usage ----
static void print_usage(const char *prog) {
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s -c <config.toml> [-d <dir>] [-v <video>] [-f <features.h5>] [-g group] [-t sec] [-r res]\n", prog);
  fprintf(stderr, "  %s -d <dir> [-g group] [-t sec] [-r res]\n", prog);
  fprintf(stderr, "  %s -v <video> [-f <features.h5>] [-g group] [-t sec] [-r res]\n", prog);
  fprintf(stderr, "  %s <video.mp4> [data.h5 group] [time_window_sec] [h3_resolution]\n", prog);
  fprintf(stderr, "\nFlags:\n");
  fprintf(stderr, "  -c <path>   TOML-style config file (defaults < config < CLI)\n");
  fprintf(stderr, "  -d <dir>    Directory containing *.mp4 and features.h5\n");
  fprintf(stderr, "  -v <path>   Video file path\n");
  fprintf(stderr, "  -f <path>   HDF5 features file path\n");
  fprintf(stderr, "  -g <name>   HDF5 group name (default: dino)\n");
  fprintf(stderr, "  -t <sec>    Time window in seconds (default: 5.0)\n");
  fprintf(stderr, "  -r <res>    H3 resolution 0-15 (default: 10)\n");
  fprintf(stderr, "  -h          Print this help\n");
  fprintf(stderr, "\nConfig keys:\n");
  fprintf(stderr, "  session_dir, video_path, features_path, group,\n");
  fprintf(stderr, "  time_window_sec, h3_resolution,\n");
  fprintf(stderr, "  scrub_sensitivity_sec, map_follow_smoothing,\n");
  fprintf(stderr, "  video_decode_budget, ingest_record_budget,\n");
  fprintf(stderr, "  imu_sample_budget, gps_point_budget,\n");
  fprintf(stderr, "  tile_uploads_per_frame, tile_style,\n");
  fprintf(stderr, "  tile_api_key, tile_url_template\n");
  fprintf(stderr, "\nTile styles:\n");
  VizConfig_print_tile_styles(stderr);
  fprintf(stderr, "\nControls:\n");
  fprintf(stderr, "  Space       Pause/resume\n");
  fprintf(stderr, "  +/-         Zoom in/out (heatmap)\n");
  fprintf(stderr, "  Left/Right  Slow down/speed up\n");
  fprintf(stderr, "  Scroll H    Scrub video (on video pane)\n");
  fprintf(stderr, "  Scroll V    Zoom map toward cursor (on map pane)\n");
  fprintf(stderr, "  Drag        Pan map (on map pane)\n");
  fprintf(stderr, "  C           Re-center map and resume follow\n");
  fprintf(stderr, "  Q/Esc       Quit\n");
}

// ---- Main ----
int main(int argc, char *argv[]) {
  const char *config_path = NULL;
  const char *dir_path;
  const char *video_path;
  const char *h5_path;
  const char *group;
  double time_window_sec;
  int h3_resolution;
  char *alloc_video = NULL;  // track malloc'd paths for cleanup
  char *alloc_h5 = NULL;
  VizConfig config;
  VizTileSource tile_source;
  VizApp app = {0};

  int opt;
  bool help_requested = false;
  bool used_runtime_flags = false;

  VizConfig_init(&config);

  opterr = 0;
  optind = 1;
  while ((opt = getopt(argc, argv, ":c:h")) != -1) {
    if (opt == 'c') config_path = optarg;
    if (opt == 'h') help_requested = true;
  }
  opterr = 1;
  optind = 1;

  if (help_requested) {
    print_usage(argv[0]);
    return 0;
  }

  if (config_path && !VizConfig_load_file(&config, config_path)) {
    return 1;
  }

  while ((opt = getopt(argc, argv, "c:d:v:f:g:t:r:h")) != -1) {
    switch (opt) {
    case 'c':
      break;
    case 'd':
      used_runtime_flags = true;
      if (!VizConfig_set_optional_text(config.session_dir,
                                       sizeof(config.session_dir),
                                       &config.has_session_dir,
                                       optarg, "session_dir")) {
        return 1;
      }
      break;
    case 'v':
      used_runtime_flags = true;
      if (!VizConfig_set_optional_text(config.video_path,
                                       sizeof(config.video_path),
                                       &config.has_video_path,
                                       optarg, "video_path")) {
        return 1;
      }
      break;
    case 'f':
      used_runtime_flags = true;
      if (!VizConfig_set_optional_text(config.features_path,
                                       sizeof(config.features_path),
                                       &config.has_features_path,
                                       optarg, "features_path")) {
        return 1;
      }
      break;
    case 'g':
      used_runtime_flags = true;
      if (!VizConfig_set_text(config.group, sizeof(config.group),
                              optarg, "group")) {
        return 1;
      }
      break;
    case 't':
      used_runtime_flags = true;
      if (!VizConfig_parse_positive_double(optarg, "time window",
                                           &config.time_window_sec)) {
        return 1;
      }
      break;
    case 'r':
      used_runtime_flags = true;
      if (!VizConfig_parse_int_in_range(optarg, "H3 resolution", 0, 15,
                                        &config.h3_resolution)) {
        return 1;
      }
      break;
    case 'h':
      print_usage(argv[0]);
      return 0;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  // Legacy positional fallback: allow -c <config> plus positional args.
  if (!used_runtime_flags && optind < argc) {
    if (!VizConfig_set_optional_text(config.video_path, sizeof(config.video_path),
                                     &config.has_video_path,
                                     argv[optind], "video_path")) {
      return 1;
    }
    if (optind + 1 < argc &&
        !VizConfig_set_optional_text(config.features_path,
                                     sizeof(config.features_path),
                                     &config.has_features_path,
                                     argv[optind + 1], "features_path")) {
      return 1;
    }
    if (optind + 2 < argc &&
        !VizConfig_set_text(config.group, sizeof(config.group),
                            argv[optind + 2], "group")) {
      return 1;
    }
    if (optind + 3 < argc &&
        !VizConfig_parse_positive_double(argv[optind + 3], "time window",
                                         &config.time_window_sec)) {
      return 1;
    }
    if (optind + 4 < argc &&
        !VizConfig_parse_int_in_range(argv[optind + 4], "H3 resolution",
                                      0, 15, &config.h3_resolution)) {
      return 1;
    }
  }

  dir_path = config.has_session_dir ? config.session_dir : NULL;
  video_path = config.has_video_path ? config.video_path : NULL;
  h5_path = config.has_features_path ? config.features_path : NULL;
  group = config.group;
  time_window_sec = config.time_window_sec;
  h3_resolution = config.h3_resolution;

  // Directory mode: scan for video and features.h5
  if (dir_path) {
    if (!video_path) {
      alloc_video = find_file_in_dir(dir_path, ".mp4", NULL);
      if (!alloc_video) {
        fprintf(stderr, "No *.mp4 found in %s\n", dir_path);
        return 1;
      }
      video_path = alloc_video;
    }
    if (!h5_path) {
      alloc_h5 = find_file_in_dir(dir_path, NULL, "features.h5");
      if (alloc_h5) h5_path = alloc_h5;
      // features.h5 is optional — no error if missing
    }
  }

  // Validate required args
  if (!video_path) {
    print_usage(argv[0]);
    free(alloc_video);
    free(alloc_h5);
    return 1;
  }

  if (time_window_sec <= 0.0) {
    fprintf(stderr, "Time window must be greater than 0\n");
    free(alloc_video);
    free(alloc_h5);
    return 1;
  }
  if (!VizConfig_resolve_tile_source(&config, &tile_source)) {
    free(alloc_video);
    free(alloc_h5);
    return 1;
  }
  app.playback_speed = 1.0;
  app.time_window_sec = time_window_sec;
  app.h3_resolution = h3_resolution;
  app.scrub_sensitivity_sec = config.scrub_sensitivity_sec;
  app.map_follow_smoothing = config.map_follow_smoothing;
  app.video_decode_budget = config.video_decode_budget;
  app.ingest_record_budget = config.ingest_record_budget;
  app.imu_sample_budget = config.imu_sample_budget;
  app.gps_point_budget = config.gps_point_budget;
  app.tile_uploads_per_frame = config.tile_uploads_per_frame;
  app.window_anchor = -1.0;
  app.first_ts = -1.0;

  // ---- Init GLFW ----
  if (!glfwInit()) {
    fprintf(stderr, "Failed to init GLFW\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  GLFWwindow *window = glfwCreateWindow(1600, 600, "psm-viz", NULL, NULL);
  if (!window) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // vsync
  glfwSetWindowUserPointer(window, &app);
  glfwSetKeyCallback(window, key_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_pos_callback);

  printf("OpenGL %s\n", glGetString(GL_VERSION));

  // ---- Load shaders ----
  GLuint video_prog = Shader_load_program("shaders/video.vert", "shaders/video.frag");
  GLuint hex_prog = Shader_load_program("shaders/hex.vert", "shaders/hex.frag");
  GLuint tile_prog = Shader_load_program("shaders/tile.vert", "shaders/tile.frag");
  if (!video_prog || !hex_prog || !tile_prog) {
    fprintf(stderr, "Failed to load shaders\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }

  // Attention overlay shader (non-fatal if missing)
  GLuint attn_prog = Shader_load_program("shaders/video.vert", "shaders/attention.frag");
  bool has_attention_overlay = (attn_prog != 0);

  // ---- Open video ----
  VideoDecoder *dec = VideoDecoder_open(video_path);
  if (!dec) {
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }
  printf("Video: %dx%d, %.1fs\n", dec->width, dec->height, dec->duration);
  app.dec = dec;

  VideoQuad vq = VideoQuad_create(video_prog);
  app.video_quad = &vq;

  // Decode first frame for initial display
  if (VideoDecoder_next_frame(dec)) {
    VideoQuad_upload(&vq, dec->rgb_buffer, dec->width, dec->height);
  }

  // ---- Open HDF5 (optional) ----
  hid_t h5_file = -1;

  if (h5_path) {
    app.sm = SpatialMemory_new(app.h3_resolution, DEFAULT_CAPACITY, DEFAULT_PRECISION);
    if (!app.sm) {
      fprintf(stderr, "Failed to initialize spatial memory\n");
    }
    h5_file = H5Fopen(h5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5_file < 0) {
      fprintf(stderr, "Failed to open HDF5: %s\n", h5_path);
    } else {
      // Always use DINO embeddings for spatial memory
      app.reader = IngestReader_open(h5_file, DINO);
      if (app.reader) {
        printf("HDF5: %zu records, %zu-d embeddings, group='%s'\n",
               app.reader->n_records, app.reader->emb_dimension, DINO);
      }

      // When -g jepa, load JEPA prediction maps for overlay interpolation
      if (strcmp(group, JEPA) == 0) {
        app.jepa_cache = JepaCache_load(h5_file);
        if (!app.jepa_cache) {
          fprintf(stderr, "Warning: -g jepa but no JEPA prediction maps found; "
                          "falling back to DINO attention overlay\n");
        }
      }
    }
  }

  // Open high-rate IMU/GPS reader (independent of embedding reader)
  if (h5_file >= 0) {
    app.imu_gps = ImuGpsReader_open(h5_file);
    if (app.imu_gps) {
      printf("ImuGps: imu=%s (%zu samples), gps=%s (%zu samples)\n",
             app.imu_gps->has_imu ? "yes" : "no", app.imu_gps->imu_n_records,
             app.imu_gps->has_gps ? "yes" : "no", app.imu_gps->gps_n_records);
    }
  }

  // Create IMU processor — prefer high-rate IMU, fall back to per-embedding
  if (app.imu_gps && app.imu_gps->has_imu) {
    app.imu_proc = ImuProcessor_new(0.3f);
    printf("IMU: high-rate (100Hz) motion coloring enabled\n");
  } else if (app.reader && app.reader->has_imu) {
    app.imu_proc = ImuProcessor_new(0.3f);
    printf("IMU: per-embedding (3Hz) motion coloring enabled\n");
  }

  // Pre-compute first_ts from the embedding stream — embeddings are extracted
  // from video frames, so emb_first_ts corresponds to video PTS 0.
  // IMU samples before the video start will have negative video_time and
  // drain harmlessly on the first frame.
  if (app.reader) {
    IngestRecord peek_rec;
    size_t saved = app.reader->cursor;
    if (IngestReader_next(app.reader, &peek_rec) == INGEST_READ_OK) {
      app.first_ts = peek_rec.timestamp;
      app.reader->cursor = saved;
    }
  }
  if (app.first_ts < 0.0 && app.imu_gps && app.imu_gps->has_imu &&
      app.imu_gps->imu_n_records > 0) {
    // No embedding reader — fall back to IMU first timestamp
    app.first_ts = app.imu_gps->imu_first_ts;
  }

  // ---- Create hex renderer, tile map, and GPS trace ----
  HexRenderer *hr = HexRenderer_new(hex_prog);
  app.hex_renderer = hr;

  TileMap *tm = TileMap_new(tile_prog, tile_source.style_name,
                            tile_source.url_template,
                            tile_source.api_key[0] ? tile_source.api_key : NULL);
  app.tile_map = tm;

  if (!tm) {
    fprintf(stderr, "Failed to initialize tile map for style '%s'\n",
            tile_source.style_name);
    HexRenderer_free(hr);
    if (app.jepa_cache) JepaCache_free(app.jepa_cache);
    if (app.imu_proc) ImuProcessor_free(app.imu_proc);
    if (app.imu_gps) ImuGpsReader_close(app.imu_gps);
    if (app.reader) IngestReader_close(app.reader);
    if (h5_file >= 0) H5Fclose(h5_file);
    if (app.sm) SpatialMemory_free(app.sm);
    VideoQuad_free(&vq);
    VideoDecoder_close(dec);
    glDeleteProgram(video_prog);
    glDeleteProgram(hex_prog);
    glDeleteProgram(tile_prog);
    if (has_attention_overlay) glDeleteProgram(attn_prog);
    glfwDestroyWindow(window);
    glfwTerminate();
    free(alloc_video);
    free(alloc_h5);
    return 1;
  }
  printf("Tiles: %s\n", tile_source.style_name);
  printf("Scrub: %.2fs/step, Map follow: %.2f, Tile uploads/frame: %d\n",
         app.scrub_sensitivity_sec, app.map_follow_smoothing,
         app.tile_uploads_per_frame);
  printf("Budgets: video=%d base/frame, ingest=%d, imu=%d, gps=%d\n",
         app.video_decode_budget, app.ingest_record_budget,
         app.imu_sample_budget, app.gps_point_budget);

  GpsTrace *gt = GpsTrace_new(hex_prog);
  app.gps_trace = gt;

  // ---- Create progress bar ----
  ProgressBar pb = ProgressBar_create(hex_prog);  // reuses hex shader (vec2 pos + vec4 color)

  // ---- Create attention overlay (if shader loaded) ----
  AttentionOverlay ao;
  if (has_attention_overlay) {
    ao = AttentionOverlay_create(attn_prog);
  }

  // ---- Render loop ----
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  app.video_start_time = glfwGetTime();
  app.video_pts_offset = dec->current_pts;
  app.video_done = false;
  double last_frame_time = glfwGetTime();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    double frame_time = glfwGetTime();
    double frame_dt = frame_time - last_frame_time;
    if (frame_dt < 0.0) frame_dt = 0.0;
    if (frame_dt > 0.25) frame_dt = 0.25;
    last_frame_time = frame_time;

    if (app.seek_pending) {
      apply_pending_seek(&app, frame_time);
    }

    int win_w, win_h;
    glfwGetFramebufferSize(window, &win_w, &win_h);
    int half_w = win_w / 2;

    // ---- Decode video frames ----
    if (!app.paused && !app.video_done) {
      double wall_elapsed = (frame_time - app.video_start_time) * app.playback_speed;
      double target_pts = app.video_pts_offset + wall_elapsed;
      bool video_frame_advanced = false;
      bool drain_backlog = false;
      int decode_steps = 0;
      int decode_budget = video_decode_budget(&app);

      while (dec->current_pts < target_pts && decode_steps < decode_budget) {
        if (!VideoDecoder_next_frame(dec)) {
          app.video_done = true;
          break;
        }
        video_frame_advanced = true;
        decode_steps++;
      }

      // Only the last decoded frame is visible, so upload once per tick.
      if (video_frame_advanced) {
        VideoQuad_upload(&vq, dec->rgb_buffer, dec->width, dec->height);
      }

      // If we still have backlog after spending this frame's decode budget,
      // drop the remaining lag and continue from the most recently decoded frame.
      if (!app.video_done && dec->current_pts < target_pts) {
        reset_playback_timing(&app, frame_time);
      }

      // ---- Drain ingest records up to displayed frame ----
      double drain_pts = dec->current_pts;
      if (app.reader && app.sm) {
        IngestRecord record;
        bool data_changed = false;
        int drained_records = 0;
        while (app.reader->cursor < app.reader->n_records) {
          if (drained_records >= app.ingest_record_budget) {
            drain_backlog = true;
            break;
          }
          size_t saved_cursor = app.reader->cursor;
          IngestReadStatus status = IngestReader_next(app.reader, &record);
          if (status == INGEST_READ_EOF) break;
          if (status == INGEST_READ_ERROR) {
            fprintf(stderr, "Failed to read ingest record at index %zu\n",
                    saved_cursor);
            break;
          }

          double record_video_time = record.timestamp - app.first_ts;
          if (record_video_time > drain_pts) {
            app.reader->cursor = saved_cursor;
            break;
          }
          drained_records++;

          SpatialMemory_advance_to_timestamp(app.sm, record.timestamp,
                                             &app.window_anchor,
                                             app.time_window_sec);

          if (!SpatialMemory_observe(app.sm, record.lat, record.lng,
                                     record.embedding,
                                     record.embedding_dim * sizeof(float))) {
            continue;
          }

          // GPS trace + IMU: skip when high-rate reader handles it
          if (!(app.imu_gps && (app.imu_gps->has_imu || app.imu_gps->has_gps))) {
            if (app.imu_proc && record.has_imu) {
              ImuPointMeta meta = ImuProcessor_update(app.imu_proc, record.accel,
                                                      record.gyro, record.timestamp,
                                                      record.lat, record.lng);
              double blended_lat, blended_lng;
              ImuProcessor_get_blended_position(app.imu_proc, &blended_lat,
                                                &blended_lng);
              GpsTrace_push(gt, blended_lat, blended_lng, &meta);
            } else {
              GpsTrace_push(gt, record.lat, record.lng, NULL);
            }
          }

          if (has_attention_overlay) {
            if (app.jepa_cache) {
              // -g jepa: lookup prediction map at this DINO timestamp
              float *interp_map = NULL;
              if (JepaCache_lookup(app.jepa_cache, record.timestamp, &interp_map))
                AttentionOverlay_upload(&ao, interp_map, JEPA_MAP_DIM);
            } else if (record.attention_map) {
              // -g dino: use DINO attention maps directly
              AttentionOverlay_upload(&ao, record.attention_map, record.attn_size);
            }
          }

          data_changed = true;
        }

        if (data_changed) {
          HexRenderer_update(hr, app.sm);
        }
      }

      // ---- Drain high-rate IMU up to displayed PTS ----
      if (app.imu_gps && app.imu_gps->has_imu && app.imu_proc) {
        ImuRecord imu_rec;
        int drained_imu = 0;
        while (app.imu_gps->imu_cursor < app.imu_gps->imu_n_records) {
          if (drained_imu >= app.imu_sample_budget) {
            drain_backlog = true;
            break;
          }
          size_t saved = app.imu_gps->imu_cursor;
          IngestReadStatus status = ImuGpsReader_next_imu(app.imu_gps, &imu_rec);
          if (status == INGEST_READ_EOF) break;
          if (status == INGEST_READ_ERROR) {
            fprintf(stderr, "Failed to read IMU sample at index %zu\n", saved);
            break;
          }

          double imu_video_time = imu_rec.timestamp - app.first_ts;
          if (imu_video_time > drain_pts) {
            app.imu_gps->imu_cursor = saved;
            break;
          }
          drained_imu++;

          // Interpolate GPS at IMU timestamp
          double gps_lat = 0.0, gps_lng = 0.0;
          if (app.imu_gps->has_gps) {
            ImuGpsReader_interpolate_gps(app.imu_gps, imu_rec.timestamp, &gps_lat,
                                         &gps_lng);
          }

          ImuPointMeta meta = ImuProcessor_update(app.imu_proc, imu_rec.accel,
                                                  imu_rec.gyro, imu_rec.timestamp,
                                                  gps_lat, gps_lng);
          double blended_lat, blended_lng;
          ImuProcessor_get_blended_position(app.imu_proc, &blended_lat,
                                            &blended_lng);
          GpsTrace_push(gt, blended_lat, blended_lng, &meta);
        }
      }

      // ---- Drain standalone GPS (no IMU) up to displayed PTS ----
      if (app.imu_gps && app.imu_gps->has_gps && !app.imu_gps->has_imu) {
        int drained_gps = 0;
        while (app.imu_gps->gps_cursor < app.imu_gps->gps_n_records) {
          if (drained_gps >= app.gps_point_budget) {
            drain_backlog = true;
            break;
          }
          double gps_video_time = app.imu_gps->gps_ts[app.imu_gps->gps_cursor] -
                                  app.first_ts;
          if (gps_video_time > drain_pts) break;
          double lat = app.imu_gps->gps_lat[app.imu_gps->gps_cursor];
          double lng = app.imu_gps->gps_lng[app.imu_gps->gps_cursor];
          app.imu_gps->gps_cursor++;
          drained_gps++;
          GpsTrace_push(gt, lat, lng, NULL);
        }
      }

      if (drain_backlog) {
        reset_playback_timing(&app, frame_time);
      }
    } else if (app.paused) {
      app.video_start_time = glfwGetTime();
      app.video_pts_offset = dec->current_pts;
    }

    // ---- Draw ----
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Left half: video
    glViewport(0, 0, half_w, win_h);
    glDisable(GL_BLEND);
    VideoQuad_update_aspect(&vq, dec->width, dec->height, half_w, win_h);
    VideoQuad_draw(&vq);

    // Attention heatmap overlay (semi-transparent over video)
    glEnable(GL_BLEND);
    if (has_attention_overlay) {
      AttentionOverlay_update_aspect(&ao, dec->width, dec->height, half_w, win_h);
      AttentionOverlay_draw(&ao, (strcmp(group, JEPA) == 0) ? 1 : 0);
    }

    // Progress bar overlay on video
    double progress = (dec->duration > 0.0) ? dec->current_pts / dec->duration : 0.0;
    ProgressBar_draw(&pb, progress);

    // Pause icon overlay
    if (app.paused) {
      ProgressBar_draw_pause_icon(&pb);
    }

    // Right half: raster tiles (background) + hex heatmap + GPS trace (overlay)
    glViewport(half_w, 0, win_w - half_w, win_h);

    double map_center_lat = 0.0;
    double map_center_lng = 0.0;
    double map_target_lat, map_target_lng;
    if (current_map_target_center(&app, &map_target_lat, &map_target_lng)) {
      if (!app.map_view_initialized) {
        snap_map_view_to(&app, map_target_lat, map_target_lng);
      } else if (!app.dragging) {
        double follow_alpha =
            1.0 - exp(-app.map_follow_smoothing * frame_dt);
        app.map_view_center_lat +=
            (map_target_lat - app.map_view_center_lat) * follow_alpha;
        app.map_view_center_lng +=
            (map_target_lng - app.map_view_center_lng) * follow_alpha;
      }
      map_center_lat = app.map_view_center_lat;
      map_center_lng = app.map_view_center_lng;
    } else if (app.map_view_initialized) {
      map_center_lat = app.map_view_center_lat;
      map_center_lng = app.map_view_center_lng;
    }

    glDisable(GL_BLEND);
    TileMap_draw(tm, map_center_lat, map_center_lng, hr->zoom,
                 win_w - half_w, win_h, app.tile_uploads_per_frame);
    glEnable(GL_BLEND);
    GpsTrace_upload(gt, map_center_lat, map_center_lng);
    HexRenderer_draw(hr, win_w - half_w, win_h, map_center_lat, map_center_lng);
    GpsTrace_draw(gt, win_w - half_w, win_h, hr->zoom);

    glfwSwapBuffers(window);
  }

  // ---- Cleanup ----
  ProgressBar_free(&pb);
  if (has_attention_overlay) {
    AttentionOverlay_free(&ao);
    glDeleteProgram(attn_prog);
  }
  GpsTrace_free(gt);
  TileMap_free(tm);
  HexRenderer_free(hr);
  VideoQuad_free(&vq);
  VideoDecoder_close(dec);
  glDeleteProgram(video_prog);
  glDeleteProgram(hex_prog);
  glDeleteProgram(tile_prog);

  if (app.jepa_cache) JepaCache_free(app.jepa_cache);
  if (app.imu_proc) ImuProcessor_free(app.imu_proc);
  if (app.imu_gps) ImuGpsReader_close(app.imu_gps);
  if (app.reader) IngestReader_close(app.reader);
  if (h5_file >= 0) H5Fclose(h5_file);
  if (app.sm) SpatialMemory_free(app.sm);

  glfwDestroyWindow(window);
  glfwTerminate();

  free(alloc_video);
  free(alloc_h5);

  return 0;
}
