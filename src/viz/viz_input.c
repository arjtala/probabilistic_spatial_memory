#include <math.h>
#include <stdio.h>
#include "viz/map_view.h"
#include "viz/viz_input.h"

static void refresh_heatmap_mode(VizApp *app) {
  if (!app || !app->hex_renderer) return;
  if (app->sm) {
    HexRenderer_update(app->hex_renderer, app->sm);
  } else {
    app->hex_renderer->vertex_count = 0;
  }
}

static VizApp *app_from_window(GLFWwindow *window) {
  return (VizApp *)glfwGetWindowUserPointer(window);
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

bool VizInput_current_map_target_center(const VizApp *app, double *out_lat,
                                        double *out_lng) {
  if (!current_auto_map_center(app, out_lat, out_lng)) return false;
  if (app->hex_renderer) {
    *out_lat += app->hex_renderer->pan_offset_lat;
    *out_lng += app->hex_renderer->pan_offset_lng;
  }
  return true;
}

void VizInput_snap_map_view_to(VizApp *app, double center_lat,
                               double center_lng) {
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
  VizInput_snap_map_view_to(app, center_lat, center_lng);
}

static bool current_render_map_center(const VizApp *app, double *out_lat,
                                      double *out_lng) {
  if (!app || !out_lat || !out_lng) return false;
  if (app->map_view_initialized) {
    *out_lat = app->map_view_center_lat;
    *out_lng = app->map_view_center_lng;
    return true;
  }
  return VizInput_current_map_target_center(app, out_lat, out_lng);
}

static void zoom_map_about_screen_point(VizApp *app, GLFWwindow *window,
                                        double zoom_factor, double cursor_x,
                                        double cursor_y) {
  double center_lat, center_lng;
  double new_center_lat, new_center_lng, new_zoom;
  int win_w, win_h;
  int half_w;
  int viewport_w;
  int viewport_h;
  double map_x;
  double map_y;

  if (!app || !app->hex_renderer) return;

  glfwGetWindowSize(window, &win_w, &win_h);
  half_w = win_w / 2;
  viewport_w = win_w - half_w;
  viewport_h = win_h;
  map_x = cursor_x - (double)half_w;
  map_y = cursor_y;

  if (viewport_w <= 0 || viewport_h <= 0 ||
      map_x < 0.0 || map_x > (double)viewport_w ||
      map_y < 0.0 || map_y > (double)viewport_h) {
    return;
  }

  if (!current_render_map_center(app, &center_lat, &center_lng)) {
    app->hex_renderer->zoom =
        VizMap_clamp_zoom(app->hex_renderer->zoom * zoom_factor);
    return;
  }
  if (!VizMap_zoom_about_viewport_point(
          center_lat, center_lng, app->hex_renderer->zoom, zoom_factor,
          viewport_w, viewport_h, map_x, map_y, &new_center_lat,
          &new_center_lng, &new_zoom)) {
    return;
  }

  app->hex_renderer->zoom = new_zoom;
  set_manual_map_center(app, new_center_lat, new_center_lng);
}

static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  VizApp *app = app_from_window(window);
  int win_w, win_h;
  double cursor_x, cursor_y;
  double half_w;

  if (!app) return;
  glfwGetWindowSize(window, &win_w, &win_h);
  glfwGetCursorPos(window, &cursor_x, &cursor_y);

  half_w = (double)win_w / 2.0;
  if (cursor_x < half_w) {
    double seek_delta;
    double base_pts;
    double target;

    if (!app->dec || app->dec->duration <= 0.0) return;
    if (fabs(xoffset) < 1e-9) return;

    seek_delta = xoffset * app->scrub_sensitivity_sec;
    base_pts = app->seek_pending ? app->pending_seek_target
                                 : app->dec->current_pts;
    target = base_pts + seek_delta;
    if (target < 0.0) target = 0.0;
    if (target > app->dec->duration) target = app->dec->duration;
    app->pending_seek_target = target;
    app->seek_pending = true;
  } else {
    if (!app->hex_renderer) return;
    if (yoffset > 0) {
      zoom_map_about_screen_point(app, window, 0.9, cursor_x, cursor_y);
    } else if (yoffset < 0) {
      zoom_map_about_screen_point(app, window, 1.1, cursor_x, cursor_y);
    }
  }
}

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
    if (!app->paused) {
      app->awaiting_initial_play = false;
      app->help_overlay_visible = false;
    }
    break;
  case GLFW_KEY_EQUAL:
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
      double center_lat, center_lng;

      app->hex_renderer->pan_offset_lat = 0.0;
      app->hex_renderer->pan_offset_lng = 0.0;
      if (VizInput_current_map_target_center(app, &center_lat, &center_lng)) {
        VizInput_snap_map_view_to(app, center_lat, center_lng);
      } else {
        app->map_view_initialized = false;
      }
    }
    break;
  case GLFW_KEY_H:
    app->debug_hud_enabled = !app->debug_hud_enabled;
    app->next_debug_title_update = 0.0;
    if (!app->debug_hud_enabled) {
      glfwSetWindowTitle(window, "psm-viz");
    }
    printf("Debug HUD: %s\n", app->debug_hud_enabled ? "on" : "off");
    break;
  case GLFW_KEY_SLASH:
    if (!(mods & GLFW_MOD_SHIFT)) {
      app->help_overlay_visible = !app->help_overlay_visible;
      printf("Help overlay: %s\n", app->help_overlay_visible ? "on" : "off");
      break;
    }
    // fall through for '?'
  case GLFW_KEY_F1:
    app->help_overlay_visible = !app->help_overlay_visible;
    printf("Help overlay: %s\n", app->help_overlay_visible ? "on" : "off");
    break;
  case GLFW_KEY_L:
    app->legend_overlay_visible = !app->legend_overlay_visible;
    printf("Legend overlay: %s\n", app->legend_overlay_visible ? "on" : "off");
    break;
  case GLFW_KEY_P:
    app->screenshot_requested = true;
    printf("Screenshot requested (.png)\n");
    break;
  case GLFW_KEY_M:
    if (app->hex_renderer) {
      HexHeatmapMode next_mode =
          HexRenderer_next_heatmap_mode(app->hex_renderer->heatmap_mode);
      HexRenderer_set_heatmap_mode(app->hex_renderer, next_mode);
      refresh_heatmap_mode(app);
      printf("Heatmap mode: %s\n",
             HexRenderer_heatmap_mode_name(app->hex_renderer->heatmap_mode));
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

static void mouse_button_callback(GLFWwindow *window, int button, int action,
                                  int mods) {
  VizApp *app = app_from_window(window);

  (void)mods;
  if (!app) return;
  if (button != GLFW_MOUSE_BUTTON_LEFT) return;

  if (action == GLFW_PRESS) {
    int win_w, win_h;
    double cx, cy;

    glfwGetWindowSize(window, &win_w, &win_h);
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

static void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
  VizApp *app = app_from_window(window);
  double center_lat, center_lng;
  double new_center_lat, new_center_lng;
  int win_w, win_h;
  int viewport_w, viewport_h;
  double dx, dy;

  if (!app || !app->dragging || !app->hex_renderer) return;

  glfwGetWindowSize(window, &win_w, &win_h);
  viewport_w = win_w - win_w / 2;
  viewport_h = win_h;

  dx = xpos - app->drag_last_x;
  dy = ypos - app->drag_last_y;
  app->drag_last_x = xpos;
  app->drag_last_y = ypos;

  if (current_render_map_center(app, &center_lat, &center_lng)) {
    if (!VizMap_pan_center(center_lat, center_lng, app->hex_renderer->zoom,
                           viewport_w, viewport_h, dx, dy, &new_center_lat,
                           &new_center_lng)) {
      return;
    }
    set_manual_map_center(app, new_center_lat, new_center_lng);
  } else {
    if (!VizMap_pan_center(0.0, 0.0, app->hex_renderer->zoom, viewport_w,
                           viewport_h, dx, dy, &new_center_lat,
                           &new_center_lng)) {
      return;
    }
    app->hex_renderer->pan_offset_lat += new_center_lat;
    app->hex_renderer->pan_offset_lng += new_center_lng;
  }
}

void VizInput_install_callbacks(GLFWwindow *window) {
  if (!window) return;
  glfwSetKeyCallback(window, key_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_pos_callback);
}
