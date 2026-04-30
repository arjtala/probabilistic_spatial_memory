#include <math.h>
#include "viz/map_view.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ISO_MAP_SKEW_X 0.55
#define ISO_MAP_SCALE_Y 0.62

double VizMap_clamp_zoom(double zoom) {
  if (zoom < 0.0001) return 0.0001;
  if (zoom > 1.0) return 1.0;
  return zoom;
}

double VizMap_follow_alpha(double smoothing, double frame_dt) {
  double alpha;

  if (smoothing <= 0.0 || frame_dt <= 0.0) return 0.0;
  alpha = 1.0 - exp(-smoothing * frame_dt);
  if (alpha < 0.0) return 0.0;
  if (alpha > 1.0) return 1.0;
  return alpha;
}

static double safe_cos_lat(double lat) {
  double value = cos(lat * M_PI / 180.0);

  if (fabs(value) < 1e-6) {
    return value < 0.0 ? -1e-6 : 1e-6;
  }
  return value;
}

const char *VizMap_projection_mode_name(MapProjectionMode mode) {
  switch (mode) {
  case MAP_PROJECTION_ISOMETRIC:
    return "isometric";
  case MAP_PROJECTION_ORTHOGRAPHIC:
  default:
    return "orthographic";
  }
}

MapProjectionMode VizMap_next_projection_mode(MapProjectionMode mode) {
  return mode == MAP_PROJECTION_ISOMETRIC
             ? MAP_PROJECTION_ORTHOGRAPHIC
             : MAP_PROJECTION_ISOMETRIC;
}

void VizMap_projection_basis(MapProjectionMode mode, double *m00, double *m01,
                             double *m10, double *m11) {
  if (!m00 || !m01 || !m10 || !m11) return;

  switch (mode) {
  case MAP_PROJECTION_ISOMETRIC:
    // Oblique/isometric-style pitch for the full map pane. This keeps the
    // basemap readable while matching the skewed 3D hex presentation.
    *m00 = 1.0;
    *m01 = ISO_MAP_SKEW_X;
    *m10 = 0.0;
    *m11 = ISO_MAP_SCALE_Y;
    break;
  case MAP_PROJECTION_ORTHOGRAPHIC:
  default:
    *m00 = 1.0;
    *m01 = 0.0;
    *m10 = 0.0;
    *m11 = 1.0;
    break;
  }
}

void VizMap_projected_half_extents(MapProjectionMode mode, double half_w,
                                   double half_h, double *out_half_w,
                                   double *out_half_h) {
  double m00, m01, m10, m11;

  if (!out_half_w || !out_half_h) return;
  VizMap_projection_basis(mode, &m00, &m01, &m10, &m11);
  *out_half_w = fabs(m00) * half_w + fabs(m01) * half_h;
  *out_half_h = fabs(m10) * half_w + fabs(m11) * half_h;
}

static bool invert_2x2(double a, double b, double c, double d,
                       double *ia, double *ib, double *ic, double *id) {
  double det = a * d - b * c;

  if (!ia || !ib || !ic || !id) return false;
  if (fabs(det) < 1e-12) return false;
  *ia = d / det;
  *ib = -b / det;
  *ic = -c / det;
  *id = a / det;
  return true;
}

static bool viewport_point_to_world_offset(MapProjectionMode projection_mode,
                                           double zoom, int viewport_w,
                                           int viewport_h, double map_x,
                                           double map_y, double *out_x,
                                           double *out_y) {
  double half_w;
  double half_h;
  double proj_half_w;
  double proj_half_h;
  double m00, m01, m10, m11;
  double im00, im01, im10, im11;
  double nx, ny, px, py;

  if (!out_x || !out_y) return false;
  if (viewport_w <= 0 || viewport_h <= 0) return false;

  half_w = VizMap_clamp_zoom(zoom);
  half_h = half_w * (double)viewport_h / (double)viewport_w;
  VizMap_projected_half_extents(projection_mode, half_w, half_h, &proj_half_w,
                                &proj_half_h);
  VizMap_projection_basis(projection_mode, &m00, &m01, &m10, &m11);
  if (!invert_2x2(m00, m01, m10, m11, &im00, &im01, &im10, &im11)) {
    return false;
  }

  nx = (map_x / (double)viewport_w) * 2.0 - 1.0;
  ny = 1.0 - (map_y / (double)viewport_h) * 2.0;
  px = nx * proj_half_w;
  py = ny * proj_half_h;
  *out_x = im00 * px + im01 * py;
  *out_y = im10 * px + im11 * py;
  return true;
}

static bool viewport_delta_to_world(MapProjectionMode projection_mode,
                                    double zoom, int viewport_w, int viewport_h,
                                    double dx, double dy, double *out_x,
                                    double *out_y) {
  double half_w;
  double half_h;
  double proj_half_w;
  double proj_half_h;
  double m00, m01, m10, m11;
  double im00, im01, im10, im11;
  double px, py;

  if (!out_x || !out_y) return false;
  if (viewport_w <= 0 || viewport_h <= 0) return false;

  half_w = VizMap_clamp_zoom(zoom);
  half_h = half_w * (double)viewport_h / (double)viewport_w;
  VizMap_projected_half_extents(projection_mode, half_w, half_h, &proj_half_w,
                                &proj_half_h);
  VizMap_projection_basis(projection_mode, &m00, &m01, &m10, &m11);
  if (!invert_2x2(m00, m01, m10, m11, &im00, &im01, &im10, &im11)) {
    return false;
  }

  px = (2.0 * dx / (double)viewport_w) * proj_half_w;
  py = (-2.0 * dy / (double)viewport_h) * proj_half_h;
  *out_x = im00 * px + im01 * py;
  *out_y = im10 * px + im11 * py;
  return true;
}

bool VizMap_zoom_about_viewport_point(double center_lat, double center_lng,
                                      double zoom,
                                      MapProjectionMode projection_mode,
                                      double zoom_factor,
                                      int viewport_w, int viewport_h,
                                      double map_x, double map_y,
                                      double *out_center_lat,
                                      double *out_center_lng,
                                      double *out_zoom) {
  double old_zoom;
  double new_zoom;
  double focus_world_x;
  double focus_world_y;
  double focus_lat;
  double focus_lng;
  double new_focus_world_x;
  double new_focus_world_y;
  double new_center_lat;

  if (!out_center_lat || !out_center_lng || !out_zoom) return false;
  if (viewport_w <= 0 || viewport_h <= 0 ||
      map_x < 0.0 || map_x > (double)viewport_w ||
      map_y < 0.0 || map_y > (double)viewport_h) {
    return false;
  }

  old_zoom = VizMap_clamp_zoom(zoom);
  new_zoom = VizMap_clamp_zoom(old_zoom * zoom_factor);
  if (fabs(new_zoom - old_zoom) < 1e-12) {
    *out_center_lat = center_lat;
    *out_center_lng = center_lng;
    *out_zoom = new_zoom;
    return true;
  }

  if (!viewport_point_to_world_offset(projection_mode, old_zoom, viewport_w,
                                      viewport_h, map_x, map_y, &focus_world_x,
                                      &focus_world_y)) {
    return false;
  }
  focus_lat = center_lat + focus_world_y;
  focus_lng = center_lng + focus_world_x / safe_cos_lat(center_lat);
  if (!viewport_point_to_world_offset(projection_mode, new_zoom, viewport_w,
                                      viewport_h, map_x, map_y,
                                      &new_focus_world_x,
                                      &new_focus_world_y)) {
    return false;
  }
  new_center_lat = focus_lat - new_focus_world_y;

  *out_center_lat = new_center_lat;
  *out_center_lng =
      focus_lng - new_focus_world_x / safe_cos_lat(new_center_lat);
  *out_zoom = new_zoom;
  return true;
}

bool VizMap_pan_center(double center_lat, double center_lng, double zoom,
                       MapProjectionMode projection_mode, int viewport_w,
                       int viewport_h, double dx, double dy,
                       double *out_center_lat, double *out_center_lng) {
  double clamped_zoom;
  double world_dx;
  double world_dy;

  if (!out_center_lat || !out_center_lng) return false;
  if (viewport_w <= 0 || viewport_h <= 0) return false;

  clamped_zoom = VizMap_clamp_zoom(zoom);
  if (!viewport_delta_to_world(projection_mode, clamped_zoom, viewport_w,
                               viewport_h, dx, dy, &world_dx, &world_dy)) {
    return false;
  }

  *out_center_lat = center_lat - world_dy;
  *out_center_lng = center_lng - world_dx / safe_cos_lat(center_lat);
  return true;
}

void VizMap_step_follow(double current_lat, double current_lng,
                        double target_lat, double target_lng,
                        double smoothing, double frame_dt,
                        double *out_lat, double *out_lng) {
  double alpha;

  if (!out_lat || !out_lng) return;

  alpha = VizMap_follow_alpha(smoothing, frame_dt);
  *out_lat = current_lat + (target_lat - current_lat) * alpha;
  *out_lng = current_lng + (target_lng - current_lng) * alpha;
}
