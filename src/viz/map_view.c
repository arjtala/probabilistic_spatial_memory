#include <math.h>
#include "viz/map_view.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

bool VizMap_zoom_about_viewport_point(double center_lat, double center_lng,
                                      double zoom, double zoom_factor,
                                      int viewport_w, int viewport_h,
                                      double map_x, double map_y,
                                      double *out_center_lat,
                                      double *out_center_lng,
                                      double *out_zoom) {
  double old_zoom;
  double new_zoom;
  double aspect;
  double nx;
  double ny;
  double old_half_h;
  double focus_lat;
  double focus_lng;
  double new_half_h;
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

  aspect = (double)viewport_h / (double)viewport_w;
  nx = (map_x / (double)viewport_w) * 2.0 - 1.0;
  ny = 1.0 - (map_y / (double)viewport_h) * 2.0;
  old_half_h = old_zoom * aspect;
  focus_lat = center_lat + ny * old_half_h;
  focus_lng = center_lng + (nx * old_zoom) / safe_cos_lat(center_lat);
  new_half_h = new_zoom * aspect;
  new_center_lat = focus_lat - ny * new_half_h;

  *out_center_lat = new_center_lat;
  *out_center_lng =
      focus_lng - (nx * new_zoom) / safe_cos_lat(new_center_lat);
  *out_zoom = new_zoom;
  return true;
}

bool VizMap_pan_center(double center_lat, double center_lng, double zoom,
                       int viewport_w, int viewport_h, double dx, double dy,
                       double *out_center_lat, double *out_center_lng) {
  double clamped_zoom;
  double aspect;
  double dlat;
  double dlng;

  if (!out_center_lat || !out_center_lng) return false;
  if (viewport_w <= 0 || viewport_h <= 0) return false;

  clamped_zoom = VizMap_clamp_zoom(zoom);
  aspect = (double)viewport_h / (double)viewport_w;
  dlat = dy * (2.0 * clamped_zoom * aspect / (double)viewport_h);
  dlng = -dx * (2.0 * clamped_zoom / (double)viewport_w) /
         safe_cos_lat(center_lat);

  *out_center_lat = center_lat + dlat;
  *out_center_lng = center_lng + dlng;
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
