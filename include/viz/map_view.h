#ifndef VIZ_MAP_VIEW_H
#define VIZ_MAP_VIEW_H

#include <stdbool.h>

typedef enum {
  MAP_PROJECTION_ORTHOGRAPHIC = 0,
  MAP_PROJECTION_ISOMETRIC = 1,
  MAP_PROJECTION_COUNT
} MapProjectionMode;

const char *VizMap_projection_mode_name(MapProjectionMode mode);
MapProjectionMode VizMap_next_projection_mode(MapProjectionMode mode);
void VizMap_projection_basis(MapProjectionMode mode, double *m00, double *m01,
                             double *m10, double *m11);
void VizMap_projected_half_extents(MapProjectionMode mode, double half_w,
                                   double half_h, double *out_half_w,
                                   double *out_half_h);

double VizMap_clamp_zoom(double zoom);
double VizMap_follow_alpha(double smoothing, double frame_dt);
bool VizMap_zoom_about_viewport_point(double center_lat, double center_lng,
                                      double zoom,
                                      MapProjectionMode projection_mode,
                                      double zoom_factor,
                                      int viewport_w, int viewport_h,
                                      double map_x, double map_y,
                                      double *out_center_lat,
                                      double *out_center_lng,
                                      double *out_zoom);
bool VizMap_pan_center(double center_lat, double center_lng, double zoom,
                       MapProjectionMode projection_mode, int viewport_w,
                       int viewport_h, double dx, double dy,
                       double *out_center_lat, double *out_center_lng);
void VizMap_step_follow(double current_lat, double current_lng,
                        double target_lat, double target_lng,
                        double smoothing, double frame_dt,
                        double *out_lat, double *out_lng);

#endif
