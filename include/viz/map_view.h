#ifndef VIZ_MAP_VIEW_H
#define VIZ_MAP_VIEW_H

#include <stdbool.h>

double VizMap_clamp_zoom(double zoom);
double VizMap_follow_alpha(double smoothing, double frame_dt);
bool VizMap_zoom_about_viewport_point(double center_lat, double center_lng,
                                      double zoom, double zoom_factor,
                                      int viewport_w, int viewport_h,
                                      double map_x, double map_y,
                                      double *out_center_lat,
                                      double *out_center_lng,
                                      double *out_zoom);
bool VizMap_pan_center(double center_lat, double center_lng, double zoom,
                       int viewport_w, int viewport_h, double dx, double dy,
                       double *out_center_lat, double *out_center_lng);
void VizMap_step_follow(double current_lat, double current_lng,
                        double target_lat, double target_lng,
                        double smoothing, double frame_dt,
                        double *out_lat, double *out_lng);

#endif
