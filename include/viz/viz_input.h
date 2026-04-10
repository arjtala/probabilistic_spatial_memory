#ifndef VIZ_INPUT_H
#define VIZ_INPUT_H

#include <stdbool.h>
#include "viz/gl_platform.h"
#include "viz/viz_app.h"

void VizInput_install_callbacks(GLFWwindow *window);
bool VizInput_current_map_target_center(const VizApp *app, double *out_lat,
                                        double *out_lng);
void VizInput_snap_map_view_to(VizApp *app, double center_lat,
                               double center_lng);

#endif
