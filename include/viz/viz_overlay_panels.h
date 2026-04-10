#ifndef VIZ_OVERLAY_PANELS_H
#define VIZ_OVERLAY_PANELS_H

#include <stdbool.h>
#include <stddef.h>
#include "viz/hex_renderer.h"
#include "viz/ui_overlay.h"

bool VizOverlayPanels_build(UiOverlayMesh *mesh, int window_w, int window_h,
                            int map_split_x, bool show_help,
                            bool show_legend, bool awaiting_initial_play,
                            HexHeatmapMode heatmap_mode, double zoom_degrees,
                            size_t tile_count, const char *status_text);

#endif
