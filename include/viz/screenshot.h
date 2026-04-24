#ifndef VIZ_SCREENSHOT_H
#define VIZ_SCREENSHOT_H

#include <stdbool.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  char output_dir[PATH_MAX];
  char prefix[64];
  unsigned long next_index;
} VizScreenshotSession;

bool VizScreenshot_ensure_directory(const char *path);
bool VizScreenshot_init(VizScreenshotSession *session, const char *directory,
                        const char *prefix, unsigned long starting_index);
bool VizScreenshot_write_png_rgba(const char *path, int width, int height,
                                  const uint8_t *rgba_pixels);
bool VizScreenshot_capture_region(VizScreenshotSession *session, int x, int y,
                                  int width, int height, char *out_path,
                                  size_t out_path_size);

#endif
