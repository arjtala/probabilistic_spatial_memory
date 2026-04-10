#define _POSIX_C_SOURCE 200809L
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "stb/stb_image.h"
#include "viz/screenshot.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

static void assert_true(bool condition, const char *message) {
  printf("%s -> %s\n", message, condition ? "ok" : "fail");
  if (!condition) {
    fprintf(stderr, "!!! %s\n", message);
    exit(EXIT_FAILURE);
  }
}

static void create_temp_dir(char *path_template) {
  int fd = mkstemp(path_template);

  if (fd < 0) {
    perror("mkstemp");
    exit(EXIT_FAILURE);
  }
  close(fd);
  unlink(path_template);
  if (mkdir(path_template, 0700) != 0) {
    perror("mkdir");
    exit(EXIT_FAILURE);
  }
}

static void assert_u8_eq(uint8_t expected, uint8_t actual, const char *message) {
  printf("%s: %u == %u\n", message, (unsigned)expected, (unsigned)actual);
  if (expected != actual) {
    fprintf(stderr, "!!! %s: expected %u but got %u\n", message,
            (unsigned)expected, (unsigned)actual);
    exit(EXIT_FAILURE);
  }
}

static void test_build_default_path_uses_directory_and_extension(void) {
  char path[PATH_MAX];

  assert_true(VizScreenshot_build_default_path(path, sizeof(path), "captures", 7),
              "build screenshot path");
  assert_true(strstr(path, "captures/psm-viz-") != NULL,
              "path contains capture directory");
  assert_true(strstr(path, "-007.png") != NULL, "path contains png suffix");
}

static void test_ensure_directory_and_write_png(void) {
  char dir_template[] = "/tmp/psm_screenshot_XXXXXX";
  char dir_path[PATH_MAX];
  char png_path[PATH_MAX];
  struct stat st;
  uint8_t rgba[16] = {
      255, 0, 0, 255, 0, 255, 0, 255,
      0, 0, 255, 255, 255, 255, 255, 255,
  };
  FILE *file;
  uint8_t header[8];
  int decoded_w = 0;
  int decoded_h = 0;
  int decoded_comp = 0;
  uint8_t *decoded = NULL;

  create_temp_dir(dir_template);
  assert_true(snprintf(dir_path, sizeof(dir_path), "%s/captures", dir_template) <
                  (int)sizeof(dir_path),
              "build capture dir path");
  assert_true(VizScreenshot_ensure_directory(dir_path),
              "ensure capture directory");
  assert_true(stat(dir_path, &st) == 0 && S_ISDIR(st.st_mode),
              "capture directory exists");

  assert_true(snprintf(png_path, sizeof(png_path), "%s/frame.png", dir_path) <
                  (int)sizeof(png_path),
              "build png path");
  assert_true(VizScreenshot_write_png_rgba(png_path, 2, 2, rgba),
              "write png");

  file = fopen(png_path, "rb");
  assert_true(file != NULL, "open png");
  assert_true(fread(header, 1, sizeof(header), file) == sizeof(header),
              "read png header");
  fclose(file);

  assert_u8_eq(0x89u, header[0], "png signature[0]");
  assert_u8_eq('P', header[1], "png signature[1]");
  assert_u8_eq('N', header[2], "png signature[2]");
  assert_u8_eq('G', header[3], "png signature[3]");
  assert_u8_eq('\r', header[4], "png signature[4]");
  assert_u8_eq('\n', header[5], "png signature[5]");
  assert_u8_eq(0x1au, header[6], "png signature[6]");
  assert_u8_eq('\n', header[7], "png signature[7]");

  decoded = stbi_load(png_path, &decoded_w, &decoded_h, &decoded_comp, 4);
  assert_true(decoded != NULL, "decode png");
  ASSERT(decoded_w == 2, 2, decoded_w);
  ASSERT(decoded_h == 2, 2, decoded_h);
  assert_u8_eq(255u, decoded[0], "top-left red");
  assert_u8_eq(0u, decoded[1], "top-left green");
  assert_u8_eq(0u, decoded[2], "top-left blue");
  assert_u8_eq(255u, decoded[3], "top-left alpha");
  assert_u8_eq(0u, decoded[4], "top-right red");
  assert_u8_eq(255u, decoded[5], "top-right green");
  assert_u8_eq(0u, decoded[6], "top-right blue");
  assert_u8_eq(255u, decoded[7], "top-right alpha");
  assert_u8_eq(0u, decoded[8], "bottom-left red");
  assert_u8_eq(0u, decoded[9], "bottom-left green");
  assert_u8_eq(255u, decoded[10], "bottom-left blue");
  assert_u8_eq(255u, decoded[11], "bottom-left alpha");
  assert_u8_eq(255u, decoded[12], "bottom-right red");
  assert_u8_eq(255u, decoded[13], "bottom-right green");
  assert_u8_eq(255u, decoded[14], "bottom-right blue");
  assert_u8_eq(255u, decoded[15], "bottom-right alpha");
  stbi_image_free(decoded);

  unlink(png_path);
  rmdir(dir_path);
  rmdir(dir_template);
}

int main(void) {
  test_build_default_path_uses_directory_and_extension();
  test_ensure_directory_and_write_png();
  return 0;
}
