#define _POSIX_C_SOURCE 200809L
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "viz/viz_config.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

static void assert_str_eq(const char *expected, const char *actual) {
  printf("'%s' == '%s'\n", expected, actual ? actual : "(null)");
  if (!actual || strcmp(expected, actual) != 0) {
    fprintf(stderr, "!!! Assertion failed: expected '%s' but got '%s'\n",
            expected, actual ? actual : "(null)");
    exit(EXIT_FAILURE);
  }
}

static void assert_contains(const char *haystack, const char *needle) {
  printf("'%s' contains '%s'\n", haystack, needle);
  if (!haystack || !strstr(haystack, needle)) {
    fprintf(stderr, "!!! Assertion failed: '%s' does not contain '%s'\n",
            haystack ? haystack : "(null)", needle);
    exit(EXIT_FAILURE);
  }
}

static void join_path(char *out, size_t out_size,
                      const char *dir, const char *name) {
  if (snprintf(out, out_size, "%s/%s", dir, name) >= (int)out_size) {
    fprintf(stderr, "Path too long\n");
    exit(EXIT_FAILURE);
  }
}

static void write_text_file(const char *path, const char *content) {
  FILE *file = fopen(path, "w");
  if (!file) {
    perror(path);
    exit(EXIT_FAILURE);
  }
  if (fputs(content, file) == EOF) {
    perror("fputs");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  fclose(file);
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

static void test_defaults_resolve_to_positron(void) {
  VizConfig config;
  VizTileSource tile_source;
  bool ok;

  VizConfig_init(&config);

  assert_str_eq(DINO, config.group);
  ASSERT(config.time_window_sec == 5.0, 1,
         config.time_window_sec == 5.0 ? 1 : 0);
  ASSERT(config.h3_resolution == DEFAULT_RESOLUTION, DEFAULT_RESOLUTION,
         config.h3_resolution);
  assert_str_eq("CartoDB.Positron", config.tile_style);

  ok = VizConfig_resolve_tile_source(&config, &tile_source);
  ASSERT(ok, 1, ok ? 1 : 0);
  assert_str_eq("CartoDB.Positron", tile_source.style_name);
  assert_contains(tile_source.url_template, "light_all");
  ASSERT(!tile_source.requires_api_key, 0, tile_source.requires_api_key ? 1 : 0);
}

static void test_load_file_resolves_relative_paths(void) {
  char dir_template[] = "/tmp/psm_viz_config_XXXXXX";
  char canonical_dir[PATH_MAX];
  const char *base_dir;
  char config_path[PATH_MAX];
  char expected_path[PATH_MAX];
  VizConfig config;
  VizTileSource tile_source;
  bool ok;

  create_temp_dir(dir_template);
  base_dir = realpath(dir_template, canonical_dir) ? canonical_dir : dir_template;

  join_path(config_path, sizeof(config_path), dir_template, "psm-viz.toml");
  write_text_file(
      config_path,
      "# Session paths resolve relative to this file\n"
      "session_dir = \"session\"\n"
      "video_path = \"clips/input.mp4\"\n"
      "features_path = \"features/features.h5\"\n"
      "group = \"jepa\"\n"
      "time_window_sec = 2.5\n"
      "h3_resolution = 8\n"
      "tile_style = \"CartoDB.DarkMatter\"\n");

  VizConfig_init(&config);
  ok = VizConfig_load_file(&config, config_path);
  ASSERT(ok, 1, ok ? 1 : 0);

  join_path(expected_path, sizeof(expected_path), base_dir, "session");
  assert_str_eq(expected_path, config.session_dir);
  join_path(expected_path, sizeof(expected_path), base_dir, "clips/input.mp4");
  assert_str_eq(expected_path, config.video_path);
  join_path(expected_path, sizeof(expected_path), base_dir,
            "features/features.h5");
  assert_str_eq(expected_path, config.features_path);
  assert_str_eq(JEPA, config.group);
  ASSERT(config.time_window_sec == 2.5, 1,
         config.time_window_sec == 2.5 ? 1 : 0);
  ASSERT(config.h3_resolution == 8, 8, config.h3_resolution);
  assert_str_eq("CartoDB.DarkMatter", config.tile_style);

  ok = VizConfig_resolve_tile_source(&config, &tile_source);
  ASSERT(ok, 1, ok ? 1 : 0);
  assert_contains(tile_source.url_template, "dark_all");

  unlink(config_path);
  rmdir(dir_template);
}

static void test_custom_tile_template_and_api_key(void) {
  VizConfig config;
  VizTileSource tile_source;
  bool ok;
  bool set_ok;

  VizConfig_init(&config);
  set_ok = VizConfig_set_optional_text(config.tile_url_template,
                                       sizeof(config.tile_url_template),
                                       &config.has_tile_url_template,
                                       "https://tiles.example.com/{z}/{x}/{y}.png?token={api_key}",
                                       "tile_url_template");
  ASSERT(set_ok, 1, set_ok ? 1 : 0);
  set_ok = VizConfig_set_optional_text(config.tile_api_key,
                                       sizeof(config.tile_api_key),
                                       &config.has_tile_api_key,
                                       "secret-token",
                                       "tile_api_key");
  ASSERT(set_ok, 1, set_ok ? 1 : 0);

  ok = VizConfig_resolve_tile_source(&config, &tile_source);
  ASSERT(ok, 1, ok ? 1 : 0);
  assert_str_eq("Custom", tile_source.style_name);
  assert_contains(tile_source.url_template, "tiles.example.com");
  assert_str_eq("secret-token", tile_source.api_key);
  ASSERT(tile_source.requires_api_key, 1, tile_source.requires_api_key ? 1 : 0);
}

static void test_stadia_without_api_key_is_rejected(void) {
  VizConfig config;
  VizTileSource tile_source;
  bool ok;
  bool set_ok;

  VizConfig_init(&config);
  set_ok = VizConfig_set_text(config.tile_style, sizeof(config.tile_style),
                              "Stadia.AlidadeSmooth", "tile_style");
  ASSERT(set_ok, 1, set_ok ? 1 : 0);
  ok = VizConfig_resolve_tile_source(&config, &tile_source);
  ASSERT(!ok, 0, ok ? 1 : 0);
}

int main(void) {
  RUN_TEST(test_defaults_resolve_to_positron);
  RUN_TEST(test_load_file_resolves_relative_paths);
  RUN_TEST(test_custom_tile_template_and_api_key);
  RUN_TEST(test_stadia_without_api_key_is_rejected);
  return 0;
}
