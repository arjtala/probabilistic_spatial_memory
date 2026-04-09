#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "viz/viz_config.h"

typedef struct {
  const char *name;
  const char *url_template;
  bool requires_api_key;
} TileStylePreset;

static const TileStylePreset k_tile_presets[] = {
    {"CartoDB.Positron",
     "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", false},
    {"CartoDB.PositronNoLabels",
     "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png", false},
    {"CartoDB.Voyager",
     "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
     false},
    {"CartoDB.DarkMatter",
     "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", false},
    {"Stadia.AlidadeSmooth",
     "https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}.png?api_key={api_key}",
     true},
    {"Stadia.AlidadeSmoothDark",
     "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}.png?api_key={api_key}",
     true},
};

static const size_t k_tile_preset_count =
    sizeof(k_tile_presets) / sizeof(k_tile_presets[0]);

static char *trim_left(char *text) {
  while (*text && isspace((unsigned char)*text)) text++;
  return text;
}

static void trim_right(char *text) {
  size_t len = strlen(text);
  while (len > 0 && isspace((unsigned char)text[len - 1])) {
    text[len - 1] = '\0';
    len--;
  }
}

static char *trim(char *text) {
  char *trimmed = trim_left(text);
  trim_right(trimmed);
  return trimmed;
}

static const TileStylePreset *find_tile_preset(const char *name) {
  if (!name || name[0] == '\0') return NULL;
  for (size_t i = 0; i < k_tile_preset_count; i++) {
    if (strcmp(k_tile_presets[i].name, name) == 0) {
      return &k_tile_presets[i];
    }
  }
  return NULL;
}

static bool has_api_key_placeholder(const char *text) {
  return text && strstr(text, "{api_key}") != NULL;
}

static void strip_inline_comment(char *line) {
  bool in_quotes = false;
  bool escaped = false;
  for (char *cursor = line; *cursor; cursor++) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (*cursor == '\\' && in_quotes) {
      escaped = true;
      continue;
    }
    if (*cursor == '"') {
      in_quotes = !in_quotes;
      continue;
    }
    if (*cursor == '#' && !in_quotes) {
      *cursor = '\0';
      return;
    }
  }
}

static bool copy_path_value(char *dst, size_t dst_size, bool *has_value,
                            const char *value, const char *field_name,
                            const char *base_dir) {
  char expanded[PATH_MAX];
  const char *src = value;
  const char *home = getenv("HOME");

  if (value[0] == '~' && (value[1] == '/' || value[1] == '\0') &&
      home && home[0] != '\0') {
    if (snprintf(expanded, sizeof(expanded), "%s%s", home, value + 1) >=
        (int)sizeof(expanded)) {
      fprintf(stderr, "%s is too long after expanding HOME\n", field_name);
      return false;
    }
    src = expanded;
  } else if (value[0] != '/' && base_dir && base_dir[0] != '\0') {
    if (snprintf(expanded, sizeof(expanded), "%s/%s", base_dir, value) >=
        (int)sizeof(expanded)) {
      fprintf(stderr, "%s is too long after resolving relative path\n",
              field_name);
      return false;
    }
    src = expanded;
  }

  return VizConfig_set_optional_text(dst, dst_size, has_value, src, field_name);
}

static bool parse_string_value(const char *text, char *out, size_t out_size,
                               const char *field_name, const char *path,
                               int line_no) {
  if (!text) {
    fprintf(stderr, "%s:%d: missing value for %s\n", path, line_no, field_name);
    return false;
  }

  if (text[0] != '"') {
    return VizConfig_set_text(out, out_size, text, field_name);
  }

  size_t out_len = 0;
  bool escaped = false;
  for (size_t i = 1;; i++) {
    char ch = text[i];
    if (ch == '\0') {
      fprintf(stderr, "%s:%d: unterminated quoted string for %s\n",
              path, line_no, field_name);
      return false;
    }

    if (escaped) {
      char translated = ch;
      switch (ch) {
      case 'n': translated = '\n'; break;
      case 'r': translated = '\r'; break;
      case 't': translated = '\t'; break;
      case '\\': break;
      case '"': break;
      default: break;
      }
      if (out_len + 1 >= out_size) {
        fprintf(stderr, "%s:%d: %s is too long\n", path, line_no, field_name);
        return false;
      }
      out[out_len++] = translated;
      escaped = false;
      continue;
    }

    if (ch == '\\') {
      escaped = true;
      continue;
    }

    if (ch == '"') {
      const char *rest = text + i + 1;
      while (*rest && isspace((unsigned char)*rest)) rest++;
      if (*rest != '\0') {
        fprintf(stderr, "%s:%d: unexpected trailing characters after %s\n",
                path, line_no, field_name);
        return false;
      }
      out[out_len] = '\0';
      return true;
    }

    if (out_len + 1 >= out_size) {
      fprintf(stderr, "%s:%d: %s is too long\n", path, line_no, field_name);
      return false;
    }
    out[out_len++] = ch;
  }
}

static void config_dir_from_path(const char *path, char *out, size_t out_size) {
  char resolved[PATH_MAX];
  const char *source = path;
  const char *slash;

  if (realpath(path, resolved)) source = resolved;
  slash = strrchr(source, '/');
  if (!slash) {
    VizConfig_set_text(out, out_size, ".", "config directory");
    return;
  }

  size_t dir_len = (size_t)(slash - source);
  if (dir_len == 0) dir_len = 1;
  if (dir_len + 1 > out_size) {
    out[0] = '\0';
    return;
  }
  memcpy(out, source, dir_len);
  out[dir_len] = '\0';
}

void VizConfig_init(VizConfig *config) {
  if (!config) return;
  memset(config, 0, sizeof(*config));
  snprintf(config->group, sizeof(config->group), "%s", DINO);
  config->time_window_sec = 5.0;
  config->h3_resolution = DEFAULT_RESOLUTION;
  config->scrub_sensitivity_sec = 2.0;
  config->map_follow_smoothing = 8.0;
  config->tile_uploads_per_frame = 1;
  snprintf(config->tile_style, sizeof(config->tile_style), "%s",
           "CartoDB.Positron");
}

bool VizConfig_set_text(char *dst, size_t dst_size, const char *value,
                        const char *field_name) {
  if (!dst || dst_size == 0 || !value) {
    fprintf(stderr, "Invalid %s\n", field_name);
    return false;
  }
  if (value[0] == '\0') {
    fprintf(stderr, "%s must not be empty\n", field_name);
    return false;
  }
  if (snprintf(dst, dst_size, "%s", value) >= (int)dst_size) {
    fprintf(stderr, "%s is too long\n", field_name);
    return false;
  }
  return true;
}

bool VizConfig_set_optional_text(char *dst, size_t dst_size, bool *has_value,
                                 const char *value, const char *field_name) {
  if (!dst || dst_size == 0 || !has_value || !value) {
    fprintf(stderr, "Invalid %s\n", field_name);
    return false;
  }
  if (value[0] == '\0') {
    dst[0] = '\0';
    *has_value = false;
    return true;
  }
  if (!VizConfig_set_text(dst, dst_size, value, field_name)) return false;
  *has_value = true;
  return true;
}

bool VizConfig_parse_positive_double(const char *text, const char *name,
                                     double *out_value) {
  char *end = NULL;
  errno = 0;
  double value = strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0') {
    fprintf(stderr, "Invalid %s: '%s'\n", name, text);
    return false;
  }
  if (value <= 0.0) {
    fprintf(stderr, "%s must be greater than 0, got '%s'\n", name, text);
    return false;
  }
  *out_value = value;
  return true;
}

bool VizConfig_parse_int_in_range(const char *text, const char *name,
                                  int min_value, int max_value,
                                  int *out_value) {
  char *end = NULL;
  errno = 0;
  long value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    fprintf(stderr, "Invalid %s: '%s'\n", name, text);
    return false;
  }
  if (value < min_value || value > max_value) {
    fprintf(stderr, "%s must be in [%d, %d], got '%s'\n",
            name, min_value, max_value, text);
    return false;
  }
  *out_value = (int)value;
  return true;
}

bool VizConfig_load_file(VizConfig *config, const char *path) {
  FILE *file;
  char base_dir[PATH_MAX];
  char line[4096];
  int line_no = 0;

  if (!config || !path) return false;

  file = fopen(path, "r");
  if (!file) {
    perror(path);
    return false;
  }

  config_dir_from_path(path, base_dir, sizeof(base_dir));

  while (fgets(line, sizeof(line), file)) {
    char value_buf[VIZ_CONFIG_URL_CAP];
    char *key;
    char *value_text;
    char *equals;

    line_no++;
    if (!strchr(line, '\n') && !feof(file)) {
      fprintf(stderr, "%s:%d: line is too long\n", path, line_no);
      fclose(file);
      return false;
    }

    strip_inline_comment(line);
    key = trim(line);
    if (key[0] == '\0') continue;

    equals = strchr(key, '=');
    if (!equals) {
      fprintf(stderr, "%s:%d: expected key = value\n", path, line_no);
      fclose(file);
      return false;
    }

    *equals = '\0';
    value_text = trim(equals + 1);
    key = trim(key);

    if (value_text[0] == '\0') {
      fprintf(stderr, "%s:%d: missing value for %s\n", path, line_no, key);
      fclose(file);
      return false;
    }

    if (!parse_string_value(value_text, value_buf, sizeof(value_buf),
                            key, path, line_no)) {
      fclose(file);
      return false;
    }

    if (strcmp(key, "session_dir") == 0) {
      if (!copy_path_value(config->session_dir, sizeof(config->session_dir),
                           &config->has_session_dir, value_buf, key, base_dir)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "video_path") == 0) {
      if (!copy_path_value(config->video_path, sizeof(config->video_path),
                           &config->has_video_path, value_buf, key, base_dir)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "features_path") == 0) {
      if (!copy_path_value(config->features_path, sizeof(config->features_path),
                           &config->has_features_path, value_buf, key, base_dir)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "group") == 0) {
      if (!VizConfig_set_text(config->group, sizeof(config->group),
                              value_buf, key)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "time_window_sec") == 0) {
      if (!VizConfig_parse_positive_double(value_buf, key,
                                           &config->time_window_sec)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "h3_resolution") == 0) {
      if (!VizConfig_parse_int_in_range(value_buf, key, 0, 15,
                                        &config->h3_resolution)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "scrub_sensitivity_sec") == 0) {
      if (!VizConfig_parse_positive_double(value_buf, key,
                                           &config->scrub_sensitivity_sec)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "map_follow_smoothing") == 0) {
      if (!VizConfig_parse_positive_double(value_buf, key,
                                           &config->map_follow_smoothing)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "tile_uploads_per_frame") == 0) {
      if (!VizConfig_parse_int_in_range(value_buf, key, 1, 8,
                                        &config->tile_uploads_per_frame)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "tile_style") == 0) {
      if (!VizConfig_set_text(config->tile_style, sizeof(config->tile_style),
                              value_buf, key)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "tile_api_key") == 0) {
      if (!VizConfig_set_optional_text(config->tile_api_key,
                                       sizeof(config->tile_api_key),
                                       &config->has_tile_api_key,
                                       value_buf, key)) {
        fclose(file);
        return false;
      }
    } else if (strcmp(key, "tile_url_template") == 0) {
      if (!VizConfig_set_optional_text(config->tile_url_template,
                                       sizeof(config->tile_url_template),
                                       &config->has_tile_url_template,
                                       value_buf, key)) {
        fclose(file);
        return false;
      }
    } else {
      fprintf(stderr, "%s:%d: unknown key '%s'\n", path, line_no, key);
      fclose(file);
      return false;
    }
  }

  fclose(file);
  return true;
}

bool VizConfig_resolve_tile_source(const VizConfig *config, VizTileSource *out) {
  const TileStylePreset *preset = NULL;
  const char *style_name;
  const char *url_template;
  bool requires_api_key = false;

  if (!config || !out) return false;

  memset(out, 0, sizeof(*out));

  if (config->has_tile_url_template) {
    style_name = "Custom";
    url_template = config->tile_url_template;
    requires_api_key = has_api_key_placeholder(url_template);
  } else {
    preset = find_tile_preset(config->tile_style);
    if (!preset) {
      fprintf(stderr, "Unknown tile_style '%s'. Available styles:\n",
              config->tile_style);
      VizConfig_print_tile_styles(stderr);
      return false;
    }
    style_name = preset->name;
    url_template = preset->url_template;
    requires_api_key = preset->requires_api_key;
  }

  if (!VizConfig_set_text(out->style_name, sizeof(out->style_name),
                          style_name, "tile style name")) {
    return false;
  }
  if (!VizConfig_set_text(out->url_template, sizeof(out->url_template),
                          url_template, "tile URL template")) {
    return false;
  }
  out->requires_api_key = requires_api_key;

  if (requires_api_key) {
    if (!config->has_tile_api_key || config->tile_api_key[0] == '\0') {
      fprintf(stderr,
              "Selected tile source '%s' requires tile_api_key in the config\n",
              style_name);
      return false;
    }
    if (!VizConfig_set_text(out->api_key, sizeof(out->api_key),
                            config->tile_api_key, "tile_api_key")) {
      return false;
    }
  }

  return true;
}

void VizConfig_print_tile_styles(FILE *stream) {
  if (!stream) return;
  for (size_t i = 0; i < k_tile_preset_count; i++) {
    fprintf(stream, "  - %s", k_tile_presets[i].name);
    if (k_tile_presets[i].requires_api_key) {
      fprintf(stream, " (requires tile_api_key)");
    }
    fprintf(stream, "\n");
  }
}
