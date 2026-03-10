#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <getopt.h>
#include <dirent.h>
#include <string.h>
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <hdf5.h>

#include "viz/shader.h"
#include "viz/video_decoder.h"
#include "viz/hex_renderer.h"
#include "viz/tile_map.h"
#include "viz/gps_trace.h"
#include "ingest/ingest.h"

// ---- State ----
static bool paused = false;
static double playback_speed = 1.0;

// ---- Video quad geometry ----
typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint texture;
  GLuint program;
} VideoQuad;

// ---- Forward declarations for scroll callback globals ----
static VideoDecoder *g_dec = NULL;
static VideoQuad *g_vq_ptr = NULL;
static HexRenderer *g_hex_renderer = NULL;
static TileMap *g_tile_map = NULL;
static GpsTrace *g_gps_trace = NULL;
static IngestReader *g_reader = NULL;
static SpatialMemory *g_sm = NULL;
static double *g_first_ts = NULL;
static double *g_last_adv = NULL;
static double g_time_window_sec = 5.0;
static int g_h3_resolution = DEFAULT_RESOLUTION;

// Timing state that scroll callback needs to reset
static double *g_video_start_time = NULL;
static double *g_video_pts_offset = NULL;
static bool *g_video_done = NULL;

static VideoQuad VideoQuad_create(GLuint program) {
  VideoQuad vq;
  vq.program = program;

  // Fullscreen quad: position (x,y) + texcoord (u,v)
  float quad[] = {
      -1.0f, -1.0f, 0.0f, 1.0f,
       1.0f, -1.0f, 1.0f, 1.0f,
      -1.0f,  1.0f, 0.0f, 0.0f,
       1.0f,  1.0f, 1.0f, 0.0f,
  };

  glGenVertexArrays(1, &vq.vao);
  glGenBuffers(1, &vq.vbo);
  glBindVertexArray(vq.vao);
  glBindBuffer(GL_ARRAY_BUFFER, vq.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);

  // Create texture
  glGenTextures(1, &vq.texture);
  glBindTexture(GL_TEXTURE_2D, vq.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);

  return vq;
}

static void VideoQuad_update_aspect(VideoQuad *vq, int video_w, int video_h,
                                    int viewport_w, int viewport_h) {
  if (video_w <= 0 || video_h <= 0 || viewport_w <= 0 || viewport_h <= 0) return;

  double video_aspect = (double)video_w / (double)video_h;
  double viewport_aspect = (double)viewport_w / (double)viewport_h;

  float sx = 1.0f, sy = 1.0f;
  if (video_aspect > viewport_aspect) {
    // Fit width, shrink height (letterbox)
    sy = (float)(viewport_aspect / video_aspect);
  } else {
    // Fit height, shrink width (pillarbox)
    sx = (float)(video_aspect / viewport_aspect);
  }

  float quad[] = {
      -sx, -sy, 0.0f, 1.0f,
       sx, -sy, 1.0f, 1.0f,
      -sx,  sy, 0.0f, 0.0f,
       sx,  sy, 1.0f, 0.0f,
  };

  glBindBuffer(GL_ARRAY_BUFFER, vq->vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(quad), quad);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void VideoQuad_upload(VideoQuad *vq, uint8_t *rgb, int w, int h) {
  glBindTexture(GL_TEXTURE_2D, vq->texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
  glBindTexture(GL_TEXTURE_2D, 0);
}

static void VideoQuad_draw(VideoQuad *vq) {
  glUseProgram(vq->program);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, vq->texture);
  glBindVertexArray(vq->vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
}

static void VideoQuad_free(VideoQuad *vq) {
  glDeleteTextures(1, &vq->texture);
  glDeleteBuffers(1, &vq->vbo);
  glDeleteVertexArrays(1, &vq->vao);
}

// ---- Progress bar ----
typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint program;
  GLint u_projection;
} ProgressBar;

static ProgressBar ProgressBar_create(GLuint program) {
  ProgressBar pb;
  pb.program = program;
  pb.u_projection = glGetUniformLocation(program, "u_projection");

  glGenVertexArrays(1, &pb.vao);
  glGenBuffers(1, &pb.vbo);

  glBindVertexArray(pb.vao);
  glBindBuffer(GL_ARRAY_BUFFER, pb.vbo);
  glBufferData(GL_ARRAY_BUFFER, 12 * 6 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

  // Layout: vec2 position, vec4 color = 6 floats
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
  return pb;
}

static void ProgressBar_draw(ProgressBar *pb, double progress) {
  if (progress < 0.0) progress = 0.0;
  if (progress > 1.0) progress = 1.0;

  float fill_x = -1.0f + 2.0f * (float)progress;
  float y0 = -1.0f, y1 = -0.95f;

  // Background: dark gray, alpha 0.6
  float bg_r = 0.15f, bg_g = 0.15f, bg_b = 0.15f, bg_a = 0.6f;
  // Fill: bright blue, alpha 0.9
  float fl_r = 0.2f, fl_g = 0.5f, fl_b = 1.0f, fl_a = 0.9f;

  // 2 quads = 4 triangles = 12 vertices
  float verts[12 * 6] = {
      // Background quad (2 triangles)
      -1.0f, y0, bg_r, bg_g, bg_b, bg_a,
       1.0f, y0, bg_r, bg_g, bg_b, bg_a,
      -1.0f, y1, bg_r, bg_g, bg_b, bg_a,
      -1.0f, y1, bg_r, bg_g, bg_b, bg_a,
       1.0f, y0, bg_r, bg_g, bg_b, bg_a,
       1.0f, y1, bg_r, bg_g, bg_b, bg_a,
      // Fill quad (2 triangles)
      -1.0f,  y0, fl_r, fl_g, fl_b, fl_a,
      fill_x, y0, fl_r, fl_g, fl_b, fl_a,
      -1.0f,  y1, fl_r, fl_g, fl_b, fl_a,
      -1.0f,  y1, fl_r, fl_g, fl_b, fl_a,
      fill_x, y0, fl_r, fl_g, fl_b, fl_a,
      fill_x, y1, fl_r, fl_g, fl_b, fl_a,
  };

  // Identity projection
  float identity[16] = {0};
  identity[0] = 1.0f;
  identity[5] = 1.0f;
  identity[10] = 1.0f;
  identity[15] = 1.0f;

  glUseProgram(pb->program);
  glUniformMatrix4fv(pb->u_projection, 1, GL_FALSE, identity);

  glBindBuffer(GL_ARRAY_BUFFER, pb->vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);

  glBindVertexArray(pb->vao);
  glDrawArrays(GL_TRIANGLES, 0, 12);
  glBindVertexArray(0);
}

static void ProgressBar_free(ProgressBar *pb) {
  glDeleteBuffers(1, &pb->vbo);
  glDeleteVertexArrays(1, &pb->vao);
}

// ---- Attention heatmap overlay ----
typedef struct {
  GLuint vao;
  GLuint vbo;
  GLuint texture;
  GLuint program;
  GLint u_opacity;
  bool has_data;
  size_t attn_size;
} AttentionOverlay;

static AttentionOverlay AttentionOverlay_create(GLuint program) {
  AttentionOverlay ao;
  ao.program = program;
  ao.u_opacity = glGetUniformLocation(program, "u_opacity");
  ao.has_data = false;
  ao.attn_size = 0;

  float quad[] = {
      -1.0f, -1.0f, 0.0f, 1.0f,
       1.0f, -1.0f, 1.0f, 1.0f,
      -1.0f,  1.0f, 0.0f, 0.0f,
       1.0f,  1.0f, 1.0f, 0.0f,
  };

  glGenVertexArrays(1, &ao.vao);
  glGenBuffers(1, &ao.vbo);
  glBindVertexArray(ao.vao);
  glBindBuffer(GL_ARRAY_BUFFER, ao.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);

  // Create single-channel float texture for attention map
  glGenTextures(1, &ao.texture);
  glBindTexture(GL_TEXTURE_2D, ao.texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);

  return ao;
}

static void AttentionOverlay_update_aspect(AttentionOverlay *ao, int video_w, int video_h,
                                           int viewport_w, int viewport_h) {
  if (video_w <= 0 || video_h <= 0 || viewport_w <= 0 || viewport_h <= 0) return;

  double video_aspect = (double)video_w / (double)video_h;
  double viewport_aspect = (double)viewport_w / (double)viewport_h;

  float sx = 1.0f, sy = 1.0f;
  if (video_aspect > viewport_aspect) {
    sy = (float)(viewport_aspect / video_aspect);
  } else {
    sx = (float)(video_aspect / viewport_aspect);
  }

  float quad[] = {
      -sx, -sy, 0.0f, 1.0f,
       sx, -sy, 1.0f, 1.0f,
      -sx,  sy, 0.0f, 0.0f,
       sx,  sy, 1.0f, 0.0f,
  };

  glBindBuffer(GL_ARRAY_BUFFER, ao->vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(quad), quad);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void AttentionOverlay_upload(AttentionOverlay *ao, float *raw_map, size_t size) {
  if (!raw_map || size == 0) return;

  size_t n = size * size;

  // Min-max normalize in-place to [0,1]
  // (safe: buffer is overwritten on next IngestReader_next call)
  float min_val = raw_map[0], max_val = raw_map[0];
  for (size_t i = 1; i < n; i++) {
    if (raw_map[i] < min_val) min_val = raw_map[i];
    if (raw_map[i] > max_val) max_val = raw_map[i];
  }

  float range = max_val - min_val;
  if (range > 1e-8f) {
    for (size_t i = 0; i < n; i++)
      raw_map[i] = (raw_map[i] - min_val) / range;
  } else {
    for (size_t i = 0; i < n; i++)
      raw_map[i] = 0.0f;
  }

  glBindTexture(GL_TEXTURE_2D, ao->texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, (GLsizei)size, (GLsizei)size, 0,
               GL_RED, GL_FLOAT, raw_map);
  glBindTexture(GL_TEXTURE_2D, 0);

  ao->has_data = true;
  ao->attn_size = size;
}

static void AttentionOverlay_draw(AttentionOverlay *ao) {
  if (!ao->has_data) return;

  glUseProgram(ao->program);
  glUniform1f(ao->u_opacity, 0.5f);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, ao->texture);
  glBindVertexArray(ao->vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
}

static void AttentionOverlay_free(AttentionOverlay *ao) {
  glDeleteTextures(1, &ao->texture);
  glDeleteBuffers(1, &ao->vbo);
  glDeleteVertexArrays(1, &ao->vao);
}

// ---- Scroll callback (scrub + zoom) ----
static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  int win_w, win_h;
  glfwGetWindowSize(window, &win_w, &win_h);

  double cursor_x, cursor_y;
  glfwGetCursorPos(window, &cursor_x, &cursor_y);

  double half_w = (double)win_w / 2.0;

  if (cursor_x < half_w) {
    // Cursor over video: horizontal scroll to scrub
    if (!g_dec || g_dec->duration <= 0.0) return;

    double seek_delta = xoffset * 2.0;  // seconds per scroll unit
    double target = g_dec->current_pts + seek_delta;
    if (target < 0.0) target = 0.0;
    if (target > g_dec->duration) target = g_dec->duration;

    if (VideoDecoder_seek(g_dec, target)) {
      VideoQuad_upload(g_vq_ptr, g_dec->rgb_buffer, g_dec->width, g_dec->height);

      // Reset timing
      if (g_video_start_time) *g_video_start_time = glfwGetTime();
      if (g_video_pts_offset) *g_video_pts_offset = g_dec->current_pts;
      if (g_video_done) *g_video_done = false;

      // On backward seek: reset ingest reader and spatial memory
      if (seek_delta < 0.0 && g_reader && g_sm) {
        g_reader->cursor = 0;
        SpatialMemory_free(g_sm);
        g_sm = SpatialMemory_new(g_h3_resolution, DEFAULT_CAPACITY, DEFAULT_PRECISION);
        if (g_first_ts) *g_first_ts = -1.0;
        if (g_last_adv) *g_last_adv = -1.0;
        if (g_gps_trace) GpsTrace_clear(g_gps_trace);
      }
    }
  } else {
    // Cursor over map: vertical scroll to zoom
    if (!g_hex_renderer) return;
    if (yoffset > 0) {
      g_hex_renderer->zoom *= 0.9;  // scroll up = zoom in
    } else if (yoffset < 0) {
      g_hex_renderer->zoom *= 1.1;  // scroll down = zoom out
    }
  }
}

// ---- Keyboard callback ----
static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                          int mods) {
  (void)scancode;
  (void)mods;
  if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

  switch (key) {
  case GLFW_KEY_ESCAPE:
  case GLFW_KEY_Q:
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    break;
  case GLFW_KEY_SPACE:
    paused = !paused;
    break;
  case GLFW_KEY_EQUAL:  // + key
    if (g_hex_renderer) g_hex_renderer->zoom *= 0.8;
    break;
  case GLFW_KEY_MINUS:
    if (g_hex_renderer) g_hex_renderer->zoom *= 1.25;
    break;
  case GLFW_KEY_RIGHT:
    playback_speed *= 2.0;
    if (playback_speed > 16.0) playback_speed = 16.0;
    printf("Speed: %.1fx\n", playback_speed);
    break;
  case GLFW_KEY_LEFT:
    playback_speed *= 0.5;
    if (playback_speed < 0.25) playback_speed = 0.25;
    printf("Speed: %.1fx\n", playback_speed);
    break;
  default:
    break;
  }
}

// ---- Directory scanning helper ----
// Returns malloc'd full path, or NULL. Caller must free().
// If 'exact_name' is non-NULL, matches that filename exactly.
// Otherwise matches the first file ending with 'extension'.
static char *find_file_in_dir(const char *dir, const char *extension,
                              const char *exact_name) {
  DIR *d = opendir(dir);
  if (!d) return NULL;

  struct dirent *ent;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.') continue;

    bool match = false;
    if (exact_name) {
      match = (strcmp(ent->d_name, exact_name) == 0);
    } else if (extension) {
      size_t nlen = strlen(ent->d_name);
      size_t elen = strlen(extension);
      if (nlen > elen && strcmp(ent->d_name + nlen - elen, extension) == 0)
        match = true;
    }

    if (match) {
      size_t dlen = strlen(dir);
      bool needs_slash = (dlen > 0 && dir[dlen - 1] != '/');
      size_t path_len = dlen + needs_slash + strlen(ent->d_name) + 1;
      char *path = malloc(path_len);
      snprintf(path, path_len, "%s%s%s", dir, needs_slash ? "/" : "",
               ent->d_name);
      closedir(d);
      return path;
    }
  }
  closedir(d);
  return NULL;
}

// ---- Usage ----
static void print_usage(const char *prog) {
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s -d <dir> [-g group] [-t sec] [-r res]\n", prog);
  fprintf(stderr, "  %s -v <video> [-f <features.h5>] [-g group] [-t sec] [-r res]\n", prog);
  fprintf(stderr, "  %s <video.mp4> [data.h5 group] [time_window_sec] [h3_resolution]\n", prog);
  fprintf(stderr, "\nFlags:\n");
  fprintf(stderr, "  -d <dir>    Directory containing *.mp4 and features.h5\n");
  fprintf(stderr, "  -v <path>   Video file path\n");
  fprintf(stderr, "  -f <path>   HDF5 features file path\n");
  fprintf(stderr, "  -g <name>   HDF5 group name (default: dino)\n");
  fprintf(stderr, "  -t <sec>    Time window in seconds (default: 5.0)\n");
  fprintf(stderr, "  -r <res>    H3 resolution 0-15 (default: 10)\n");
  fprintf(stderr, "  -h          Print this help\n");
  fprintf(stderr, "\nControls:\n");
  fprintf(stderr, "  Space       Pause/resume\n");
  fprintf(stderr, "  +/-         Zoom in/out (heatmap)\n");
  fprintf(stderr, "  Left/Right  Slow down/speed up\n");
  fprintf(stderr, "  Scroll H    Scrub video (on video pane)\n");
  fprintf(stderr, "  Scroll V    Zoom map (on map pane)\n");
  fprintf(stderr, "  Q/Esc       Quit\n");
}

// ---- Main ----
int main(int argc, char *argv[]) {
  const char *dir_path = NULL;
  const char *video_path = NULL;
  const char *h5_path = NULL;
  const char *group = DINO;
  double time_window_sec = 5.0;
  int h3_resolution = DEFAULT_RESOLUTION;
  char *alloc_video = NULL;  // track malloc'd paths for cleanup
  char *alloc_h5 = NULL;

  int opt;
  bool used_flags = false;
  while ((opt = getopt(argc, argv, "d:v:f:g:t:r:h")) != -1) {
    used_flags = true;
    switch (opt) {
    case 'd': dir_path = optarg; break;
    case 'v': video_path = optarg; break;
    case 'f': h5_path = optarg; break;
    case 'g': group = optarg; break;
    case 't': time_window_sec = atof(optarg); break;
    case 'r': h3_resolution = atoi(optarg); break;
    case 'h':
      print_usage(argv[0]);
      return 0;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  // Directory mode: scan for video and features.h5
  if (dir_path) {
    if (!video_path) {
      alloc_video = find_file_in_dir(dir_path, ".mp4", NULL);
      if (!alloc_video) {
        fprintf(stderr, "No *.mp4 found in %s\n", dir_path);
        return 1;
      }
      video_path = alloc_video;
    }
    if (!h5_path) {
      alloc_h5 = find_file_in_dir(dir_path, NULL, "features.h5");
      if (alloc_h5) h5_path = alloc_h5;
      // features.h5 is optional — no error if missing
    }
  }

  // Legacy positional fallback: no flags were used and positional args remain
  if (!used_flags && optind < argc) {
    video_path = argv[optind];
    if (optind + 1 < argc) h5_path = argv[optind + 1];
    if (optind + 2 < argc) group = argv[optind + 2];
    if (optind + 3 < argc) time_window_sec = atof(argv[optind + 3]);
    if (optind + 4 < argc) h3_resolution = atoi(argv[optind + 4]);
  }

  // Validate required args
  if (!video_path) {
    print_usage(argv[0]);
    free(alloc_video);
    free(alloc_h5);
    return 1;
  }

  if (h3_resolution < 0 || h3_resolution > 15) {
    fprintf(stderr, "H3 resolution must be 0-15, got %d\n", h3_resolution);
    free(alloc_video);
    free(alloc_h5);
    return 1;
  }
  g_time_window_sec = time_window_sec;
  g_h3_resolution = h3_resolution;

  // ---- Init GLFW ----
  if (!glfwInit()) {
    fprintf(stderr, "Failed to init GLFW\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  GLFWwindow *window = glfwCreateWindow(1600, 600, "psm-viz", NULL, NULL);
  if (!window) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // vsync
  glfwSetKeyCallback(window, key_callback);
  glfwSetScrollCallback(window, scroll_callback);

  printf("OpenGL %s\n", glGetString(GL_VERSION));

  // ---- Load shaders ----
  GLuint video_prog = Shader_load_program("shaders/video.vert", "shaders/video.frag");
  GLuint hex_prog = Shader_load_program("shaders/hex.vert", "shaders/hex.frag");
  GLuint tile_prog = Shader_load_program("shaders/tile.vert", "shaders/tile.frag");
  if (!video_prog || !hex_prog || !tile_prog) {
    fprintf(stderr, "Failed to load shaders\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }

  // Attention overlay shader (non-fatal if missing)
  GLuint attn_prog = Shader_load_program("shaders/video.vert", "shaders/attention.frag");
  bool has_attention_overlay = (attn_prog != 0);

  // ---- Open video ----
  VideoDecoder *dec = VideoDecoder_open(video_path);
  if (!dec) {
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }
  printf("Video: %dx%d, %.1fs\n", dec->width, dec->height, dec->duration);
  g_dec = dec;

  VideoQuad vq = VideoQuad_create(video_prog);
  g_vq_ptr = &vq;

  // Decode first frame for initial display
  if (VideoDecoder_next_frame(dec)) {
    VideoQuad_upload(&vq, dec->rgb_buffer, dec->width, dec->height);
  }

  // ---- Open HDF5 (optional) ----
  SpatialMemory *sm = NULL;
  IngestReader *reader = NULL;
  hid_t h5_file = -1;
  double last_adv = -1.0;
  double first_ts = -1.0;

  if (h5_path) {
    sm = SpatialMemory_new(g_h3_resolution, DEFAULT_CAPACITY, DEFAULT_PRECISION);
    h5_file = H5Fopen(h5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5_file < 0) {
      fprintf(stderr, "Failed to open HDF5: %s\n", h5_path);
    } else {
      reader = IngestReader_open(h5_file, group);
      printf("HDF5: %zu records, %zu-d embeddings, group='%s'\n", reader->n_records,
             reader->emb_dimension, group);
    }
  }

  // Set up global pointers for scroll callback
  g_sm = sm;
  g_reader = reader;
  g_first_ts = &first_ts;
  g_last_adv = &last_adv;

  // ---- Create hex renderer, tile map, and GPS trace ----
  HexRenderer *hr = HexRenderer_new(hex_prog);
  g_hex_renderer = hr;

  TileMap *tm = TileMap_new(tile_prog);
  g_tile_map = tm;

  GpsTrace *gt = GpsTrace_new(hex_prog);
  g_gps_trace = gt;

  // ---- Create progress bar ----
  ProgressBar pb = ProgressBar_create(hex_prog);  // reuses hex shader (vec2 pos + vec4 color)

  // ---- Create attention overlay (if shader loaded) ----
  AttentionOverlay ao;
  if (has_attention_overlay) {
    ao = AttentionOverlay_create(attn_prog);
  }

  // ---- Render loop ----
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  double video_start_time = glfwGetTime();
  double video_pts_offset = dec->current_pts;
  bool video_done = false;

  g_video_start_time = &video_start_time;
  g_video_pts_offset = &video_pts_offset;
  g_video_done = &video_done;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    int win_w, win_h;
    glfwGetFramebufferSize(window, &win_w, &win_h);
    int half_w = win_w / 2;

    // Re-read sm in case scroll callback replaced it
    sm = g_sm;

    // ---- Decode video frames ----
    if (!paused && !video_done) {
      double wall_elapsed = (glfwGetTime() - video_start_time) * playback_speed;
      double target_pts = video_pts_offset + wall_elapsed;

      while (dec->current_pts < target_pts) {
        if (!VideoDecoder_next_frame(dec)) {
          video_done = true;
          break;
        }
        VideoQuad_upload(&vq, dec->rgb_buffer, dec->width, dec->height);
      }

      // ---- Drain ingest records up to video PTS ----
      if (reader && sm) {
        IngestRecord record;
        bool data_changed = false;
        while (reader->cursor < reader->n_records) {
          size_t saved_cursor = reader->cursor;
          if (!IngestReader_next(reader, &record)) break;

          if (first_ts < 0.0) first_ts = record.timestamp;

          double record_video_time = record.timestamp - first_ts;
          if (record_video_time > dec->current_pts) {
            reader->cursor = saved_cursor;
            break;
          }

          if (last_adv < 0.0) {
            last_adv = record.timestamp;
          }
          if (record.timestamp - last_adv >= time_window_sec) {
            SpatialMemory_advance_all(sm);
            last_adv = record.timestamp;
          }

          SpatialMemory_observe(sm, record.lat, record.lng, record.embedding,
                                record.embedding_dim * sizeof(float));
          GpsTrace_push(gt, record.lat, record.lng);

          if (has_attention_overlay && record.attention_map) {
            AttentionOverlay_upload(&ao, record.attention_map, record.attn_size);
          }

          data_changed = true;
        }

        if (data_changed) {
          HexRenderer_update(hr, sm);
        }
      }
    } else if (paused) {
      video_start_time = glfwGetTime();
      video_pts_offset = dec->current_pts;
    }

    // ---- Draw ----
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Left half: video
    glViewport(0, 0, half_w, win_h);
    glDisable(GL_BLEND);
    VideoQuad_update_aspect(&vq, dec->width, dec->height, half_w, win_h);
    VideoQuad_draw(&vq);

    // Attention heatmap overlay (semi-transparent over video)
    glEnable(GL_BLEND);
    if (has_attention_overlay) {
      AttentionOverlay_update_aspect(&ao, dec->width, dec->height, half_w, win_h);
      AttentionOverlay_draw(&ao);
    }

    // Progress bar overlay on video
    double progress = (dec->duration > 0.0) ? dec->current_pts / dec->duration : 0.0;
    ProgressBar_draw(&pb, progress);

    // Right half: OSM tiles (background) + hex heatmap + GPS trace (overlay)
    glViewport(half_w, 0, win_w - half_w, win_h);

    // Use GPS trace center when hex renderer has no data yet
    double map_center_lat = hr->center_lat;
    double map_center_lng = hr->center_lng;
    if (hr->vertex_count == 0 && gt->count > 0) {
      map_center_lat = gt->center_lat;
      map_center_lng = gt->center_lng;
    }

    glDisable(GL_BLEND);
    TileMap_draw(tm, map_center_lat, map_center_lng, hr->zoom,
                 win_w - half_w, win_h);
    glEnable(GL_BLEND);
    GpsTrace_upload(gt, map_center_lat, map_center_lng);
    HexRenderer_draw(hr, win_w - half_w, win_h);
    GpsTrace_draw(gt, win_w - half_w, win_h, hr->zoom);

    glfwSwapBuffers(window);
  }

  // ---- Cleanup ----
  ProgressBar_free(&pb);
  if (has_attention_overlay) {
    AttentionOverlay_free(&ao);
    glDeleteProgram(attn_prog);
  }
  GpsTrace_free(gt);
  TileMap_free(tm);
  HexRenderer_free(hr);
  VideoQuad_free(&vq);
  VideoDecoder_close(dec);
  glDeleteProgram(video_prog);
  glDeleteProgram(hex_prog);
  glDeleteProgram(tile_prog);

  if (reader) IngestReader_close(reader);
  if (h5_file >= 0) H5Fclose(h5_file);
  if (sm) SpatialMemory_free(sm);

  glfwDestroyWindow(window);
  glfwTerminate();

  free(alloc_video);
  free(alloc_h5);

  return 0;
}
