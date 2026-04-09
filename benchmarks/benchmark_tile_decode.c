#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>
#include "stb/stb_image.h"

#define SAMPLE_TILE_WIDTH 256
#define SAMPLE_TILE_HEIGHT 256
#define DEFAULT_TOTAL_DECODES 4000
#define DEFAULT_THREAD_COUNT 1
#define PNG_SIGNATURE_SIZE 8

typedef struct {
  uint8_t *data;
  size_t size;
  size_t capacity;
} ByteBuffer;

typedef struct {
  bool use_disk_path;
  const uint8_t *png_data;
  size_t png_size;
  const char *png_path;
  size_t iterations;
  int expected_width;
  int expected_height;
  size_t failures;
} DecodeWorkerArgs;

typedef enum {
  DECODE_SOURCE_MEMORY = 0,
  DECODE_SOURCE_DISK = 1,
} DecodeSourceMode;

static double monotonic_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool parse_size_arg(const char *text, const char *name, size_t *out) {
  char *end = NULL;
  unsigned long long value;

  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0ULL) {
    fprintf(stderr, "Invalid %s: '%s'\n", name, text);
    return false;
  }
  *out = (size_t)value;
  return true;
}

static bool parse_source_mode_arg(const char *text, DecodeSourceMode *out_mode) {
  if (!text || !out_mode) return false;
  if (strcmp(text, "memory") == 0) {
    *out_mode = DECODE_SOURCE_MEMORY;
    return true;
  }
  if (strcmp(text, "disk") == 0) {
    *out_mode = DECODE_SOURCE_DISK;
    return true;
  }
  fprintf(stderr, "Invalid source mode: '%s' (expected memory or disk)\n", text);
  return false;
}

static bool buffer_reserve(ByteBuffer *buf, size_t extra) {
  size_t needed;
  size_t new_capacity;
  uint8_t *new_data;

  if (!buf) return false;
  if (buf->size > SIZE_MAX - extra) return false;
  needed = buf->size + extra;
  if (needed <= buf->capacity) return true;

  new_capacity = buf->capacity ? buf->capacity : 256;
  while (new_capacity < needed) {
    if (new_capacity > SIZE_MAX / 2) {
      new_capacity = needed;
      break;
    }
    new_capacity *= 2;
  }

  new_data = realloc(buf->data, new_capacity);
  if (!new_data) return false;
  buf->data = new_data;
  buf->capacity = new_capacity;
  return true;
}

static bool buffer_append(ByteBuffer *buf, const void *data, size_t size) {
  if (!buf || !data) return false;
  if (!buffer_reserve(buf, size)) return false;
  memcpy(buf->data + buf->size, data, size);
  buf->size += size;
  return true;
}

static bool buffer_append_u32_be(ByteBuffer *buf, uint32_t value) {
  uint8_t bytes[4];

  bytes[0] = (uint8_t)((value >> 24) & 0xffu);
  bytes[1] = (uint8_t)((value >> 16) & 0xffu);
  bytes[2] = (uint8_t)((value >> 8) & 0xffu);
  bytes[3] = (uint8_t)(value & 0xffu);
  return buffer_append(buf, bytes, sizeof(bytes));
}

static bool append_png_chunk(ByteBuffer *png, const char tag[4],
                             const uint8_t *data, size_t size) {
  uLong crc;

  if (!png || !tag || (!data && size != 0)) return false;
  if (size > UINT_MAX) return false;

  crc = crc32(0L, Z_NULL, 0);
  crc = crc32(crc, (const Bytef *)tag, 4);
  if (size > 0) {
    crc = crc32(crc, (const Bytef *)data, (uInt)size);
  }

  return buffer_append_u32_be(png, (uint32_t)size) &&
         buffer_append(png, tag, 4) &&
         (size == 0 || buffer_append(png, data, size)) &&
         buffer_append_u32_be(png, (uint32_t)crc);
}

static bool build_sample_tile_png(ByteBuffer *png, int width, int height) {
  static const uint8_t k_png_signature[PNG_SIGNATURE_SIZE] = {
      0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n',
  };
  const size_t row_stride = 1u + (size_t)width * 4u;
  const size_t raw_size = row_stride * (size_t)height;
  uint8_t ihdr[13];
  uint8_t *raw = NULL;
  uint8_t *compressed = NULL;
  uLongf compressed_size;
  bool ok = false;

  if (!png || width <= 0 || height <= 0) return false;

  raw = malloc(raw_size);
  if (!raw) goto done;

  for (int y = 0; y < height; y++) {
    size_t row_offset = (size_t)y * row_stride;
    raw[row_offset] = 0;
    for (int x = 0; x < width; x++) {
      size_t pixel_offset = row_offset + 1u + (size_t)x * 4u;
      raw[pixel_offset + 0] = (uint8_t)(((unsigned)x * 5u +
                                         (unsigned)y * 3u) & 0xffu);
      raw[pixel_offset + 1] = (uint8_t)(((unsigned)x * 7u +
                                         (unsigned)y * 11u) & 0xffu);
      raw[pixel_offset + 2] =
          (uint8_t)(((((unsigned)x >> 4) ^ ((unsigned)y >> 4)) & 1u) ? 255u
                                                                      : 32u);
      raw[pixel_offset + 3] = 255u;
    }
  }

  compressed_size = compressBound((uLong)raw_size);
  compressed = malloc((size_t)compressed_size);
  if (!compressed) goto done;
  if (compress2((Bytef *)compressed, &compressed_size, (const Bytef *)raw,
                (uLong)raw_size, Z_BEST_SPEED) != Z_OK) {
    goto done;
  }

  ihdr[0] = (uint8_t)((width >> 24) & 0xff);
  ihdr[1] = (uint8_t)((width >> 16) & 0xff);
  ihdr[2] = (uint8_t)((width >> 8) & 0xff);
  ihdr[3] = (uint8_t)(width & 0xff);
  ihdr[4] = (uint8_t)((height >> 24) & 0xff);
  ihdr[5] = (uint8_t)((height >> 16) & 0xff);
  ihdr[6] = (uint8_t)((height >> 8) & 0xff);
  ihdr[7] = (uint8_t)(height & 0xff);
  ihdr[8] = 8;
  ihdr[9] = 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  ok = buffer_append(png, k_png_signature, sizeof(k_png_signature)) &&
       append_png_chunk(png, "IHDR", ihdr, sizeof(ihdr)) &&
       append_png_chunk(png, "IDAT", compressed, (size_t)compressed_size) &&
       append_png_chunk(png, "IEND", NULL, 0);

done:
  free(raw);
  free(compressed);
  return ok;
}

static bool decode_png_once(const uint8_t *png_data, size_t png_size,
                            int expected_width, int expected_height) {
  int width = 0;
  int height = 0;
  int channels = 0;
  uint8_t *decoded = NULL;

  if (!png_data || png_size == 0 || png_size > (size_t)INT_MAX) return false;

  decoded = stbi_load_from_memory(png_data, (int)png_size,
                                  &width, &height, &channels, 4);
  if (!decoded) return false;
  stbi_image_free(decoded);
  return width == expected_width && height == expected_height && channels >= 3;
}

static bool read_file_bytes(const char *path, ByteBuffer *buf) {
  struct stat st;
  FILE *file = NULL;
  size_t bytes_read;

  if (!path || !buf) return false;
  memset(buf, 0, sizeof(*buf));

  if (stat(path, &st) != 0 || st.st_size <= 0) return false;
  if ((uintmax_t)st.st_size > (uintmax_t)SIZE_MAX) return false;

  buf->capacity = (size_t)st.st_size;
  buf->size = buf->capacity;
  buf->data = malloc(buf->capacity);
  if (!buf->data) return false;

  file = fopen(path, "rb");
  if (!file) {
    free(buf->data);
    memset(buf, 0, sizeof(*buf));
    return false;
  }

  bytes_read = fread(buf->data, 1, buf->size, file);
  fclose(file);
  if (bytes_read != buf->size) {
    free(buf->data);
    memset(buf, 0, sizeof(*buf));
    return false;
  }
  return true;
}

static bool write_file_bytes(const char *path, const uint8_t *data, size_t size) {
  FILE *file;

  if (!path || !data || size == 0) return false;
  file = fopen(path, "wb");
  if (!file) return false;
  if (fwrite(data, 1, size, file) != size) {
    fclose(file);
    return false;
  }
  return fclose(file) == 0;
}

static bool decode_png_from_disk_once(const char *png_path,
                                      int expected_width,
                                      int expected_height) {
  ByteBuffer file_bytes = {0};
  bool ok;

  if (!read_file_bytes(png_path, &file_bytes)) return false;
  ok = decode_png_once(file_bytes.data, file_bytes.size,
                       expected_width, expected_height);
  free(file_bytes.data);
  return ok;
}

static bool create_temp_png_path(char *out_path, size_t out_size) {
  char template_path[] = "/tmp/psm_tile_decode_XXXXXX";
  int fd;

  if (!out_path || out_size == 0) return false;
  fd = mkstemp(template_path);
  if (fd < 0) return false;
  close(fd);
  unlink(template_path);
  if (snprintf(out_path, out_size, "%s.png", template_path) >= (int)out_size) {
    return false;
  }
  return true;
}

static void *decode_worker(void *userdata) {
  DecodeWorkerArgs *args = (DecodeWorkerArgs *)userdata;

  if (!args) return NULL;
  for (size_t i = 0; i < args->iterations; i++) {
    bool ok;

    if (args->use_disk_path) {
      ok = decode_png_from_disk_once(args->png_path,
                                     args->expected_width,
                                     args->expected_height);
    } else {
      ok = decode_png_once(args->png_data, args->png_size,
                           args->expected_width, args->expected_height);
    }
    if (!ok) {
      args->failures++;
      break;
    }
  }
  return NULL;
}

static void fail_benchmark(const char *message) {
  fprintf(stderr, "%s\n", message);
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  ByteBuffer png = {0};
  size_t total_decodes = DEFAULT_TOTAL_DECODES;
  size_t thread_count = DEFAULT_THREAD_COUNT;
  DecodeSourceMode source_mode = DECODE_SOURCE_MEMORY;
  pthread_t *threads = NULL;
  DecodeWorkerArgs *workers = NULL;
  size_t total_failures = 0;
  char temp_png_path[PATH_MAX] = {0};
  bool wrote_temp_png = false;
  double start;
  double elapsed;

  if (argc > 1 && !parse_size_arg(argv[1], "total_decodes", &total_decodes)) {
    return 1;
  }
  if (argc > 2 && !parse_size_arg(argv[2], "thread_count", &thread_count)) {
    return 1;
  }
  if (argc > 3 && !parse_source_mode_arg(argv[3], &source_mode)) {
    return 1;
  }
  if (thread_count == 0 || thread_count > (size_t)INT_MAX) {
    fprintf(stderr, "thread_count must be in [1, %d]\n", INT_MAX);
    return 1;
  }

  if (!build_sample_tile_png(&png, SAMPLE_TILE_WIDTH, SAMPLE_TILE_HEIGHT)) {
    free(png.data);
    fail_benchmark("Failed to build sample tile PNG");
  }
  if (!decode_png_once(png.data, png.size, SAMPLE_TILE_WIDTH, SAMPLE_TILE_HEIGHT)) {
    free(png.data);
    fail_benchmark("Sample tile PNG failed to decode");
  }
  if (source_mode == DECODE_SOURCE_DISK) {
    if (!create_temp_png_path(temp_png_path, sizeof(temp_png_path)) ||
        !write_file_bytes(temp_png_path, png.data, png.size) ||
        !decode_png_from_disk_once(temp_png_path, SAMPLE_TILE_WIDTH,
                                   SAMPLE_TILE_HEIGHT)) {
      free(png.data);
      unlink(temp_png_path);
      fail_benchmark("Sample tile PNG failed to round-trip through disk");
    }
    wrote_temp_png = true;
  }

  threads = calloc(thread_count, sizeof(*threads));
  workers = calloc(thread_count, sizeof(*workers));
  if (!threads || !workers) {
    free(threads);
    free(workers);
    free(png.data);
    fail_benchmark("Failed to allocate worker state");
  }

  for (size_t i = 0; i < thread_count; i++) {
    workers[i].use_disk_path = (source_mode == DECODE_SOURCE_DISK);
    workers[i].png_data = png.data;
    workers[i].png_size = png.size;
    workers[i].png_path = temp_png_path;
    workers[i].expected_width = SAMPLE_TILE_WIDTH;
    workers[i].expected_height = SAMPLE_TILE_HEIGHT;
    workers[i].iterations = total_decodes / thread_count;
    if (i < total_decodes % thread_count) {
      workers[i].iterations++;
    }
  }

  start = monotonic_seconds();
  for (size_t i = 0; i < thread_count; i++) {
    if (pthread_create(&threads[i], NULL, decode_worker, &workers[i]) != 0) {
      free(threads);
      free(workers);
      free(png.data);
      fail_benchmark("Failed to create decode worker");
    }
  }
  for (size_t i = 0; i < thread_count; i++) {
    pthread_join(threads[i], NULL);
    total_failures += workers[i].failures;
  }
  elapsed = monotonic_seconds() - start;

  printf("Tile decode benchmark\n");
  printf("  sample_tile:     %dx%d PNG (%zu bytes compressed)\n",
         SAMPLE_TILE_WIDTH, SAMPLE_TILE_HEIGHT, png.size);
  printf("  source_mode:     %s\n",
         source_mode == DECODE_SOURCE_DISK ? "disk" : "memory");
  printf("  total_decodes:   %zu\n", total_decodes);
  printf("  thread_count:    %zu\n", thread_count);
  printf("  failures:        %zu\n", total_failures);
  printf("  elapsed_sec:     %.3f\n", elapsed);
  printf("  decodes_per_sec: %.0f\n",
         elapsed > 0.0 ? (double)total_decodes / elapsed : 0.0);
  printf("  megapixels/sec:  %.2f\n",
         elapsed > 0.0 ? ((double)total_decodes * SAMPLE_TILE_WIDTH *
                          SAMPLE_TILE_HEIGHT) / (elapsed * 1000000.0)
                       : 0.0);

  free(threads);
  free(workers);
  free(png.data);
  if (wrote_temp_png) {
    unlink(temp_png_path);
  }

  return total_failures == 0 ? 0 : 1;
}
