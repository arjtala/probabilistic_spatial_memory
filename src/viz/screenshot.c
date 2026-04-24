#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "viz/gl_platform.h"
#include "viz/screenshot.h"

#define PNG_SIGNATURE_SIZE 8
#define PNG_MAX_STORE_BLOCK 65535u
#define PNG_ZLIB_MODULUS 65521u

static void write_u16_le(uint8_t *dst, uint16_t value) {
  dst[0] = (uint8_t)(value & 0xffu);
  dst[1] = (uint8_t)((value >> 8) & 0xffu);
}

static void write_u32_be(uint8_t *dst, uint32_t value) {
  dst[0] = (uint8_t)((value >> 24) & 0xffu);
  dst[1] = (uint8_t)((value >> 16) & 0xffu);
  dst[2] = (uint8_t)((value >> 8) & 0xffu);
  dst[3] = (uint8_t)(value & 0xffu);
}

static bool write_all(FILE *file, const void *data, size_t size) {
  return file && data && fwrite(data, 1, size, file) == size;
}

static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t size) {
  static uint32_t table[256];
  static bool initialized = false;

  if (!initialized) {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t value = i;
      for (int bit = 0; bit < 8; bit++) {
        value = (value & 1u) ? (0xedb88320u ^ (value >> 1)) : (value >> 1);
      }
      table[i] = value;
    }
    initialized = true;
  }

  for (size_t i = 0; i < size; i++) {
    crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
  }
  return crc;
}

static uint32_t adler32_update(uint32_t adler, const uint8_t *data,
                               size_t size) {
  uint32_t sum1 = adler & 0xffffu;
  uint32_t sum2 = (adler >> 16) & 0xffffu;

  for (size_t i = 0; i < size; i++) {
    sum1 += data[i];
    if (sum1 >= PNG_ZLIB_MODULUS) sum1 -= PNG_ZLIB_MODULUS;
    sum2 += sum1;
    if (sum2 >= PNG_ZLIB_MODULUS) sum2 -= PNG_ZLIB_MODULUS;
  }

  return (sum2 << 16) | sum1;
}

static bool write_chunk(FILE *file, const char type[4], const uint8_t *data,
                        uint32_t size) {
  uint8_t size_be[4];
  uint8_t crc_be[4];
  uint32_t crc = 0xffffffffu;

  write_u32_be(size_be, size);
  if (!write_all(file, size_be, sizeof(size_be)) ||
      !write_all(file, type, 4)) {
    return false;
  }

  crc = crc32_update(crc, (const uint8_t *)type, 4);
  if (size > 0) {
    if (!data || !write_all(file, data, size)) return false;
    crc = crc32_update(crc, data, size);
  }

  crc ^= 0xffffffffu;
  write_u32_be(crc_be, crc);
  return write_all(file, crc_be, sizeof(crc_be));
}

static bool write_idat_chunk(FILE *file, int width, int height,
                             const uint8_t *rgba_pixels) {
  const char type[4] = {'I', 'D', 'A', 'T'};
  const uint8_t zlib_header[2] = {0x78u, 0x01u};
  uint8_t size_be[4];
  uint8_t crc_be[4];
  uint8_t adler_be[4];
  uint8_t *block = NULL;
  size_t row_bytes;
  size_t row_span;
  size_t raw_size;
  size_t block_count;
  size_t chunk_size;
  size_t remaining_raw;
  size_t row = 0;
  size_t row_pos = 0;
  uint32_t crc = 0xffffffffu;
  uint32_t adler = 1u;
  bool ok = false;

  if (!file || !rgba_pixels || width <= 0 || height <= 0) return false;
  if ((size_t)width > SIZE_MAX / 4u) return false;

  row_bytes = (size_t)width * 4u;
  row_span = row_bytes + 1u;
  if ((size_t)height > SIZE_MAX / row_span) return false;

  raw_size = row_span * (size_t)height;
  block_count = (raw_size + PNG_MAX_STORE_BLOCK - 1u) / PNG_MAX_STORE_BLOCK;
  if (block_count > (SIZE_MAX - raw_size - 6u) / 5u) return false;
  chunk_size = 2u + raw_size + block_count * 5u + 4u;
  if (chunk_size > 0xffffffffu) return false;

  block = malloc(PNG_MAX_STORE_BLOCK);
  if (!block) return false;

  write_u32_be(size_be, (uint32_t)chunk_size);
  if (!write_all(file, size_be, sizeof(size_be)) ||
      !write_all(file, type, 4) ||
      !write_all(file, zlib_header, sizeof(zlib_header))) {
    goto cleanup;
  }

  crc = crc32_update(crc, (const uint8_t *)type, 4);
  crc = crc32_update(crc, zlib_header, sizeof(zlib_header));

  remaining_raw = raw_size;
  while (remaining_raw > 0) {
    size_t block_len = remaining_raw > PNG_MAX_STORE_BLOCK
                           ? PNG_MAX_STORE_BLOCK
                           : remaining_raw;
    uint8_t block_header[5];
    uint16_t len16 = (uint16_t)block_len;
    uint16_t nlen16 = (uint16_t)~len16;
    size_t used = 0;

    while (used < block_len) {
      if (row_pos == 0) {
        block[used++] = 0u;
        row_pos = 1;
        continue;
      }

      size_t bytes_left = row_bytes - (row_pos - 1u);
      size_t copy_size = block_len - used;
      const uint8_t *src;

      if (copy_size > bytes_left) copy_size = bytes_left;
      src = rgba_pixels + row * row_bytes + (row_pos - 1u);
      memcpy(block + used, src, copy_size);
      used += copy_size;
      row_pos += copy_size;
      if (row_pos == row_bytes + 1u) {
        row++;
        row_pos = 0;
      }
    }

    block_header[0] = (remaining_raw == block_len) ? 0x01u : 0x00u;
    write_u16_le(&block_header[1], len16);
    write_u16_le(&block_header[3], nlen16);

    if (!write_all(file, block_header, sizeof(block_header)) ||
        !write_all(file, block, block_len)) {
      goto cleanup;
    }

    crc = crc32_update(crc, block_header, sizeof(block_header));
    crc = crc32_update(crc, block, block_len);
    adler = adler32_update(adler, block, block_len);
    remaining_raw -= block_len;
  }

  write_u32_be(adler_be, adler);
  if (!write_all(file, adler_be, sizeof(adler_be))) goto cleanup;
  crc = crc32_update(crc, adler_be, sizeof(adler_be));
  crc ^= 0xffffffffu;
  write_u32_be(crc_be, crc);
  ok = write_all(file, crc_be, sizeof(crc_be));

cleanup:
  free(block);
  return ok;
}

static bool flip_rgba_rows(uint8_t *rgba_pixels, int width, int height) {
  uint8_t *scratch = NULL;
  size_t row_bytes;

  if (!rgba_pixels || width <= 0 || height <= 0) return false;
  if (height == 1) return true;
  if ((size_t)width > SIZE_MAX / 4u) return false;

  row_bytes = (size_t)width * 4u;
  scratch = malloc(row_bytes);
  if (!scratch) return false;

  for (int top = 0, bottom = height - 1; top < bottom; top++, bottom--) {
    uint8_t *top_row = rgba_pixels + (size_t)top * row_bytes;
    uint8_t *bottom_row = rgba_pixels + (size_t)bottom * row_bytes;
    memcpy(scratch, top_row, row_bytes);
    memcpy(top_row, bottom_row, row_bytes);
    memcpy(bottom_row, scratch, row_bytes);
  }

  free(scratch);
  return true;
}

bool VizScreenshot_ensure_directory(const char *path) {
  struct stat st;

  if (!path || path[0] == '\0') return false;
  if (stat(path, &st) == 0) {
    return S_ISDIR(st.st_mode);
  }
  if (mkdir(path, 0755) == 0) return true;
  return errno == EEXIST;
}

bool VizScreenshot_init(VizScreenshotSession *session, const char *directory,
                        const char *prefix, unsigned long starting_index) {
  if (!session || !directory || directory[0] == '\0' || !prefix ||
      prefix[0] == '\0') {
    return false;
  }
  if (!VizScreenshot_ensure_directory(directory)) return false;
  if (snprintf(session->output_dir, sizeof(session->output_dir), "%s",
               directory) >= (int)sizeof(session->output_dir)) {
    return false;
  }
  if (snprintf(session->prefix, sizeof(session->prefix), "%s", prefix) >=
      (int)sizeof(session->prefix)) {
    return false;
  }
  session->next_index = starting_index;
  return true;
}

bool VizScreenshot_write_png_rgba(const char *path, int width, int height,
                                  const uint8_t *rgba_pixels) {
  static const uint8_t png_signature[PNG_SIGNATURE_SIZE] = {
      0x89u, 'P', 'N', 'G', '\r', '\n', 0x1au, '\n'};
  static const char iend_type[4] = {'I', 'E', 'N', 'D'};
  FILE *file;
  uint8_t ihdr[13];
  bool ok = false;

  if (!path || !rgba_pixels || width <= 0 || height <= 0) return false;

  file = fopen(path, "wb");
  if (!file) return false;

  write_u32_be(&ihdr[0], (uint32_t)width);
  write_u32_be(&ihdr[4], (uint32_t)height);
  ihdr[8] = 8u;   // bit depth
  ihdr[9] = 6u;   // color type: RGBA
  ihdr[10] = 0u;  // compression
  ihdr[11] = 0u;  // filter
  ihdr[12] = 0u;  // interlace

  ok = write_all(file, png_signature, sizeof(png_signature)) &&
       write_chunk(file, "IHDR", ihdr, sizeof(ihdr)) &&
       write_idat_chunk(file, width, height, rgba_pixels) &&
       write_chunk(file, iend_type, NULL, 0);
  if (fclose(file) != 0) ok = false;
  return ok;
}

bool VizScreenshot_capture_region(VizScreenshotSession *session, int x, int y,
                                  int width, int height, char *out_path,
                                  size_t out_path_size) {
  uint8_t *rgba_pixels;
  bool wrote_ok = false;
  size_t pixel_count;

  if (!session || !out_path || out_path_size == 0 || width <= 0 || height <= 0) {
    return false;
  }
  if ((size_t)width > SIZE_MAX / 4u || (size_t)height > SIZE_MAX / (size_t)width ||
      (size_t)width * (size_t)height > SIZE_MAX / 4u) {
    return false;
  }
  if (snprintf(out_path, out_path_size, "%s/%s-%06lu.png",
               session->output_dir, session->prefix, session->next_index) >=
      (int)out_path_size) {
    return false;
  }

  pixel_count = (size_t)width * (size_t)height * 4u;
  rgba_pixels = malloc(pixel_count);
  if (!rgba_pixels) return false;

  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_BACK);
  glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba_pixels);
  if (!flip_rgba_rows(rgba_pixels, width, height)) {
    free(rgba_pixels);
    return false;
  }

  wrote_ok = VizScreenshot_write_png_rgba(out_path, width, height, rgba_pixels);
  free(rgba_pixels);

  if (wrote_ok) {
    session->next_index++;
  }
  return wrote_ok;
}
