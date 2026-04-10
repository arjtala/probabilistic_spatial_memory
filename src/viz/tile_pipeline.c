#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stb/stb_image.h"
#include "viz/tile_pipeline.h"

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
  size_t total = size * nmemb;
  MemBuffer *buf = (MemBuffer *)userdata;
  uint8_t *tmp = realloc(buf->data, buf->size + total);

  if (!tmp) return 0;
  buf->data = tmp;
  memcpy(buf->data + buf->size, ptr, total);
  buf->size += total;
  return total;
}

static bool queue_tile_from_disk(TileMap *tm, PendingDownload *pd,
                                 int x, int y, int z) {
  MemBuffer disk_buf = {0};

  if (!tm || !pd) return false;
  if (!TileDiskCache_read(&tm->disk_cache, x, y, z,
                          &disk_buf.data, &disk_buf.size)) {
    return false;
  }

  pthread_mutex_lock(&tm->pending_mutex);
  if (!pd->active || pd->easy || pd->ready || pd->decoding || pd->decoded ||
      pd->x != x || pd->y != y || pd->z != z) {
    pthread_mutex_unlock(&tm->pending_mutex);
    free(disk_buf.data);
    return false;
  }
  pd->buf = disk_buf;
  pd->active = false;
  pd->ready = true;
  pthread_cond_signal(&tm->pending_cond);
  pthread_mutex_unlock(&tm->pending_mutex);
  return true;
}

static bool slot_is_busy(const PendingDownload *pd) {
  return pd && (pd->active || pd->ready || pd->decoding || pd->decoded);
}

static bool is_pending_locked(const TileMap *tm, int x, int y, int z) {
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    const PendingDownload *pd = &tm->pending[i];
    if (slot_is_busy(pd) && pd->x == x && pd->y == y && pd->z == z) {
      return true;
    }
  }
  return false;
}

static PendingDownload *find_free_slot_locked(TileMap *tm) {
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    if (!slot_is_busy(&tm->pending[i])) return &tm->pending[i];
  }
  return NULL;
}

static void clear_pending_download(PendingDownload *pd) {
  if (!pd) return;
  free(pd->buf.data);
  free(pd->decoded_pixels);
  pd->buf.data = NULL;
  pd->buf.size = 0;
  pd->decoded_pixels = NULL;
  pd->decoded_w = 0;
  pd->decoded_h = 0;
  pd->easy = NULL;
  pd->x = 0;
  pd->y = 0;
  pd->z = 0;
  pd->active = false;
  pd->ready = false;
  pd->decoding = false;
  pd->decoded = false;
}

static bool has_decode_work(const TileMap *tm) {
  if (!tm) return false;
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    const PendingDownload *pd = &tm->pending[i];
    if (pd->ready && !pd->decoding && !pd->decoded) {
      return true;
    }
  }
  return false;
}

static void *tile_decode_worker(void *userdata) {
  TileMap *tm = (TileMap *)userdata;

  for (;;) {
    PendingDownload *pd = NULL;
    uint8_t *compressed = NULL;
    size_t compressed_size = 0;
    int slot = -1;

    pthread_mutex_lock(&tm->pending_mutex);
    while (!tm->shutdown_worker && !has_decode_work(tm)) {
      pthread_cond_wait(&tm->pending_cond, &tm->pending_mutex);
    }
    if (tm->shutdown_worker) {
      pthread_mutex_unlock(&tm->pending_mutex);
      return NULL;
    }

    for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
      PendingDownload *candidate = &tm->pending[i];
      if (candidate->ready && !candidate->decoding && !candidate->decoded) {
        pd = candidate;
        slot = i;
        pd->decoding = true;
        pd->ready = false;
        compressed = pd->buf.data;
        compressed_size = pd->buf.size;
        pd->buf.data = NULL;
        pd->buf.size = 0;
        break;
      }
    }
    pthread_mutex_unlock(&tm->pending_mutex);

    if (!pd || !compressed || compressed_size == 0) {
      free(compressed);
      pthread_mutex_lock(&tm->pending_mutex);
      if (slot >= 0) {
        clear_pending_download(&tm->pending[slot]);
      }
      pthread_mutex_unlock(&tm->pending_mutex);
      continue;
    }

    {
      int w = 0;
      int h = 0;
      int channels = 0;
      uint8_t *decoded = stbi_load_from_memory(compressed, (int)compressed_size,
                                               &w, &h, &channels, 4);
      free(compressed);

      pthread_mutex_lock(&tm->pending_mutex);
      pd = &tm->pending[slot];
      pd->decoding = false;
      if (decoded) {
        pd->decoded_pixels = decoded;
        pd->decoded_w = w;
        pd->decoded_h = h;
        pd->decoded = true;
      } else {
        clear_pending_download(pd);
      }
      pthread_mutex_unlock(&tm->pending_mutex);
    }
  }
}

static bool append_text(char *dst, size_t dst_size, size_t *dst_len,
                        const char *text) {
  size_t text_len = strlen(text);

  if (*dst_len + text_len + 1 > dst_size) return false;
  memcpy(dst + *dst_len, text, text_len);
  *dst_len += text_len;
  dst[*dst_len] = '\0';
  return true;
}

static bool append_int(char *dst, size_t dst_size, size_t *dst_len, int value) {
  char buf[32];

  snprintf(buf, sizeof(buf), "%d", value);
  return append_text(dst, dst_size, dst_len, buf);
}

static bool format_tile_url(const TileMap *tm, int x, int y, int z,
                            char *out, size_t out_size) {
  size_t out_len = 0;
  const char *cursor;
  const char *subdomains = "abcd";
  char subdomain[2] = {subdomains[(x + y + z) & 3], '\0'};

  if (!tm || !out || out_size == 0 || tm->url_template[0] == '\0') return false;
  cursor = tm->url_template;
  out[0] = '\0';

  while (*cursor) {
    if (strncmp(cursor, "{x}", 3) == 0) {
      if (!append_int(out, out_size, &out_len, x)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{y}", 3) == 0) {
      if (!append_int(out, out_size, &out_len, y)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{z}", 3) == 0) {
      if (!append_int(out, out_size, &out_len, z)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{s}", 3) == 0) {
      if (!append_text(out, out_size, &out_len, subdomain)) return false;
      cursor += 3;
    } else if (strncmp(cursor, "{api_key}", 9) == 0) {
      if (!append_text(out, out_size, &out_len, tm->api_key)) return false;
      cursor += 9;
    } else {
      char ch[2] = {*cursor, '\0'};
      if (!append_text(out, out_size, &out_len, ch)) return false;
      cursor++;
    }
  }

  return true;
}

bool TilePipeline_init(TileMap *tm) {
  if (!tm) return false;

  if (pthread_mutex_init(&tm->pending_mutex, NULL) != 0) {
    return false;
  }
  if (pthread_cond_init(&tm->pending_cond, NULL) != 0) {
    pthread_mutex_destroy(&tm->pending_mutex);
    return false;
  }

  curl_global_init(CURL_GLOBAL_DEFAULT);
  tm->multi = curl_multi_init();
  if (!tm->multi) {
    pthread_mutex_destroy(&tm->pending_mutex);
    pthread_cond_destroy(&tm->pending_cond);
    curl_global_cleanup();
    return false;
  }

  if (pthread_create(&tm->decode_thread, NULL, tile_decode_worker, tm) != 0) {
    curl_multi_cleanup(tm->multi);
    tm->multi = NULL;
    pthread_mutex_destroy(&tm->pending_mutex);
    pthread_cond_destroy(&tm->pending_cond);
    curl_global_cleanup();
    return false;
  }

  return true;
}

void TilePipeline_shutdown(TileMap *tm) {
  if (!tm) return;

  pthread_mutex_lock(&tm->pending_mutex);
  tm->shutdown_worker = true;
  pthread_cond_signal(&tm->pending_cond);
  pthread_mutex_unlock(&tm->pending_mutex);
  pthread_join(tm->decode_thread, NULL);

  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    PendingDownload *pd = &tm->pending[i];
    if (pd->active && pd->easy) {
      curl_multi_remove_handle(tm->multi, pd->easy);
      curl_easy_cleanup(pd->easy);
    }
    clear_pending_download(pd);
  }
  if (tm->multi) {
    curl_multi_cleanup(tm->multi);
    tm->multi = NULL;
  }
  pthread_mutex_destroy(&tm->pending_mutex);
  pthread_cond_destroy(&tm->pending_cond);
  curl_global_cleanup();
}

void TilePipeline_start_download(TileMap *tm, int x, int y, int z) {
  char url[TILE_MAP_URL_CAP];
  CURL *easy = NULL;
  PendingDownload *pd = NULL;

  if (!tm) return;

  pthread_mutex_lock(&tm->pending_mutex);
  if (is_pending_locked(tm, x, y, z)) {
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  pd = find_free_slot_locked(tm);
  if (!pd) {
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  clear_pending_download(pd);
  pd->x = x;
  pd->y = y;
  pd->z = z;
  pd->active = true;
  pthread_mutex_unlock(&tm->pending_mutex);

  if (queue_tile_from_disk(tm, pd, x, y, z)) {
    return;
  }

  if (!format_tile_url(tm, x, y, z, url, sizeof(url))) {
    pthread_mutex_lock(&tm->pending_mutex);
    clear_pending_download(pd);
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  easy = curl_easy_init();
  if (!easy) {
    pthread_mutex_lock(&tm->pending_mutex);
    clear_pending_download(pd);
    pthread_mutex_unlock(&tm->pending_mutex);
    return;
  }

  pthread_mutex_lock(&tm->pending_mutex);
  if (!pd->active || pd->easy || pd->ready || pd->decoding || pd->decoded ||
      pd->x != x || pd->y != y || pd->z != z) {
    pthread_mutex_unlock(&tm->pending_mutex);
    curl_easy_cleanup(easy);
    return;
  }
  pd->easy = easy;
  pthread_mutex_unlock(&tm->pending_mutex);

  curl_easy_setopt(easy, CURLOPT_URL, url);
  curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(easy, CURLOPT_WRITEDATA, &pd->buf);
  curl_easy_setopt(easy, CURLOPT_USERAGENT, "psm-viz/1.0");
  curl_easy_setopt(easy, CURLOPT_TIMEOUT, 10L);
  curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(easy, CURLOPT_PRIVATE, pd);

  if (curl_multi_add_handle(tm->multi, easy) != CURLM_OK) {
    pthread_mutex_lock(&tm->pending_mutex);
    if (pd->easy == easy) {
      clear_pending_download(pd);
    }
    pthread_mutex_unlock(&tm->pending_mutex);
    curl_easy_cleanup(easy);
  }
}

void TilePipeline_poll_downloads(TileMap *tm) {
  CURLMsg *msg;
  int msgs_left;

  if (!tm || !tm->multi) return;

  curl_multi_perform(tm->multi, &tm->running_transfers);

  while ((msg = curl_multi_info_read(tm->multi, &msgs_left))) {
    PendingDownload *pd = NULL;
    CURL *easy;
    bool ready;

    if (msg->msg != CURLMSG_DONE) continue;

    easy = msg->easy_handle;
    curl_easy_getinfo(easy, CURLINFO_PRIVATE, &pd);
    if (!pd) continue;

    curl_multi_remove_handle(tm->multi, easy);
    curl_easy_cleanup(easy);
    ready = (msg->data.result == CURLE_OK && pd->buf.size > 0);
    if (ready) {
      TileDiskCache_write(&tm->disk_cache, pd->x, pd->y, pd->z,
                          pd->buf.data, pd->buf.size);
    }
    pthread_mutex_lock(&tm->pending_mutex);
    pd->easy = NULL;
    pd->active = false;
    pd->ready = ready;
    if (pd->ready) {
      pthread_cond_signal(&tm->pending_cond);
    } else {
      clear_pending_download(pd);
    }
    pthread_mutex_unlock(&tm->pending_mutex);
  }
}

bool TilePipeline_take_decoded(TileMap *tm, TileDecodedImage *out_image) {
  if (!tm || !out_image) return false;

  memset(out_image, 0, sizeof(*out_image));
  pthread_mutex_lock(&tm->pending_mutex);
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    PendingDownload *pd = &tm->pending[i];
    if (!pd->decoded || !pd->decoded_pixels) continue;

    out_image->pixels = pd->decoded_pixels;
    pd->decoded_pixels = NULL;
    out_image->width = pd->decoded_w;
    out_image->height = pd->decoded_h;
    out_image->x = pd->x;
    out_image->y = pd->y;
    out_image->z = pd->z;
    clear_pending_download(pd);
    pthread_mutex_unlock(&tm->pending_mutex);
    return true;
  }
  pthread_mutex_unlock(&tm->pending_mutex);
  return false;
}

void TilePipeline_accumulate_stats(TileMap *tm, TileMapStats *out_stats) {
  if (!tm || !out_stats) return;

  pthread_mutex_lock(&tm->pending_mutex);
  for (int i = 0; i < TILE_MAP_MAX_PENDING_DOWNLOADS; i++) {
    const PendingDownload *pd = &tm->pending[i];
    if (pd->active) out_stats->active_downloads++;
    if (pd->ready) out_stats->ready_downloads++;
    if (pd->decoding) out_stats->decoding_downloads++;
    if (pd->decoded) out_stats->decoded_downloads++;
  }
  pthread_mutex_unlock(&tm->pending_mutex);
}
