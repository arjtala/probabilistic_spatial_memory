#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "core/tile.h"

// xoshiro256** PRNG used by the reservoir sampler. Public domain, fast,
// good statistical quality, and — crucially for evaluation reproducibility
// — explicitly seedable via Tile_set_random_seed(). Thread safety is not
// required: Tile observes are single-threaded per SpatialMemory. The
// reservoir does not need cryptographic quality.
//
// Reference: https://prng.di.unimi.it/xoshiro256starstar.c
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__NetBSD__) || defined(__DragonFly__)
#define TILE_HAS_ARC4RANDOM 1
#endif

static uint64_t tile_prng_state[4];
static int tile_prng_seeded = 0;

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static void tile_prng_seed_from(uint64_t seed) {
  uint64_t s = seed;
  // splitmix64 expansion gives four well-distributed seed words even for a
  // small input seed (e.g. 0).
  tile_prng_state[0] = splitmix64(&s);
  tile_prng_state[1] = splitmix64(&s);
  tile_prng_state[2] = splitmix64(&s);
  tile_prng_state[3] = splitmix64(&s);
  tile_prng_seeded = 1;
}

static void tile_prng_seed_entropic(void) {
  uint64_t seed;
#if defined(TILE_HAS_ARC4RANDOM)
  seed = ((uint64_t)arc4random() << 32) | (uint64_t)arc4random();
#else
  seed = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)&tile_prng_state;
#endif
  tile_prng_seed_from(seed);
}

void Tile_set_random_seed(uint64_t seed) {
  tile_prng_seed_from(seed);
}

static inline uint64_t tile_rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static uint64_t tile_rand_uint64(void) {
  if (!tile_prng_seeded) tile_prng_seed_entropic();
  const uint64_t result = tile_rotl(tile_prng_state[1] * 5, 7) * 9;
  const uint64_t t = tile_prng_state[1] << 17;
  tile_prng_state[2] ^= tile_prng_state[0];
  tile_prng_state[3] ^= tile_prng_state[1];
  tile_prng_state[1] ^= tile_prng_state[2];
  tile_prng_state[0] ^= tile_prng_state[3];
  tile_prng_state[2] ^= t;
  tile_prng_state[3] = tile_rotl(tile_prng_state[3], 45);
  return result;
}

bool Tile_coords_to_cell(double lat, double lng, int resolution,
                         H3Index *out_cell_id, const char *context) {
  if (!out_cell_id) return false;
  if (resolution < 0 || resolution > 15) {
    fprintf(stderr, "%s: H3 resolution must be in [0, 15], got %d\n",
            context, resolution);
    return false;
  }
  if (!isfinite(lat) || !isfinite(lng) || lat < -90.0 || lat > 90.0 ||
      lng < -180.0 || lng > 180.0) {
    fprintf(stderr, "%s: invalid lat/lng\n", context);
    return false;
  }

  LatLng loc;
  loc.lat = degsToRads(lat);
  loc.lng = degsToRads(lng);
  if (latLngToCell(&loc, resolution, out_cell_id)) {
    fprintf(stderr, "%s: invalid lat/lng or resolution\n", context);
    return false;
  }
  return true;
}

Tile *Tile_new(const double lat, const double lng, const int resolution,
               const size_t capacity, const size_t precision,
               const size_t exemplar_capacity) {
  H3Index cellId;
  if (!Tile_coords_to_cell(lat, lng, resolution, &cellId, "Tile_new")) {
    return NULL;
  }

  Tile *tile = (Tile *)malloc(sizeof(Tile));
  if (NULL == tile) {
    fprintf(stderr, "Tile_new: out of memory\n");
    return NULL;
  }

  RingBuffer *rb = RingBuffer_new(capacity, precision);
  if (!rb) {
    free(tile);
    return NULL;
  }

  tile->cellId = cellId;
  tile->rb = rb;
  tile->exemplars = NULL;
  tile->exemplar_capacity = exemplar_capacity;
  tile->exemplar_count = 0;
  tile->exemplar_seen = 0;
  if (exemplar_capacity > 0) {
    tile->exemplars =
        (TileExemplar *)calloc(exemplar_capacity, sizeof(TileExemplar));
    if (!tile->exemplars) {
      fprintf(stderr, "Tile_new: out of memory allocating exemplar reservoir\n");
      RingBuffer_free(rb);
      free(tile);
      return NULL;
    }
  }
  return tile;
}

static void tile_exemplars_free(Tile *tile) {
  if (!tile || !tile->exemplars) return;
  for (size_t i = 0; i < tile->exemplar_count; ++i) {
    free(tile->exemplars[i].data);
    tile->exemplars[i].data = NULL;
    tile->exemplars[i].size = 0;
  }
  free(tile->exemplars);
  tile->exemplars = NULL;
  tile->exemplar_count = 0;
}

void Tile_free(Tile *tile) {
  if (!tile) return;
  tile_exemplars_free(tile);
  RingBuffer_free(tile->rb);
  free(tile);
}

static void tile_exemplars_sample(Tile *tile, double t, const void *data,
                                  size_t size) {
  if (!tile || tile->exemplar_capacity == 0 || !data || size == 0) return;

  tile->exemplar_seen++;

  // Reservoir sampling (Algorithm R): fill the buffer first, then each
  // subsequent candidate evicts a uniformly chosen slot with probability
  // capacity / seen.
  if (tile->exemplar_count < tile->exemplar_capacity) {
    void *copy = malloc(size);
    if (!copy) {
      fprintf(stderr, "Tile: failed to copy exemplar bytes (size=%zu)\n", size);
      return;
    }
    memcpy(copy, data, size);
    TileExemplar *slot = &tile->exemplars[tile->exemplar_count++];
    slot->t = t;
    slot->data = copy;
    slot->size = size;
    return;
  }

  uint64_t idx = tile_rand_uint64() % tile->exemplar_seen;
  if (idx < tile->exemplar_capacity) {
    void *copy = malloc(size);
    if (!copy) {
      fprintf(stderr, "Tile: failed to copy exemplar bytes (size=%zu)\n", size);
      return;
    }
    memcpy(copy, data, size);
    TileExemplar *slot = &tile->exemplars[(size_t)idx];
    free(slot->data);
    slot->t = t;
    slot->data = copy;
    slot->size = size;
  }
}

// Record an observation at timestamp t. Drives both the HLL ring buffer
// (including the [t_min, t_max] interval on the current slot) and the
// per-tile reservoir sampler.
void Tile_observe(Tile *tile, double t, const void *data, size_t size) {
  if (!tile) return;
  RingBufferHLL *current = RingBuffer_current(tile->rb);
  if (current) {
    RingBufferHLL_add(current, t, data, size);
    RingBufferHLL_release(current);
  }
  tile_exemplars_sample(tile, t, data, size);
}

// Rotate to the next time window. This is called on a timer (e.g. every 5 minutes). The
// current HLL slot is left as-is, head moves forward, and the new slot is reset. Old observations naturally
// fall off when the buffer wraps.
void Tile_advance(Tile *tile) {
  if (!tile) return;
  RingBuffer_advance(tile->rb);
}

//  Tile_query(tile, n) — "How many distinct things were seen here over the last N time windows?" Merges N slots
//  from the ring buffer, gets the count, frees the merged HLL, and returns the estimate.
double Tile_query(Tile *tile, const size_t n) {
  if (!tile) return 0.0;
  RingBufferWindow window = RingBuffer_merge_window(tile->rb, n);
  if (!window.sketch) return 0.0;
  double count = RingBufferHLL_count(window.sketch);
  RingBufferHLL_release(window.sketch);
  return count;
}

size_t Tile_exemplar_count(const Tile *tile) {
  if (!tile) return 0;
  return tile->exemplar_count;
}

const TileExemplar *Tile_exemplar_at(const Tile *tile, size_t idx) {
  if (!tile || !tile->exemplars) return NULL;
  if (idx >= tile->exemplar_count) return NULL;
  return &tile->exemplars[idx];
}
