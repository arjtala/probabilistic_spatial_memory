#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdbool.h>
#include <stddef.h>
#include "vendor/probabilistic_data_structures/hyperloglog/hll.h"

#define DEFAULT_CAPACITY 12
#define DEFAULT_PRECISION 10

typedef struct RingBufferHLL RingBufferHLL;

// Data structure that holds a fixed-size array of hyperloglog counters
// This is the core of the time-decay mechanism, each slot is a time window
// and merging slots gives you "memory over the last N intervals."
typedef struct {
  size_t capacity;
  size_t head;
  size_t precision;
  RingBufferHLL *hlls[];
} RingBuffer;

size_t RingBuffer_precision_min(void);
size_t RingBuffer_precision_max(void);
bool RingBuffer_precision_is_valid(size_t precision);

void RingBufferHLL_release(RingBufferHLL *hll);
void RingBufferHLL_add(RingBufferHLL *hll, const void *data, size_t size);
double RingBufferHLL_count(const RingBufferHLL *hll);

RingBuffer *RingBuffer_new(const size_t capacity, const size_t precision);
void RingBuffer_free(RingBuffer *rb);
void RingBuffer_advance(RingBuffer *rb);
// Caller owns one retained reference and must release it with
// RingBufferHLL_release.
RingBufferHLL *RingBuffer_current(RingBuffer *rb);
// Merge the last n ring-buffer slots (plus the head slot) into a newly
// allocated handle. Returns NULL on either "no HLLs to merge" or allocation
// failure; the optional out_empty distinguishes them so callers can surface
// OOM instead of silently reporting zero.
//   non-NULL return  -> success; *out_empty == false if provided
//   NULL + *out_empty = true  -> empty ring (rb NULL/empty or head unset)
//   NULL + *out_empty = false -> allocation failure (stderr already logged)
// Caller owns the returned merged handle and must release it with
// RingBufferHLL_release. out_empty may be NULL if the caller doesn't care.
RingBufferHLL *RingBuffer_merge_window(RingBuffer *rb, size_t n, bool *out_empty);

// Test-only hook: when set non-zero, the internal handle-wrapping allocation
// is forced to fail so the OOM branch can be exercised deterministically.
void RingBuffer_test_force_wrap_failure(int on);

#endif
