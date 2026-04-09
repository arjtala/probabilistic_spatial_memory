#ifndef RING_BUFFER_H
#define RING_BUFFER_H

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
// Caller owns the returned merged handle and must release it with
// RingBufferHLL_release.
RingBufferHLL *RingBuffer_merge_window(RingBuffer *rb, size_t n);

#endif
