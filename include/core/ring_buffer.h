#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include "vendor/probabilistic_data_structures/hyperloglog/hll.h"

#define DEFAULT_CAPACITY 12
#define DEFAULT_PRECISION 10

// Data structure that holds a fixed-size array of hyperloglog counters
// This is the core of the time-decay mechanism, each slot is a time window
// and merging slots gives you "memory over the last N intervals."
typedef struct {
  size_t capacity;
  size_t head;
  size_t precision;
  HLL *hlls[];
} RingBuffer;

RingBuffer *RingBuffer_new(const size_t capacity, const size_t precision);
void RingBuffer_free(RingBuffer *rb);
void RingBuffer_advance(RingBuffer *rb);
HLL *RingBuffer_current(RingBuffer *rb);
HLL *RingBuffer_merge_window(RingBuffer *rb, size_t n);

#endif
