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

// Result of merging a ring-buffer window. The sketch is owned by the caller
// and must be released with RingBufferHLL_release. The [t_min, t_max] interval
// spans every observation that contributed to the merged sketch. When
// is_empty is true the window contained no observations: t_min = DBL_MAX,
// t_max = -DBL_MAX, and sketch is still a valid (but empty) handle so
// callers can query the count uniformly.
typedef struct {
  RingBufferHLL *sketch;
  double t_min;
  double t_max;
  bool is_empty;
} RingBufferWindow;

size_t RingBuffer_precision_min(void);
size_t RingBuffer_precision_max(void);
bool RingBuffer_precision_is_valid(size_t precision);

void RingBufferHLL_release(RingBufferHLL *hll);
// Record an observation in this slot. t is the observation timestamp and
// extends the slot's [t_min, t_max] interval. Safe to call with any finite
// double — the first observation seeds both bounds.
void RingBufferHLL_add(RingBufferHLL *hll, double t, const void *data,
                       size_t size);
double RingBufferHLL_count(const RingBufferHLL *hll);
// Read the [t_min, t_max] interval of this slot. Returns true and sets
// *out_t_min/*out_t_max when the slot has at least one observation. Returns
// false when the slot is empty (and leaves the out-params untouched).
bool RingBufferHLL_interval(const RingBufferHLL *hll, double *out_t_min,
                            double *out_t_max);

RingBuffer *RingBuffer_new(const size_t capacity, const size_t precision);
void RingBuffer_free(RingBuffer *rb);
void RingBuffer_advance(RingBuffer *rb);
// Caller owns one retained reference and must release it with
// RingBufferHLL_release.
RingBufferHLL *RingBuffer_current(RingBuffer *rb);
// Merge the current slot plus the n prior slots (up to capacity-1). Returns
// a RingBufferWindow whose sketch is owned by the caller and must be
// released with RingBufferHLL_release. On allocation failure the sketch is
// NULL — is_empty is false in that case; callers should treat NULL sketch
// as an error. When the window has no observations is_empty is true,
// t_min == DBL_MAX, t_max == -DBL_MAX, and the sketch is an empty handle.
RingBufferWindow RingBuffer_merge_window(RingBuffer *rb, size_t n);

#endif
