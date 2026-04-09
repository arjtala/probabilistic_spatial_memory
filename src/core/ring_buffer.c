#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/ring_buffer.h"

size_t RingBuffer_precision_min(void) {
  return 4;
}

size_t RingBuffer_precision_max(void) {
  return 8 * sizeof(uint64_t);
}

bool RingBuffer_precision_is_valid(size_t precision) {
  return precision >= RingBuffer_precision_min() &&
         precision <= RingBuffer_precision_max();
}

RingBuffer *RingBuffer_new(const size_t capacity, const size_t precision) {
  if (capacity == 0) {
    fprintf(stderr, "RingBuffer_new: capacity must be greater than 0\n");
    return NULL;
  }
  if (!RingBuffer_precision_is_valid(precision)) {
    fprintf(stderr, "RingBuffer_new: precision %zu is out of range [%zu, %zu]\n",
            precision, RingBuffer_precision_min(), RingBuffer_precision_max());
    return NULL;
  }

  RingBuffer *rb = (RingBuffer *)malloc(sizeof(RingBuffer) + capacity * sizeof(HLL *));
  if (NULL == rb) {
    fprintf(stderr, "RingBuffer_new: out of memory\n");
    return NULL;
  }
  rb->head = 0;
  rb->capacity = capacity;
  rb->precision = precision;
  for (size_t i = 0; i < capacity; ++i) {
    rb->hlls[i] = HLL_default(rb->precision);
    if (!rb->hlls[i]) {
      for (size_t j = 0; j < i; ++j) {
        freeHLL(rb->hlls[j]);
      }
      free(rb);
      return NULL;
    }
  }
  return rb;
}

void RingBuffer_free(RingBuffer *rb) {
  if (!rb) return;
  for (size_t i = 0; i < rb->capacity; ++i) {
    if (rb->hlls[i]) {
      freeHLL(rb->hlls[i]);
    }
  }
  free(rb);
}
void RingBuffer_advance(RingBuffer *rb) {
  if (!rb || rb->capacity == 0) return;
  HLL *replacement = HLL_default(rb->precision);
  if (!replacement) return;
  rb->head = (rb->head + 1) % rb->capacity;
  freeHLL(rb->hlls[rb->head]);
  rb->hlls[rb->head] = replacement;
}
HLL *RingBuffer_current(RingBuffer *rb) {
  if (!rb || rb->capacity == 0) return NULL;
  return rb->hlls[rb->head];
}
HLL *RingBuffer_merge_window(RingBuffer *rb, size_t n) {
  if (!rb || rb->capacity == 0) return NULL;
  HLL *curr_hll = RingBuffer_current(rb);
  if (!curr_hll) return NULL;
  if (n > rb->capacity) {
    n = rb->capacity;
  }

  HLL *merged_hll = HLL_merge_copy(curr_hll, curr_hll);
  if (!merged_hll || n == 0) return merged_hll;

  size_t idx = rb->head;
  for (size_t i = 0; i < n; ++i) {
    idx = (idx - 1 + rb->capacity) % rb->capacity;
    HLL *next = HLL_merge_copy(merged_hll, rb->hlls[idx]);
    freeHLL(merged_hll);
    if (!next) {
      return NULL;
    }
    merged_hll = next;
  }
  return merged_hll;
}
