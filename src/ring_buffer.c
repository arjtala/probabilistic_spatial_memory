#include "ring_buffer.h"

RingBuffer *RingBuffer_new(size_t capacity, size_t precision) {
  RingBuffer *rb = (RingBuffer*)malloc(sizeof(RingBuffer) + capacity * sizeof(HLL*));
  if (NULL==rb) {
    fprintf(stderr, "Out of memory.\n");
    exit(EXIT_FAILURE);
  }
  rb->head = 0;
  rb->capacity = capacity;
  rb->precision = precision;
  for (size_t i = 0; i < capacity; ++i) {
    rb->hlls[i] = HLL_default(rb->precision);
  }
  return rb;
}

void RingBuffer_free(RingBuffer *rb) {
  for (size_t i = 0; i < rb->capacity; ++i) {
    freeHLL(rb->hlls[i]);
  }
  free(rb);
}
void RingBuffer_advance(RingBuffer *rb) {
  rb->head = (rb->head + 1) % rb->capacity;
  freeHLL(rb->hlls[rb->head]);
  rb->hlls[rb->head] = HLL_default(rb->precision);
}
HLL *RingBuffer_current(RingBuffer *rb) { return rb->hlls[rb->head]; }
HLL *RingBuffer_merge_window(RingBuffer *rb, size_t n) {
  HLL *curr_hll = RingBuffer_current(rb);
  // This is a hacky way of return a deep copy as it performs an element-wise
  // max of registers of the HLL with itself
  if (n == 0) return HLL_merge_copy(curr_hll, curr_hll);
  if (n > rb->capacity) n = rb->capacity;
  HLL *merged_hll;
  HLL *prev_hll;
  size_t idx = rb->head;
  for (size_t i = 0; i < n; ++i) {
    idx = (idx - 1 + rb->capacity) % rb->capacity;
    prev_hll = rb->hlls[idx];
    merged_hll = HLL_merge_copy(curr_hll, prev_hll);
    if (i>0) freeHLL(curr_hll);
    curr_hll = merged_hll;

  }
  return curr_hll;
}
