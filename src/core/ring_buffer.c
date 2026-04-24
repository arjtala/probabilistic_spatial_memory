#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/ring_buffer.h"

struct RingBufferHLL {
  HLL *hll;
  size_t refcount;
};

// Test-only allocation fault injection. Non-zero skips the wrapper malloc so
// merge can exercise its OOM path without relying on actual memory exhaustion.
static int ring_buffer_force_wrap_failure = 0;

void RingBuffer_test_force_wrap_failure(int on) {
  ring_buffer_force_wrap_failure = on ? 1 : 0;
}

// Vendor HLL has no native clone primitive. Previous code relied on
// HLL_merge_copy(src, src) as an idempotent self-merge "clone" trick; this
// wrapper makes intent explicit at call sites so the surprise doesn't leak.
static HLL *HLL_clone(const HLL *src) {
  if (!src) return NULL;
  HLL *dst = HLL_default(src->p);
  if (!dst) return NULL;
  HLL *merged = HLL_merge_copy(dst, src);
  freeHLL(dst);
  return merged;
}

static RingBufferHLL *ring_buffer_hll_wrap(HLL *hll) {
  RingBufferHLL *wrapped;

  if (!hll) return NULL;
  if (ring_buffer_force_wrap_failure) {
    freeHLL(hll);
    return NULL;
  }
  wrapped = malloc(sizeof(*wrapped));
  if (!wrapped) {
    freeHLL(hll);
    return NULL;
  }
  wrapped->hll = hll;
  wrapped->refcount = 1;
  return wrapped;
}

static RingBufferHLL *ring_buffer_hll_new(size_t precision) {
  return ring_buffer_hll_wrap(HLL_default(precision));
}

static RingBufferHLL *ring_buffer_hll_retain(RingBufferHLL *hll) {
  if (hll) {
    hll->refcount++;
  }
  return hll;
}

void RingBufferHLL_release(RingBufferHLL *hll) {
  if (!hll) return;
  if (hll->refcount == 0) return;
  hll->refcount--;
  if (hll->refcount == 0) {
    freeHLL(hll->hll);
    free(hll);
  }
}

void RingBufferHLL_add(RingBufferHLL *hll, const void *data, size_t size) {
  if (!hll || !hll->hll) return;
  HLL_add(hll->hll, data, size);
}

double RingBufferHLL_count(const RingBufferHLL *hll) {
  if (!hll || !hll->hll) return 0.0;
  return HLL_count(hll->hll);
}

size_t RingBuffer_precision_min(void) {
  return 4;
}

size_t RingBuffer_precision_max(void) {
  // Project-level practical cap. Higher values explode HLL register memory
  // (2^p bytes per sketch) long before they are useful here.
  return 18;
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

  RingBuffer *rb =
      (RingBuffer *)malloc(sizeof(RingBuffer) + capacity * sizeof(RingBufferHLL *));
  if (NULL == rb) {
    fprintf(stderr, "RingBuffer_new: out of memory\n");
    return NULL;
  }
  rb->head = 0;
  rb->capacity = capacity;
  rb->precision = precision;
  for (size_t i = 0; i < capacity; ++i) {
    rb->hlls[i] = ring_buffer_hll_new(rb->precision);
    if (!rb->hlls[i]) {
      for (size_t j = 0; j < i; ++j) {
        RingBufferHLL_release(rb->hlls[j]);
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
      RingBufferHLL_release(rb->hlls[i]);
    }
  }
  free(rb);
}
void RingBuffer_advance(RingBuffer *rb) {
  RingBufferHLL *replacement;
  RingBufferHLL *expired;

  if (!rb || rb->capacity == 0) return;
  replacement = ring_buffer_hll_new(rb->precision);
  if (!replacement) return;
  rb->head = (rb->head + 1) % rb->capacity;
  expired = rb->hlls[rb->head];
  rb->hlls[rb->head] = replacement;
  RingBufferHLL_release(expired);
}
RingBufferHLL *RingBuffer_current(RingBuffer *rb) {
  if (!rb || rb->capacity == 0) return NULL;
  return ring_buffer_hll_retain(rb->hlls[rb->head]);
}
RingBufferHLL *RingBuffer_merge_window(RingBuffer *rb, size_t n, bool *out_empty) {
  HLL *merged_hll;
  HLL *next;
  HLL *curr_hll;
  RingBufferHLL *current;
  RingBufferHLL *wrapped;

  if (out_empty) *out_empty = false;

  if (!rb || rb->capacity == 0) {
    if (out_empty) *out_empty = true;
    return NULL;
  }
  current = rb->hlls[rb->head];
  curr_hll = current ? current->hll : NULL;
  if (!curr_hll) {
    if (out_empty) *out_empty = true;
    return NULL;
  }
  if (n > rb->capacity) {
    n = rb->capacity;
  }

  merged_hll = HLL_clone(curr_hll);
  if (!merged_hll) {
    fprintf(stderr, "RingBuffer_merge_window: out of memory\n");
    return NULL;
  }
  if (n == 0) {
    wrapped = ring_buffer_hll_wrap(merged_hll);
    if (!wrapped) {
      fprintf(stderr, "RingBuffer_merge_window: out of memory\n");
    }
    return wrapped;
  }

  size_t idx = rb->head;
  for (size_t i = 0; i < n; ++i) {
    idx = (idx - 1 + rb->capacity) % rb->capacity;
    next = HLL_merge_copy(merged_hll, rb->hlls[idx]->hll);
    freeHLL(merged_hll);
    if (!next) {
      fprintf(stderr, "RingBuffer_merge_window: out of memory\n");
      return NULL;
    }
    merged_hll = next;
  }
  wrapped = ring_buffer_hll_wrap(merged_hll);
  if (!wrapped) {
    fprintf(stderr, "RingBuffer_merge_window: out of memory\n");
  }
  return wrapped;
}
