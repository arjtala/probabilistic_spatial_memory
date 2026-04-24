#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/ring_buffer.h"

struct RingBufferHLL {
  HLL *hll;
  size_t refcount;
  // [t_min, t_max] timestamp interval for observations in this slot. Use
  // DBL_MAX / -DBL_MAX as the "no observations" sentinel — NAN is unsafe
  // because several build profiles compile with -ffast-math, under which
  // NaN comparison semantics are unreliable.
  double t_min;
  double t_max;
};

static void ring_buffer_hll_reset_interval(RingBufferHLL *wrapped) {
  if (!wrapped) return;
  wrapped->t_min = DBL_MAX;
  wrapped->t_max = -DBL_MAX;
}

static RingBufferHLL *ring_buffer_hll_wrap(HLL *hll) {
  RingBufferHLL *wrapped;

  if (!hll) return NULL;
  wrapped = malloc(sizeof(*wrapped));
  if (!wrapped) {
    freeHLL(hll);
    return NULL;
  }
  wrapped->hll = hll;
  wrapped->refcount = 1;
  ring_buffer_hll_reset_interval(wrapped);
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

void RingBufferHLL_add(RingBufferHLL *hll, double t, const void *data,
                       size_t size) {
  if (!hll || !hll->hll) return;
  HLL_add(hll->hll, data, size);
  // The DBL_MAX / -DBL_MAX sentinels make the first observation "Just Work":
  // fmin(DBL_MAX, t) = t and fmax(-DBL_MAX, t) = t.
  if (t < hll->t_min) hll->t_min = t;
  if (t > hll->t_max) hll->t_max = t;
}

double RingBufferHLL_count(const RingBufferHLL *hll) {
  if (!hll || !hll->hll) return 0.0;
  return HLL_count(hll->hll);
}

bool RingBufferHLL_interval(const RingBufferHLL *hll, double *out_t_min,
                            double *out_t_max) {
  if (!hll) return false;
  // Slot is empty exactly when no observation has extended the interval
  // away from the sentinels.
  if (hll->t_min == DBL_MAX || hll->t_max == -DBL_MAX) return false;
  if (out_t_min) *out_t_min = hll->t_min;
  if (out_t_max) *out_t_max = hll->t_max;
  return true;
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
RingBufferWindow RingBuffer_merge_window(RingBuffer *rb, size_t n) {
  RingBufferWindow window = {
      .sketch = NULL,
      .t_min = DBL_MAX,
      .t_max = -DBL_MAX,
      .is_empty = false,
  };
  HLL *merged_hll;
  HLL *next;
  HLL *curr_hll;
  RingBufferHLL *current;

  if (!rb || rb->capacity == 0) return window;
  current = rb->hlls[rb->head];
  curr_hll = current ? current->hll : NULL;
  if (!curr_hll) return window;
  if (n > rb->capacity) {
    n = rb->capacity;
  }

  // HLL_merge_copy(x, x) is the canonical "clone" path in this vendor API;
  // this gives the caller an owned sketch independent of the ring buffer.
  merged_hll = HLL_merge_copy(curr_hll, curr_hll);
  if (!merged_hll) return window;

  // Fold the current slot's interval in first.
  if (current->t_min <= current->t_max) {
    if (current->t_min < window.t_min) window.t_min = current->t_min;
    if (current->t_max > window.t_max) window.t_max = current->t_max;
  }

  if (n == 0) {
    window.sketch = ring_buffer_hll_wrap(merged_hll);
    if (!window.sketch) return window;  // sketch NULL signals OOM
    window.is_empty = (window.t_min == DBL_MAX || window.t_max == -DBL_MAX);
    return window;
  }

  size_t idx = rb->head;
  for (size_t i = 0; i < n; ++i) {
    idx = (idx - 1 + rb->capacity) % rb->capacity;
    next = HLL_merge_copy(merged_hll, rb->hlls[idx]->hll);
    freeHLL(merged_hll);
    if (!next) {
      return window;  // sketch left NULL to signal OOM
    }
    merged_hll = next;

    RingBufferHLL *slot = rb->hlls[idx];
    if (slot && slot->t_min <= slot->t_max) {
      if (slot->t_min < window.t_min) window.t_min = slot->t_min;
      if (slot->t_max > window.t_max) window.t_max = slot->t_max;
    }
  }

  window.sketch = ring_buffer_hll_wrap(merged_hll);
  if (!window.sketch) return window;
  window.is_empty = (window.t_min == DBL_MAX || window.t_max == -DBL_MAX);
  return window;
}
