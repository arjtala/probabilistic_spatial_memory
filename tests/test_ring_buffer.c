#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "core/ring_buffer.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

#define CAPACITY 3
#define PRECISION 4

void test_ring_buffer_new(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  ASSERT(NULL != rb, 1, NULL != rb);
  ASSERT(CAPACITY == rb->capacity, CAPACITY, (int)rb->capacity);
  ASSERT(PRECISION == rb->precision, PRECISION, (int)rb->precision);
  ASSERT(0 == rb->head, 0, (int)rb->head);
  RingBuffer_free(rb);
}

void test_ring_buffer_new_invalid_args(void) {
  RingBuffer *no_capacity = RingBuffer_new(0, PRECISION);
  RingBuffer *too_small_precision =
      RingBuffer_new(CAPACITY, RingBuffer_precision_min() - 1);
  RingBuffer *too_large_precision =
      RingBuffer_new(CAPACITY, RingBuffer_precision_max() + 1);
  ASSERT(NULL == no_capacity, 1, NULL == no_capacity);
  ASSERT(NULL == too_small_precision, 1, NULL == too_small_precision);
  ASSERT(NULL == too_large_precision, 1, NULL == too_large_precision);
}

void test_ring_buffer_precision_limits(void) {
  RingBuffer *min_precision = RingBuffer_new(CAPACITY, RingBuffer_precision_min());
  RingBuffer *max_precision = RingBuffer_new(CAPACITY, RingBuffer_precision_max());
  ASSERT(NULL != min_precision, 1, NULL != min_precision);
  ASSERT(NULL != max_precision, 1, NULL != max_precision);
  RingBuffer_free(min_precision);
  RingBuffer_free(max_precision);
}

void test_ring_buffer_current(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  RingBufferHLL *hll = RingBuffer_current(rb);
  int curr_count = (int)RingBufferHLL_count(hll);
  ASSERT(0 == curr_count, 0, curr_count);
  const char *pb = "peanut butter";
  RingBufferHLL_add(hll, pb, strlen(pb));
  curr_count = (int)RingBufferHLL_count(hll);
  ASSERT(1 <= curr_count, 1, curr_count);
  RingBufferHLL_release(hll);
  RingBuffer_free(rb);
}

void test_ring_buffer_advance(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  RingBufferHLL *hll = RingBuffer_current(rb);
  const char *pb = "peanut butter";
  RingBufferHLL_add(hll, pb, strlen(pb));

  RingBufferHLL *before_advance = RingBuffer_current(rb);
  int orig_data_size = (int)RingBufferHLL_count(before_advance);
  ASSERT(0 == rb->head, 0, (int)rb->head);
  ASSERT(1 <= orig_data_size, 1, orig_data_size);


  RingBuffer_advance(rb);
  RingBufferHLL *after_advance = RingBuffer_current(rb);
  int new_data_size = (int)RingBufferHLL_count(after_advance);
  RingBufferHLL *merged = RingBuffer_merge_window(rb, 1);
  orig_data_size = (int)RingBufferHLL_count(merged);
  ASSERT(1 == rb->head, 1, (int)rb->head);
  ASSERT(1 <= orig_data_size, 1, orig_data_size);
  ASSERT(0 == new_data_size, 0, new_data_size);

  RingBufferHLL_release(hll);
  RingBufferHLL_release(before_advance);
  RingBufferHLL_release(after_advance);
  RingBufferHLL_release(merged);
  RingBuffer_free(rb);
}

void test_ring_buffer_wrap(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  RingBufferHLL *hll = RingBuffer_current(rb);
  const char *pb = "peanut butter";
  RingBufferHLL_add(hll, pb, strlen(pb));
  int curr_size = (int)RingBufferHLL_count(hll);
  ASSERT(curr_size >= 1, curr_size, 1);

  for (size_t i = 0; i < CAPACITY; ++i) RingBuffer_advance(rb);
  RingBufferHLL *wrapped = RingBuffer_current(rb);
  int wrap_size = (int)RingBufferHLL_count(wrapped);
  ASSERT(0 == rb->head, 0, (int)rb->head);
  ASSERT(0 == wrap_size, 0, wrap_size);

  RingBufferHLL_release(hll);
  RingBufferHLL_release(wrapped);
  RingBuffer_free(rb);
}

void test_ring_buffer_merge_window(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  const char *sammich[] = {"peanut butter", "jelly", "toast"};
  int sizes[] = {0,0,0};
  RingBufferHLL *hll;
  for (size_t i = 0; i < CAPACITY-1; ++i) {
    hll = RingBuffer_current(rb);
    RingBufferHLL_add(hll, sammich[i], strlen(sammich[i]));
    sizes[i] = (int)RingBufferHLL_count(hll);
    RingBufferHLL_release(hll);
    RingBuffer_advance(rb);
  }
  RingBufferHLL *two_merged = RingBuffer_merge_window(rb, 1);
  int two_merged_count = (int)RingBufferHLL_count(two_merged);
  RingBufferHLL *all_merged = RingBuffer_merge_window(rb, CAPACITY);
  int all_merged_count = (int)RingBufferHLL_count(all_merged);

  ASSERT(sizes[1] <= two_merged_count, sizes[1], two_merged_count);
  ASSERT(all_merged_count >= two_merged_count, all_merged_count, two_merged_count);

  RingBufferHLL_release(two_merged);
  RingBufferHLL_release(all_merged);
  RingBuffer_free(rb);
}

void test_ring_buffer_retained_handle_survives_advance(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  RingBufferHLL *held = RingBuffer_current(rb);
  const char *pb = "peanut butter";

  RingBufferHLL_add(held, pb, strlen(pb));
  RingBuffer_advance(rb);

  ASSERT(1 <= (int)RingBufferHLL_count(held), 1, (int)RingBufferHLL_count(held));

  RingBufferHLL *current = RingBuffer_current(rb);
  ASSERT(0 == (int)RingBufferHLL_count(current), 0, (int)RingBufferHLL_count(current));

  RingBufferHLL_release(current);
  RingBufferHLL_release(held);
  RingBuffer_free(rb);
}

void test_ring_buffer_retained_handle_survives_free(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  RingBufferHLL *held = RingBuffer_current(rb);
  const char *pb = "peanut butter";

  RingBufferHLL_add(held, pb, strlen(pb));
  RingBuffer_free(rb);

  ASSERT(1 <= (int)RingBufferHLL_count(held), 1, (int)RingBufferHLL_count(held));
  RingBufferHLL_release(held);
}

int main(void) {
  RUN_TEST(test_ring_buffer_new);
  RUN_TEST(test_ring_buffer_new_invalid_args);
  RUN_TEST(test_ring_buffer_precision_limits);
  RUN_TEST(test_ring_buffer_current);
  RUN_TEST(test_ring_buffer_advance);
  RUN_TEST(test_ring_buffer_wrap);
  RUN_TEST(test_ring_buffer_merge_window);
  RUN_TEST(test_ring_buffer_retained_handle_survives_advance);
  RUN_TEST(test_ring_buffer_retained_handle_survives_free);

  return 0;
}
