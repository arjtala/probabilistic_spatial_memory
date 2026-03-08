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
  ASSERT(CAPACITY == rb->capacity, CAPACITY, (int)rb->capacity);
  ASSERT(PRECISION == rb->precision, PRECISION, (int)rb->precision);
  ASSERT(0 == rb->head, 0, (int)rb->head);
  RingBuffer_free(rb);
}

void test_ring_buffer_current(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  HLL *hll = RingBuffer_current(rb);
  int curr_count = (int)HLL_count(hll);
  ASSERT(0 == curr_count, 0, curr_count);
  const char *pb = "peanut butter";
  HLL_add(hll, pb, strlen(pb));
  curr_count = (int)HLL_count(hll);
  ASSERT(1 <= curr_count, 1, curr_count);
  RingBuffer_free(rb);
}

void test_ring_buffer_advance(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  HLL *hll = RingBuffer_current(rb);
  const char *pb = "peanut butter";
  HLL_add(hll, pb, strlen(pb));

  int orig_data_size = (int)HLL_count(RingBuffer_current(rb));
  ASSERT(0 == rb->head, 0, (int)rb->head);
  ASSERT(1 <= orig_data_size, 1, orig_data_size);


  RingBuffer_advance(rb);
  int new_data_size = (int)HLL_count(RingBuffer_current(rb));
  orig_data_size = (int)HLL_count(rb->hlls[0]);
  ASSERT(1 == rb->head, 1, (int)rb->head);
  ASSERT(1 <= orig_data_size, 1, orig_data_size);
  ASSERT(0 == new_data_size, 0, new_data_size);

  RingBuffer_free(rb);
}

void test_ring_buffer_wrap(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  HLL *hll = RingBuffer_current(rb);
  const char *pb = "peanut butter";
  HLL_add(hll, pb, strlen(pb));
  int curr_size = (int)HLL_count(hll);
  ASSERT(curr_size >= 1, curr_size, 1);

  for (size_t i = 0; i < CAPACITY; ++i) RingBuffer_advance(rb);
  int wrap_size = (int)HLL_count(RingBuffer_current(rb));
  ASSERT(0 == rb->head, 0, (int)rb->head);
  ASSERT(0 == wrap_size, 0, wrap_size);

  RingBuffer_free(rb);
}

void test_ring_buffer_merge_window(void) {
  RingBuffer *rb = RingBuffer_new(CAPACITY, PRECISION);
  const char *sammich[] = {"peanut butter", "jelly", "toast"};
  int sizes[] = {0,0,0};
  HLL *hll;
  for (size_t i = 0; i < CAPACITY-1; ++i) {
    hll = RingBuffer_current(rb);
    HLL_add(hll, sammich[i], strlen(sammich[i]));
    sizes[i] = (int)HLL_count(RingBuffer_current(rb));
    RingBuffer_advance(rb);
  }
  HLL *two_merged = RingBuffer_merge_window(rb, 1);
  int two_merged_count = (int)HLL_count(two_merged);
  HLL *all_merged = RingBuffer_merge_window(rb, CAPACITY);
  int all_merged_count = (int)HLL_count(all_merged);

  ASSERT(sizes[1] <= two_merged_count, sizes[1], two_merged_count);
  ASSERT(all_merged_count >= two_merged_count, all_merged_count, two_merged_count);

  freeHLL(two_merged);
  freeHLL(all_merged);
  RingBuffer_free(rb);
}

int main(void) {
  RUN_TEST(test_ring_buffer_new);
  RUN_TEST(test_ring_buffer_current);
  RUN_TEST(test_ring_buffer_advance);
  RUN_TEST(test_ring_buffer_wrap);
  RUN_TEST(test_ring_buffer_merge_window);

  return 0;
}
