#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/exemplar_codec.h"
#include "vendor/probabilistic_data_structures/lib/utilities.h"

// xorshift PRNG — small, deterministic, no header dependency.
static uint32_t xs_state = 0xC0FFEE01u;
static uint32_t xs_next(void) {
  uint32_t x = xs_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  xs_state = x;
  return x;
}
static float xs_normal(void) {
  // Box-Muller. Reasonable test-only quality.
  float u1 = ((float)(xs_next() & 0xFFFFFF) + 1.0f) / 16777217.0f;
  float u2 = ((float)(xs_next() & 0xFFFFFF) + 1.0f) / 16777217.0f;
  return sqrtf(-2.0f * logf(u1)) * cosf((float)(2.0 * M_PI) * u2);
}
static void xs_seed(uint32_t s) { xs_state = s ? s : 1u; }

static double cosine_double(const float *a, const float *b, size_t n) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double ai = a[i], bi = b[i];
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  if (na <= 0.0 || nb <= 0.0) return 0.0;
  return dot / (sqrt(na) * sqrt(nb));
}

void test_codec_name_roundtrip(void) {
  const char *names[] = {"raw", "turboquant_2b", "turboquant_3b", "turboquant_4b"};
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    ExemplarCodec c;
    bool ok = ExemplarCodec_from_string(names[i], &c);
    ASSERT(ok, 1, ok);
    ASSERT(strcmp(ExemplarCodec_name(c), names[i]) == 0, 1,
           strcmp(ExemplarCodec_name(c), names[i]) == 0);
  }
  ExemplarCodec c;
  ASSERT(!ExemplarCodec_from_string("turboquant_5b", &c), 0,
         ExemplarCodec_from_string("turboquant_5b", &c));
  ASSERT(!ExemplarCodec_from_string(NULL, &c), 0,
         ExemplarCodec_from_string(NULL, &c));
}

void test_payload_size_matches_packing(void) {
  // RAW: dim * 4 bytes.
  ASSERT(ExemplarCodec_payload_size(EXEMPLAR_CODEC_RAW, 768) == 768 * 4,
         (int)(768 * 4),
         (int)ExemplarCodec_payload_size(EXEMPLAR_CODEC_RAW, 768));
  // TurboQuant 4-bit on dim=768 -> dim_pad=1024 -> 4*1024/8 = 512 body
  // bytes + 16 header + 1 slack = 529.
  size_t p4_768 = ExemplarCodec_payload_size(EXEMPLAR_CODEC_TURBOQUANT_4B, 768);
  ASSERT(p4_768 == 16 + 512 + 1, (int)(16 + 512 + 1), (int)p4_768);
  // Compression vs RAW at dim=768: 4-bit ~6x, 2-bit ~12x.
  size_t p2_768 = ExemplarCodec_payload_size(EXEMPLAR_CODEC_TURBOQUANT_2B, 768);
  size_t raw_768 = ExemplarCodec_payload_size(EXEMPLAR_CODEC_RAW, 768);
  ASSERT(p2_768 < p4_768, 1, p2_768 < p4_768);
  ASSERT(p4_768 * 5 < raw_768, 1, p4_768 * 5 < raw_768);
}

void test_raw_roundtrip(void) {
  const size_t dim = 64;
  float v[64];
  xs_seed(1);
  for (size_t i = 0; i < dim; ++i) v[i] = xs_normal();

  void *enc = NULL;
  size_t enc_size = 0;
  bool ok = ExemplarCodec_encode(EXEMPLAR_CODEC_RAW, v, dim * sizeof(float),
                                 &enc, &enc_size);
  ASSERT(ok, 1, ok);
  ASSERT(enc_size == dim * sizeof(float), (int)(dim * sizeof(float)),
         (int)enc_size);

  ExemplarCodecQuery *q = ExemplarCodecQuery_new(EXEMPLAR_CODEC_RAW, dim, v);
  ASSERT(NULL != q, 1, NULL != q);
  double sim = 0.0;
  ok = ExemplarCodec_cosine(q, enc, enc_size, &sim);
  ASSERT(ok, 1, ok);
  // Self-cosine must be ~1.
  ASSERT(sim > 0.9999, 1, sim > 0.9999);
  ExemplarCodecQuery_free(q);
  free(enc);
}

static void run_self_cosine(ExemplarCodec codec, double tolerance) {
  // Self-cosine on ~realistic dims (CLIP-L = 768). The codec should preserve
  // self-similarity within `tolerance` even after lossy quantization.
  const size_t dim = 768;
  float *v = (float *)malloc(dim * sizeof(float));
  xs_seed(42);
  for (size_t i = 0; i < dim; ++i) v[i] = xs_normal();

  void *enc = NULL;
  size_t enc_size = 0;
  bool ok = ExemplarCodec_encode(codec, v, dim * sizeof(float), &enc, &enc_size);
  ASSERT(ok, 1, ok);

  ExemplarCodecQuery *q = ExemplarCodecQuery_new(codec, dim, v);
  ASSERT(NULL != q, 1, NULL != q);
  double sim = 0.0;
  ok = ExemplarCodec_cosine(q, enc, enc_size, &sim);
  ASSERT(ok, 1, ok);
  // Quantization noise lowers self-similarity; the bound depends on bits.
  ASSERT(sim > 1.0 - tolerance, 1, sim > 1.0 - tolerance);

  ExemplarCodecQuery_free(q);
  free(enc);
  free(v);
}

void test_turboquant_4b_self_cosine(void) {
  run_self_cosine(EXEMPLAR_CODEC_TURBOQUANT_4B, 0.02);
}

void test_turboquant_3b_self_cosine(void) {
  run_self_cosine(EXEMPLAR_CODEC_TURBOQUANT_3B, 0.05);
}

void test_turboquant_2b_self_cosine(void) {
  run_self_cosine(EXEMPLAR_CODEC_TURBOQUANT_2B, 0.20);
}

void test_turboquant_preserves_top_ranking(void) {
  // On a small bank of random vectors plus one clear winner (a noisy copy of
  // the query), TurboQuant 4-bit should pick the same top-1 as float32.
  // We don't claim ranking parity for tied/near-tied raw similarities — the
  // 0.02 self-cosine bound earlier is the real correctness statement.
  const size_t dim = 256;
  const size_t bank = 32;
  float *vs = (float *)malloc(bank * dim * sizeof(float));
  float *q = (float *)malloc(dim * sizeof(float));
  xs_seed(7);
  for (size_t i = 0; i < bank * dim; ++i) vs[i] = xs_normal();
  for (size_t i = 0; i < dim; ++i) q[i] = xs_normal();
  // Plant a clear winner at index 0: 90% query + 10% noise. Float32 raw cosine
  // ~0.95; the next best random match in 256d is typically ~0.1.
  for (size_t i = 0; i < dim; ++i) {
    vs[0 * dim + i] = 0.9f * q[i] + 0.1f * vs[0 * dim + i];
  }

  size_t best_raw = 0;
  double best_raw_sim = -2.0;
  for (size_t b = 0; b < bank; ++b) {
    double s = cosine_double(q, vs + b * dim, dim);
    if (s > best_raw_sim) { best_raw_sim = s; best_raw = b; }
  }
  ASSERT(best_raw == 0, 0, (int)best_raw);

  ExemplarCodecQuery *prep =
      ExemplarCodecQuery_new(EXEMPLAR_CODEC_TURBOQUANT_4B, dim, q);
  ASSERT(NULL != prep, 1, NULL != prep);
  size_t best_tq = 0;
  double best_tq_sim = -2.0;
  for (size_t b = 0; b < bank; ++b) {
    void *enc = NULL;
    size_t enc_size = 0;
    bool ok = ExemplarCodec_encode(EXEMPLAR_CODEC_TURBOQUANT_4B,
                                   vs + b * dim, dim * sizeof(float),
                                   &enc, &enc_size);
    ASSERT(ok, 1, ok);
    double s = 0.0;
    ok = ExemplarCodec_cosine(prep, enc, enc_size, &s);
    ASSERT(ok, 1, ok);
    if (s > best_tq_sim) { best_tq_sim = s; best_tq = b; }
    free(enc);
  }
  ExemplarCodecQuery_free(prep);
  ASSERT(best_tq == best_raw, (int)best_raw, (int)best_tq);

  free(vs);
  free(q);
}

void test_cosine_rejects_dim_mismatch(void) {
  // Encoder produced bytes for dim=128; query says dim=64. Must reject, not
  // crash, not return garbage.
  const size_t enc_dim = 128;
  float *enc_vec = (float *)malloc(enc_dim * sizeof(float));
  xs_seed(11);
  for (size_t i = 0; i < enc_dim; ++i) enc_vec[i] = xs_normal();
  void *enc = NULL;
  size_t enc_size = 0;
  bool ok = ExemplarCodec_encode(EXEMPLAR_CODEC_TURBOQUANT_4B, enc_vec,
                                 enc_dim * sizeof(float), &enc, &enc_size);
  ASSERT(ok, 1, ok);

  float q_vec[64];
  xs_seed(12);
  for (size_t i = 0; i < 64; ++i) q_vec[i] = xs_normal();
  ExemplarCodecQuery *q =
      ExemplarCodecQuery_new(EXEMPLAR_CODEC_TURBOQUANT_4B, 64, q_vec);
  ASSERT(NULL != q, 1, NULL != q);
  double sim = 999.0;
  ok = ExemplarCodec_cosine(q, enc, enc_size, &sim);
  ASSERT(!ok, 0, ok);
  ASSERT(sim == 999.0, 1, sim == 999.0);

  ExemplarCodecQuery_free(q);
  free(enc);
  free(enc_vec);
}

void test_zero_norm_query_rejected(void) {
  float zero[16] = {0};
  ExemplarCodecQuery *q =
      ExemplarCodecQuery_new(EXEMPLAR_CODEC_RAW, 16, zero);
  ASSERT(NULL == q, 1, NULL == q);
  q = ExemplarCodecQuery_new(EXEMPLAR_CODEC_TURBOQUANT_4B, 16, zero);
  ASSERT(NULL == q, 1, NULL == q);
}

int main(void) {
  RUN_TEST(test_codec_name_roundtrip);
  RUN_TEST(test_payload_size_matches_packing);
  RUN_TEST(test_raw_roundtrip);
  RUN_TEST(test_turboquant_4b_self_cosine);
  RUN_TEST(test_turboquant_3b_self_cosine);
  RUN_TEST(test_turboquant_2b_self_cosine);
  RUN_TEST(test_turboquant_preserves_top_ranking);
  RUN_TEST(test_cosine_rejects_dim_mismatch);
  RUN_TEST(test_zero_norm_query_rejected);
  return 0;
}
