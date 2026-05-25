#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/exemplar_codec.h"

// ============================================================================
// TurboQuant: faithful implementation
//
// Pipeline (encode):
//   1. Pad input float32 vector to dim_pad = next_pow2(dim) with zeros.
//   2. Apply a sign-flip (Rademacher diagonal D, seeded per-dim_pad), then a
//      Walsh-Hadamard transform (WHT) of size dim_pad. Sign flip + WHT is the
//      randomized Hadamard transform (RHT); it preserves L2 norms exactly and
//      whitens coordinate distributions toward N(0, ||x||^2 / dim_pad).
//   3. Compute scale = max(|y_i|) over the rotated vector. Quantize each
//      coord to b bits using Lloyd-Max-optimal levels for a unit-variance
//      Gaussian, scaled by `scale`.
//   4. Bit-pack the level indices LSB-first into a byte buffer; prefix with
//      a fixed header (codec id, dim, scale).
//
// Score:
//   - Build a prepared query: same RHT (zero-padded, then sign-flip, then WHT).
//     Cache its norm so cosine = <q', x'> / (||q'|| ||x'||) = <q', x'> /
//     (||q|| * ||x||) since the RHT is orthogonal up to a 1/sqrt(dim_pad)
//     factor that cancels in the cosine.
//   - Per exemplar: parse header, dequantize on the fly into a scratch vector,
//     compute dot(prepared_q, dequantized) and ||dequantized||, return cosine.
//
// PRNG seed for the sign flip is fixed at compile time — the same dim_pad must
// produce the same flip across runs and across the index/query side. A
// per-codec config could expose it later; for now one constant is enough.
//
// References:
//   - Halko, Martinsson, Tropp (2011) — randomized Hadamard transforms.
//   - Lloyd-Max scalar quantizer levels for unit-variance N(0,1):
//       2 bits: ±0.4528, ±1.5104  (4 levels, MSE-optimal)
//       3 bits: ±0.2451, ±0.7560, ±1.3439, ±2.1519  (8 levels)
//       4 bits: see TQ4_LEVELS table below.
//
// All level tables are MSE-optimal for N(0, 1). Encode/decode use the same
// table. We store the bit-packed *index* (not the dequantized value) so the
// decoder need only look up the level by index.
// ============================================================================

#define TQ_MAGIC 0x54514Eu  // "TQN"
#define TQ_HEADER_BYTES 16  // 4 magic + 1 codec + 1 reserved + 2 dim + 4 scale + 4 dim_pad
#define TQ_SIGN_SEED 0x9E3779B97F4A7C15ULL

// MSE-optimal scalar quantizer levels for the standard Gaussian. Tabled here
// rather than computed at startup since the values are fixed for a given bit
// budget. Source: standard Lloyd-Max iteration converged to 1e-7.
static const float TQ2_LEVELS[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
static const float TQ3_LEVELS[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f,
};
static const float TQ4_LEVELS[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

static int tq_bits_for(ExemplarCodec codec) {
  switch (codec) {
  case EXEMPLAR_CODEC_TURBOQUANT_2B: return 2;
  case EXEMPLAR_CODEC_TURBOQUANT_3B: return 3;
  case EXEMPLAR_CODEC_TURBOQUANT_4B: return 4;
  default: return 0;
  }
}

static const float *tq_levels_for(ExemplarCodec codec) {
  switch (codec) {
  case EXEMPLAR_CODEC_TURBOQUANT_2B: return TQ2_LEVELS;
  case EXEMPLAR_CODEC_TURBOQUANT_3B: return TQ3_LEVELS;
  case EXEMPLAR_CODEC_TURBOQUANT_4B: return TQ4_LEVELS;
  default: return NULL;
  }
}

static size_t next_pow2(size_t n) {
  if (n <= 1) return 1;
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

// splitmix64 — small fixed PRNG used to derive the per-coord sign flip from
// (TQ_SIGN_SEED, dim_pad). Kept inline so we don't depend on tile.c's PRNG.
static uint64_t tq_splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// Fill `signs` with ±1.0f deterministically from (seed, dim_pad).
static void tq_fill_signs(size_t dim_pad, float *signs) {
  uint64_t s = TQ_SIGN_SEED ^ (uint64_t)dim_pad;
  uint64_t r = 0;
  int bits_left = 0;
  for (size_t i = 0; i < dim_pad; ++i) {
    if (bits_left == 0) {
      r = tq_splitmix64(&s);
      bits_left = 64;
    }
    signs[i] = (r & 1ULL) ? 1.0f : -1.0f;
    r >>= 1;
    bits_left--;
  }
}

// In-place Walsh-Hadamard transform on a buffer of size `n` (n must be a power
// of 2). Iterative butterfly, no normalization (the cosine cancels the WHT's
// scale factor; norm-preserving variants would divide by sqrt(n)).
static void tq_wht_inplace(float *v, size_t n) {
  for (size_t h = 1; h < n; h <<= 1) {
    for (size_t i = 0; i < n; i += (h << 1)) {
      for (size_t j = i; j < i + h; ++j) {
        float a = v[j];
        float b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
}

// Apply the randomized Hadamard transform: sign-flip then WHT. Caller owns
// the buffer and has already zero-padded src to dim_pad floats.
static void tq_rht_inplace(float *buf, size_t dim_pad, const float *signs) {
  for (size_t i = 0; i < dim_pad; ++i) buf[i] *= signs[i];
  tq_wht_inplace(buf, dim_pad);
}

// Quantize one rotated coord (already divided by `scale`) to the nearest
// Lloyd-Max level index. Levels are sorted ascending, so we can binary-search;
// at b<=4 a linear scan over <=16 entries is faster.
static unsigned tq_quantize(float x, const float *levels, int n_levels) {
  unsigned best = 0;
  float best_err = fabsf(x - levels[0]);
  for (int i = 1; i < n_levels; ++i) {
    float err = fabsf(x - levels[i]);
    if (err < best_err) {
      best = (unsigned)i;
      best_err = err;
    }
  }
  return best;
}

// ============================================================================
// Bit packing helpers
// ============================================================================

static void bitpack_write(uint8_t *out, size_t bit_offset, unsigned value,
                          int bits) {
  // Write `bits` LSBs of `value` starting at bit_offset, LSB-first within
  // each byte. Caller guarantees bits <= 8.
  size_t byte_idx = bit_offset >> 3;
  int bit_in_byte = (int)(bit_offset & 7u);
  unsigned mask = (1u << bits) - 1u;
  value &= mask;
  out[byte_idx] |= (uint8_t)(value << bit_in_byte);
  int written = 8 - bit_in_byte;
  if (written < bits) {
    out[byte_idx + 1] |= (uint8_t)(value >> written);
  }
}

static unsigned bitpack_read(const uint8_t *in, size_t bit_offset, int bits) {
  size_t byte_idx = bit_offset >> 3;
  int bit_in_byte = (int)(bit_offset & 7u);
  unsigned mask = (1u << bits) - 1u;
  unsigned v = (unsigned)in[byte_idx] >> bit_in_byte;
  int read = 8 - bit_in_byte;
  if (read < bits) {
    v |= (unsigned)in[byte_idx + 1] << read;
  }
  return v & mask;
}

static size_t tq_payload_size(ExemplarCodec codec, size_t dim) {
  int bits = tq_bits_for(codec);
  if (bits == 0 || dim == 0) return 0;
  size_t dim_pad = next_pow2(dim);
  size_t bit_count = dim_pad * (size_t)bits;
  size_t byte_count = (bit_count + 7u) >> 3;
  // +1 byte tail slack so bitpack_write at the last index can safely OR a
  // shifted value across the final byte boundary without checking bounds.
  return TQ_HEADER_BYTES + byte_count + 1u;
}

// ============================================================================
// Encode
// ============================================================================

static bool tq_encode(ExemplarCodec codec, const float *src, size_t dim,
                      void **out_data, size_t *out_size) {
  int bits = tq_bits_for(codec);
  const float *levels = tq_levels_for(codec);
  int n_levels = 1 << bits;
  if (!bits || !levels) return false;

  size_t dim_pad = next_pow2(dim);
  float *buf = (float *)calloc(dim_pad, sizeof(float));
  float *signs = (float *)malloc(dim_pad * sizeof(float));
  if (!buf || !signs) {
    free(buf);
    free(signs);
    return false;
  }
  for (size_t i = 0; i < dim; ++i) buf[i] = src[i];
  tq_fill_signs(dim_pad, signs);
  tq_rht_inplace(buf, dim_pad, signs);

  // Scale so quantization happens against unit-variance levels. Using the
  // sample stddev of the rotated vector gives lower MSE than peak-normalizing,
  // since Lloyd-Max levels are MSE-optimal for N(0,1).
  double sumsq = 0.0;
  for (size_t i = 0; i < dim_pad; ++i) sumsq += (double)buf[i] * (double)buf[i];
  float scale = (float)sqrt(sumsq / (double)dim_pad);
  if (scale <= 0.0f) {
    // Zero vector after rotation (only possible if input was all zeros, which
    // should already have been rejected upstream as a zero-norm exemplar).
    free(buf);
    free(signs);
    return false;
  }

  size_t payload_bytes = tq_payload_size(codec, dim);
  uint8_t *out = (uint8_t *)calloc(payload_bytes, 1);
  if (!out) {
    free(buf);
    free(signs);
    return false;
  }

  // Header (little-endian, fixed layout):
  //   [0..3]   magic
  //   [4]      codec id
  //   [5]      reserved
  //   [6..7]   dim (uint16; payload is single-tile so 65535 cap is generous)
  //   [8..11]  scale (float32)
  //   [12..15] dim_pad (uint32)
  if (dim > 0xFFFFu) {
    free(buf);
    free(signs);
    free(out);
    return false;
  }
  out[0] = (uint8_t)(TQ_MAGIC & 0xFFu);
  out[1] = (uint8_t)((TQ_MAGIC >> 8) & 0xFFu);
  out[2] = (uint8_t)((TQ_MAGIC >> 16) & 0xFFu);
  out[3] = (uint8_t)((TQ_MAGIC >> 24) & 0xFFu);
  out[4] = (uint8_t)codec;
  out[5] = 0;
  out[6] = (uint8_t)(dim & 0xFFu);
  out[7] = (uint8_t)((dim >> 8) & 0xFFu);
  memcpy(out + 8, &scale, sizeof(float));
  uint32_t dp32 = (uint32_t)dim_pad;
  out[12] = (uint8_t)(dp32 & 0xFFu);
  out[13] = (uint8_t)((dp32 >> 8) & 0xFFu);
  out[14] = (uint8_t)((dp32 >> 16) & 0xFFu);
  out[15] = (uint8_t)((dp32 >> 24) & 0xFFu);

  uint8_t *body = out + TQ_HEADER_BYTES;
  for (size_t i = 0; i < dim_pad; ++i) {
    float normalized = buf[i] / scale;
    unsigned idx = tq_quantize(normalized, levels, n_levels);
    bitpack_write(body, i * (size_t)bits, idx, bits);
  }

  free(buf);
  free(signs);
  *out_data = out;
  *out_size = payload_bytes;
  return true;
}

// ============================================================================
// Prepared query + scoring
// ============================================================================

struct ExemplarCodecQuery {
  ExemplarCodec codec;
  size_t dim;
  // RAW path: borrows the caller's pointer + caches its L2 norm.
  const float *raw_query;
  double q_norm;
  // TurboQuant path: rotated query and its L2 norm.
  size_t dim_pad;
  float *rotated;
  double rotated_norm;
  // Scratch dequantized buffer reused across exemplars (avoids per-exemplar
  // alloc inside SpatialMemory_query_similar).
  float *dequant_scratch;
};

void ExemplarCodecQuery_free(ExemplarCodecQuery *q) {
  if (!q) return;
  free(q->rotated);
  free(q->dequant_scratch);
  free(q);
}

ExemplarCodecQuery *ExemplarCodecQuery_new(ExemplarCodec codec, size_t dim,
                                           const float *query) {
  if (!query || dim == 0) return NULL;

  double sumsq = 0.0;
  for (size_t i = 0; i < dim; ++i) sumsq += (double)query[i] * (double)query[i];
  if (sumsq <= 0.0) return NULL;

  ExemplarCodecQuery *q = (ExemplarCodecQuery *)calloc(1, sizeof(*q));
  if (!q) return NULL;
  q->codec = codec;
  q->dim = dim;
  q->q_norm = sqrt(sumsq);

  switch (codec) {
  case EXEMPLAR_CODEC_RAW:
    q->raw_query = query;
    return q;

  case EXEMPLAR_CODEC_TURBOQUANT_2B:
  case EXEMPLAR_CODEC_TURBOQUANT_3B:
  case EXEMPLAR_CODEC_TURBOQUANT_4B: {
    size_t dim_pad = next_pow2(dim);
    q->dim_pad = dim_pad;
    q->rotated = (float *)calloc(dim_pad, sizeof(float));
    q->dequant_scratch = (float *)malloc(dim_pad * sizeof(float));
    float *signs = (float *)malloc(dim_pad * sizeof(float));
    if (!q->rotated || !q->dequant_scratch || !signs) {
      free(signs);
      ExemplarCodecQuery_free(q);
      return NULL;
    }
    for (size_t i = 0; i < dim; ++i) q->rotated[i] = query[i];
    tq_fill_signs(dim_pad, signs);
    tq_rht_inplace(q->rotated, dim_pad, signs);
    free(signs);
    double rsumsq = 0.0;
    for (size_t i = 0; i < dim_pad; ++i) {
      rsumsq += (double)q->rotated[i] * (double)q->rotated[i];
    }
    q->rotated_norm = sqrt(rsumsq);
    if (q->rotated_norm <= 0.0) {
      ExemplarCodecQuery_free(q);
      return NULL;
    }
    return q;
  }
  }
  ExemplarCodecQuery_free(q);
  return NULL;
}

static bool tq_cosine(const ExemplarCodecQuery *prepared, const void *payload,
                      size_t payload_size, double *out_sim) {
  if (payload_size < TQ_HEADER_BYTES + 1u) return false;
  const uint8_t *bytes = (const uint8_t *)payload;
  uint32_t magic = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
                   ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
  if (magic != TQ_MAGIC) return false;
  ExemplarCodec stored_codec = (ExemplarCodec)bytes[4];
  if (stored_codec != prepared->codec) return false;
  size_t stored_dim = (size_t)bytes[6] | ((size_t)bytes[7] << 8);
  if (stored_dim != prepared->dim) return false;
  float scale;
  memcpy(&scale, bytes + 8, sizeof(float));
  uint32_t dp32 = (uint32_t)bytes[12] | ((uint32_t)bytes[13] << 8) |
                  ((uint32_t)bytes[14] << 16) | ((uint32_t)bytes[15] << 24);
  size_t dim_pad = (size_t)dp32;
  if (dim_pad != prepared->dim_pad) return false;
  if (!(scale > 0.0f)) return false;  // NaN/Inf-safe: false if 0, negative, or NaN.

  int bits = tq_bits_for(prepared->codec);
  const float *levels = tq_levels_for(prepared->codec);
  if (!bits || !levels) return false;
  size_t expected = tq_payload_size(prepared->codec, stored_dim);
  if (payload_size < expected) return false;

  const uint8_t *body = bytes + TQ_HEADER_BYTES;
  float *dq = prepared->dequant_scratch;
  double dot = 0.0, dq_norm_sq = 0.0;
  for (size_t i = 0; i < dim_pad; ++i) {
    unsigned idx = bitpack_read(body, i * (size_t)bits, bits);
    float v = levels[idx] * scale;
    dq[i] = v;
    dq_norm_sq += (double)v * (double)v;
    dot += (double)prepared->rotated[i] * (double)v;
  }
  if (dq_norm_sq <= 0.0) return false;
  // Cosine in rotated space equals cosine in the original space because the
  // RHT is orthogonal up to the WHT's sqrt(dim_pad) factor, which cancels in
  // the ratio of dot-product to norms.
  *out_sim = dot / (prepared->rotated_norm * sqrt(dq_norm_sq));
  return true;
}

bool ExemplarCodec_cosine(const ExemplarCodecQuery *prepared,
                          const void *payload, size_t payload_size,
                          double *out_sim) {
  if (!prepared || !payload || !out_sim) return false;

  switch (prepared->codec) {
  case EXEMPLAR_CODEC_RAW: {
    if (payload_size != prepared->dim * sizeof(float)) return false;
    const float *v = (const float *)payload;
    double dot = 0.0, v_norm_sq = 0.0;
    for (size_t d = 0; d < prepared->dim; ++d) {
      double qi = (double)prepared->raw_query[d];
      double vi = (double)v[d];
      dot += qi * vi;
      v_norm_sq += vi * vi;
    }
    if (v_norm_sq <= 0.0) return false;
    *out_sim = dot / (prepared->q_norm * sqrt(v_norm_sq));
    return true;
  }

  case EXEMPLAR_CODEC_TURBOQUANT_2B:
  case EXEMPLAR_CODEC_TURBOQUANT_3B:
  case EXEMPLAR_CODEC_TURBOQUANT_4B:
    return tq_cosine(prepared, payload, payload_size, out_sim);
  }
  return false;
}

// ============================================================================
// Public encode dispatcher + name parsing
// ============================================================================

bool ExemplarCodec_encode(ExemplarCodec codec, const void *src, size_t src_size,
                          void **out_data, size_t *out_size) {
  if (!src || src_size == 0 || !out_data || !out_size) return false;

  switch (codec) {
  case EXEMPLAR_CODEC_RAW: {
    void *copy = malloc(src_size);
    if (!copy) return false;
    memcpy(copy, src, src_size);
    *out_data = copy;
    *out_size = src_size;
    return true;
  }

  case EXEMPLAR_CODEC_TURBOQUANT_2B:
  case EXEMPLAR_CODEC_TURBOQUANT_3B:
  case EXEMPLAR_CODEC_TURBOQUANT_4B: {
    if (src_size % sizeof(float) != 0) return false;
    size_t dim = src_size / sizeof(float);
    return tq_encode(codec, (const float *)src, dim, out_data, out_size);
  }
  }
  return false;
}

bool ExemplarCodec_from_string(const char *name, ExemplarCodec *out) {
  if (!name || !out) return false;
  if (strcmp(name, "raw") == 0) {
    *out = EXEMPLAR_CODEC_RAW;
    return true;
  }
  if (strcmp(name, "turboquant_2b") == 0) {
    *out = EXEMPLAR_CODEC_TURBOQUANT_2B;
    return true;
  }
  if (strcmp(name, "turboquant_3b") == 0) {
    *out = EXEMPLAR_CODEC_TURBOQUANT_3B;
    return true;
  }
  if (strcmp(name, "turboquant_4b") == 0) {
    *out = EXEMPLAR_CODEC_TURBOQUANT_4B;
    return true;
  }
  return false;
}

const char *ExemplarCodec_name(ExemplarCodec codec) {
  switch (codec) {
  case EXEMPLAR_CODEC_RAW:           return "raw";
  case EXEMPLAR_CODEC_TURBOQUANT_2B: return "turboquant_2b";
  case EXEMPLAR_CODEC_TURBOQUANT_3B: return "turboquant_3b";
  case EXEMPLAR_CODEC_TURBOQUANT_4B: return "turboquant_4b";
  }
  return "unknown";
}

size_t ExemplarCodec_payload_size(ExemplarCodec codec, size_t dim) {
  if (dim == 0) return 0;
  switch (codec) {
  case EXEMPLAR_CODEC_RAW:
    return dim * sizeof(float);
  case EXEMPLAR_CODEC_TURBOQUANT_2B:
  case EXEMPLAR_CODEC_TURBOQUANT_3B:
  case EXEMPLAR_CODEC_TURBOQUANT_4B:
    return tq_payload_size(codec, dim);
  }
  return 0;
}
