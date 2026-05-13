#ifndef EXEMPLAR_CODEC_H
#define EXEMPLAR_CODEC_H

#include <stdbool.h>
#include <stddef.h>

// Pluggable encoding for per-tile exemplar embeddings. The HLL counting layer
// always sees the original observation bytes; only the reservoir-stored
// exemplar payload runs through this codec. New codecs (e.g. TurboQuant
// 2/3/4-bit) plug in here without touching Tile_observe / RingBuffer.
//
// TurboQuant variants apply a randomized Hadamard rotation (sign-flip + WHT)
// on the embedding padded to the next power of 2, then per-coordinate scalar
// quantization with Lloyd-Max-optimal levels for a unit-variance Gaussian.
// Cosine similarity is preserved up to quantization noise; the dequantized
// vector is never reconstructed end-to-end during scoring.
typedef enum {
  EXEMPLAR_CODEC_RAW = 0,        // verbatim bytes; pre-codec default.
  EXEMPLAR_CODEC_TURBOQUANT_2B,  // 2 bits/coord on padded dim
  EXEMPLAR_CODEC_TURBOQUANT_3B,  // 3 bits/coord
  EXEMPLAR_CODEC_TURBOQUANT_4B,  // 4 bits/coord
} ExemplarCodec;

// Opaque per-query workspace. Built once per SpatialMemory_query_similar call
// (which may score across many tiles), then handed to ExemplarCodec_cosine for
// each exemplar. For TurboQuant this carries the rotated query so the n log n
// transform is paid once, not per-exemplar.
typedef struct ExemplarCodecQuery ExemplarCodecQuery;

// Encode `src_size` source bytes into a reservoir-storable payload. On success
// allocates a fresh buffer (caller-owned, free with free()) and writes its
// pointer + length to *out_data / *out_size. Returns false on OOM, invalid
// inputs, or unknown codec. For TurboQuant codecs `src_size` must be a positive
// multiple of `sizeof(float)`.
bool ExemplarCodec_encode(ExemplarCodec codec, const void *src, size_t src_size,
                          void **out_data, size_t *out_size);

// Build a prepared query for `dim` float32 elements. Returns NULL on OOM,
// invalid inputs (dim == 0, NULL query, zero-norm query), or unknown codec.
// For RAW the prepared object borrows `query`; the caller must keep it alive
// until ExemplarCodecQuery_free returns.
ExemplarCodecQuery *ExemplarCodecQuery_new(ExemplarCodec codec, size_t dim,
                                           const float *query);
void ExemplarCodecQuery_free(ExemplarCodecQuery *q);

// Score a stored payload against a prepared query. Writes cosine similarity in
// [-1, 1] to *out_sim and returns true on a usable payload; returns false when
// the payload's encoded shape does not match the prepared query's dim, the
// payload yields a zero-norm vector, or the codec is unknown.
bool ExemplarCodec_cosine(const ExemplarCodecQuery *prepared,
                          const void *payload, size_t payload_size,
                          double *out_sim);

// CLI <-> enum helpers. Names: "raw", "turboquant_2b", "turboquant_3b",
// "turboquant_4b".
bool ExemplarCodec_from_string(const char *name, ExemplarCodec *out);
const char *ExemplarCodec_name(ExemplarCodec codec);

// Bytes the codec emits per encoded exemplar at `dim` float32 inputs. Returns
// 0 on unknown codec or invalid dim. Useful for E9-style memory accounting.
size_t ExemplarCodec_payload_size(ExemplarCodec codec, size_t dim);

#endif
