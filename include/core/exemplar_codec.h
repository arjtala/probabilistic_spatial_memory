#ifndef EXEMPLAR_CODEC_H
#define EXEMPLAR_CODEC_H

#include <stdbool.h>
#include <stddef.h>

// Pluggable encoding for per-tile exemplar embeddings. The HLL counting layer
// always sees the original observation bytes; only the reservoir-stored
// exemplar payload runs through this codec. New codecs (e.g. TurboQuant
// 2/3/4-bit) can plug in here without touching Tile_observe / RingBuffer.
typedef enum {
  EXEMPLAR_CODEC_RAW = 0,  // verbatim bytes; current default behavior.
} ExemplarCodec;

// Encode `src_size` source bytes into a reservoir-storable payload. On success
// allocates a fresh buffer (caller-owned, free with free()) and writes its
// pointer + length to *out_data / *out_size. Returns false on OOM or invalid
// inputs.
bool ExemplarCodec_encode(ExemplarCodec codec, const void *src, size_t src_size,
                          void **out_data, size_t *out_size);

// Score a stored payload against a float32 query of `dim` elements. `q_norm`
// is the precomputed L2 norm of `query`. Writes cosine similarity in [-1, 1]
// to *out_sim and returns true on a usable payload; returns false when the
// payload's encoded shape does not match `dim`, the payload has zero norm,
// or the codec is unknown.
bool ExemplarCodec_cosine(ExemplarCodec codec, const float *query, size_t dim,
                          double q_norm, const void *payload,
                          size_t payload_size, double *out_sim);

#endif
