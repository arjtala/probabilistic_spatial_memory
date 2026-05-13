#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "core/exemplar_codec.h"

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
  }
  return false;
}

bool ExemplarCodec_cosine(ExemplarCodec codec, const float *query, size_t dim,
                          double q_norm, const void *payload,
                          size_t payload_size, double *out_sim) {
  if (!query || dim == 0 || !payload || !out_sim || q_norm <= 0.0) {
    return false;
  }

  switch (codec) {
  case EXEMPLAR_CODEC_RAW: {
    if (payload_size != dim * sizeof(float)) return false;
    const float *v = (const float *)payload;
    double dot = 0.0, v_norm_sq = 0.0;
    for (size_t d = 0; d < dim; ++d) {
      double qi = (double)query[d];
      double vi = (double)v[d];
      dot += qi * vi;
      v_norm_sq += vi * vi;
    }
    if (v_norm_sq <= 0.0) return false;
    *out_sim = dot / (q_norm * sqrt(v_norm_sq));
    return true;
  }
  }
  return false;
}
