/**
 * @file encoding.c
 * @brief Sequence to residue indices, bounded by the model's max_length.
 */
#include <stdio.h>

#include "beam/internal.h"


bool beam_encode(const BeamModel *m, const char *seq,
                     uint8_t *out, uint32_t *out_len) {
    if (!seq) return false;
    /* One lookup decides both the index and the validity, because the
     * descriptor's table marks a non-residue rather than reporting 0 for it.
     * The amino-acid table this replaced could not: it mapped every unknown
     * byte to 0, which is 'A', so it needed a separate predicate beside it. */
    const uint8_t *to_idx = m->alpha->char_to_idx;
    uint32_t n = 0;
    for (const unsigned char *p = (const unsigned char *)seq; *p; ++p) {
        if (n >= m->max_len) return false;              /* too long */
        const uint8_t ix = to_idx[*p];
        if (ix == BEAM_RESIDUE_INVALID) return false;   /* not in this alphabet */
        out[n++] = ix;
    }
    if (n == 0) return false;
    *out_len = n;
    return true;
}

bool beam_encode_n(const BeamModel *m, const char *seq, size_t n,
                   uint8_t *out, uint32_t *out_len) {
    if (!seq && n) return false;
    if (n == 0 || n > m->max_len) return false;         /* empty, too long */
    const uint8_t *to_idx = m->alpha->char_to_idx;
    const unsigned char *p = (const unsigned char *)seq;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t ix = to_idx[p[i]];
        if (ix == BEAM_RESIDUE_INVALID) return false;    /* not in this alphabet */
        out[i] = ix;
    }
    *out_len = (uint32_t)n;
    return true;
}
