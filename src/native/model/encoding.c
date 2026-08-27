/**
 * @file encoding.c
 * @brief Sequence to residue indices, bounded by the model's max_length.
 */
#include <stdio.h>

#include "beam/internal.h"


bool beam_encode(const BeamModel *m, const char *seq,
                     uint8_t *out, uint32_t *out_len) {
    if (!seq) return false;
    uint32_t n = 0;
    for (const unsigned char *p = (const unsigned char *)seq; *p; ++p) {
        if (n >= m->max_len) return false;              /* too long */
        if (!beam_is_residue(*p)) return false;     /* not an amino acid */
        out[n++] = beam_char_to_idx[*p];
    }
    if (n == 0) return false;
    *out_len = n;
    return true;
}
