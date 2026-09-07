/**
 * @file packed_rows.c
 * @brief Building the dictionary form of a finalized payload.
 *
 * See `cell_mask` in sparse_rows.h for what this exploits and why it is
 * exact. The build is two passes over the payload: one to learn the distinct
 * values and how many cells exist, one to write the indices.
 */
#include "beam/internal.h"

/* A value is absent exactly when it is -inf. Compared by bits, because that
 * is also how the dictionary decides two values are the same one. */
static inline bool cell_finite(double v) {
    return v != -INFINITY;
}

static inline uint64_t bits_of(double v) {
    uint64_t b; memcpy(&b, &v, sizeof b); return b;
}

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}

BeamError beam_sparse_pack_cells(BeamSparseRows *t) {
    if (!t) return BEAM_ENULL("sparse table");
    /* Like sealing and interning: a table that cannot take this keeps what it
     * has and reports OK. */
    if (t->mutable_rows || t->cell_mask || !t->values || !t->values[0]
        || !t->size || !t->capacity)
        return BEAM_OK;
    const uint32_t rows = t->n_pool ? t->n_pool : t->size[0];
    const uint32_t R = t->stride;
    if (rows == 0u || R == 0u || R > 32u) return BEAM_OK;

    const double *src = t->values[0];

    /* Pass one: the distinct values, and how many cells exist.
     *
     * Open addressing over slots that hold "value index + 1", so zero means
     * empty and the table needs no separate occupancy array. Sized for the
     * worst case -- every cell distinct -- and abandoned if that many really
     * do turn up, because then this form would cost more than it saves. */
    uint64_t cells = 0;
    for (uint32_t s = 0; s < rows; ++s)
        for (uint32_t x = 0; x < R; ++x)
            if (cell_finite(src[(size_t)s * R + x])) cells++;
    if (cells == 0u) return BEAM_OK;

    size_t slots = 1u;
    while (slots < (size_t)BEAM_PACK_MAX_DICT * 2u) slots <<= 1;
    uint32_t *ht = (uint32_t *)calloc(slots, sizeof *ht);
    double *dict = (double *)malloc((size_t)BEAM_PACK_MAX_DICT * sizeof *dict);
    if (!ht || !dict) { free(ht); free(dict);
                        return BEAM_EALLOC("the value-dictionary scratch"); }

    uint32_t n_dict = 0;
    bool too_many = false;
    for (uint32_t s = 0; s < rows && !too_many; ++s)
        for (uint32_t x = 0; x < R; ++x) {
            const double v = src[(size_t)s * R + x];
            if (!cell_finite(v)) continue;
            const uint64_t b = bits_of(v);
            size_t j = (size_t)mix64(b) & (slots - 1u);
            for (;;) {
                const uint32_t held = ht[j];
                if (held == 0u) {
                    if (n_dict == BEAM_PACK_MAX_DICT) { too_many = true; break; }
                    dict[n_dict] = v;
                    ht[j] = ++n_dict;          /* stored as index + 1 */
                    break;
                }
                if (bits_of(dict[held - 1u]) == b) break;
                j = (j + 1u) & (slots - 1u);
            }
            if (too_many) break;
        }
    if (too_many) { free(ht); free(dict); return BEAM_OK; }

    /* Would this actually be smaller? A table whose values barely repeat is
     * better left alone rather than given an indirection for nothing. */
    const uint64_t now = (uint64_t)rows * R * sizeof(double);
    const uint64_t packed = (uint64_t)rows * sizeof(uint32_t)
                          + ((uint64_t)rows + 1u) * sizeof(uint32_t)
                          + cells * sizeof(uint16_t)
                          + (uint64_t)n_dict * sizeof(double);
    if (packed >= now) { free(ht); free(dict); return BEAM_OK; }

    uint32_t *mask  = (uint32_t *)malloc((size_t)rows * sizeof *mask);
    uint32_t *start = (uint32_t *)malloc(((size_t)rows + 1u) * sizeof *start);
    uint16_t *index = (uint16_t *)malloc((size_t)cells * sizeof *index);
    double   *final = (double *)malloc((size_t)n_dict * sizeof *final);
    if (!mask || !start || !index || !final) {
        free(ht); free(dict); free(mask); free(start); free(index); free(final);
        return BEAM_EALLOC("the packed payload");
    }
    memcpy(final, dict, (size_t)n_dict * sizeof *final);

    /* Pass two: the mask and the indices, in ascending residue order so a
     * whole-row read walks them without computing a rank per residue. */
    uint64_t at = 0;
    for (uint32_t s = 0; s < rows; ++s) {
        start[s] = (uint32_t)at;
        uint32_t mk = 0;
        for (uint32_t x = 0; x < R; ++x) {
            const double v = src[(size_t)s * R + x];
            if (!cell_finite(v)) continue;
            mk |= 1u << x;
            const uint64_t b = bits_of(v);
            size_t j = (size_t)mix64(b) & (slots - 1u);
            while (bits_of(final[ht[j] - 1u]) != b) j = (j + 1u) & (slots - 1u);
            index[at++] = (uint16_t)(ht[j] - 1u);
        }
        mask[s] = mk;
    }
    start[rows] = (uint32_t)at;
    free(ht); free(dict);

    t->cell_mask = mask; t->cell_start = start; t->cell_index = index;
    t->dict = final; t->n_dict = n_dict; t->n_cell_rows = rows;

    /* Release the array this replaces. Installed first, so a table is never
     * without a payload: after this point `cell_mask` is what answers, and
     * `values0` being NULL is what says so. */
    free(t->values[0]);
    t->values[0] = NULL;
    t->values0 = NULL;
    t->capacity[0] = 0u;
    return BEAM_OK;
}
