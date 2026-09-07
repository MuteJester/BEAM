/**
 * @file frozen_counts.c
 * @brief Freezing and thawing count tables. See frozen_counts.h for why.
 */
#include "beam/internal.h"
#include "beam/frozen_counts.h"

/* A cell is present when its bits are not those of +0.0. Bitwise, not `!= 0`,
 * so a -0.0 survives the round trip as the distinct value it is. */
static inline bool cell_present(double v) {
    uint64_t bits, zero = 0u;
    const double pz = 0.0;
    memcpy(&bits, &v, sizeof bits);
    memcpy(&zero, &pz, sizeof zero);
    return bits != zero;
}

/**
 * The narrowest tier that stores every present cell of this row EXACTLY.
 *
 * An integer tier is offered only to a value that is a whole number of at
 * least one, so a fraction, a negative, a -0.0 or a NaN all fall through to
 * binary64 and are stored as the bits they already are. That is what keeps
 * fractional abundances and posterior pseudo-counts exact.
 */
static uint32_t row_tier(const double *row, uint32_t stride, uint32_t *out_mask,
                         uint32_t *out_present) {
    uint32_t mask = 0u, present = 0u, tier = BEAM_FROZEN_U8;
    for (uint32_t x = 0; x < stride; ++x) {
        if (!cell_present(row[x])) continue;
        mask |= 1u << x;
        present++;
        const double v = row[x];
        if (!(v >= 1.0) || v != floor(v)) { tier = BEAM_FROZEN_F64; continue; }
        if (tier == BEAM_FROZEN_F64) continue;
        const uint32_t need = (v <= 255.0) ? BEAM_FROZEN_U8
                            : (v <= 65535.0) ? BEAM_FROZEN_U16
                            : (v <= 4294967295.0) ? BEAM_FROZEN_U32
                            : BEAM_FROZEN_F64;
        if (need > tier) tier = need;
    }
    *out_mask = mask;
    *out_present = present;
    return tier;
}

static void encode_row(const double *row, uint32_t stride, uint32_t mask,
                       uint32_t tier, unsigned char *dst) {
    for (uint32_t x = 0; x < stride; ++x) {
        if (!(mask & (1u << x))) continue;
        const double v = row[x];
        switch (tier) {
        case BEAM_FROZEN_U8:  { uint8_t  t = (uint8_t)v;  memcpy(dst, &t, 1); dst += 1; break; }
        case BEAM_FROZEN_U16: { uint16_t t = (uint16_t)v; memcpy(dst, &t, 2); dst += 2; break; }
        case BEAM_FROZEN_U32: { uint32_t t = (uint32_t)v; memcpy(dst, &t, 4); dst += 4; break; }
        default:              {                           memcpy(dst, &v, 8); dst += 8; break; }
        }
    }
}

void beam_frozen_row(const BeamFrozenCounts *f, uint32_t k, double *out) {
    const uint32_t mt = f->mask_tier[k];
    const uint32_t mask = mt & ((1u << BEAM_FROZEN_MASK_BITS) - 1u);
    const uint32_t tier = mt >> BEAM_FROZEN_TIER_SHIFT;
    const unsigned char *src = f->values
                             + f->block_base[k / BEAM_FROZEN_BLOCK_ROWS]
                             + f->sub_offset[k];
    for (uint32_t x = 0; x < f->stride; ++x) {
        if (!(mask & (1u << x))) { out[x] = 0.0; continue; }
        switch (tier) {
        case BEAM_FROZEN_U8:  { uint8_t  t; memcpy(&t, src, 1); src += 1; out[x] = (double)t; break; }
        case BEAM_FROZEN_U16: { uint16_t t; memcpy(&t, src, 2); src += 2; out[x] = (double)t; break; }
        case BEAM_FROZEN_U32: { uint32_t t; memcpy(&t, src, 4); src += 4; out[x] = (double)t; break; }
        default:              { double   t; memcpy(&t, src, 8); src += 8; out[x] = t;         break; }
        }
    }
}

/* One occupied row, carried with its payload so freezing never has to ask
 * which partition owns it -- that is a fact about the model's table kind,
 * not about the table. */
typedef struct { uint32_t id; const double *values; } FrozenRow;

static int cmp_frozen_row(const void *a, const void *b) {
    const uint32_t x = ((const FrozenRow *)a)->id;
    const uint32_t y = ((const FrozenRow *)b)->id;
    return (x > y) - (x < y);
}

void beam_frozen_destroy(BeamFrozenCounts *f) {
    if (!f) return;
    free(f->row_id); free(f->mask_tier); free(f->sub_offset);
    free(f->block_base); free(f->values);
    memset(f, 0, sizeof *f);
}

uint64_t beam_frozen_memory_bytes(const BeamFrozenCounts *f) {
    if (!f) return 0u;
    return (uint64_t)f->n_rows * (sizeof(uint32_t) + sizeof(uint32_t)
                                  + sizeof(uint16_t))
         + (uint64_t)(f->n_blocks + 1u) * sizeof(uint64_t)
         + f->n_value_bytes
         + sizeof(BeamFrozenCounts);
}

BeamError beam_frozen_freeze(const BeamSparseRows *src, uint32_t n_parts,
                             BeamFrozenCounts *out) {
    if (!src || !out) return BEAM_ENULL("frozen counts");
    memset(out, 0, sizeof *out);
    const uint32_t stride = src->stride;
    if (stride == 0u || stride > BEAM_FROZEN_MASK_BITS)
        return BEAM_ERANGE("%s: stride %u does not fit a %u-bit mask",
                           __func__, stride, BEAM_FROZEN_MASK_BITS);
    out->stride = stride;

    uint64_t total = 0;
    for (uint32_t p = 0; p < n_parts && p < src->n_parts; ++p)
        total += src->size ? src->size[p] : 0u;
    if (total == 0u) return BEAM_OK;
    if (total > UINT32_MAX) return BEAM_ERANGE("%s: too many rows", __func__);
    const uint32_t n = (uint32_t)total;

    /* Rows in ascending dense order, each carrying its payload. Observation
     * filled the partitions in arrival order, so this is where that order is
     * discarded -- safe because nothing reads a count table by slot:
     * write_count_table walks dense rows and finalize sorts by destination. */
    FrozenRow *rows = (FrozenRow *)malloc((size_t)n * sizeof *rows);
    if (!rows) return BEAM_EALLOC("the frozen row list");
    uint32_t k = 0;
    for (uint32_t p = 0; p < n_parts && p < src->n_parts; ++p) {
        if (!src->row_ids || !src->row_ids[p] || !src->values
            || !src->values[p]) continue;
        for (uint32_t s = 0; s < src->size[p]; ++s) {
            rows[k].id = src->row_ids[p][s];
            rows[k].values = src->values[p] + (size_t)s * stride;
            k++;
        }
    }
    if (k != n) { free(rows); return BEAM_ERANGE("%s: listed %u of %u rows",
                                                 __func__, k, n); }
    qsort(rows, n, sizeof *rows, cmp_frozen_row);

    uint32_t *ids = (uint32_t *)malloc((size_t)n * sizeof *ids);
    if (!ids) { free(rows); return BEAM_EALLOC("the frozen row list"); }
    for (uint32_t i = 0; i < n; ++i) ids[i] = rows[i].id;

    const uint32_t n_blocks = (n + BEAM_FROZEN_BLOCK_ROWS - 1u)
                            / BEAM_FROZEN_BLOCK_ROWS;
    uint32_t *mask_tier = (uint32_t *)malloc((size_t)n * sizeof *mask_tier);
    uint16_t *sub = (uint16_t *)malloc((size_t)n * sizeof *sub);
    uint64_t *base = (uint64_t *)malloc(((size_t)n_blocks + 1u) * sizeof *base);
    if (!mask_tier || !sub || !base) {
        free(rows); free(ids); free(mask_tier); free(sub); free(base);
        return BEAM_EALLOC("the frozen row index");
    }

    /* Pass one: classify each row and lay out its bytes. */
    uint64_t bytes = 0, block_start = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (i % BEAM_FROZEN_BLOCK_ROWS == 0u) {
            base[i / BEAM_FROZEN_BLOCK_ROWS] = bytes;
            block_start = bytes;
        }
        const double *row = rows[i].values;
        uint32_t mask = 0, present = 0;
        const uint32_t tier = row_tier(row, stride, &mask, &present);
        mask_tier[i] = mask | (tier << BEAM_FROZEN_TIER_SHIFT);
        const uint64_t off = bytes - block_start;
        if (off > UINT16_MAX) { free(rows); free(ids); free(mask_tier);
            free(sub); free(base);
            return BEAM_ERANGE("%s: block offset %llu overflows",
                               __func__, (unsigned long long)off); }
        sub[i] = (uint16_t)off;
        bytes += (uint64_t)present * beam_frozen_tier_width(tier);
    }
    base[n_blocks] = bytes;

    unsigned char *values = (unsigned char *)malloc(bytes ? (size_t)bytes : 1u);
    if (!values) { free(rows); free(ids); free(mask_tier); free(sub); free(base);
                   return BEAM_EALLOC("the frozen value payload"); }

    /* Pass two: write them. */
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t mt = mask_tier[i];
        encode_row(rows[i].values, stride,
                   mt & ((1u << BEAM_FROZEN_MASK_BITS) - 1u),
                   mt >> BEAM_FROZEN_TIER_SHIFT,
                   values + base[i / BEAM_FROZEN_BLOCK_ROWS] + sub[i]);
    }
    free(rows);

    out->row_id = ids; out->mask_tier = mask_tier; out->sub_offset = sub;
    out->block_base = base; out->values = values;
    out->n_rows = n; out->n_blocks = n_blocks; out->n_value_bytes = bytes;
    return BEAM_OK;
}

BeamError beam_frozen_thaw(const BeamFrozenCounts *src, uint32_t n_dense_rows,
                           uint32_t n_parts,
                           uint32_t (*part_of)(uint32_t row, void *ctx),
                           void *ctx, BeamSparseRows *out) {
    if (!src || !out || !part_of) return BEAM_ENULL("frozen counts");
    BeamError e = beam_sparse_init(out, n_dense_rows, n_parts, src->stride, true);
    if (e != BEAM_OK) return e;
    double *row = (double *)malloc((size_t)src->stride * sizeof *row);
    if (!row) { beam_sparse_destroy(out); return BEAM_EALLOC("a thaw buffer"); }
    for (uint32_t i = 0; i < src->n_rows; ++i) {
        beam_frozen_row(src, i, row);
        const uint32_t id = src->row_id[i];
        double *dst = beam_sparse_row_mut(out, id, part_of(id, ctx), true);
        if (!dst) { free(row); beam_sparse_destroy(out);
                    return BEAM_EALLOC("a thawed count row"); }
        memcpy(dst, row, (size_t)src->stride * sizeof *row);
    }
    free(row);
    beam_sparse_shrink(out);
    return BEAM_OK;
}
