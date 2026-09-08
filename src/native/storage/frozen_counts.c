/**
 * @file frozen_counts.c
 * @brief Freezing and thawing count tables. See frozen_counts.h for why.
 */
#include "beam/internal.h"
#include "beam/frozen_counts.h"

/* ── Bits ────────────────────────────────────────────────────── */

/* Values are packed with no padding, so one can straddle a byte and a row can
 * start mid-byte. These are the only two places that knows, and both work a
 * byte at a time rather than assuming an alignment the packing does not have. */

/*
 * A value can straddle a byte and a row can start mid-byte, so both of these
 * work on the two 64-bit words a value can touch rather than on one bit at a
 * time. The bit-at-a-time version of this was 23% of a foundation load.
 *
 * Reading eight bytes from an arbitrary offset runs past the last value, so
 * the payload carries BEAM_FROZEN_PAD bytes of slack. They are never read as
 * data -- `n_value_bits` says where the values stop.
 */
#define BEAM_FROZEN_PAD 16u

static inline void put_bits(unsigned char *dst, uint64_t at, uint64_t v,
                            uint32_t width) {
    if (width == 0u) return;
    const uint64_t byte = at >> 3;
    const uint32_t off = (uint32_t)(at & 7u);
    uint64_t lo;
    memcpy(&lo, dst + byte, sizeof lo);
    lo |= v << off;
    memcpy(dst + byte, &lo, sizeof lo);
    if (off != 0u && off + width > 64u) {
        uint64_t hi;
        memcpy(&hi, dst + byte + 8u, sizeof hi);
        hi |= v >> (64u - off);
        memcpy(dst + byte + 8u, &hi, sizeof hi);
    }
}

static inline uint64_t get_bits(const unsigned char *src, uint64_t at,
                                uint32_t width) {
    if (width == 0u) return 0u;
    const uint64_t byte = at >> 3;
    const uint32_t off = (uint32_t)(at & 7u);
    uint64_t lo;
    memcpy(&lo, src + byte, sizeof lo);
    uint64_t v = lo >> off;
    if (off != 0u && off + width > 64u) {
        uint64_t hi;
        memcpy(&hi, src + byte + 8u, sizeof hi);
        v |= hi << (64u - off);
    }
    return width == 64u ? v : (v & ((1ull << width) - 1u));
}

/* A cell is present when its bits are not those of +0.0. Bitwise, not
 * `!= 0`, so a -0.0 survives the round trip as the distinct value it is. */
static inline bool cell_present(double v) {
    uint64_t bits, zero = 0u;
    const double pz = 0.0;
    memcpy(&bits, &v, sizeof bits);
    memcpy(&zero, &pz, sizeof zero);
    return bits != zero;
}

static inline uint64_t bits_of(double v) {
    uint64_t b; memcpy(&b, &v, sizeof b); return b;
}

/**
 * The mask and width one row needs.
 *
 * An integer width is offered only to a whole number of at least one that
 * binary64 holds exactly, so a fraction, a negative, a -0.0, a NaN or
 * anything past 2^53 all fall to the escape and keep their own bits.
 */
static uint32_t row_header(const double *row, uint32_t stride,
                           uint32_t *out_present) {
    uint32_t mask = 0u, present = 0u, width = 1u;
    bool integral = true;
    for (uint32_t x = 0; x < stride; ++x) {
        if (!cell_present(row[x])) continue;
        mask |= 1u << x;
        present++;
        const double v = row[x];
        if (!(v >= 1.0) || v != floor(v) || v > 9007199254740992.0) {
            integral = false;
            continue;
        }
        if (!integral) continue;
        const uint64_t u = (uint64_t)v;
        const uint32_t need = 64u - (uint32_t)__builtin_clzll(u);
        if (need > width) width = need;
    }
    if (!integral) width = BEAM_FROZEN_WIDTH_F64;
    *out_present = present;
    return mask | (width << BEAM_FROZEN_WIDTH_SHIFT);
}

static inline uint32_t header_mask(uint32_t h) {
    return h & ((1u << BEAM_FROZEN_MASK_BITS) - 1u);
}
static inline uint32_t header_width(uint32_t h) {
    return (h >> BEAM_FROZEN_WIDTH_SHIFT) & BEAM_FROZEN_WIDTH_MASK;
}
/** Bits one row's values occupy. */
static inline uint64_t row_bits(uint32_t h) {
    const uint32_t w = header_width(h);
    return (uint64_t)__builtin_popcount(header_mask(h))
         * (w == BEAM_FROZEN_WIDTH_F64 ? 64u : w);
}

static void encode_row(const double *row, uint32_t stride, uint32_t h,
                       unsigned char *dst, uint64_t at) {
    const uint32_t mask = header_mask(h), width = header_width(h);
    const bool raw = (width == BEAM_FROZEN_WIDTH_F64);
    for (uint32_t x = 0; x < stride; ++x) {
        if (!((mask >> x) & 1u)) continue;
        const uint64_t v = raw ? bits_of(row[x]) : (uint64_t)row[x];
        put_bits(dst, at, v, raw ? 64u : width);
        at += raw ? 64u : width;
    }
}

static void decode_row(const BeamFrozenCounts *f, uint32_t h, uint64_t at,
                       double *out) {
    const uint32_t mask = header_mask(h), width = header_width(h);
    const bool raw = (width == BEAM_FROZEN_WIDTH_F64);
    const uint32_t w = raw ? 64u : width;
    for (uint32_t x = 0; x < f->stride; ++x) {
        if (!((mask >> x) & 1u)) { out[x] = 0.0; continue; }
        const uint64_t v = get_bits(f->values, at, w);
        if (raw) memcpy(&out[x], &v, sizeof out[x]);
        else out[x] = (double)v;
        at += w;
    }
}

/* ── Walking ─────────────────────────────────────────────────── */

void beam_frozen_cursor_init(BeamFrozenCursor *c, const BeamFrozenCounts *f) {
    c->f = f; c->k = 0u; c->widx = 0u; c->bit = 0u;
    c->bits = (f && f->present && f->n_dense_rows) ? f->present[0] : 0u;
}

bool beam_frozen_cursor_next(BeamFrozenCursor *c, uint32_t *row, double *out) {
    const BeamFrozenCounts *f = c->f;
    if (!f || c->k >= f->n_rows) return false;
    uint32_t id;
    if (f->row_id) {
        id = f->row_id[c->k];
    } else {
        const uint32_t words = beam_frozen_bitwords(f->n_dense_rows);
        while (c->bits == 0u) {
            if (++c->widx >= words) return false;
            c->bits = f->present[c->widx];
        }
        const uint32_t b = (uint32_t)__builtin_ctzll(c->bits);
        c->bits &= c->bits - 1u;                   /* clear the lowest set bit */
        id = c->widx * 64u + b;
    }
    const uint32_t h = f->mask_width[c->k];
    if (row) *row = id;
    if (out) decode_row(f, h, c->bit, out);
    c->bit += row_bits(h);
    c->k++;
    return true;
}

/** Where block `blk` starts, and which row that is. */
static void seek_block(const BeamFrozenCounts *f, uint32_t k,
                       uint32_t *out_first, uint64_t *out_bit) {
    const uint32_t blk = k / BEAM_FROZEN_BLOCK_ROWS;
    *out_first = blk * BEAM_FROZEN_BLOCK_ROWS;
    *out_bit = f->block_bit[blk];
}

void beam_frozen_row(const BeamFrozenCounts *f, uint32_t k, double *out) {
    if (!f || k >= f->n_rows) return;
    uint32_t i; uint64_t at;
    seek_block(f, k, &i, &at);
    for (; i < k; ++i) at += row_bits(f->mask_width[i]);
    decode_row(f, f->mask_width[k], at, out);
}

uint32_t beam_frozen_row_id(const BeamFrozenCounts *f, uint32_t k) {
    if (!f || k >= f->n_rows) return BEAM_ROW_MISSING;
    if (f->row_id) return f->row_id[k];
    /* Select the (k+1)-th set bit, starting from the block's first row so the
     * scan is bounded by the block rather than by the table. */
    const uint32_t first = (k / BEAM_FROZEN_BLOCK_ROWS) * BEAM_FROZEN_BLOCK_ROWS;
    const uint32_t words = beam_frozen_bitwords(f->n_dense_rows);
    uint32_t seen = 0u;
    for (uint32_t w = 0; w < words; ++w) {
        uint64_t bits = f->present[w];
        const uint32_t pc = (uint32_t)__builtin_popcountll(bits);
        if (seen + pc <= first) { seen += pc; continue; }
        while (bits) {
            const uint32_t b = (uint32_t)__builtin_ctzll(bits);
            if (seen == k) return w * 64u + b;
            bits &= bits - 1u;
            seen++;
        }
    }
    return BEAM_ROW_MISSING;
}

/* ── Lifecycle ───────────────────────────────────────────────── */

void beam_frozen_destroy(BeamFrozenCounts *f) {
    if (!f) return;
    free(f->present); free(f->row_id);
    free(f->mask_width); free(f->block_bit); free(f->values);
    memset(f, 0, sizeof *f);
}

uint64_t beam_frozen_memory_bytes(const BeamFrozenCounts *f) {
    if (!f) return 0u;
    return (f->present
            ? (uint64_t)beam_frozen_bitwords(f->n_dense_rows) * sizeof(uint64_t)
            : (uint64_t)f->n_rows * sizeof(uint32_t))
         + (uint64_t)f->n_rows * sizeof(uint32_t)
         + (uint64_t)(f->n_blocks + 1u) * sizeof(uint64_t)
         + (f->n_value_bits + 7u) / 8u
         + sizeof(BeamFrozenCounts);
}

typedef struct { uint32_t id; const double *values; } FrozenRow;
static int cmp_frozen_row(const void *a, const void *b) {
    const uint32_t x = ((const FrozenRow *)a)->id;
    const uint32_t y = ((const FrozenRow *)b)->id;
    return (x > y) - (x < y);
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
    out->n_dense_rows = src->n_dense_rows;

    uint64_t total = 0;
    for (uint32_t p = 0; p < n_parts && p < src->n_parts; ++p)
        total += src->size ? src->size[p] : 0u;
    if (total == 0u) return BEAM_OK;
    if (total > UINT32_MAX) return BEAM_ERANGE("%s: too many rows", __func__);
    const uint32_t n = (uint32_t)total;

    /* Rows ascending, each carrying its payload. Observation filled the
     * partitions in arrival order; that order is discarded here, which is safe
     * because nothing reads a count table by slot -- write_count_table walks
     * dense rows and finalize sorts by destination. */
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

    const uint32_t words = beam_frozen_bitwords(src->n_dense_rows);
    const uint32_t n_blocks = beam_frozen_blocks(n);
    const bool bitmap = beam_frozen_prefers_bitmap(src->n_dense_rows, n);
    uint64_t *present = bitmap
        ? (uint64_t *)calloc(words, sizeof *present) : NULL;
    uint32_t *ids = bitmap
        ? NULL : (uint32_t *)malloc((size_t)n * sizeof *ids);
    uint32_t *mask_width = (uint32_t *)malloc((size_t)n * sizeof *mask_width);
    uint64_t *block_bit = (uint64_t *)malloc(((size_t)n_blocks + 1u)
                                             * sizeof *block_bit);
    if ((bitmap ? (void *)present : (void *)ids) == NULL
        || !mask_width || !block_bit) {
        free(rows); free(present); free(ids); free(mask_width); free(block_bit);
        return BEAM_EALLOC("the frozen row index");
    }

    /* Pass one: headers, the presence bitmap, and where each block starts. */
    uint64_t bits = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (i % BEAM_FROZEN_BLOCK_ROWS == 0u)
            block_bit[i / BEAM_FROZEN_BLOCK_ROWS] = bits;
        const uint32_t id = rows[i].id;
        if (id >= src->n_dense_rows) {
            free(rows); free(present); free(ids); free(mask_width);
            free(block_bit);
            return BEAM_ERANGE("%s: row %u is outside the table", __func__, id);
        }
        if (bitmap) present[id >> 6] |= 1ull << (id & 63u);
        else ids[i] = id;
        uint32_t np = 0;
        mask_width[i] = row_header(rows[i].values, stride, &np);
        bits += row_bits(mask_width[i]);
    }
    block_bit[n_blocks] = bits;

    const size_t vbytes = (size_t)((bits + 7u) / 8u);
    unsigned char *values = (unsigned char *)calloc(vbytes + BEAM_FROZEN_PAD, 1u);
    if (!values) {
        free(rows); free(present); free(ids); free(mask_width); free(block_bit);
        return BEAM_EALLOC("the frozen value payload");
    }

    /* Pass two: the values. calloc above matters -- put_bits only ever sets
     * bits, so the zeros it leaves are the zero bits of every value. */
    uint64_t at = 0;
    for (uint32_t i = 0; i < n; ++i) {
        encode_row(rows[i].values, stride, mask_width[i], values, at);
        at += row_bits(mask_width[i]);
    }
    free(rows);

    out->present = present; out->row_id = ids; out->mask_width = mask_width;
    out->block_bit = block_bit; out->values = values;
    out->n_rows = n; out->n_blocks = n_blocks; out->n_value_bits = bits;
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

    BeamFrozenCursor c;
    beam_frozen_cursor_init(&c, src);
    uint32_t id = 0;
    while (beam_frozen_cursor_next(&c, &id, row)) {
        double *dst = beam_sparse_row_mut(out, id, part_of(id, ctx), true);
        if (!dst) { free(row); beam_sparse_destroy(out);
                    return BEAM_EALLOC("a thawed count row"); }
        memcpy(dst, row, (size_t)src->stride * sizeof *row);
    }
    free(row);
    beam_sparse_shrink(out);
    return BEAM_OK;
}
