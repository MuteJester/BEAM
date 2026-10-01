/**
 * @file model_file.c
 * @brief Persistence for BEAM models.
 *
 * Counts are written, not the derived log-probability tables. A loaded model
 * is therefore a first-class one: it can still be observed into, subtracted
 * from, and re-finalized under different prior strengths. Writing the tables
 * instead would produce an artefact that could only be scored against. A load
 * re-finalizes from the counts -- under the prior strengths the header stores
 * when the model was finalized as written (`flags`), since estimating them
 * again from the same counts only reproduces them.
 *
 * Layout: the fixed header below, then the two count tables, then the length
 * histogram, then a CRC-32C of everything before it. A dense table is its flat
 * double block; a sparse one is the frozen encoding, written as it stands in
 * memory. Little-endian throughout, which is every platform this library
 * targets.
 *
 * ONE format is read: the one this build writes. BEAM has not been released,
 * so no older file is owed a reader, and a file from an earlier development
 * build is refused with a sentence saying so rather than read through a
 * compatibility path nobody maintains.
 */
#include <stdio.h>
#include <stddef.h>

#include "beam/crc32c.h"
#include "beam/internal.h"
#include "beam/memory.h"

/* 0x4245414D is 'B','E','A','M' as ASCII, most significant byte first. */
#define BEAM_MAGIC   0x4245414Du
#define BEAM_FORMAT  9u

typedef struct {
    uint32_t magic;
    uint32_t format;
    /* Decides every table dimension in the payload, so it is checked before
     * anything else is interpreted. */
    uint32_t alphabet;
    uint32_t order;
    uint32_t max_length;
    uint32_t storage_sparse;   /* 0: dense flat blocks, 1: frozen sparse */
    /* Bit 0: the level strengths are estimated from the counts; bit 1: the
     * length strength is. The values below are then the estimates the build
     * made. */
    uint32_t prior_auto;
    /* Bit 0: the model was FINALIZED when written, so the strengths below
     * are the estimates for exactly these counts and a load finalizes under
     * them instead of re-deriving them -- the same numbers, 48.5 s against
     * 7.8 s a load on the foundation. Written zero before 2026-09-30, and
     * such a file is re-derived as it always was. Readers from before then
     * never read this word, so the bit is invisible to them. */
    uint32_t flags;
    /*
     * Provenance: three facts a finished model cannot recompute about itself.
     * Contiguous, and pinned so below, because
     * `beam_model_file_provenance_span` hands their window to callers that
     * compare two files for everything else.
     */
    int64_t  created_unix;
    int64_t  observe_nanos;
    int64_t  finalize_nanos;
    double   weight_forward, weight_backward;
    /* The hierarchical prior. Stored because it is configuration: the same
     * counts under other strengths are a different model. */
    double   alpha[BEAM_MAX_ORDER + 1];
    double   alpha_length;
    double   total_weight;
    uint64_t n_observed;
    uint64_t n_skipped;
} BeamHeader;

/* Pinned because these numbers are the file format. */
_Static_assert(sizeof(BeamHeader) == 184, "the format-9 header is 184 bytes");

void beam_model_file_provenance_span(size_t *offset, size_t *bytes) {
    const size_t first = offsetof(BeamHeader, created_unix);
    const size_t last_end = offsetof(BeamHeader, finalize_nanos)
                          + sizeof(((BeamHeader *)0)->finalize_nanos);
    if (offset) *offset = first;
    if (bytes) *bytes = last_end - first;
}
_Static_assert(offsetof(BeamHeader, observe_nanos)
                   == offsetof(BeamHeader, created_unix) + 8,
               "the provenance fields must stay contiguous");
_Static_assert(offsetof(BeamHeader, finalize_nanos)
                   == offsetof(BeamHeader, created_unix) + 16,
               "the provenance fields must stay contiguous");

static bool write_all(FILE *f, const void *p, size_t n, uint32_t *crc) {
    if (n && fwrite(p, 1, n, f) != n) return false;
    *crc = beam_crc32c_update(*crc, p, n);
    return true;
}

static bool read_all(FILE *f, void *p, size_t n, uint32_t *crc) {
    if (n && fread(p, 1, n, f) != n) return false;
    *crc = beam_crc32c_update(*crc, p, n);
    return true;
}

/**
 * The frozen encoding, as it stands in memory.
 *
 * Five arrays and a byte block, each already contiguous, so this is a length
 * header and six writes. Widths are fixed by the format rather than by the
 * host: `uint32` row ids and mask/tier words, `uint16` sub-offsets, `uint64`
 * block bases and lengths, all little-endian like every other field here.
 */
static bool write_frozen_count_table(FILE *f, const BeamFrozenCounts *z,
                                     uint32_t *crc) {
    const uint64_t rows = z->n_rows, blocks = z->n_blocks;
    const uint64_t vbits = z->n_value_bits, dense = z->n_dense_rows;
    if (!write_all(f, &rows, sizeof rows, crc)) return false;
    if (!write_all(f, &dense, sizeof dense, crc)) return false;
    if (!write_all(f, &blocks, sizeof blocks, crc)) return false;
    if (!write_all(f, &vbits, sizeof vbits, crc)) return false;
    if (rows == 0u) return true;
    /* Which form identifies the rows. Chosen by occupancy at freeze time, so
     * the file has to say which one it holds rather than let the reader guess
     * from dimensions that may not reproduce the same choice. */
    const uint64_t form = z->present ? 1u : 0u;
    if (!write_all(f, &form, sizeof form, crc)) return false;
    const size_t vbytes = (size_t)((vbits + 7u) / 8u);
    return (form
            ? write_all(f, z->present,
                        (size_t)beam_frozen_bitwords(z->n_dense_rows)
                        * sizeof(uint64_t), crc)
            : write_all(f, z->row_id, (size_t)rows * sizeof(uint32_t), crc))
        && write_all(f, z->mask_width, (size_t)rows * sizeof(uint32_t), crc)
        && write_all(f, z->block_bit, ((size_t)blocks + 1u) * sizeof(uint64_t), crc)
        && (vbytes == 0u || write_all(f, z->values, vbytes, crc));
}

static bool write_count_table(FILE *f, const BeamModel *m, uint32_t dir, uint32_t *crc) {
    const uint32_t nr = beam_count_n_rows(m);
    const size_t row_bytes = (size_t)m->n_residues * sizeof(double);
    if (!m->cfg.sparse) {
        const double *p = m->cnt[dir];
        return write_all(f, p, (size_t)nr * row_bytes, crc);
    }

    /*
     * Format 4 writes the frozen encoding itself rather than expanding it.
     *
     * Every earlier format wrote a row id and all twenty doubles: 164 bytes
     * per row for rows that are 85.6% zeros. The frozen form already holds
     * exactly the same information without them -- presence as a bit, values
     * in the narrowest type that is exact -- so writing it out is 12.6x
     * smaller on a real repertoire, 98.0 MB to about 7.8 MB, and needs no
     * encoder of its own.
     *
     * A model whose counts are still mutable is frozen for the duration of
     * the write. Nothing else can see that, and it costs less than expanding
     * every row would.
     */
    if (m->counts_frozen)
        return write_frozen_count_table(
                   f, beam_model_frozen_table(m, dir), crc);

    /*
     * A sparse model whose counts are still mutable -- saved before it was
     * ever finalized, or after something thawed them -- is frozen into a
     * temporary for the write. The alternative is a format-4 header over a
     * format-3 payload, which is what this did at first: the two paths
     * produced different bytes for the same model and only the header said
     * which one a reader should expect.
     *
     * A temporary rather than freezing the model itself, because `m` is
     * const here and saving is not a reason to change what the caller holds.
     */
    BeamFrozenCounts tmp;
    const BeamSparseRows *src = &m->sp_cnt[dir];
    if (beam_frozen_freeze(src, src->n_parts, &tmp) != BEAM_OK) return false;
    const bool wrote = write_frozen_count_table(f, &tmp, crc);
    beam_frozen_destroy(&tmp);
    return wrote;
}

/**
 * Read a frozen count table straight into the model.
 *
 * A file is not a trusted input, so everything the encoding rests on is
 * checked before anything dereferences it: the row count against the table,
 * the bitmap's population against the row count, each row's width against the
 * widths that exist, every block offset against the running position, and the
 * total the headers imply against the total the file declares. A file failing
 * any of them is refused rather than decoded into whatever the bytes happen
 * to address.
 */
static bool read_frozen_count_table(FILE *f, BeamModel *m, uint32_t dir, uint32_t *crc) {
    const uint32_t nr = beam_count_n_rows(m);
    uint64_t rows = 0, dense = 0, blocks = 0, vbits = 0;
    if (!read_all(f, &rows, sizeof rows, crc)
        || !read_all(f, &dense, sizeof dense, crc)
        || !read_all(f, &blocks, sizeof blocks, crc)
        || !read_all(f, &vbits, sizeof vbits, crc))
        return false;
    if (rows > nr || dense != nr) return false;

    BeamFrozenCounts z;
    memset(&z, 0, sizeof z);
    z.stride = m->n_residues;
    z.n_dense_rows = nr;
    if (rows == 0u) {
        *(BeamFrozenCounts *)beam_model_frozen_table(m, dir) = z;
        return true;
    }
    if (blocks != beam_frozen_blocks((uint32_t)rows)) return false;

    uint64_t form = 0;
    if (!read_all(f, &form, sizeof form, crc) || form > 1u) return false;
    z.n_rows = (uint32_t)rows;
    z.n_blocks = (uint32_t)blocks;
    z.n_value_bits = vbits;
    const uint32_t words = beam_frozen_bitwords(nr);
    const size_t vbytes = (size_t)((vbits + 7u) / 8u);
    if (form) z.present = (uint64_t *)malloc((size_t)words * sizeof(uint64_t));
    else      z.row_id  = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
    z.mask_width = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
    z.block_bit  = (uint64_t *)malloc(((size_t)blocks + 1u) * sizeof(uint64_t));
    /* Padded like the freeze path: the word-at-a-time reader touches up to
     * eight bytes past the last value. */
    z.values     = (unsigned char *)calloc(vbytes + 16u, 1u);
    if ((form ? (void *)z.present : (void *)z.row_id) == NULL
        || !z.mask_width || !z.block_bit || !z.values) {
        beam_frozen_destroy(&z); return false;
    }
    if (!(form
          ? read_all(f, z.present, (size_t)words * sizeof(uint64_t), crc)
          : read_all(f, z.row_id, (size_t)rows * sizeof(uint32_t), crc))
        || !read_all(f, z.mask_width, (size_t)rows * sizeof(uint32_t), crc)
        || !read_all(f, z.block_bit, ((size_t)blocks + 1u) * sizeof(uint64_t), crc)
        || (vbytes && !read_all(f, z.values, vbytes, crc))) {
        beam_frozen_destroy(&z); return false;
    }

    uint64_t at = 0;
    if (form) {
        uint64_t seen = 0;
        for (uint32_t w = 0; w < words; ++w)
            seen += (uint64_t)__builtin_popcountll(z.present[w]);
        if (seen != rows) { beam_frozen_destroy(&z); return false; }
    } else {
        for (uint64_t i = 0; i < rows; ++i)
            if (z.row_id[i] >= nr || (i && z.row_id[i] <= z.row_id[i - 1u])) {
                beam_frozen_destroy(&z); return false;
            }
    }
    const uint32_t stride_mask = (m->n_residues >= 32u)
                               ? UINT32_MAX : ((1u << m->n_residues) - 1u);
    for (uint64_t i = 0; i < rows; ++i) {
        const uint32_t h = z.mask_width[i];
        const uint32_t mask = h & ((1u << BEAM_FROZEN_MASK_BITS) - 1u);
        const uint32_t width = (h >> BEAM_FROZEN_WIDTH_SHIFT)
                             & BEAM_FROZEN_WIDTH_MASK;
        if (mask == 0u || (mask & ~stride_mask)) {
            beam_frozen_destroy(&z); return false;
        }
        if (width == 0u || (width > 53u && width != BEAM_FROZEN_WIDTH_F64)) {
            beam_frozen_destroy(&z); return false;
        }
        if (i % BEAM_FROZEN_BLOCK_ROWS == 0u
            && z.block_bit[i / BEAM_FROZEN_BLOCK_ROWS] != at) {
            beam_frozen_destroy(&z); return false;
        }
        at += (uint64_t)__builtin_popcount(mask)
            * (width == BEAM_FROZEN_WIDTH_F64 ? 64u : width);
        if (at > vbits) { beam_frozen_destroy(&z); return false; }
    }
    if (at != vbits || z.block_bit[blocks] != vbits) {
        beam_frozen_destroy(&z); return false;
    }

    *(BeamFrozenCounts *)beam_model_frozen_table(m, dir) = z;
    return true;
}

BeamError beam_model_save(const BeamModel *m, const char *path) {
    if (!m || !path) return BEAM_ENULL(!m ? "model" : "path");
    if (m->compacted)
        return BEAM_ECOMPACTED("there are no counts left to write; "
                                   "save before compacting");

    FILE *f = fopen(path, "wb");
    if (!f) return BEAM_FAIL(BEAM_ERR_IO_OPEN,
                            "%s: cannot open '%s' for writing",
                            __func__, path);

    BeamHeader h;
    memset(&h, 0, sizeof h);
    h.magic           = BEAM_MAGIC;
    h.format          = BEAM_FORMAT;
    h.alphabet        = (uint32_t)m->cfg.alphabet;
    h.order           = m->cfg.order;
    h.max_length      = m->cfg.max_length;
    h.storage_sparse  = m->cfg.sparse ? 1u : 0u;
    h.prior_auto      = (m->cfg.alpha_auto ? 1u : 0u)
                      | (m->cfg.alpha_length_auto ? 2u : 0u);
    /* Any change to the counts clears `finalized`, so a finalized model's
     * strengths are the estimates for the counts written below. */
    h.flags           = m->finalized ? 1u : 0u;
    h.created_unix    = m->created_unix;
    h.observe_nanos   = m->observe_nanos;
    h.finalize_nanos  = m->finalize_nanos;
    h.weight_forward  = m->cfg.weight_forward;
    h.weight_backward = m->cfg.weight_backward;
    for (uint32_t k = 0; k <= BEAM_MAX_ORDER; ++k) h.alpha[k] = m->cfg.alpha[k];
    h.alpha_length    = m->cfg.alpha_length;
    h.total_weight    = m->total_weight;
    h.n_observed      = m->n_observed;
    h.n_skipped       = m->n_skipped;

    uint32_t crc = 0;
    bool ok = write_all(f, &h, sizeof h, &crc);
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d)
        ok = write_count_table(f, m, d, &crc);
    if (ok) ok = write_all(f, m->cnt_length,
                           (size_t)m->n_len_slots * sizeof(double), &crc);
    if (ok) ok = fwrite(&crc, 1, sizeof crc, f) == sizeof crc;

    /*
     * Naming the flush separately matters: buffered writes fail at `fclose`,
     * so a full disk almost always surfaces there rather than at one of the
     * `write_all` calls above.
     */
    const bool flushed = fclose(f) == 0;
    if (!ok || !flushed)
        return BEAM_FAIL(BEAM_ERR_IO_WRITE,
                        "%s: writing '%s' failed%s, so the file is incomplete "
                        "and must not be loaded; the disk may be full or the "
                        "path may have become unwritable",
                        __func__, path,
                        flushed ? "" : " while flushing the last buffer");
    return BEAM_OK;
}

/*
 * Every failure below names itself, except the two that must not:
 * `beam_model_new` and `beam_model_finalize` already set precise errors (a
 * memory-budget refusal among them), and overwriting those would send the
 * reader to rebuild a file that was never damaged.
 */
BeamModel *beam_model_load(const char *path) {
    beam_memory_init();       /* see beam_memory_init: static links drop it otherwise */
    if (!path) return (void)BEAM_ENULL("path"), NULL;
    beam_clear_error();
    FILE *f = fopen(path, "rb");
    if (!f) return (void)BEAM_FAIL(BEAM_ERR_IO_OPEN,
                                  "beam_model_load: cannot open '%s' for "
                                  "reading", path), NULL;

    BeamHeader h;
    memset(&h, 0, sizeof h);
    uint32_t crc = 0;
    /* Magic and format first, so a file of another kind or version is named
     * as such rather than reported as a short read of a longer header. */
    const size_t id_bytes = offsetof(BeamHeader, alphabet);
    if (!read_all(f, &h, id_bytes, &crc)) {
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_READ,
                              "beam_model_load: '%s' is too short to be a "
                              "BEAM model file", path), NULL;
    }
    if (h.magic != BEAM_MAGIC) {
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_CORRUPT,
                              "beam_model_load: '%s' is not a BEAM model file "
                              "(wrong magic number)", path), NULL;
    }
    if (h.format != BEAM_FORMAT) {
        const unsigned got = (unsigned)h.format;
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_VERSION,
                              "beam_model_load: '%s' is a format %u BEAM model "
                              "from an earlier development build, and this "
                              "build reads only format %u. Rebuild the model "
                              "from its sequences", path, got,
                              (unsigned)BEAM_FORMAT), NULL;
    }
    if (!read_all(f, (unsigned char *)&h + id_bytes, sizeof h - id_bytes, &crc)) {
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_READ,
                              "beam_model_load: '%s' ends inside its header, "
                              "so it was truncated in writing", path), NULL;
    }
    if (!beam_alphabet_info((BeamAlphabet)h.alphabet)) {
        const unsigned a = (unsigned)h.alphabet;
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_VERSION,
                              "beam_model_load: '%s' was written with "
                              "alphabet id %u, which this build does not know",
                              path, a), NULL;
    }
    if (h.storage_sparse > 1u || h.prior_auto > 3u || h.order > 255u
        || h.max_length > 255u) {
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_CORRUPT,
                              "beam_model_load: '%s' has a header field out of "
                              "range, so its header is corrupt", path), NULL;
    }

    BeamModelConfig cfg;
    beam_model_config_default_for((BeamAlphabet)h.alphabet, &cfg);
    cfg.order           = (uint8_t)h.order;
    cfg.max_length      = (uint8_t)h.max_length;
    cfg.sparse          = h.storage_sparse == 1u;
    cfg.weight_forward  = h.weight_forward;
    cfg.weight_backward = h.weight_backward;
    for (uint32_t k = 0; k <= BEAM_MAX_ORDER; ++k) cfg.alpha[k] = h.alpha[k];
    cfg.alpha_length    = h.alpha_length;
    cfg.alpha_auto        = (h.prior_auto & 1u) != 0u;
    cfg.alpha_length_auto = (h.prior_auto & 2u) != 0u;

    /* `beam_model_new` validates the configuration -- the prior's strengths
     * included -- and sets a precise error, passed through untouched. */
    BeamModel *m = beam_model_new(&cfg);
    if (!m) { fclose(f); return NULL; }

    bool ok = true;
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
        ok = m->cfg.sparse
           ? read_frozen_count_table(f, m, d, &crc)
           : read_all(f, m->cnt[d], beam_n_table(m) * sizeof(double), &crc);
    }
    /* Set after the loop: a failure part way through leaves the flag false,
     * and destroy releases the frozen tables either way. */
    if (ok && m->cfg.sparse) m->counts_frozen = true;
    if (ok) ok = read_all(f, m->cnt_length,
                          (size_t)m->n_len_slots * sizeof(double), &crc);

    uint32_t stored = 0;
    if (ok) ok = fread(&stored, 1, sizeof stored, f) == sizeof stored;
    fclose(f);
    if (!ok) {
        beam_model_destroy(m);
        return (void)BEAM_FAIL(BEAM_ERR_IO_READ,
                              "beam_model_load: '%s' ends before its counts "
                              "do, so it was truncated in writing", path), NULL;
    }
    if (stored != crc) {
        beam_model_destroy(m);
        return (void)BEAM_FAIL(BEAM_ERR_IO_CORRUPT,
                              "beam_model_load: '%s' fails its checksum "
                              "(stored %08x, computed %08x), so its contents "
                              "changed after it was written",
                              path, stored, crc), NULL;
    }

    m->total_weight   = h.total_weight;
    m->n_observed     = h.n_observed;
    m->n_skipped      = h.n_skipped;
    /* Provenance survives the round trip: a loaded model reports when it was
     * FIRST built and what the whole build cost. */
    m->created_unix   = h.created_unix;
    m->observe_nanos  = h.observe_nanos;
    m->finalize_nanos = h.finalize_nanos;

    /* Under the stored strengths when the file says they fit its counts
     * (see `flags`); the auto flags come back afterwards, so a later change
     * to the counts is re-estimated as it would have been. */
    const bool auto_k = m->cfg.alpha_auto, auto_len = m->cfg.alpha_length_auto;
    if (h.flags & 1u) {
        m->cfg.alpha_auto = false;
        m->cfg.alpha_length_auto = false;
    }
    const BeamError fe = beam_model_finalize(m);
    m->cfg.alpha_auto = auto_k;
    m->cfg.alpha_length_auto = auto_len;
    if (fe != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    /*
     * The finalize just above is NOT charged to the model. It is an artefact
     * of storing counts rather than tables, and charging it would make a
     * build time that rises every time the model is opened.
     */
    m->finalize_nanos = h.finalize_nanos;
    return m;
}
