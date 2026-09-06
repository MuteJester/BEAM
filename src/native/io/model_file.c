/**
 * @file model_file.c
 * @brief Persistence for BEAM models.
 *
 * Counts are written, not the derived log-probability tables. That costs
 * nothing in file size (they are the same shape) and keeps a loaded model a
 * first-class one: it can still be subtracted from, observed into, and
 * re-finalized under different smoothing. Writing the tables instead would
 * produce an artefact that could only be scored against.
 *
 * Layout: a fixed header, then the count tables in a declared order, then a
 * CRC-32C of everything before it. A dense table is its flat double block. A
 * sparse table is a uint64 row count followed by sorted `(uint32 row,
 * double[20])` records. Little-endian doubles and integers, which is every
 * platform this library targets.
 */
#include <stdio.h>
#include <stddef.h>

#include "beam/crc32c.h"
#include "beam/internal.h"

/* 0x4245414D is 'B','E','A','M' as ASCII, most significant byte first. The
 * field is written in host byte order, so those bytes appear reversed in a
 * hex dump on a little-endian machine; the check is on the integer, not on
 * the text, so that is harmless.
 *
 * This model shipped as EDGE before the split into its own repository, under
 * magic 0x45444745. That value is retained below ONLY so the loader can tell
 * a pre-BEAM file apart from a corrupt one and say so. Such files are not
 * readable: the change is deliberate and the foundation models are being
 * rebuilt. */
#define BEAM_MAGIC        0x4245414Du   /* "BEAM" */
#define BEAM_MAGIC_LEGACY 0x45444745u   /* "EDGE", pre-split; rejected */
#define BEAM_FORMAT   3u
#define BEAM_FORMAT_V2       2u
#define BEAM_FORMAT_DENSE_V1 1u

/* ── Checksum ────────────────────────────────────────────────── */

/*
 * An BEAM model at order 4 is a gigabyte of counts, and the shared
 * byte-at-a-time CRC-32C runs at about a gigabyte a second, so the checksum
 * was costing as much as the write itself. Every x86-64 chip since Nehalem
 * has the polynomial in silicon; this uses it when the running CPU says it
 * has it and falls back to the shared implementation otherwise.
 *
 * Same polynomial, same convention, same answer — `test_crc_matches_the
 * _reference` checks that against `beam_crc32c_update` on unaligned buffers of
 * every length class, which is the only thing that makes swapping the
 * implementation safe.
 */
#if defined(__x86_64__) || defined(__i386__)
#define BEAM_HAVE_CRC_HW 1
#include <x86intrin.h>

__attribute__((target("sse4.2")))
static uint32_t crc32c_hw(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = ~crc;
    while (len && ((uintptr_t)p & 7u)) { c = __builtin_ia32_crc32qi(c, *p++); --len; }
#if defined(__x86_64__)
    uint64_t c64 = c;
    while (len >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        c64 = __builtin_ia32_crc32di(c64, v);
        p += 8; len -= 8;
    }
    c = (uint32_t)c64;
#endif
    while (len--) c = __builtin_ia32_crc32qi(c, *p++);
    return ~c;
}
#endif

uint32_t beam_model_crc32c(uint32_t crc, const void *data, size_t len) {
#ifdef BEAM_HAVE_CRC_HW
    if (__builtin_cpu_supports("sse4.2")) return crc32c_hw(crc, data, len);
#endif
    return beam_crc32c_update(crc, data, len);
}

typedef struct {
    uint32_t magic;
    uint32_t format;
    uint32_t order;
    uint32_t depth_cap;
    uint32_t max_length;
    /*
     * Closed-world flag. Occupies what used to be a reserved word, so the
     * layout and the format number are unchanged and a file written before
     * `strict` existed reads back with 0 -- which is exactly the open-world
     * default it was written under. Persisting it is not optional: the file
     * stores COUNTS, not finalized tables, so load re-finalizes, and a strict
     * model reloaded without this flag re-finalizes with the add-one and the
     * probability floor back on. That silently produces a THIRD model --
     * kappa 0 but nothing dead -- whose D0 returns to sum_L 20**L and whose
     * floored cells each contribute ~690 nats, blowing the surprisal range
     * from ~230 to ~17,500 nats.
     */
    uint32_t strict;
    double   coef_wide, coef_mid, coef_len;
    double   kappa0, kappa1, kappa2;
    double   kappa_wide, kappa_len;
    double   weight_forward, weight_backward;
    /* Format 2: 0.0 for dense count blocks, 1.0 for sparse row records.
     * Reuses the old reserved word, so the fixed header itself stays stable. */
    double   storage_sparse;
    double   total_weight;
    uint64_t n_observed;
    uint64_t n_skipped;

    /*
     * Format 3. APPENDED, rather than grouped with the other uint32s where it
     * belongs, and the awkwardness buys the whole compatibility story: a
     * format-1 or -2 header is then exactly a PREFIX of this one, so the
     * reader takes the common bytes and asks for the rest only when the
     * format says they are there. Grouping it would have made two layouts to
     * parse and two chances to get an offset wrong.
     *
     * This field is not cosmetic. Alphabet size fully determines the payload
     * layout -- row width, record size, every table's extent -- so without it
     * a nucleotide model reads back either as CRC corruption or, worse, as an
     * amino-acid model of the same shape. BEAM_ALPHABET_AA is 0, so the
     * formats written before this field existed read back as what they are.
     *
     * `reserved` is written zero and ignored. Both of this header's earlier
     * spare words were spent -- one on `strict`, one on `storage_sparse` --
     * which is why adding the alphabet needed a format bump at all. Leaving
     * one costs eight bytes a file and saves the next bump.
     */
    uint32_t alphabet;
    uint32_t reserved;
} BeamHeader;

/* The bytes formats 1 and 2 wrote: everything up to the appended tail. */
#define BEAM_HEADER_V2_BYTES offsetof(BeamHeader, alphabet)

/* Pinned because these numbers are the file format. Trailing padding would be
 * written to disk uninitialised-looking and could move with the compiler, so
 * the tail is two explicit words rather than one word and a hole. */
_Static_assert(sizeof(BeamHeader) == 144, "the format-3 header is 144 bytes");
_Static_assert(BEAM_HEADER_V2_BYTES == 136, "formats 1 and 2 wrote 136 bytes");

static bool write_all(FILE *f, const void *p, size_t n, uint32_t *crc) {
    if (n && fwrite(p, 1, n, f) != n) return false;
    *crc = beam_model_crc32c(*crc, p, n);
    return true;
}

static bool read_all(FILE *f, void *p, size_t n, uint32_t *crc) {
    if (n && fread(p, 1, n, f) != n) return false;
    *crc = beam_model_crc32c(*crc, p, n);
    return true;
}

#define BEAM_SPARSE_IO_ROWS 256u

/*
 * A record is a row index plus one double per emitted residue, so its size is
 * the alphabet's and only known at run time. The staging block is sized by the
 * widest alphabet instead, which is why every length below is computed rather
 * than taken with sizeof: `sizeof block` is now the capacity, not the payload,
 * and using it would write the tail of an amino-acid-sized buffer into a
 * nucleotide file.
 */
#define BEAM_SPARSE_RECORD_MAX \
    (sizeof(uint32_t) + BEAM_MAX_RESIDUES * sizeof(double))

static bool write_count_table(FILE *f, const BeamModel *m, uint32_t dir,
                              BeamTableKind kind, uint32_t *crc) {
    const uint32_t nr = beam_count_n_rows(m, kind);
    const size_t row_bytes = (size_t)m->n_residues * sizeof(double);
    const size_t rec = sizeof(uint32_t) + row_bytes;
    if (!m->cfg.sparse) {
        const double *p = beam_dense_count_table_const(m, dir, kind);
        return write_all(f, p, (size_t)nr * row_bytes, crc);
    }

    const BeamSparseRows *t = beam_sparse_count_table_const(m, dir, kind);
    const uint64_t occupied = beam_sparse_n_rows(t);
    if (!write_all(f, &occupied, sizeof occupied, crc)) return false;
    uint64_t written = 0;
    unsigned char block[BEAM_SPARSE_IO_ROWS * BEAM_SPARSE_RECORD_MAX];
    uint32_t buffered = 0;
    for (uint32_t row = 0; row < nr; ++row) {
        const double *values = beam_count_row(m, dir, kind, row);
        if (!values) continue;
        unsigned char *record = block + (size_t)buffered * rec;
        memcpy(record, &row, sizeof row);
        memcpy(record + sizeof row, values, row_bytes);
        buffered++;
        written++;
        if (buffered == BEAM_SPARSE_IO_ROWS) {
            if (!write_all(f, block, (size_t)BEAM_SPARSE_IO_ROWS * rec, crc))
                return false;
            buffered = 0;
        }
    }
    if (buffered > 0
        && !write_all(f, block, (size_t)buffered * rec, crc))
        return false;
    return written == occupied;
}

static bool read_sparse_count_table(FILE *f, BeamModel *m, uint32_t dir,
                                    BeamTableKind kind, uint32_t *crc) {
    const uint32_t nr = beam_count_n_rows(m, kind);
    const size_t row_bytes = (size_t)m->n_residues * sizeof(double);
    const size_t rec = sizeof(uint32_t) + row_bytes;
    uint64_t occupied = 0;
    if (!read_all(f, &occupied, sizeof occupied, crc) || occupied > nr)
        return false;
    uint32_t previous = 0;
    unsigned char block[BEAM_SPARSE_IO_ROWS * BEAM_SPARSE_RECORD_MAX];
    uint64_t consumed = 0;
    while (consumed < occupied) {
        const uint32_t take = (occupied - consumed) < BEAM_SPARSE_IO_ROWS
                            ? (uint32_t)(occupied - consumed)
                            : BEAM_SPARSE_IO_ROWS;
        if (!read_all(f, block, (size_t)take * rec, crc))
            return false;
        for (uint32_t j = 0; j < take; ++j) {
            const uint64_t i = consumed + j;
            const unsigned char *record = block + (size_t)j * rec;
            uint32_t row = 0;
            double values[BEAM_MAX_RESIDUES];
            memcpy(&row, record, sizeof row);
            memcpy(values, record + sizeof row, row_bytes);
            if (row >= nr || (i > 0 && row <= previous)) return false;
            bool any = false;
            for (uint32_t x = 0; x < m->n_residues; ++x) {
                if (!(values[x] >= 0.0) || !isfinite(values[x])) return false;
                if (values[x] > 0.0) any = true;
            }
            if (!any) return false;
            double *dst = beam_count_row_mut(m, dir, kind, row, true);
            if (!dst) return false;
            memcpy(dst, values, row_bytes);
            previous = row;
        }
        consumed += take;
    }
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
    h.order           = m->cfg.order;
    h.depth_cap       = m->cfg.depth_cap;
    h.max_length      = m->cfg.max_length;
    h.strict          = m->cfg.strict ? 1u : 0u;
    h.coef_wide       = m->cfg.coef_wide;
    h.coef_mid        = m->cfg.coef_mid;
    h.coef_len        = m->cfg.coef_len;
    h.kappa0          = m->cfg.kappa[0];
    h.kappa1          = m->cfg.kappa[1];
    h.kappa2          = m->cfg.kappa[2];
    h.kappa_wide      = m->cfg.kappa_wide;
    h.kappa_len       = m->cfg.kappa_len;
    h.weight_forward  = m->cfg.weight_forward;
    h.weight_backward = m->cfg.weight_backward;
    h.storage_sparse  = m->cfg.sparse ? 1.0 : 0.0;
    h.alphabet        = (uint32_t)m->cfg.alphabet;
    h.reserved        = 0u;
    h.total_weight    = m->total_weight;
    h.n_observed      = m->n_observed;
    h.n_skipped       = m->n_skipped;

    uint32_t crc = 0;
    bool ok = write_all(f, &h, sizeof h, &crc);
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
        ok = write_count_table(f, m, d, BEAM_TABLE_WIDE, &crc)
          && write_count_table(f, m, d, BEAM_TABLE_MID, &crc)
          && write_count_table(f, m, d, BEAM_TABLE_LEN, &crc);
    }
    if (ok) ok = write_all(f, m->cnt_length,
                           (size_t)m->n_len_slots * sizeof(double), &crc);
    if (ok) ok = fwrite(&crc, 1, sizeof crc, f) == sizeof crc;

    /*
     * A bare code here served whatever the thread-local message buffer last
     * held, because `beam_set_error` is the only thing that writes it and this
     * path never called it. Saving to a full disk reported "beam_model_path_count:
     * length 0 is not one this model generates" -- a specific, actionable and
     * entirely unrelated diagnosis, which is worse than none because it will
     * be acted on. Naming the flush separately matters: buffered writes fail
     * at `fclose`, so a full disk almost always surfaces there rather than at
     * one of the `write_all` calls above.
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

bool beam_model_is_legacy_file(const char *path) {
    if (!path) return false;
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0;
    const bool got = fread(&magic, 1, sizeof magic, f) == sizeof magic;
    fclose(f);
    return got && magic == BEAM_MAGIC_LEGACY;
}

/*
 * Every failure below names itself.
 *
 * All nine used to `return NULL` without touching the error context, so the
 * binding hard-coded one sentence -- "not a model file, truncated, or
 * corrupt" -- for every one of them. A missing file, a directory, a text
 * file, and a PERFECTLY GOOD model that merely did not fit the memory budget
 * all produced that same sentence, and the last of those sent the reader to
 * rebuild a file that was never damaged.
 *
 * Two of the nine must NOT set an error of their own: `beam_model_new` and
 * `beam_model_finalize` already set precise ones (the budget refusal among
 * them), and overwriting those is how the information was lost.
 */
BeamModel *beam_model_load(const char *path) {
    if (!path) return (void)BEAM_ENULL("path"), NULL;
    beam_clear_error();
    FILE *f = fopen(path, "rb");
    if (!f) return (void)BEAM_FAIL(BEAM_ERR_IO_OPEN,
                                  "beam_model_load: cannot open '%s' for "
                                  "reading", path), NULL;

    /*
     * Read the bytes every format wrote, decide from `format` whether there
     * is a tail, and only then read it. Reading `sizeof h` unconditionally
     * would swallow the first eight bytes of a format-2 payload and then
     * disagree with its CRC -- a corrupt-file report for a perfectly good
     * file. Zeroed first, so an older format leaves `alphabet` at
     * BEAM_ALPHABET_AA, which is what it is.
     */
    BeamHeader h;
    memset(&h, 0, sizeof h);
    uint32_t crc = 0;
    if (!read_all(f, &h, BEAM_HEADER_V2_BYTES, &crc)
        || h.magic != BEAM_MAGIC
        || (h.format != BEAM_FORMAT
            && h.format != BEAM_FORMAT_V2
            && h.format != BEAM_FORMAT_DENSE_V1)) {
        const bool short_read = h.magic != BEAM_MAGIC && h.magic == 0u;
        const bool bad_magic  = h.magic != BEAM_MAGIC;
        fclose(f);
        if (short_read)
            return (void)BEAM_FAIL(BEAM_ERR_IO_READ,
                                  "beam_model_load: '%s' is too short to be a "
                                  "BEAM model file", path), NULL;
        if (bad_magic)
            return (void)BEAM_FAIL(BEAM_ERR_IO_CORRUPT,
                                  "beam_model_load: '%s' is not a BEAM model "
                                  "file (wrong magic number)", path), NULL;
        return (void)BEAM_FAIL(BEAM_ERR_IO_VERSION,
                              "beam_model_load: '%s' is a BEAM model file in "
                              "format %u, which this build does not read",
                              path, (unsigned)h.format), NULL;
    }
    if (h.format >= BEAM_FORMAT
        && !read_all(f, (unsigned char *)&h + BEAM_HEADER_V2_BYTES,
                     sizeof h - BEAM_HEADER_V2_BYTES, &crc)) {
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_READ,
                              "beam_model_load: '%s' ends inside its header, "
                              "so it was truncated in writing", path), NULL;
    }
    /* An alphabet this build does not know makes every table dimension in the
     * payload unreadable, so it is refused here rather than misinterpreted. */
    if (!beam_alphabet_info((BeamAlphabet)h.alphabet)) {
        const unsigned a = (unsigned)h.alphabet;
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_VERSION,
                              "beam_model_load: '%s' was written with "
                              "alphabet id %u, which this build does not know",
                              path, a), NULL;
    }

    BeamModelConfig cfg;
    beam_model_config_default_for((BeamAlphabet)h.alphabet, &cfg);
    cfg.order           = (uint8_t)h.order;
    cfg.depth_cap       = (uint8_t)h.depth_cap;
    cfg.max_length      = (uint8_t)h.max_length;
    cfg.strict          = (h.strict != 0u);
    /* Format 1 predates sparse storage entirely; from format 2 the word is
     * meaningful and must be exactly 0 or 1. */
    cfg.sparse          = h.format >= BEAM_FORMAT_V2
                        && h.storage_sparse == 1.0;
    if (h.format >= BEAM_FORMAT_V2
        && h.storage_sparse != 0.0 && h.storage_sparse != 1.0) {
        fclose(f);
        return (void)BEAM_FAIL(BEAM_ERR_IO_CORRUPT,
                              "beam_model_load: '%s' has a storage flag that "
                              "is neither 0 nor 1, so its header is corrupt",
                              path), NULL;
    }
    cfg.coef_wide       = h.coef_wide;
    cfg.coef_mid        = h.coef_mid;
    cfg.coef_len        = h.coef_len;
    cfg.kappa[0]        = h.kappa0;
    cfg.kappa[1]        = h.kappa1;
    cfg.kappa[2]        = h.kappa2;
    cfg.kappa_wide      = h.kappa_wide;
    cfg.kappa_len       = h.kappa_len;
    cfg.weight_forward  = h.weight_forward;
    cfg.weight_backward = h.weight_backward;

    /* `beam_model_new` has already set a precise error -- a memory-budget
     * refusal names the figure and the remedy -- so it is passed through
     * untouched rather than replaced with a report about the file. */
    BeamModel *m = beam_model_new(&cfg);
    if (!m) { fclose(f); return NULL; }

    bool ok = true;
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
        if (m->cfg.sparse) {
            ok = read_sparse_count_table(f, m, d, BEAM_TABLE_WIDE, &crc)
              && read_sparse_count_table(f, m, d, BEAM_TABLE_MID, &crc)
              && read_sparse_count_table(f, m, d, BEAM_TABLE_LEN, &crc);
        } else {
            ok = read_all(f, m->cnt_wide[d],
                          beam_n_wide(m) * sizeof(double), &crc)
              && read_all(f, m->cnt_mid[d],
                          beam_n_mid(m) * sizeof(double), &crc)
              && read_all(f, m->cnt_len[d],
                          beam_n_len(m) * sizeof(double), &crc);
        }
    }
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

    m->total_weight = h.total_weight;
    m->n_observed   = h.n_observed;
    m->n_skipped    = h.n_skipped;

    /* Same as `beam_model_new`: finalize's own error is the useful one. */
    if (beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}
