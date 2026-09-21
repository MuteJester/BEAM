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
#include "beam/memory.h"
#include "beam/crc32c.h"

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
#define BEAM_FORMAT   7u
#define BEAM_FORMAT_V6       6u
#define BEAM_FORMAT_V5       5u
#define BEAM_FORMAT_V4       4u
#define BEAM_FORMAT_V3       3u
#define BEAM_FORMAT_V2       2u
#define BEAM_FORMAT_DENSE_V1 1u

/* ── Checksum ────────────────────────────────────────────────── */

typedef struct {
    uint32_t magic;
    uint32_t format;
    uint32_t order;
    uint32_t dead_depth_cap;   /* format 5 and older; zero since 6 */
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
    /*
     * Format 7: provenance. Three facts a finished model cannot recompute
     * about itself, in the bytes formats 1 to 5 spent on the expert mixing
     * coefficients. Same size, same alignment, so the header stays 144 bytes
     * and its static assertion keeps working as a tripwire.
     *
     * Persisted because they are not derivable. Everything else a caller
     * might want -- size, occupancy, what the model can spell -- is computed
     * from the tables on demand, and storing it would be a second copy to
     * keep in step.
     */
    int64_t  created_unix;
    int64_t  observe_nanos;
    int64_t  finalize_nanos;
    double   dead_kappa3[3];   /* format 5 and older; zero since 6 */
    double   dead_kappa[2];    /* format 5 and older; zero since 6 */
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

/*
 * The three provenance fields are contiguous and stay that way: the span is
 * taken from the first and the last rather than written as a number, so
 * reordering the struct moves the window instead of silently mis-aiming it.
 */
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
 * Read a format-5 frozen count table straight into the model.
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

/**
 * Read the format-4 frozen encoding into the MUTABLE tables.
 *
 * Format 4 packed values into byte tiers with a per-row byte offset; format 5
 * packs them at the width each row actually needs. Rather than translate one
 * frozen encoding into the other, this decodes format 4 the way every
 * pre-frozen format is read -- into count rows -- and lets finalize freeze
 * them into the current form afterwards. A legacy path is exercised once per
 * file and should be the simplest thing that is correct, not the fastest.
 */
static bool read_sparse_count_table(FILE *f, BeamModel *m, uint32_t dir, uint32_t *crc) {
    const uint32_t nr = beam_count_n_rows(m);
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
            double *dst = beam_cnt_row_mut(m, dir, row, true);
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
    h.dead_depth_cap  = 0u;
    h.max_length      = m->cfg.max_length;
    h.strict          = m->cfg.strict ? 1u : 0u;
    h.created_unix    = m->created_unix;
    h.observe_nanos   = m->observe_nanos;
    h.finalize_nanos  = m->finalize_nanos;





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
        ok = write_count_table(f, m, d, &crc);
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
    beam_memory_init();       /* see beam_memory_init: static links drop it otherwise */
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
        || h.format != BEAM_FORMAT) {
        const bool short_read = h.magic != BEAM_MAGIC && h.magic == 0u;
        const bool bad_magic  = h.magic != BEAM_MAGIC;
        const uint32_t got    = h.format;
        fclose(f);
        if (short_read)
            return (void)BEAM_FAIL(BEAM_ERR_IO_READ,
                                  "beam_model_load: '%s' is too short to be a "
                                  "BEAM model file", path), NULL;
        if (bad_magic)
            return (void)BEAM_FAIL(BEAM_ERR_IO_CORRUPT,
                                  "beam_model_load: '%s' is not a BEAM model "
                                  "file (wrong magic number)", path), NULL;
        if (got >= BEAM_FORMAT_DENSE_V1 && got <= BEAM_FORMAT_V6)
            return (void)BEAM_FAIL(BEAM_ERR_IO_CORRUPT,
                                  "beam_model_load: '%s' is a format %u BEAM "
                                  "model, written by a build with three "
                                  "experts per direction and a capped depth "
                                  "index. It cannot be converted: the cap "
                                  "merged every depth at or beyond it, so "
                                  "which depth an observation came from was "
                                  "discarded when the model was trained, and "
                                  "no table in the file holds order-%u "
                                  "context at exact depth. Retrain the model "
                                  "with this build",
                                  path, (unsigned)got,
                                  (unsigned)h.order), NULL;
        return (void)BEAM_FAIL(BEAM_ERR_IO_VERSION,
                              "beam_model_load: '%s' is a BEAM model file in "
                              "format %u, which this build does not read",
                              path, (unsigned)h.format), NULL;
    }
    /* Formats 3 and 4 append the alphabet field; 1 and 2 stop at 136 bytes.
     * Keyed on V3 rather than on BEAM_FORMAT, because the current format is
     * not the only one that has this tail -- and once it stopped being, this
     * read was silently skipped for format 3 and the payload was then read
     * from the wrong offset. */
    if (h.format >= BEAM_FORMAT_V3
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
    cfg.weight_forward  = h.weight_forward;
    cfg.weight_backward = h.weight_backward;

    /* `beam_model_new` has already set a precise error -- a memory-budget
     * refusal names the figure and the remedy -- so it is passed through
     * untouched rather than replaced with a report about the file. */
    BeamModel *m = beam_model_new(&cfg);
    if (!m) { fclose(f); return NULL; }

    /* Format 4 stores counts frozen; every earlier one stores expanded rows.
     * The frozen reader installs them directly, so a loaded model arrives in
     * the same shape a finalized one is in rather than being expanded and
     * frozen again. */
    const bool frozen_on_disk = (h.format >= BEAM_FORMAT) && m->cfg.sparse;
    bool ok = true;
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
        if (frozen_on_disk) {
            ok = read_frozen_count_table(f, m, d, &crc);
        } else if (m->cfg.sparse) {
            ok = read_sparse_count_table(f, m, d, &crc);
        } else {
            ok = read_all(f, m->cnt[d], beam_n_table(m) * sizeof(double), &crc);
        }
    }
    /* Both installed, so the model may now say it is carrying them. Set
     * after the loop rather than inside it: a failure part way through leaves
     * the flag false, and destroy releases the frozen tables either way. */
    if (ok && frozen_on_disk) m->counts_frozen = true;
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
    /*
     * Provenance survives the round trip, and that is the point of storing
     * it. A model loaded from disk reports when it was FIRST built and what
     * the whole build cost, not when it was loaded and what loading cost.
     */
    m->created_unix   = h.created_unix;
    m->observe_nanos  = h.observe_nanos;
    m->finalize_nanos = h.finalize_nanos;

    /* Same as `beam_model_new`: finalize's own error is the useful one. */
    if (beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    /*
     * And the finalize just above is NOT charged to the model.
     *
     * It is an artefact of the format rather than work anyone asked for: the
     * file stores counts so that a loaded model can still be subtracted from
     * and re-finalized under different smoothing, so every load has to
     * rebuild the tables. Charging it would make a build time that rises
     * every time the model is opened, and that is permanent once such a model
     * is saved again -- the foundation model would accrue a load's finalize
     * per session for ever.
     *
     * A re-finalize the CALLER asks for still counts, which is the difference
     * this restore draws: `finalize_nanos` means finalizes of this model's
     * content, not finalizes of its representation.
     */
    m->finalize_nanos = h.finalize_nanos;
    return m;
}
