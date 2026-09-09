/**
 * @file frozen_counts.h
 * @brief Counts after observation stops: sorted, cell-sparse, typed.
 *
 * Internal to src/native/. Not a public header and not installed.
 */
#ifndef BEAM_INTERNAL_FROZEN_COUNTS_H
#define BEAM_INTERNAL_FROZEN_COUNTS_H

#include "beam/model.h"
#include "beam/sparse_rows.h"

/**
 * A count table that is no longer being written to.
 *
 * `BeamSparseRows` is built for the observation loop: a dense directory so a
 * row is found in one load, sixteen partitions so sixteen depth tasks can
 * grow their payloads without a lock, and twenty contiguous doubles per row
 * so an increment is one store. That is the right shape while counts are
 * being accumulated -- `observe_one_position` performs six of those lookups
 * and six increments per residue per direction, which is the hottest loop in
 * training.
 *
 * It is the wrong shape afterwards. Once observation stops, every consumer of
 * counts is a bulk sequential pass -- finalize scans them, save writes them,
 * add/subtract/posterior merge them -- and none of them needs a directory or
 * random access. Meanwhile the shape costs, measured on a 197,139-sequence
 * repertoire at order 4:
 *
 *   dense row directory   29,561,112 bytes   for 597,373 occupied rows
 *   slot -> row map        2,389,492 bytes
 *   20-double payloads    95,579,680 bytes   of which 85.58% are zero
 *
 * This form keeps the same information and drops the shape:
 *
 *   - rows sorted by dense id, so a consumer walks them in order and two
 *     tables can be merged as sorted streams;
 *   - no directory at all;
 *   - one bit per residue saying whether that cell is present, so the 85.58%
 *     that are zero cost a bit rather than eight bytes;
 *   - present values stored in the narrowest type that holds them EXACTLY,
 *     chosen per row, with binary64 as the escape hatch.
 *
 * The escape hatch is not decoration. `beam_model_observe` accepts arbitrary
 * positive `double` abundances and `beam_model_posterior` contributes
 * `kappa * normalise(prior)`, which is fractional in general. Integer counts
 * are a property of a corpus, never a property of BEAM, so a row containing
 * anything an integer cannot hold exactly is stored as the doubles it
 * already was.
 *
 * Round-tripping is bit-exact by construction and is checked as such:
 * thawing a frozen table reproduces every cell's bit pattern, including the
 * zeros the mask implies.
 */

/** Rows per block. A block is what makes random access possible at all. */
#define BEAM_FROZEN_BLOCK_ROWS 256u

/** Bits of the row header given to the value width. */
#define BEAM_FROZEN_MASK_BITS  20u
#define BEAM_FROZEN_WIDTH_SHIFT BEAM_FROZEN_MASK_BITS
#define BEAM_FROZEN_WIDTH_MASK 63u

/**
 * A row whose values are not integers keeps them as they are.
 *
 * The width field is otherwise the number of bits one value occupies, at most
 * 53 because that is the largest integer binary64 holds exactly. 63 is not a
 * width; it says the row's values are raw binary64 and 64 bits each.
 */
#define BEAM_FROZEN_WIDTH_F64  63u

/** Words in a presence bitmap over `n` dense rows. */
static inline uint32_t beam_frozen_bitwords(uint32_t n) {
    return (n + 63u) / 64u;
}

/** Is a bitmap over `n_dense_rows` smaller than a list of `n_rows` ids? */
static inline bool beam_frozen_prefers_bitmap(uint32_t n_dense_rows,
                                              uint32_t n_rows) {
    return (uint64_t)beam_frozen_bitwords(n_dense_rows) * sizeof(uint64_t)
         < (uint64_t)n_rows * sizeof(uint32_t);
}

/** Blocks needed to cover `n_rows` occupied rows. */
static inline uint32_t beam_frozen_blocks(uint32_t n_rows) {
    return (n_rows + BEAM_FROZEN_BLOCK_ROWS - 1u) / BEAM_FROZEN_BLOCK_ROWS;
}

/**
 * Counts after observation stops.
 *
 * Four arrays, and each of them exists because a whole class of consumer
 * asked for it. Nothing here supports random access to a cell, because
 * nothing reads counts that way: `beam_frozen_thaw`, `write_frozen_count_table`
 * and `beam_model_n_nonzero_cells` all walk rows in order, and every other
 * count consumer -- observe, add, subtract, posterior, re-finalize -- thaws
 * first.
 *
 * `present`  one bit per DENSE row, replacing a sorted array of row ids.
 *            The ids were four bytes each for rows that are already
 *            identified by their position: 17,114,944 bytes on the
 *            foundation against 923,792 for the bitmap.
 *
 * `mask_width` per occupied row, a 20-bit residue mask and a 6-bit value
 *            width. The mask is which residues this row emitted; the width is
 *            how many bits one of its values takes, which is the row's own
 *            largest value rather than a tier rounded up to a byte.
 *
 * `values`   the present values, `width` bits each, packed end to end with no
 *            padding between rows. Byte tiers cost 104,992,360 bytes on the
 *            foundation; the widths the rows actually need cost about
 *            74,000,000.
 *
 * `block_bit` where each 256-row block's values begin. A sequential walk does
 *            not need it -- the cursor carries its own position -- but
 *            reaching row k without walking from zero does.
 *
 * The escape is not decoration. `beam_model_observe` takes arbitrary positive
 * doubles and `beam_model_posterior` contributes `kappa * normalise(prior)`,
 * so integer counts are a property of a corpus and never of BEAM. A row
 * holding anything an integer cannot represent exactly is stored as the
 * binary64 it already was, and the round trip is bit-exact either way.
 */
/**
 * Which way a table identifies its occupied rows.
 *
 * A bitmap costs one bit per DENSE row whatever the occupancy; a sorted list
 * costs four bytes per OCCUPIED row. They cross at one row in thirty-two, and
 * both sides of that line are real here: the foundation occupies 57.90% of
 * its rows and the bitmap is 18x smaller, while a per-repertoire model
 * occupies a fraction of a percent and the bitmap is several times larger.
 *
 * Chosen per table from its own occupancy, like the finalized directory and
 * for the same reason. Exactly one of `present` and `row_id` is non-NULL.
 */
typedef struct {
    uint64_t      *present;     /**< [bitwords(n_dense_rows)], or NULL     */
    uint32_t      *row_id;      /**< [n_rows] ascending, or NULL           */
    uint32_t      *mask_width;  /**< [n_rows] 20-bit mask, 6-bit width     */
    uint64_t      *block_bit;   /**< [n_blocks + 1] value bit offsets      */
    unsigned char *values;      /**< bit-packed, no padding between rows   */
    uint32_t       n_rows;
    uint32_t       n_blocks;
    uint32_t       n_dense_rows;
    uint32_t       stride;      /**< residues per row                      */
    uint64_t       n_value_bits;
} BeamFrozenCounts;

/**
 * A walk over the occupied rows in ascending dense order.
 *
 * The only shape any consumer wants, and the reason `sub_offset` is gone: a
 * walker carries its own position in the value stream, so storing one per row
 * was two bytes apiece to avoid arithmetic nobody was doing.
 */
typedef struct {
    const BeamFrozenCounts *f;
    uint32_t k;        /**< rows already yielded                          */
    uint32_t widx;     /**< bitmap word `bits` was taken from             */
    uint64_t bits;     /**< that word, with yielded rows cleared          */
    uint64_t bit;      /**< where the next row's values start             */
} BeamFrozenCursor;

void beam_frozen_cursor_init(BeamFrozenCursor *c, const BeamFrozenCounts *f);

/**
 * The next occupied row, decoded into `out` (`stride` doubles, zeros
 * included). False when there are none left.
 */
bool beam_frozen_cursor_next(BeamFrozenCursor *c, uint32_t *row, double *out);

/**
 * Freeze every occupied row of `src` into `out`.
 *
 * `src` is left untouched; the caller decides when to release it. Rows come
 * out sorted by dense id regardless of the order observation produced, which
 * is safe because nothing reads a count table by slot: `write_count_table`
 * walks dense rows and `finalize_sparse_table` sorts by destination.
 */
BeamError beam_frozen_freeze(const BeamSparseRows *src, uint32_t n_parts,
                             BeamFrozenCounts *out);

/**
 * Rebuild a mutable table from `src`, partitioning rows with `part_of`.
 *
 * The result holds the same rows with the same bits, including the zeros the
 * mask implied. It is a fresh table: the caller owns it and must destroy it.
 */
BeamError beam_frozen_thaw(const BeamFrozenCounts *src, uint32_t n_dense_rows,
                           uint32_t n_parts,
                           uint32_t (*part_of)(uint32_t row, void *ctx),
                           void *ctx, BeamSparseRows *out);

void beam_frozen_destroy(BeamFrozenCounts *f);
uint64_t beam_frozen_memory_bytes(const BeamFrozenCounts *f);

/**
 * Row `k`'s values, decoded into `out`, without walking from row zero.
 *
 * Reaches the block that holds `k` in one step and walks at most 255 rows
 * inside it. For a pass over every row use the cursor: this would make one
 * O(n) into O(n * 256).
 */
void beam_frozen_row(const BeamFrozenCounts *f, uint32_t k, double *out);

/** The dense row id of occupied row `k`. Same cost as `beam_frozen_row`. */
uint32_t beam_frozen_row_id(const BeamFrozenCounts *f, uint32_t k);

/* ── Model-level freezing ────────────────────────────────────── */

BeamError beam_model_freeze_counts(BeamModel *m);
BeamError beam_model_thaw_counts(BeamModel *m);

/**
 * Thaw ONE count table into a caller-owned mutable table, leaving the model
 * untouched.
 *
 * `beam_model_thaw_counts` expands all six at once and installs them, which
 * is what finalize used to do: on the foundation that turns 29.6 MB of frozen
 * counts into 731.3 MB of mutable rows, all six alive for the whole build
 * while 714.2 MB of log tables accumulate beside them. Measured peak 1,969.5
 * MB for a model that ends up occupying 806.7.
 *
 * A table is dead the moment its own log table is written, so finalize takes
 * them one at a time through this and destroys each before the next. The
 * model stays frozen throughout -- `counts_frozen` is a whole-model flag and
 * a half-thawed model would contradict it, which is why this hands back a
 * local rather than installing anything.
 *
 * The caller owns `out` and must `beam_sparse_destroy` it.
 */
BeamError beam_model_thaw_count_table(const BeamModel *m, uint32_t dir,
                                      uint32_t kind, BeamSparseRows *out);
void      beam_model_release_frozen(BeamModel *m);
uint64_t  beam_model_frozen_bytes(const BeamModel *m);
const BeamFrozenCounts *beam_model_frozen_table(const BeamModel *m,
                                                uint32_t dir, uint32_t kind);

#endif /* BEAM_INTERNAL_FROZEN_COUNTS_H */
