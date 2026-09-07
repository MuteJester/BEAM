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

/** Rows per offset block. One `uint16` sub-offset can address any block. */
#define BEAM_FROZEN_BLOCK_ROWS 256u

/** How a row's present values are stored. Ascending width; f64 is exact. */
typedef enum {
    BEAM_FROZEN_U8  = 0,
    BEAM_FROZEN_U16 = 1,
    BEAM_FROZEN_U32 = 2,
    BEAM_FROZEN_F64 = 3
} BeamFrozenTier;

/** Bits 0..19 are the residue mask; bits 20..21 are the tier. */
#define BEAM_FROZEN_MASK_BITS 20u
#define BEAM_FROZEN_TIER_SHIFT BEAM_FROZEN_MASK_BITS

typedef struct {
    uint32_t      *row_id;      /**< [n_rows] ascending dense row ids       */
    uint32_t      *mask_tier;   /**< [n_rows] residue mask plus tier        */
    uint16_t      *sub_offset;  /**< [n_rows] value offset within its block */
    uint64_t      *block_base;  /**< [n_blocks + 1] block value offsets     */
    unsigned char *values;      /**< packed present values, per-row tier    */
    uint32_t       n_rows;
    uint32_t       n_blocks;
    uint64_t       n_value_bytes;
    uint32_t       stride;      /**< residues per row                       */
} BeamFrozenCounts;

/** Bytes one present value occupies in each tier. */
static inline uint32_t beam_frozen_tier_width(uint32_t tier) {
    static const uint32_t w[4] = { 1u, 2u, 4u, 8u };
    return w[tier & 3u];
}

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

/** The values of one row, decoded into `out` (`stride` doubles). */
void beam_frozen_row(const BeamFrozenCounts *f, uint32_t k, double *out);

#endif /* BEAM_INTERNAL_FROZEN_COUNTS_H */
