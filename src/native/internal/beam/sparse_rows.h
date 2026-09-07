/**
 * @file sparse_rows.h
 * @brief Sparse per-residue row storage.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_SPARSE_ROWS_H
#define BEAM_INTERNAL_SPARSE_ROWS_H

#include "beam/model.h"

/* ── Sparse per-residue row storage ─────────────────────────── */

#define BEAM_ROW_MISSING UINT32_MAX

/**
 * A finalized directory in rank form: one bit per dense row, plus enough
 * cached prefix counts to turn "how many rows come before this one" into a
 * popcount.
 *
 * The direct directory spends four bytes on EVERY dense row, occupied or not.
 * That is 28.19 MiB per generation at order 4 whatever the corpus, which is
 * fine for the foundation -- it occupies 57.90% of its rows -- and is most of
 * a per-repertoire model, which occupies about 3%. This form spends 1.25 bits
 * per row instead: 40 bytes per 256 rows, 1.10 MiB for the same directory.
 *
 * It works only because a finalized table assigns slots in ascending row
 * order, so a row's slot IS the number of occupied rows before it. `sub`
 * packs the within-block prefix for words 1..3 into 9 bits each (at most 192,
 * so 9 bits is exact); word 0's prefix is zero and is not stored.
 *
 * Build-once by construction: there is no way to insert a row without
 * renumbering every slot after it, which is why only finalized tables can
 * hold one.
 */
typedef struct {
    uint64_t bits[4];   /* 256 dense rows, ascending                        */
    uint32_t rank;      /* occupied rows strictly before this block         */
    uint32_t sub;       /* prefix for words 1..3, nine bits each            */
} BeamRankBlock;

#define BEAM_RANK_BLOCK_ROWS 256u

/**
 * Seal a finalized table when at most one row in this many is occupied.
 *
 * Measured on this machine, order 4, two donor models per point, one scheme
 * per process so neither warms the other's cache, best of seven interleaved
 * repetitions. The trade is stable in shape and varies only in how much
 * memory it returns:
 *
 *   occupancy   tables        scoring   inner product   Hill numbers
 *     0.86%   37.5 -> 10.7 MB  1.6x       1.32x           0.80x
 *     3.39%   65.7 -> 38.9 MB  2.2x       1.17x           0.91x
 *     8.74%  125.2 -> 98.4 MB  1.5x       1.03x           0.87x
 *    17.67%  224.7 -> 198  MB  1.5x       1.16x           0.95x
 *    31.23%  375.8 -> 349  MB  1.3x       1.06x           0.93x
 *
 * Scoring and the mixture inner product get faster because a 1.1 MiB
 * directory stays in cache where a 28.19 MiB one cannot. The Hill sweep gets
 * 5-20% slower: it walks rows in order, so it was never paying for the misses
 * the rank form avoids, and only pays the popcount it adds.
 *
 * The saving itself is a constant 28.4 MiB per table set, so what changes with
 * occupancy is only its share of the model -- 76% at 0.86%, 7.6% at 31%. One
 * in eight is where that share stops being worth a slower Hill sweep. It also
 * puts every per-repertoire model on the rank form and leaves the foundation,
 * which occupies 57.90% of its rows, on the direct one.
 */
#define BEAM_RANK_SEAL_MAX_OCCUPANCY_DENOM 8u

/** Blocks needed to cover `n_dense_rows`. */
static inline uint32_t beam_rank_blocks(uint32_t n_dense_rows) {
    return (n_dense_rows + BEAM_RANK_BLOCK_ROWS - 1u) / BEAM_RANK_BLOCK_ROWS;
}

/**
 * Dense row directory with sparse per-row payloads, one double per
 * emitted residue.
 *
 * Count tables are partitioned by the depth task that owns their rows. That
 * lets the existing parallel observation loop grow each payload without a
 * lock: one task is the sole writer of one partition. Finalized tables have a
 * single payload whose rows are emitted in dense-index order, preserving the
 * sequential access pattern used by the dynamic programs.
 */
typedef struct {
    /**
     * dense row -> slot within its owning partition.
     *
     * NULL exactly when `rank` is non-NULL: a sealed finalized table answers
     * the same question from its bitmap and keeps no direct directory.
     */
    uint32_t  *index;
    /**
     * The same directory in rank form, or NULL. Set only by
     * `beam_sparse_seal_rank`, only on a finalized table, and only when the
     * table is sparse enough that the trade is worth taking.
     */
    BeamRankBlock *rank;
    /**
     * `values[0]` cached, for the read-only lookup below. Maintained wherever
     * `values[0]` is assigned, and NULL on a mutable table -- which is what
     * makes it a safe discriminator rather than a second thing to keep in
     * step by hand.
     */
    double    *values0;
    double   **values;      /* [n_parts][capacity[p] * stride]               */
    /**
     * slot -> dense row. MUTABLE TABLES ONLY, and NULL on a finalized one --
     * which is what every site tests, rather than `mutable_rows`.
     *
     * A mutable table needs it to recover a moved row's identity after an
     * erase, and to turn a wrong-partition lookup into a miss instead of an
     * alias. A finalized table carried one so its occupied rows could be
     * listed; `beam_layer_max` was the only caller that ever listed them and
     * now reads the directory slice for its depth bucket instead, so the map
     * is 16.32 MiB of the foundation that nothing would load.
     */
    uint32_t **row_ids;
    uint32_t  *size;        /* occupied rows per partition                   */
    uint32_t  *capacity;    /* allocated rows per partition                  */
    uint32_t   n_dense_rows;
    uint32_t   n_parts;
    /**
     * Doubles per row: the emitting alphabet's residue count.
     *
     * Held here rather than read from the model, because the innermost
     * lookup below has no model to read it from, and threading one through
     * would cost a pointer chase in the hottest loop in the library to
     * recover a number that cannot change for the life of the table.
     */
    uint32_t   stride;
    bool       mutable_rows;
} BeamSparseRows;

BeamError beam_sparse_init(BeamSparseRows *t, uint32_t n_dense_rows,
                              uint32_t n_parts, uint32_t stride,
                              bool mutable_rows);
void beam_sparse_destroy(BeamSparseRows *t);
const double *beam_sparse_row(const BeamSparseRows *t, uint32_t row,
                                  uint32_t part);
double *beam_sparse_row_mut(BeamSparseRows *t, uint32_t row,
                                uint32_t part, bool create);
bool beam_sparse_erase(BeamSparseRows *t, uint32_t row, uint32_t part);
void beam_sparse_shrink(BeamSparseRows *t);
/**
 * Replace a finalized table's direct directory with the rank form.
 *
 * Verifies the slot-ordering invariant it depends on rather than trusting it,
 * and fails without touching the table if it does not hold. A table that is
 * not finalized, is already sealed, or holds no rows is left alone and
 * reported OK: sealing is an optimisation, not a state callers depend on.
 */
BeamError beam_sparse_seal_rank(BeamSparseRows *t);
uint64_t beam_sparse_memory_bytes(const BeamSparseRows *t);
uint64_t beam_sparse_n_rows(const BeamSparseRows *t);

/**
 * Does this table hold `row`? Answered by whichever directory it carries.
 *
 * Callers that only need presence -- `beam_layer_max` walking one depth
 * bucket -- should ask this rather than read `index`, which a sealed table
 * does not have. On the rank form one 40-byte block answers 256 consecutive
 * rows, so a scan over a slice moves 1.25 bits per row instead of 32: less
 * memory traffic than the direct directory, not more.
 *
 * The caller guarantees `row < n_dense_rows`.
 */
static inline bool beam_sparse_present(const BeamSparseRows *t, uint32_t row) {
    if (t->rank) {
        const BeamRankBlock *b = &t->rank[row >> 8];
        return ((b->bits[(row >> 6) & 3u] >> (row & 63u)) & 1ull) != 0ull;
    }
    return t->index[row] != BEAM_ROW_MISSING;
}

/**
 * A finalized table's row, without the validation `beam_sparse_row` performs.
 *
 * That function is written for BOTH generations, so it re-establishes on every
 * call what only a mutable table can violate: seven null and range tests, a
 * `double **` indirection, and two loads from one-element `size` and
 * `capacity` arrays. A finalized table is immutable, has exactly one partition
 * (`beam_sparse_init` refuses any other), and has `capacity == size` after
 * `beam_sparse_shrink`, so not one of those tests can fail -- and this is the
 * innermost lookup of every dynamic program in the library, so re-proving it
 * tens of millions of times per analytic is most of what the lookup costs.
 * Measured at 1.53-2.26x on the directory and 1.03-1.40x on the whole layer
 * kernel, for the identical bytes.
 *
 * `values0` is NULL on a mutable table, so the fallback is not a convenience:
 * it is what keeps a caller that reaches here with the wrong table generation
 * correct rather than fast and wrong.
 *
 * The caller guarantees `row < n_dense_rows`, which every `beam_lp_*_row`
 * addressing helper does by construction. A table holding no rows at all has
 * every index entry MISSING, so `values0` is never dereferenced when it would
 * be NULL for that reason.
 */
static inline const double *beam_sparse_row_final(const BeamSparseRows *t,
                                                      uint32_t row) {
    if (!t->values0) return beam_sparse_row(t, row, 0u);
    if (t->rank) {
        /* One block is 40 bytes and the whole directory is about a megabyte,
         * so this lands in cache where the 28 MiB direct array does not --
         * which is why the extra popcount is not the cost it looks like. */
        const BeamRankBlock *b = &t->rank[row >> 8];
        const uint32_t ofs  = row & 255u;
        const uint32_t w    = ofs >> 6;
        const uint64_t word = b->bits[w];
        const uint64_t bit  = 1ull << (ofs & 63u);
        if (!(word & bit)) return NULL;
        const uint32_t sub = w ? ((b->sub >> (9u * (w - 1u))) & 511u) : 0u;
        const uint32_t slot = b->rank + sub
                            + (uint32_t)__builtin_popcountll(word & (bit - 1ull));
        return t->values0 + (size_t)slot * t->stride;
    }
    const uint32_t slot = t->index[row];
    if (slot == BEAM_ROW_MISSING) return NULL;
    return t->values0 + (size_t)slot * t->stride;
}

#endif /* BEAM_INTERNAL_SPARSE_ROWS_H */
