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
 * a per-repertoire model, which occupies far less -- 8.08% on P00322, and
 * 0.86% to 31.23% across the donors measured below. This form spends 1.25 bits
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

/**
 * Intern a finalized table's rows when at least this fraction repeat.
 *
 * Measured on the Emerson cohort, order 4: 73.1-79.1% of a donor's finalized
 * rows are byte-identical to another row, and 49-56% are the same twenty
 * bytes -- one residue at exactly 0.0 and nineteen at -inf -- of which only
 * twenty distinct ones can exist. A 197k-sequence donor holds 597,373 rows
 * that are 158,872 distinct.
 *
 * The foundation is the other case: 411,757 of 4,278,736 rows repeat, 9.6%,
 * which does not pay for the indirection interning adds. So this is decided
 * per table from what that table actually contains, like the directory above.
 */
#define BEAM_INTERN_MIN_DUPLICATE_NUM   1u
#define BEAM_INTERN_MIN_DUPLICATE_DENOM 4u

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
     * Dense position -> row in a shared pool, or NULL when rows are not
     * interned.
     *
     * The directory answers "which of the occupied rows is this", 0..size-1.
     * Without a pool that IS the payload row. With one it is an index into
     * this map, whose entries name `n_pool` distinct payload rows that
     * several dense positions may share. One extra load, and the rows a
     * caller receives are bit-identical either way -- interning shares
     * storage between rows that were already equal to the byte.
     */
    uint32_t  *pool_id;
    uint32_t   n_pool;      /* distinct payload rows; 0 when not interned    */

    /**
     * The payload as a dictionary of values rather than an array of them.
     * NULL together when the table stores plain 20-double rows.
     *
     * Every value in a finalized table is `log(c / t)` for integers c <= t,
     * and a repertoire's counts are small -- 80% of cells are at most 255 --
     * so the reachable values are the logs of a bounded set of small
     * rationals. Measured on real repertoires, 1,175,396 finite cells take
     * 59,008 distinct values: 95% of the stored doubles repeat one already
     * stored elsewhere.
     *
     * Row interning above collapses rows identical in ALL twenty cells. That
     * is the same idea one level too coarse: two rows differing in a single
     * residue share nineteen values and it catches neither. The numbers above
     * are what remains AFTER it runs.
     *
     * So: one bit per residue saying whether the cell exists, a two-byte
     * index per existing cell, and the distinct values once each. Exact --
     * the dictionary holds the same bit patterns -- and 5.35x smaller.
     *
     * IT IS NOT FREE, and the cost is on the hottest path. Measured on a
     * 197,139-sequence per-repertoire model, all six tables packed, best of
     * five, identical checksums either way:
     *
     *                       packed      plain     ratio
     *     model          13,829,157  37,898,429   2.74x smaller
     *     scoring            15.4 ms     8.8 ms   1.75x SLOWER
     *     Hill sweep        357.2 ms   315.1 ms   1.13x slower
     *     build             713 ms     672 ms     1.06x slower
     *
     * Scoring pays most because it reads single cells: a mask test, a
     * popcount over the bits below the residue, then an indirection through
     * the dictionary, where a plain row is one load. The sweeps read whole
     * rows and amortise that over twenty cells.
     *
     * Kept deliberately, and the trade was reviewed with the numbers above
     * rather than assumed: a caller holds many per-repertoire models at once
     * and 24 MB each accumulates faster than 7 microseconds per scored
     * sequence. A scoring-bound workload would want the opposite, which is
     * what `beam_sparse_pack_cells` declining leaves available.
     *
     * The foundation is unaffected -- it declines, having more distinct
     * values than a two-byte index can name.
     */
    /**
     * Doubles between one residue of a row and the next: 1 when the row is
     * twenty contiguous doubles, or the payload's row count when the table is
     * stored plane-major.
     *
     * A table read one residue at a time wants the latter. The backward half
     * of the edge kernel reads exactly one residue from each of twenty rows
     * whose dense ids are consecutive -- and, measured on the foundation, whose
     * SLOTS are consecutive in every one of the 78,538 tiles where all twenty
     * are present. Laid out by row that is fifty cache lines to use a hundred
     * and sixty bytes; laid out by residue it is those bytes contiguous.
     *
     * 1 for every other table, which leaves them exactly as they were.
     */
    uint32_t   plane_step;
    uint32_t  *cell_mask;   /* [n_cell_rows] residue presence, bit per x     */
    uint32_t  *cell_start;  /* [n_cell_rows + 1] first index of each row     */
    uint16_t  *cell_index;  /* [cell_start[n]] positions in `dict`           */
    double    *dict;        /* [n_dict] the distinct values, once each       */
    uint32_t   n_dict;
    uint32_t   n_cell_rows; /* payload rows the three arrays above describe  */
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
/**
 * Give a partition room for exactly `rows` rows, now.
 *
 * For a caller that already knows how many rows it is about to add. Filling
 * after this never grows, so it never allocates a second payload to copy the
 * first into, and leaves no slack for `beam_sparse_shrink` to reclaim.
 * Growth from nothing to 1,846,623 rows otherwise doubles twenty-one times,
 * copying every row already written each time and overshooting the answer by
 * up to a factor of two.
 *
 * Never shrinks: a partition already larger than `rows` is left alone.
 */
BeamError beam_sparse_reserve(BeamSparseRows *t, uint32_t part, uint32_t rows);

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
/**
 * Share storage between a finalized table's byte-identical rows.
 *
 * Exact by construction: two rows are pooled only when all `stride * 8` bytes
 * match, so every caller receives the same doubles it received before. Leaves
 * the table alone -- and reports OK -- when too few rows repeat to pay for
 * the indirection, or when the table cannot be interned at all.
 */
BeamError beam_sparse_intern_rows(BeamSparseRows *t);
/**
 * Replace a finalized table's plain rows with mask-and-dictionary cells.
 *
 * Exact: the dictionary stores the same bit patterns, so every value handed
 * back is the one that was there. Declines -- reporting OK -- when the table
 * cannot take it or the value set is too large for a two-byte index, which is
 * how a corpus with many distinct counts keeps the plain rows it is better
 * served by.
 */
BeamError beam_sparse_pack_cells(BeamSparseRows *t);

/**
 * Re-lay a finalized payload by residue rather than by row.
 *
 * For a table read one residue at a time. Declines -- reporting OK -- for a
 * table that is packed, mutable, already transposed or empty, so a caller may
 * ask without checking.
 */
BeamError beam_sparse_plane_transpose(BeamSparseRows *t);

/** The most distinct values a packed table may hold, so an index is two bytes. */
#define BEAM_PACK_MAX_DICT 65536u
uint64_t beam_sparse_memory_bytes(const BeamSparseRows *t);
uint64_t beam_sparse_n_rows(const BeamSparseRows *t);

/**
 * A finalized row, in whichever form the table holds it.
 *
 * The kernels used to take a `const double *` and index it, which only a
 * table of contiguous twenty-double rows can supply. A packed table has no
 * such array to point at, so what they take now is this: either that pointer,
 * or the three things a packed row needs to answer the same question.
 *
 * `dense` is the discriminator and it is uniform for the life of a table, so
 * the branch in `beam_log_cell` is the same way every time it is reached in a
 * given run -- which is the only reason it is acceptable in that loop.
 */
/*
 * Initialise this DESIGNATED, never positionally. `step` and `mask` are
 * adjacent uint32_t, so a reordering or an inserted field turns `step = 1`
 * into `step = 0` silently -- and a zero step makes every residue of a dense
 * row read residue 0, which is a plausible number for all twenty. `-Wall
 * -Wextra` says nothing, because every member is still initialised with a
 * compatible type. Three of the four initialisers live in table_access.h,
 * where a change here is not visible.
 */
typedef struct {
    const double   *dense;   /* the row's residue 0, or NULL if packed       */
    const uint16_t *index;   /* this row's dictionary positions, if packed   */
    const double   *dict;
    uint32_t        step;    /* doubles between residues: 1, or one plane    */
    uint32_t        mask;    /* residues present, if packed                  */
    bool            present; /* false when the table holds no such row       */
} BeamLogRow;

/**
 * All `stride` residues of `r` at once.
 *
 * The forward kernel wants a whole row, and asking `beam_log_cell` for each
 * residue would rank the same mask twenty times. A packed row's indices are
 * stored in ascending residue order, so one walk does it with no popcount at
 * all.
 */
static inline void beam_log_row_into(const BeamLogRow *r, uint32_t stride,
                                     double *out) {
    if (r->dense) {
        const double *v = r->dense;
        for (uint32_t x = 0; x < stride; ++x, v += r->step) out[x] = *v;
        return;
    }
    const uint16_t *ix = r->index;
    const uint32_t mask = r->mask;
    for (uint32_t x = 0; x < stride; ++x)
        out[x] = ((mask >> x) & 1u) ? r->dict[*ix++] : -INFINITY;
}

/** Residue `x` of `r`, or `-INFINITY` when this row never emitted it. */
static inline double beam_log_cell(const BeamLogRow *r, uint32_t x) {
    if (r->dense) return r->dense[(size_t)x * r->step];
    if (!((r->mask >> x) & 1u)) return -INFINITY;
    return r->dict[r->index[__builtin_popcount(r->mask & ((1u << x) - 1u))]];
}

/**
 * One cell of a packed payload row, or `-INFINITY` when it is absent.
 *
 * `slot` is a payload row -- what the directory returns, after any pool
 * mapping. Used where the caller wants a single residue rather than a row,
 * which is what the backward half of the edge kernel does: it reads one
 * residue from each of twenty different wide rows.
 *
 * A PLAIN backward table no longer pays the scatter that motivated this:
 * `beam_sparse_plane_transpose` lays it out by residue, so those twenty
 * values are contiguous. A packed table declines that transpose -- there is
 * no array of rows to transpose -- so here the twenty reads remain twenty
 * separate index runs, and this is the form that still wants a better
 * layout.
 */
static inline double beam_sparse_cell(const BeamSparseRows *t, uint32_t slot,
                                      uint32_t x) {
    const uint32_t mask = t->cell_mask[slot];
    if (!((mask >> x) & 1u)) return -INFINITY;
    const uint32_t at = t->cell_start[slot]
                      + (uint32_t)__builtin_popcount(mask & ((1u << x) - 1u));
    return t->dict[t->cell_index[at]];
}

/**
 * A whole packed row, expanded into `out` (`stride` doubles).
 *
 * The indices of a row are stored in ascending residue order, so this walks
 * them once rather than computing a rank per residue.
 */
static inline void beam_sparse_row_into(const BeamSparseRows *t, uint32_t slot,
                                        uint32_t stride, double *out) {
    const uint32_t mask = t->cell_mask[slot];
    const uint16_t *ix = t->cell_index + t->cell_start[slot];
    for (uint32_t x = 0; x < stride; ++x)
        out[x] = ((mask >> x) & 1u) ? t->dict[*ix++] : -INFINITY;
}

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
 * A forward walk over the occupied rows of a half-open dense range.
 *
 * `beam_sparse_present` answers one row, and a caller sweeping a range asks it
 * once per row -- including for every row that is not there. On the rank form
 * that is a block index, a word load, a shift and a mask to discover an
 * absence; on the foundation's forward wide table a single depth bucket asks
 * it 194,481 times and roughly 82,000 of those find nothing.
 *
 * This asks the same question of sixty-four rows at a time. One word load
 * yields every occupied row in it through `ctz`, so absent rows cost nothing
 * beyond the bit they occupy, and a fully empty word is skipped whole. The
 * sparser the table the larger the saving, which is the opposite of where the
 * rank form's other costs fall.
 *
 * Ascending order, so a caller that also wants payload slots can keep a
 * running counter instead of a rank lookup per row -- see
 * `beam_row_cursor_slot0`. `layer_max` wants presence only.
 *
 * The direct form keeps its row-at-a-time scan: there is no bitmap to walk,
 * and `index` is one 32-bit load per row either way.
 */
typedef struct {
    const BeamSparseRows *t;
    uint64_t word;      /* unconsumed bits of word `w`, low bit = row w*64 */
    uint32_t w;         /* current word index, rank form only              */
    uint32_t w_last;    /* last word index this range touches              */
    uint32_t row;       /* next row to examine, direct form only           */
    uint32_t end;       /* one past the last row of the range              */
} BeamRowCursor;

/** Word `w` of a sealed directory's bitmap, across the 256-row blocks. */
static inline uint64_t beam_rank_word(const BeamSparseRows *t, uint32_t w) {
    return t->rank[w >> 2].bits[w & 3u];
}

static inline void beam_row_cursor_init(BeamRowCursor *c,
                                        const BeamSparseRows *t,
                                        uint32_t first, uint32_t end) {
    c->t = t; c->end = end; c->row = first; c->word = 0u;
    c->w = first >> 6; c->w_last = end ? (end - 1u) >> 6 : 0u;
    if (t->rank && first < end) {
        /* Drop the bits below `first`; bits at or past `end` are rejected by
         * the range test in `next`, which costs nothing on the common word. */
        c->word = beam_rank_word(t, c->w) & (~0ull << (first & 63u));
    }
}

/** The next occupied row at or after the cursor, or false when the range ends. */
static inline bool beam_row_cursor_next(BeamRowCursor *c, uint32_t *out) {
    if (c->t->rank) {
        for (;;) {
            while (c->word == 0ull) {
                if (c->w >= c->w_last) return false;
                c->word = beam_rank_word(c->t, ++c->w);
            }
            const uint32_t r = (c->w << 6) + (uint32_t)__builtin_ctzll(c->word);
            c->word &= c->word - 1ull;               /* clear the low set bit */
            if (r >= c->end) return false;
            *out = r; return true;
        }
    }
    while (c->row < c->end) {
        const uint32_t r = c->row++;
        if (c->t->index[r] != BEAM_ROW_MISSING) { *out = r; return true; }
    }
    return false;
}

/**
 * Occupied rows strictly before `row`, which is that row's payload slot when
 * it is occupied and the slot the next occupied row will take when it is not.
 *
 * Exact only on a finalized table, where slots are assigned in ascending row
 * order. Lets a parallel walk split a range on block boundaries and recover
 * each chunk's starting slot in constant time rather than counting up to it.
 */
static inline uint32_t beam_row_cursor_slot0(const BeamSparseRows *t,
                                             uint32_t row) {
    if (!t->rank) return 0u;                 /* direct form reads `index`   */
    const BeamRankBlock *b = &t->rank[row >> 8];
    const uint32_t w = (row >> 6) & 3u;
    const uint32_t sub = w ? ((b->sub >> (9u * (w - 1u))) & 511u) : 0u;
    return b->rank + sub
         + (uint32_t)__builtin_popcountll(b->bits[w] & ((1ull << (row & 63u)) - 1ull));
}

/**
 * `beam_sparse_row_final` as a reference rather than a pointer, so a packed
 * table can answer it too. Same directory work; only what it hands back
 * differs.
 */
static inline BeamLogRow beam_sparse_log_row(const BeamSparseRows *t,
                                             uint32_t row) {
    BeamLogRow r = { .step = 1u };
    uint32_t slot;
    if (t->rank) {
        const BeamRankBlock *b = &t->rank[row >> 8];
        const uint32_t ofs  = row & 255u;
        const uint32_t w    = ofs >> 6;
        const uint64_t word = b->bits[w];
        const uint64_t bit  = 1ull << (ofs & 63u);
        if (!(word & bit)) return r;
        const uint32_t sub = w ? ((b->sub >> (9u * (w - 1u))) & 511u) : 0u;
        slot = b->rank + sub
             + (uint32_t)__builtin_popcountll(word & (bit - 1ull));
    } else {
        if (!t->index) return r;
        slot = t->index[row];
        if (slot == BEAM_ROW_MISSING) return r;
    }
    if (t->pool_id) slot = t->pool_id[slot];
    r.present = true;
    if (t->cell_mask) {
        r.mask  = t->cell_mask[slot];
        r.index = t->cell_index + t->cell_start[slot];
        r.dict  = t->dict;
    } else if (t->plane_step > 1u) {
        r.dense = t->values0 + slot;          /* residue x is `step` further on */
        r.step  = t->plane_step;
    } else {
        r.dense = t->values0 + (size_t)slot * t->stride;
        r.step  = 1u;
    }
    return r;
}


/* ── Compaction of a finalized table ─────────────────────────── */

/**
 * Which of the four transforms actually happened.
 *
 * All four keep the table as it is and report `BEAM_OK` when they cannot
 * apply, which for a single call is right -- none is needed for correctness,
 * only for size. The cost of that is a pipeline whose failure to compact is
 * invisible: a model can keep a 28.19 MiB directory on each of six tables with
 * no error, no log, and `beam_model_memory_bytes` reporting the larger figure
 * with nothing to compare it against. This is the signal.
 *
 * `packed` and `transposed` are never both true. The transpose has no array of
 * rows to work on once the cells are packed, so the two are alternatives and
 * the pipeline gives packing priority -- see `storage/compaction.c`.
 */
typedef struct {
    bool sealed;        /**< directory swapped for the rank form   */
    bool interned;      /**< byte-identical rows now share storage */
    bool packed;        /**< cells replaced by dictionary indices  */
    bool transposed;    /**< backward table relaid plane-major     */
} BeamCompactionApplied;

/**
 * Apply all four, in the one order that works, to a finalized table.
 *
 * `n_dense` is the table's dense row count, which the occupancy test compares
 * against. `backward` selects the transpose, which is for C-anchored tables
 * alone. `applied` may be NULL.
 */
BeamError beam_sparse_compact_finalized(BeamSparseRows *t, uint32_t n_dense,
                                        bool backward,
                                        BeamCompactionApplied *applied);

#endif /* BEAM_INTERNAL_SPARSE_ROWS_H */
