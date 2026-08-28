/**
 * @file sparse_rows.h
 * @brief Sparse 20-value row storage.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_SPARSE_ROWS_H
#define BEAM_INTERNAL_SPARSE_ROWS_H

#include "beam/model.h"

/* ── Sparse 20-value row storage ────────────────────────────── */

#define BEAM_ROW_MISSING UINT32_MAX

/**
 * Dense row directory with sparse 20-double payloads.
 *
 * Count tables are partitioned by the depth task that owns their rows. That
 * lets the existing parallel observation loop grow each payload without a
 * lock: one task is the sole writer of one partition. Finalized tables have a
 * single payload whose rows are emitted in dense-index order, preserving the
 * sequential access pattern used by the dynamic programs.
 */
typedef struct {
    uint32_t  *index;       /* dense row -> slot within its owning partition */
    /**
     * `values[0]` cached, for the read-only lookup below. Maintained wherever
     * `values[0]` is assigned, and NULL on a mutable table -- which is what
     * makes it a safe discriminator rather than a second thing to keep in
     * step by hand.
     */
    double    *values0;
    double   **values;      /* [n_parts][capacity[p] * 20]                   */
    uint32_t **row_ids;     /* slot -> dense row, for every table.  Finalized
                             * tables fill ascending so theirs is sorted;
                             * mutable count tables fill in observation order
                             * and theirs is not. */
    uint32_t  *size;        /* occupied rows per partition                   */
    uint32_t  *capacity;    /* allocated rows per partition                  */
    uint32_t   n_dense_rows;
    uint32_t   n_parts;
    bool       mutable_rows;
} BeamSparseRows;

BeamError beam_sparse_init(BeamSparseRows *t, uint32_t n_dense_rows,
                              uint32_t n_parts, bool mutable_rows);
void beam_sparse_destroy(BeamSparseRows *t);
const uint32_t *beam_sparse_rows(const BeamSparseRows *t, uint32_t part,
                                     uint32_t *n);
const double *beam_sparse_row(const BeamSparseRows *t, uint32_t row,
                                  uint32_t part);
double *beam_sparse_row_mut(BeamSparseRows *t, uint32_t row,
                                uint32_t part, bool create);
bool beam_sparse_erase(BeamSparseRows *t, uint32_t row, uint32_t part);
void beam_sparse_shrink(BeamSparseRows *t);
uint64_t beam_sparse_memory_bytes(const BeamSparseRows *t);
uint64_t beam_sparse_n_rows(const BeamSparseRows *t);

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
    const uint32_t slot = t->index[row];
    if (slot == BEAM_ROW_MISSING) return NULL;
    return t->values0 + (size_t)slot * BEAM_RESIDUES;
}

#endif /* BEAM_INTERNAL_SPARSE_ROWS_H */
