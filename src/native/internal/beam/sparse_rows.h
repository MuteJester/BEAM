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

#endif /* BEAM_INTERNAL_SPARSE_ROWS_H */
