/**
 * @file table_access.h
 * @brief Dense and sparse row access.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_TABLE_ACCESS_H
#define BEAM_INTERNAL_TABLE_ACCESS_H

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/sparse_rows.h"
#include "beam/indexing.h"

/* ── Dense/sparse row access ────────────────────────────────── */

/*
 * A row is `(depth, context)` and its sparse partition IS its depth. There is
 * no clamp and no second stride, so this is arithmetic rather than a case
 * analysis: the whole point of one table per direction is that there is one
 * answer to "which partition does this row live in".
 */
static inline uint32_t beam_part_for_row(const BeamModel *m, uint32_t row) {
    return row / m->n_ctx;
}

static inline uint32_t beam_count_n_rows(const BeamModel *m) {
    return m->max_len * m->n_ctx;
}

static inline const double *beam_cnt_row(const BeamModel *m, uint32_t dir,
                                         uint32_t row) {
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_cnt[dir], row, beam_part_for_row(m, row))
        : m->cnt[dir] + (size_t)row * m->n_residues;
}

static inline double *beam_cnt_row_mut(BeamModel *m, uint32_t dir,
                                       uint32_t row, bool create) {
    return m->cfg.sparse
        ? beam_sparse_row_mut(&m->sp_cnt[dir], row,
                              beam_part_for_row(m, row), create)
        : m->cnt[dir] + (size_t)row * m->n_residues;
}

static inline bool beam_count_erase(BeamModel *m, uint32_t dir,
                                    uint32_t row) {
    if (!m->cfg.sparse) return false;
    return beam_sparse_erase(&m->sp_cnt[dir], row, beam_part_for_row(m, row));
}

/*
 * One row of the finalized log table. `present` is false only in the sparse
 * form, where an unobserved row is absent rather than twenty cells of -inf.
 * A caller must test it before reading: that test is the closed world.
 *
 * BEAM_INLINE, not plain `static inline`. At -O2 GCC declined to inline the
 * three-expert ancestors of this function and emitted real calls, measured at
 * 3.75% of cycles inside the edge-weight kernel. There is one row accessor
 * now rather than three, but the reason is unchanged: it is three loads and a
 * branch, and the kernel reaches it tens of millions of times per analytic,
 * so a call would dominate the body.
 */
BEAM_INLINE BeamLogRow beam_lp_row(const BeamModel *m, uint32_t dir,
                                   uint32_t depth, uint32_t ctx) {
    const uint32_t row = beam_row(m, depth, ctx);
    if (m->cfg.sparse) return beam_sparse_log_row(&m->sp_lp[dir], row);
    BeamLogRow r = { .dense = m->lp[dir] + (size_t)row * m->n_residues,
                     .step = 1u, .present = true };
    return r;
}

#endif /* BEAM_INTERNAL_TABLE_ACCESS_H */
