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
 * The row an unobserved top row IS under the hierarchical prior: its nearest
 * observed ancestor, dropping the most distant residue one level at a time,
 * and the uniform row below level 0.
 *
 * The forward table is packed most-distant-first, so a level-k context is the
 * top context's low k digits; the backward table is packed nearest-first, so
 * each level up drops the LAST digit. Kept out of line: it runs only on a
 * miss, and inlining it into every kernel that reads a row would double their
 * size for the path that stored rows never take.
 */
static BeamLogRow beam_lp_ancestor(const BeamModel *m, uint32_t dir,
                                   uint32_t depth, uint32_t ctx)
    __attribute__((unused, noinline));
static BeamLogRow beam_lp_ancestor(const BeamModel *m, uint32_t dir,
                                   uint32_t depth, uint32_t ctx) {
    uint32_t c = ctx;
    for (uint32_t k = m->order; k-- > 0; ) {
        c = (dir == BEAM_BWD) ? c / m->n_alphabet : c % m->lo_ctx[k];
        const BeamLogRow r = beam_sparse_log_row(&m->sp_lo[dir][k],
                                                 depth * m->lo_ctx[k] + c);
        if (r.present) return r;
    }
    BeamLogRow u = { .dense = m->base_row, .step = 1u, .present = true };
    return u;
}

/*
 * One row of the finalized log table. Always `present`: under the
 * hierarchical prior every row is a distribution, a stored one or the
 * ancestor it resolves to. The flag survives because the kernels still test
 * it, and a row that could not be resolved at all would be a bug they should
 * keep reading as dead rather than as a distribution.
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
    if (m->cfg.sparse) {
        const BeamLogRow r = beam_sparse_log_row(&m->sp_lp[dir], row);
        if (__builtin_expect(r.present, 1)) return r;
        return beam_lp_ancestor(m, dir, depth, ctx);
    }
    BeamLogRow r = { .dense = m->lp[dir] + (size_t)row * m->n_residues,
                     .step = 1u, .present = true };
    return r;
}

#endif /* BEAM_INTERNAL_TABLE_ACCESS_H */
