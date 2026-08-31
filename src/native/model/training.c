/**
 * @file training.c
 * @brief Observation and count accumulation.
 */
#include <stdio.h>

#include "beam/internal.h"


/* ── Count accumulation ──────────────────────────────────────── */

/**
 * Record one sequence into one direction's tables.
 *
 * `depth_of(p)` is the residue's distance from this direction's anchor and
 * `ctx_of(p, k)` its order-`k` context on the anchor side. Everything else is
 * identical between the directions, which is the point of the design.
 */
static bool observe_one_position(BeamModel *m, uint32_t dir,
                                 const uint8_t *s, uint32_t len, uint32_t p,
                                 uint32_t depth, double w) {
    uint32_t cw_ix, cm_ix, cl_ix;
    if (dir == BEAM_FWD) {
        cw_ix = beam_ctx_forward(s, (int32_t)p, m->order, m->n_alphabet);
        cm_ix = beam_ctx_forward(s, (int32_t)p, 3u, m->n_alphabet);
        cl_ix = beam_ctx_forward(s, (int32_t)p, 2u, m->n_alphabet);
    } else {
        cw_ix = beam_ctx_backward(s, (int32_t)p, len, m->order, m->n_alphabet);
        cm_ix = beam_ctx_backward(s, (int32_t)p, len, 3u, m->n_alphabet);
        cl_ix = beam_ctx_backward(s, (int32_t)p, len, 2u, m->n_alphabet);
    }
    const uint32_t x = s[p];
    if (!m->cfg.sparse) {
        m->cnt_wide[dir][beam_ix_wide(m, depth, cw_ix, x)] += w;
        m->cnt_mid[dir][beam_ix_mid(m, depth, cm_ix, x)]   += w;
        m->cnt_len[dir][beam_ix_len(m, len, depth, cl_ix, x)] += w;
        return true;
    }

    const uint32_t wd = beam_depth_part(m, depth);
    const uint32_t rw = wd * m->n_ctx_wide + cw_ix;
    const uint32_t rm = depth * m->n_ctx_mid + cm_ix;
    const uint32_t rl = (len * m->max_len + depth) * m->n_ctx_len + cl_ix;
    double *wide = beam_cnt_wide_row_mut(m, dir, rw, true);
    double *mid  = beam_cnt_mid_row_mut(m, dir, rm, true);
    double *lng  = beam_cnt_len_row_mut(m, dir, rl, true);
    if (!wide || !mid || !lng) return false;
    wide[x] += w;
    mid[x]  += w;
    lng[x]  += w;
    return true;
}

/** Sequences encoded at a time. Bounds the scratch at a few MB whatever `n`. */
#define BEAM_OBS_CHUNK 262144u

/**
 * Count one chunk of pre-encoded sequences into the tables, in parallel.
 *
 * The parallel axis is the DEPTH, not the sequence, and that is what makes
 * this exact rather than merely fast. Every cell of every table carries the
 * depth in its index, so partitioning the depths partitions the cells: no two
 * threads can touch the same cell, no atomics are needed, and — because each
 * thread still walks the sequences in ascending index — every cell accumulates
 * its contributions in exactly the order the serial loop would have added
 * them. The counts are bit-identical, not just equivalent.
 *
 * Two details make that true rather than nearly true. The wide table caps its
 * depth index at `depth_cap`, so every depth at or beyond the cap shares one
 * cell and they must be handled by ONE task, not spread across several. And
 * within that merged task the depths have to be visited in the order the
 * serial pass met them, which is ascending position: ascending depth going
 * forwards, descending depth going backwards.
 */
static bool observe_chunk(BeamModel *m, const uint8_t *enc,
                          const uint32_t *len_of, const double *wt,
                          uint32_t n) {
    const uint32_t dc = m->cfg.depth_cap;
    const uint32_t nb = dc + 1u;
    const uint32_t stride = m->max_len;
    const int64_t ntask = 2 * (int64_t)nb;
    int failed = 0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t task = 0; task < ntask; ++task) {
        const uint32_t dir = (uint32_t)(task / nb);
        const uint32_t b   = (uint32_t)(task % nb);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t len = len_of[i];
            if (len == 0 || b >= len) continue;
            const uint8_t *s = enc + (size_t)i * stride;
            const double w = wt[i];
            const uint32_t hi = (b == dc) ? len : b + 1u;   /* half-open */
            if (dir == BEAM_FWD) {
                for (uint32_t d = b; d < hi; ++d)
                    if (!observe_one_position(m, dir, s, len, d, d, w))
                        failed = 1;
            } else {
                for (uint32_t d = hi; d-- > b; )
                    if (!observe_one_position(m, dir, s, len,
                                              len - 1u - d, d, w))
                        failed = 1;
            }
        }
    }
    return failed == 0;
}

BeamError beam_model_observe(BeamModel *m,
                                const char *const *seqs,
                                const double *abundances,
                                uint32_t n) {
    if (!m || !seqs)
        return BEAM_FAIL(BEAM_ERR_NULL_ARG, "observe: %s must not be NULL",
                        !m ? "model" : "sequences");
    if (m->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "observe: the model is compacted, so its counts are "
                        "gone; build a fresh model instead");
    if (n == 0) return BEAM_OK;
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);

    const uint32_t stride = m->max_len;
    const uint32_t cap = n < BEAM_OBS_CHUNK ? n : BEAM_OBS_CHUNK;
    uint8_t  *enc    = (uint8_t *)malloc((size_t)cap * stride);
    uint32_t *len_of = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    double   *wt     = (double *)malloc((size_t)cap * sizeof(double));
    if (!enc || !len_of || !wt) {
        free(enc); free(len_of); free(wt);
        return BEAM_EALLOC("the observation scratch buffers");
    }

    for (uint32_t base = 0; base < n; base += cap) {
        const uint32_t cnt = (n - base) < cap ? (n - base) : cap;
        /* Validation and the scalar totals stay in sequence order: they are
         * cheap, and `total_weight` is a running sum whose value depends on
         * the order it is taken in. */
        for (uint32_t k = 0; k < cnt; ++k) {
            uint32_t len = 0;
            const double w = abundances ? abundances[base + k] : 1.0;
            if (!beam_encode(m, seqs[base + k], enc + (size_t)k * stride, &len)
                || !(w > 0.0) || !isfinite(w)) {
                len_of[k] = 0; wt[k] = 0.0; m->n_skipped++;
                continue;
            }
            len_of[k] = len;
            wt[k] = w;
            m->cnt_length[len] += w;
            m->total_weight    += w;
            m->n_observed++;
        }
        if (!observe_chunk(m, enc, len_of, wt, cnt)) {
            free(enc); free(len_of); free(wt);
            m->finalized = false;
            return BEAM_EALLOC("a sparse count-table row");
        }
    }
    free(enc); free(len_of); free(wt);
    m->finalized = false;
    return BEAM_OK;
}
