/**
 * @file training.c
 * @brief Observation and count accumulation.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/provenance.h"


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
    /*
     * Counts are packed most-distant-first in BOTH directions. The finalized
     * C-anchored table is repacked nearest-first for cache reasons at the end
     * of finalize; these are counts, so they are not. Reading the two forms
     * as interchangeable is the one mistake here that produces a plausible,
     * wrongly-addressed model, which is why the two context helpers are named
     * apart.
     */
    const uint32_t ctx = (dir == BEAM_FWD)
        ? beam_ctx_forward(s, (int32_t)p, m->order, m->n_alphabet)
        : beam_ctx_backward_counts(s, (int32_t)p, len, m->order,
                                   m->n_alphabet);
    const uint32_t x = s[p];
    if (!m->cfg.sparse) {
        m->cnt[dir][beam_ix(m, depth, ctx, x)] += w;
        return true;
    }

    double *row = beam_cnt_row_mut(m, dir, beam_row(m, depth, ctx), true);
    if (!row) return false;
    row[x] += w;
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
 * That used to need two qualifications, and no longer does. The wide table
 * capped its depth index, so every depth at or beyond the cap shared one cell
 * and a single task had to own all of them, visiting them in the order the
 * serial pass met them. With depth exact there is no merged bucket: one task
 * owns one depth, one depth owns its own cells, and each is written by one
 * task in sequence order.
 */
static bool observe_chunk(BeamModel *m, const uint8_t *enc,
                          const uint32_t *len_of, const double *wt,
                          uint32_t n) {
    const uint32_t nb = m->max_len;
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
            const uint32_t hi = b + 1u;                     /* half-open */
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

static BeamError observe_body(BeamModel *m,
                                const char *const *seqs,
                                const double *abundances,
                                uint32_t n) {
    /* Counts may be frozen from a previous finalize; observation writes them,
     * so restore the mutable form first. Cheap to ask and a no-op when they
     * are already mutable, which is every call during a normal build. */
    {
        const BeamError thaw = beam_model_thaw_counts(m);
        if (thaw != BEAM_OK) return thaw;
    }

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

/*
 * The public entry, which exists only to time the body.
 *
 * Wrapped rather than stamped at each exit: the body has seven returns and a
 * build that is counted at six of them is worse than one counted at none,
 * because it looks right. See internal/beam/provenance.h for why the clock
 * is monotonic.
 */
BeamError beam_model_observe(BeamModel *m,
                                const char *const *seqs,
                                const double *abundances,
                                uint32_t n) {
    const int64_t t0 = beam_now_nanos();
    const BeamError e = observe_body(m, seqs, abundances, n);
    if (m) m->observe_nanos += beam_now_nanos() - t0;
    return e;
}
