/**
 * @file extrema.c
 * @brief Accumulator bounds and the surprisal range.
 */
#include "beam/internal.h"


/* ── Accumulator range ───────────────────────────────────────── */

/**
 * Max-plus sweep giving the largest value the spectrum DP must be able to
 * represent, which is the larger of two things that are NOT the same:
 *
 *   - the running accumulator, which grows monotonically along the path;
 *   - the final surprisal, which is the accumulator plus the per-length
 *     constant `ln Z(L) - ln pi(L) - tail`.
 *
 * Neither dominates in general, so the larger of the two is returned. On these
 * models the surprisal is the binding one — the sink term `-tail` is positive
 * and the per-length constant is a fraction of a nat either way — so bounding
 * only the accumulator would size a grid too small. A strongly negative
 * constant flips it, and a grid sized to the surprisal alone would then index
 * past its own top. Both are covered here.
 */
static double accumulator_max_at(const BeamModel *m, uint32_t len) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)malloc((size_t)S * sizeof(double));
    double *next = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !cur || !next || !tail) {
        free(w); free(cur); free(next); free(tail);
        (void)BEAM_EALLOC("the max-plus accumulator sweep buffers");
        return 0.0;
    }
    double hi = 0.0;
    for (uint32_t s = 0; s < S; ++s) cur[s] = -INFINITY;
    cur[S - 1] = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_layer_weights(m, i, len, w, NULL);
        for (uint32_t s = 0; s < S; ++s) next[s] = -INFINITY;
        for (uint32_t s = 0; s < S; ++s) {
            if (cur[s] == -INFINITY) continue;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY) continue;
                const double v = cur[s] - row[x];      /* surprisal accumulates */
                if (v > next[base + x]) next[base + x] = v;
            }
        }
        double *tmp = cur; cur = next; next = tmp;
    }
    beam_tail_weights(m, len, tail);
    {
        const double off = m->log_z[len] - m->log_pi[len];
        for (uint32_t s = 0; s < S; ++s) {
            if (cur[s] == -INFINITY) continue;
            if (cur[s] > hi) hi = cur[s];
            if (tail[s] == -INFINITY) continue;
            const double surprisal = cur[s] - tail[s] + off;
            if (surprisal > hi) hi = surprisal;
        }
    }
    free(w); free(cur); free(next); free(tail);
    return hi;
}

/**
 * Largest value a spectrum grid has to represent.
 *
 * Returns 0.0 both for a model that generates nothing and for an allocation
 * failure. Both set the thread-local error context, so `beam_last_error()`
 * separates them after a zero.
 */
double beam_model_accumulator_max(const BeamModel *m) {
    if (!m || !m->finalized) {
        if (!m) beam_set_error(BEAM_ERR_NULL_ARG,
                              "accumulator_max: model must not be NULL");
        else    (void)BEAM_ENOTBUILT();
        return 0.0;
    }
    beam_clear_error();
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!work) { (void)BEAM_EALLOC("the per-length work list"); return 0.0; }
    const uint32_t nw = beam_live_lengths_desc(m, work);
    double hi = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:hi) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        const double v = accumulator_max_at(m, work[k]);
        if (v > hi) hi = v;
    }
    free(work);
    return hi;
}

/* ── Surprisal range ─────────────────────────────────────────── */

/**
 * Min-plus and max-plus sweeps over one length, giving the exact extremes of
 * `-ln P` among producible sequences of that length.
 */
static BeamError range_at(const BeamModel *m, uint32_t len,
                         double *out_min, double *out_max) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *lo   = (double *)malloc((size_t)S * sizeof(double));
    double *hi   = (double *)malloc((size_t)S * sizeof(double));
    double *nlo  = (double *)malloc((size_t)S * sizeof(double));
    double *nhi  = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !lo || !hi || !nlo || !nhi || !tail) {
        free(w); free(lo); free(hi); free(nlo); free(nhi); free(tail);
        return BEAM_EALLOC("the min-plus / max-plus sweep buffers");
    }
    for (uint32_t s = 0; s < S; ++s) { lo[s] = INFINITY; hi[s] = -INFINITY; }
    lo[S - 1] = 0.0; hi[S - 1] = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_layer_weights(m, i, len, w, NULL);
        for (uint32_t s = 0; s < S; ++s) { nlo[s] = INFINITY; nhi[s] = -INFINITY; }
        for (uint32_t s = 0; s < S; ++s) {
            if (hi[s] == -INFINITY) continue;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY) continue;
                const double a = lo[s] - row[x];      /* surprisal accumulates */
                const double b = hi[s] - row[x];
                if (a < nlo[base + x]) nlo[base + x] = a;
                if (b > nhi[base + x]) nhi[base + x] = b;
            }
        }
        memcpy(lo, nlo, (size_t)S * sizeof(double));
        memcpy(hi, nhi, (size_t)S * sizeof(double));
    }

    beam_tail_weights(m, len, tail);
    const double off = m->log_z[len] - m->log_pi[len];
    double best = INFINITY, worst = -INFINITY;
    for (uint32_t s = 0; s < S; ++s) {
        if (tail[s] == -INFINITY || hi[s] == -INFINITY) continue;
        const double a = lo[s] - tail[s] + off;
        const double b = hi[s] - tail[s] + off;
        if (a < best) best = a;
        if (b > worst) worst = b;
    }
    free(w); free(lo); free(hi); free(nlo); free(nhi); free(tail);
    *out_min = best;
    *out_max = worst;
    return BEAM_OK;
}

BeamError beam_model_surprisal_range(const BeamModel *m,
                                        double *out_min, double *out_max) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!work) return BEAM_EALLOC("the per-length work list");
    const uint32_t nw = beam_live_lengths_desc(m, work);
    double best = INFINITY, worst = -INFINITY;
    int failed = BEAM_OK;
    /* Minimum and maximum are order-independent, so reducing them across
     * threads gives the same answer the serial scan did, bit for bit. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(min:best) \
            reduction(max:worst) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double a = 0.0, b = 0.0;
        const BeamError e = range_at(m, work[k], &a, &b);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; continue; }
        if (a < best) best = a;
        if (b > worst) worst = b;
    }
    free(work);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                        "surprisal_range: a per-length sweep failed");
    if (best == INFINITY)
        return BEAM_ENOPATHS("no length has a producible sequence, so "
                                 "the surprisal has no range");
    if (out_min) *out_min = best;
    if (out_max) *out_max = worst;
    return BEAM_OK;
}
