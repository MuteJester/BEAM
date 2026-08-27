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
