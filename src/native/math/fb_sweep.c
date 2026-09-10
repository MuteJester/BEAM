/**
 * @file fb_sweep.c
 * @brief The forward-backward sweep the two exact per-position answers share.
 *
 * See internal/beam/fb_sweep.h for what this is and why it is its own file.
 * Everything here produces the per-layer edge posterior; nothing here decides
 * what to do with it.
 *
 * ## Log space throughout, and the comment that said otherwise
 *
 * Both sweeps carry logs and combine with log-sum-exp. There is no per-layer
 * rescale and no running scale factor anywhere in this file.
 *
 * `positional.c` said there was, in a section titled "The rescale, and why the
 * clamp changes it", describing a linear forward sweep renormalised by the
 * layer maximum and the strided pass a clamped layer paid to correct it. That
 * mechanism was removed -- see the note on `beam_fb_advance` below, which
 * records why and what was tried first -- and the file header went on
 * describing it, cost and all, for code that no longer existed. It survived
 * because the two comments were four hundred lines apart and only one of them
 * sat next to the loop.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"
#include "beam/fb_sweep.h"

#ifdef _OPENMP
#include <omp.h>
#endif


BeamError beam_fb_betas(const BeamModel *m, uint32_t len,
                               const uint8_t *clamp, double **beta,
                               double *w) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    beam_tail_weights(m, len, beta[len]);

    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, len, w, NULL);
        uint32_t lo, hi;
        beam_clamp_range(clamp, (uint32_t)i, R, &lo, &hi);
        const double *nb = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
            const uint32_t s = (uint32_t)s64;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (uint32_t x = lo; x < hi; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            if (mx == -INFINITY) { beta[i][s] = -INFINITY; continue; }
            double acc = 0.0;
            for (uint32_t x = lo; x < hi; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                acc += exp(row[x] + nb[base + x] - mx);
            }
            beta[i][s] = mx + log(acc);
        }
    }
    return BEAM_OK;
}


/**
 * Advance the LOG forward vector across one layer.
 *
 * `cur[s]` is `ln alpha_i(s)`, and on return `next` holds the same for layer
 * `i + 1`. Accumulated by TARGET, for the reason `compute_log_z` gives: a
 * target's twenty-one sources are its private business, so no two threads
 * write the same accumulator.
 *
 * There is no rescale and no running scale factor, and that is the point.
 * This sweep used to carry `alpha` linearly, renormalised once per layer by
 * the largest edge weight in it -- and no single factor per layer can hold a
 * trellis whose live states span more than 745 nats. At the C API's own
 * defaults a length-60 nucleotide sweep spans that easily, and the states that
 * underflowed to zero were not marginal: they included PINNED positions, whose
 * row must be one-hot and came back summing to 2e-54 with BEAM_OK.
 *
 * Three attempts were made to choose that factor well -- over the whole layer,
 * over edges a live path can use, and weighted by the mass each source carries
 * -- before the factor was removed instead. The backward sweep above never had
 * the problem, having always been log-space; now neither half does.
 *
 * Returns false when the layer leaves no mass at all, which under a clamp is a
 * real outcome rather than an error.
 */
bool beam_fb_advance(const BeamModel *m, const double *w,
                    const double *cur, double *next,
                    uint32_t lo, uint32_t hi) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    bool any = false;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:any) \
            num_threads(beam_threads())
#endif
    for (int64_t t = 0; t < (int64_t)S; ++t) {
        const uint32_t x = (uint32_t)t % A;
        if (x < lo || x >= hi) { next[t] = -INFINITY; continue; }
        const uint32_t rest = (uint32_t)t / A;
        double mx = -INFINITY, sum = 0.0;
        for (uint32_t a = 0; a < A; ++a) {
            const uint32_t src = a * SK + rest;
            if (cur[src] == -INFINITY) continue;
            const double wv = w[(size_t)src * R + x];
            if (wv == -INFINITY) continue;
            lse_add(&mx, &sum, cur[src] + wv);
        }
        next[t] = (mx == -INFINITY) ? -INFINITY : mx + log(sum);
        if (next[t] != -INFINITY) any = true;
    }
    return any;
}


void beam_fb_free(BeamFBWork *k) {
    if (k->beta) {
        for (uint32_t i = 0; i < k->n_beta; ++i) free(k->beta[i]);
        free(k->beta);
    }
    free(k->w); free(k->cur); free(k->next); free(k->tail);
    if (k->pool) beam_account_release(k->pool, k->held);   /* NULL only if unused */
    memset(k, 0, sizeof(*k));
}

BeamError beam_fb_alloc(const BeamModel *m, uint32_t len, BeamFBWork *k,
                        BeamMemAccount *pool) {
    const uint32_t S = m->n_states;
    const uint32_t R = m->n_residues;
    memset(k, 0, sizeof(*k));

    /*
     * Refuse before allocating, for the reason `beam_model_new` refuses:
     * `calloc` on Linux grants lazily, so an oversized request does not fail
     * where it is made, it fails much later on first write with the OOM killer
     * choosing a victim from the whole machine.
     *
     * The whole backward field is held at once because the posterior needs
     * layer `i + 1` while the forward sweep is at layer `i`, and the forward
     * sweep only runs once the backward one is finished. That is `(len + 1) *
     * n_states` doubles, which is 44 MB on the default amino-acid model and
     * 6.6 GB on a nucleotide model at the alphabet's ceiling order -- so the
     * bound is a real one at the top of the range rather than a formality.
     */
    const uint64_t need = ((uint64_t)len + 1u) * S * sizeof(double)
                        + (uint64_t)S * (R + 3u) * sizeof(double);
    if (!beam_account_take(pool, need)) {
        const char *nu, *bu;
        const double nv = beam_budget_scaled(need, &nu);
        if (pool->held) {
            /* The accumulators are already holding part of the budget and are
             * live for the whole sweep, so what this field has to fit in is
             * what they left, not the whole figure. Naming both is what makes
             * the message actionable: raising the budget helps, and so does
             * asking for one length rather than the pooled answer. */
            const char *hu;
            const double hv = beam_budget_scaled(pool->held, &hu);
            const double bv = beam_budget_scaled(beam_model_memory_budget(),
                                                 &bu);
            return BEAM_FAIL(BEAM_ERR_BUDGET,
                        "a forward-backward sweep at length %u needs about "
                        "%.2f %s for the backward field of %u states, and the "
                        "expected-count accumulators already hold %.2f %s of "
                        "the %.2f %s budget; raise it with "
                        "BEAM.set_memory_budget, or ask at a lower order",
                        (unsigned)len, nv, nu, (unsigned)S, hv, hu, bv, bu);
        }
        const double bv = beam_budget_scaled(beam_account_room(pool), &bu);
        return BEAM_FAIL(BEAM_ERR_BUDGET,
                        "a forward-backward sweep at length %u needs about "
                        "%.2f %s for the backward field of %u states, and the "
                        "memory budget is %.2f %s; raise it with "
                        "BEAM.set_memory_budget, or ask at a lower order",
                        (unsigned)len, nv, nu, (unsigned)S, bv, bu);
    }
    k->pool = pool;
    k->held = need;

    k->beta = (double **)calloc((size_t)len + 1u, sizeof(double *));
    if (!k->beta) return BEAM_EALLOC("the backward-value layer table");
    k->n_beta = len + 1u;
    for (uint32_t i = 0; i <= len; ++i) {
        k->beta[i] = (double *)malloc((size_t)S * sizeof(double));
        if (!k->beta[i]) { beam_fb_free(k); return BEAM_EALLOC("a backward layer"); }
    }
    k->w    = (double *)malloc((size_t)S * R * sizeof(double));
    k->cur  = (double *)calloc(S, sizeof(double));
    k->next = (double *)malloc((size_t)S * sizeof(double));
    k->tail = (double *)malloc((size_t)S * sizeof(double));
    if (!k->w || !k->cur || !k->next || !k->tail) {
        beam_fb_free(k);
        return BEAM_EALLOC("the forward-backward buffers");
    }
    return BEAM_OK;
}
