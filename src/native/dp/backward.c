/**
 * @file backward.c
 * @brief Backward values, plain and q-tilted.
 */
#include "beam/internal.h"
#include "beam/mixing.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* ── Backward values ─────────────────────────────────────────── */

/**
 * `beta[i][s]` is the natural log of the total weight of every completion of a
 * length-`len` path from state `s` at layer `i`, including the sink term.
 *
 * Kept in log space rather than rescaled: the sampler indexes it randomly
 * rather than sweeping it, so a per-layer scale factor would have to be
 * carried alongside and applied at every draw.
 */
BeamError beam_betas(const BeamModel *m, uint32_t len, double **beta) {
    return beam_betas_tilted(m, len, 1.0, beta);
}

BeamError beam_betas_tilted(const BeamModel *m, uint32_t len, double q,
                               double **beta) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    /*
     * One layer, `S * R` doubles: 31 MB on an order-4 amino-acid model. The
     * backward FIELD is the caller's -- it is `(len + 1) * S` doubles and
     * dwarfs this -- so it is budgeted where it is allocated, not here.
     */
    BEAM_BUDGET_OR_REFUSE("a backward sweep", len,
                          (uint64_t)S * R * sizeof(double));

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    if (!w) return BEAM_EALLOC("one layer of edge weights");

    beam_tail_weights(m, len, beta[len]);
    if (q != 1.0)
        for (uint32_t s = 0; s < S; ++s)
            if (beta[len][s] != -INFINITY) beta[len][s] *= q;

    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, len, w, NULL);
        const double *nb = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
            const uint32_t s = (uint32_t)s64;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = q * row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            if (mx == -INFINITY) { beta[i][s] = -INFINITY; continue; }
            double acc = 0.0;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                acc += exp(q * row[x] + nb[base + x] - mx);
            }
            beta[i][s] = mx + log(acc);
        }
    }
    free(w);
    return BEAM_OK;
}


/* ── Best completions ────────────────────────────────────────── */

/*
 * The third member of the family above, and the one that is not a sum.
 *
 * `beam_betas` and `beam_betas_tilted` total every completion; this takes the
 * largest. The skeleton is the same -- `beam_tail_weights` seed, descending
 * layers, `base = (s % SK) * A`, the same edge-skip -- and three things are
 * not: the semiring is max-plus rather than log-sum-exp, there is no tilt, and
 * it visits only the states the forward frontier reached rather than all of
 * them.
 *
 * Those three are why this is a separate function rather than a flag on the
 * others. A merged version would carry a semiring parameter and a liveness
 * parameter for four combinations of which two are used, in the innermost loop
 * of the search. What the sharing needs is not one function but one FILE, so
 * that a change to the shape is made three times or deliberately once.
 *
 * `test_the_best_completion_never_exceeds_the_total` is the link between them:
 * a maximum cannot exceed a sum over the same set, at every layer and every
 * state, and the seed layer must agree exactly.
 */
BeamError beam_best_completions(const BeamModel *m, uint32_t len,
                                double **best) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    BEAM_BUDGET_OR_REFUSE("a best-completion sweep", len,
                          (uint64_t)S * R * sizeof(double) + S);

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    uint8_t *alive = (uint8_t *)malloc(S);
    if (!w || !alive) { free(w); free(alive);
        return BEAM_EALLOC("one layer of edge weights"); }

    /* Only states the root can reach are ever expanded by the search below,
     * so only their completions are read. This asks for no `out_max`, which
     * is the case `beam_layer_weights_live` documents as safe: dead rows are
     * provably unread, because a reached state's finite edge always lands on
     * a reached target, and an infinite one is rejected before the target is
     * touched.
     *
     * Unreached entries are filled with -inf rather than left stale. Nothing
     * should read them; if anything ever does, an impossible completion is a
     * loud wrong answer instead of a quiet one. */
    LiveHandle h;
    if (beam_live_acquire(m, len, &h) != BEAM_OK) {
        free(w); free(alive);
        return BEAM_EALLOC("the forward frontier");
    }
    const BeamLive *L = h.L;

    beam_tail_weights(m, len, best[len]);
    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        const uint32_t  nr = L->n_reached[i];
        const uint32_t *rv = L->reached[i];
        beam_reached_mask(L, (uint32_t)i, S, alive);
        beam_layer_weights_live(m, (uint32_t)i, len, w, NULL, alive);
        const double *nb = best[i + 1];
        double *cu = best[i];
        for (uint32_t t = 0; t < S; ++t) cu[t] = -INFINITY;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nr; ++k) {
            const uint32_t s = rv[k];
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            cu[s] = mx;
        }
    }
    free(w); free(alive);
    beam_live_release(&h);
    return BEAM_OK;
}
