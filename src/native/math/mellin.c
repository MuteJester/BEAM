/**
 * @file mellin.c
 * @brief The Mellin transform and its per-length terms.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"



/* ── Mellin transform ────────────────────────────────────────── */

/**
 * `ln sum_s P(s)^q`, assembled from the per-length tilted partition functions.
 *
 * `sum over length-L sequences of P^q = exp(q * c_L) * Z_q(L)` with
 * `c_L = ln pi(L) - ln Z(L)`, so the lengths combine as a log-sum-exp. Doing
 * that in the log domain rather than by summing exponentials is what keeps a
 * long tail of rare lengths from underflowing out of the total.
 */
/**
 * Live lengths, longest first: the per-length sweeps are the unit of parallel
 * work and cost grows with the length, so the longest must be handed out
 * first or fifteen cores wait on it at the end.
 */
BeamError beam_mellin_terms(const BeamModel *m, double q, double *terms,
                             double *out_max) {
    const uint32_t n = m->n_len_slots;
    for (uint32_t L = 0; L < n; ++L) terms[L] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *lzs = (double *)malloc((size_t)n * sizeof(double));
    if (!work || !lzs) {
        free(work); free(lzs);
        return BEAM_EALLOC("the per-length Mellin work list");
    }
    const uint32_t nw = beam_live_lengths_desc(m, work);
    int failed = BEAM_OK;
    /* Concurrent sweeps DIVIDE the budget rather than multiplying it. A
     * concurrency of 0 means one sweep does not fit at all, which the sweep
     * itself refuses with the better message, so run one and report that. */
    const uint32_t conc = beam_budget_concurrency(
        beam_jet_sweep_bytes(m->n_states, m->n_residues, 1u), nw ? nw : 1u);
    if (nw && conc == 0u)
        return beam_budget_refuse("a tilted per-length sweep", work[0],
                                  beam_jet_sweep_bytes(m->n_states, m->n_residues, 1u));

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc)
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double mom[1], lz = 0.0;
        const BeamError e = jet_sweep(m, work[k], q, 0u, mom, &lz);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
        else lzs[work[k]] = lz;
    }
    free(work);
    if (failed != BEAM_OK) {
        free(lzs);
        return BEAM_FAIL((BeamError)failed,
                        "a per-length tilted sweep could not be computed");
    }
    /* Scanned in ascending length, so the maximum and the terms are exactly
     * what the serial loop produced. */
    double mx = -INFINITY;
    for (uint32_t L = 1; L < n; ++L) {
        if (!beam_length_live(m, L) || lzs[L] == -INFINITY) continue;
        terms[L] = q * (m->log_pi[L] - m->log_z[L]) + lzs[L];
        if (terms[L] > mx) mx = terms[L];
    }
    free(lzs);
    *out_max = mx;
    return BEAM_OK;
}

BeamError beam_model_log_mellin(const BeamModel *m, double q, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    if (!terms) return BEAM_EALLOC("the per-length Mellin terms");
    double mx = 0.0;
    BeamError e = beam_mellin_terms(m, q, terms, &mx);
    if (e != BEAM_OK) { free(terms); return e; }
    if (mx == -INFINITY) { free(terms); *out = -INFINITY; return BEAM_OK; }

    double acc = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        if (terms[L] != -INFINITY) acc += exp(terms[L] - mx);
    free(terms);
    *out = mx + log(acc);
    return BEAM_OK;
}
