/**
 * @file marginals.c
 * @brief Per-length marginals.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"



/* ── Per-length marginals ────────────────────────────────────── */

BeamError beam_model_length_marginals(const BeamModel *m,
                                         double *counting, double *generated,
                                         double *mean_surprisal,
                                         double *var_surprisal) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();

    for (uint32_t L = 0; L < m->n_len_slots; ++L) {
        if (counting)       counting[L] = 0.0;
        if (generated)      generated[L] = 0.0;
        if (mean_surprisal) mean_surprisal[L] = 0.0;
        if (var_surprisal)  var_surprisal[L] = 0.0;
    }
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!work) return BEAM_EALLOC("the per-length work list");
    const uint32_t nw = beam_live_lengths_desc(m, work);
    int failed = BEAM_OK;
    /* Every length is an independent pair of sweeps writing its own slot, so
     * this needs no reduction at all. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        const uint32_t L = work[k];
        if (counting)  counting[L]  = beam_model_path_count(m, L);
        if (generated) generated[L] = exp(m->log_pi[L]);
        if (mean_surprisal || var_surprisal) {
            double mom[3] = {0}, lz = 0.0;
            BeamError e = jet_sweep(m, L, 1.0, 2u, mom, &lz);
            if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; continue; }
            /* u = -(c_L + raw), so the shift is the same for both moments. */
            const double cL = m->log_pi[L] - m->log_z[L];
            const double e1 = -(cL + mom[1]);
            const double e2 = cL * cL + 2.0 * cL * mom[1] + mom[2];
            if (mean_surprisal) mean_surprisal[L] = e1;
            if (var_surprisal) {
                const double v = e2 - e1 * e1;
                var_surprisal[L] = v > 0.0 ? v : 0.0;
            }
        }
    }
    free(work);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                        "length_marginals: a per-length sweep failed");
    return BEAM_OK;
}
