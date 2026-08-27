/**
 * @file tilt.c
 * @brief One-sweep tilt summary, for saddlepoint stepping.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"



/* ── One-sweep tilt summary, for saddlepoint stepping ────────── */

BeamError beam_model_surprisal_tilt(const BeamModel *m, double q,
                                       double *out_log_mellin,
                                       double *out_mean, double *out_var) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();

    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    double *m1    = (double *)calloc(m->n_len_slots, sizeof(double));
    double *m2    = (double *)calloc(m->n_len_slots, sizeof(double));
    if (!terms || !m1 || !m2) {
        free(terms); free(m1); free(m2);
        return BEAM_EALLOC("the per-length tilt summaries");
    }

    /* One jet sweep per length gives that length's tilted mass and its first
     * two moments of the surprisal at the same time. */
    double mx = -INFINITY;
    for (uint32_t L = 0; L < m->n_len_slots; ++L) terms[L] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    double *jets = (double *)calloc((size_t)m->n_len_slots * 4u, sizeof(double));
    if (!work || !jets) {
        free(work); free(jets); free(terms); free(m1); free(m2);
        return BEAM_EALLOC("the per-length tilt jets");
    }
    const uint32_t nw = beam_live_lengths_desc(m, work);
    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double mom[3] = {0}, lz = 0.0;
        const BeamError e = jet_sweep(m, work[k], q, 2u, mom, &lz);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; continue; }
        double *j = jets + (size_t)work[k] * 4u;
        j[0] = lz; j[1] = mom[1]; j[2] = mom[2];
    }
    free(work);
    if (failed != BEAM_OK) {
        free(jets); free(terms); free(m1); free(m2);
        return BEAM_FAIL((BeamError)failed,
                        "surprisal_tilt: a per-length jet sweep failed");
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        const double *j = jets + (size_t)L * 4u;
        if (j[0] == -INFINITY) continue;
        terms[L] = q * (m->log_pi[L] - m->log_z[L]) + j[0];
        if (terms[L] > mx) mx = terms[L];

        /* u = -(c_L + raw), so E[u] = -(c_L + E[raw]) and
         * E[u^2] = c_L^2 + 2 c_L E[raw] + E[raw^2]. */
        const double cL = m->log_pi[L] - m->log_z[L];
        m1[L] = -(cL + j[1]);
        m2[L] = cL * cL + 2.0 * cL * j[1] + j[2];
    }
    free(jets);
    if (mx == -INFINITY) {
        free(terms); free(m1); free(m2);
        return BEAM_ENOPATHS("no length carries tilted mass at this q");
    }

    double wsum = 0.0, e1 = 0.0, e2 = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L) {
        if (terms[L] == -INFINITY) continue;
        const double w = exp(terms[L] - mx);
        wsum += w;
        e1 += w * m1[L];
        e2 += w * m2[L];
    }
    e1 /= wsum; e2 /= wsum;

    if (out_log_mellin) *out_log_mellin = mx + log(wsum);
    if (out_mean)       *out_mean = e1;
    if (out_var)        *out_var  = e2 - e1 * e1;

    free(terms); free(m1); free(m2);
    return BEAM_OK;
}
