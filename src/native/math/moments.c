/**
 * @file moments.c
 * @brief Surprisal moments, pooled and per length.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"



/* ── Surprisal moments ───────────────────────────────────────── */

BeamError beam_model_surprisal_moments(const BeamModel *m, double q,
                                          uint32_t order, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (order > BEAM_MAX_MOMENT)
        return BEAM_ERANGE("%s: moment order %u exceeds the supported "
                               "maximum of %u", __func__, (unsigned)order,
                               (unsigned)BEAM_MAX_MOMENT);

    const uint32_t NM = order + 1u;
    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    beam_binomials(bc, order);

    double *mom   = (double *)calloc(NM, sizeof(double));
    double *acc   = (double *)calloc(NM, sizeof(double));
    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    if (!mom || !acc || !terms) {
        free(mom); free(acc); free(terms);
        return BEAM_EALLOC("the per-length moment accumulators");
    }

    /* Length weights under the q-tilted distribution. */
    double mx = 0.0;
    BeamError e = beam_mellin_terms(m, q, terms, &mx);
    if (e != BEAM_OK || mx == -INFINITY) {
        free(mom); free(acc); free(terms);
        return e != BEAM_OK ? e
             : BEAM_ENOPATHS("no length carries tilted mass at this q");
    }
    double wsum = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        if (terms[L] != -INFINITY) wsum += exp(terms[L] - mx);

    /* One jet per length, all of them independent. They are computed up front
     * so the mixing below still runs in ascending length and the raw moments
     * accumulate in exactly the serial order. */
    double *jets = (double *)calloc((size_t)m->n_len_slots * NM, sizeof(double));
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!jets || !work) {
        free(jets); free(work); free(mom); free(acc); free(terms);
        return BEAM_EALLOC("the per-length jets");
    }
    uint32_t nw = 0;
    for (uint32_t L = m->n_len_slots; L-- > 1; )
        if (terms[L] != -INFINITY) work[nw++] = L;
    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double lz = 0.0;
        const BeamError je = jet_sweep(m, work[k], q, order,
                                      jets + (size_t)work[k] * NM, &lz);
        if (je != BEAM_OK) { if ((int)je > failed) failed = (int)je; }
    }
    if (failed != BEAM_OK) {
        free(jets); free(work); free(mom); free(acc); free(terms);
        return BEAM_FAIL((BeamError)failed,
                        "surprisal_moments: a per-length jet sweep failed");
    }
    free(work);

    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (terms[L] == -INFINITY) continue;
        const double weight = exp(terms[L] - mx) / wsum;
        memcpy(mom, jets + (size_t)L * NM, NM * sizeof(double));

        /*
         * The jet gives moments of `raw`; the surprisal is
         * `u = -(c_L + raw)`, so shift and negate:
         *   E[u^k] = sum_j C(k,j) * (-c_L)^(k-j) * (-1)^j * E[raw^j].
         *
         * A mixture's RAW moments are the weighted average of its components'
         * raw moments, which is why this is summed here and converted to
         * cumulants only at the very end. Averaging cumulants across lengths
         * instead would be quietly wrong.
         */
        const double cL = m->log_pi[L] - m->log_z[L];
        double pow_negc[BEAM_MAX_MOMENT + 1];
        pow_negc[0] = 1.0;
        for (uint32_t t = 1; t <= order; ++t) pow_negc[t] = pow_negc[t - 1] * (-cL);

        for (uint32_t k = 0; k < NM; ++k) {
            double v = 0.0;
            for (uint32_t j = 0; j <= k; ++j) {
                double term = bc[k][j] * mom[j] * pow_negc[k - j];
                v += (j & 1u) ? -term : term;
            }
            acc[k] += weight * v;
        }
    }

    for (uint32_t k = 0; k < NM; ++k) out[k] = acc[k];
    out[0] = 1.0;                                   /* exact by construction */
    free(jets); free(mom); free(acc); free(terms);
    return BEAM_OK;
}

/* ── Per-length moments, unmixed ─────────────────────────────── */

BeamError beam_model_length_derivatives(const BeamModel *m, double q,
                                           uint32_t order, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (order > BEAM_MAX_MOMENT)
        return BEAM_ERANGE("%s: moment order %u exceeds the supported "
                               "maximum of %u", __func__, (unsigned)order,
                               (unsigned)BEAM_MAX_MOMENT);

    const uint32_t n  = m->n_len_slots;
    const uint32_t NM = order + 1u;
    memset(out, 0, (size_t)n * NM * sizeof(double));

    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    beam_binomials(bc, order);

    double *terms = (double *)malloc((size_t)n * sizeof(double));
    double *jets  = (double *)calloc((size_t)n * NM, sizeof(double));
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!terms || !jets || !work) {
        free(terms); free(jets); free(work);
        return BEAM_EALLOC("the per-length moment buffers");
    }

    double mx = 0.0;
    BeamError e = beam_mellin_terms(m, q, terms, &mx);
    if (e != BEAM_OK || mx == -INFINITY) {
        free(terms); free(jets); free(work);
        return e != BEAM_OK ? e
             : BEAM_ENOPATHS("no length carries tilted mass at this q");
    }
    double wsum = 0.0;
    for (uint32_t L = 0; L < n; ++L)
        if (terms[L] != -INFINITY) wsum += exp(terms[L] - mx);

    uint32_t nw = 0;
    for (uint32_t L = n; L-- > 1; ) if (terms[L] != -INFINITY) work[nw++] = L;

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double lz = 0.0;
        const BeamError je = jet_sweep(m, work[k], q, order,
                                      jets + (size_t)work[k] * NM, &lz);
        if (je != BEAM_OK) { if ((int)je > failed) failed = (int)je; }
    }
    if (failed != BEAM_OK) {
        free(terms); free(jets); free(work);
        return BEAM_FAIL((BeamError)failed,
                        "length_derivatives: a per-length jet sweep failed");
    }
    free(work);

    for (uint32_t L = 1; L < n; ++L) {
        if (terms[L] == -INFINITY) continue;
        const double *mom = jets + (size_t)L * NM;
        double *row = out + (size_t)L * NM;

        /* Slot 0 is the length's own share of the tilted law, not the
         * constant 1 the jet carries there: with the share in hand the
         * mixture is `sum_L row[0] * row[k]`, which is exactly what
         * `beam_model_surprisal_moments` returns, and the caller can
         * check that rather than take it on trust. */
        row[0] = exp(terms[L] - mx) / wsum;

        /* The jet gives moments of `raw`; the surprisal is `u = -(c_L + raw)`,
         * so shift and negate, exactly as the pooled form does before mixing. */
        const double cL = m->log_pi[L] - m->log_z[L];
        double pow_negc[BEAM_MAX_MOMENT + 1];
        pow_negc[0] = 1.0;
        for (uint32_t t = 1; t <= order; ++t) pow_negc[t] = pow_negc[t - 1] * (-cL);
        for (uint32_t k = 1; k < NM; ++k) {
            double v = 0.0;
            for (uint32_t j = 0; j <= k; ++j) {
                const double term = bc[k][j] * mom[j] * pow_negc[k - j];
                v += (j & 1u) ? -term : term;
            }
            row[k] = v;
        }
    }
    free(terms); free(jets);
    return BEAM_OK;
}
