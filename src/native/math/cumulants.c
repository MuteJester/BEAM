/**
 * @file cumulants.c
 * @brief Surprisal cumulants.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"



BeamError beam_model_surprisal_cumulants(const BeamModel *m, double q,
                                            uint32_t order, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (order < 1 || order > BEAM_MAX_MOMENT)
        return BEAM_ERANGE("%s: cumulant order %u is outside 1..%u",
                               __func__, (unsigned)order,
                               (unsigned)BEAM_MAX_MOMENT);

    double *mu = (double *)calloc(order + 1u, sizeof(double));
    if (!mu) return BEAM_EALLOC("the raw-moment buffer");
    BeamError e = beam_model_surprisal_moments(m, q, order, mu);
    if (e != BEAM_OK) { free(mu); return e; }

    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    beam_binomials(bc, order);

    /* k_n = mu_n - sum_{i=1}^{n-1} C(n-1, i-1) k_i mu_{n-i} */
    for (uint32_t n = 1; n <= order; ++n) {
        double v = mu[n];
        for (uint32_t i = 1; i < n; ++i)
            v -= bc[n - 1][i - 1] * out[i - 1] * mu[n - i];
        out[n - 1] = v;
    }
    free(mu);
    return BEAM_OK;
}
