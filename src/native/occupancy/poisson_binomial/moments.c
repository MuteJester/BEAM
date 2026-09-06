/**
 * @file moments.c
 * @brief Closed-form mean and variance of the occupancy count.
 */
#include <math.h>
#include <stdlib.h>

#include "beam/publicness.h"
#include "beam/error_macros.h"
#include "beam/occupancy_internal.h"

/* 2*pi to the last representable bit; the codebase spells such constants out
 * rather than relying on M_PI, which is not in C11 proper. */

/* ── Closed-form moments ───────────────────────────────────── */

BeamError beam_publicness_moments(const double *p, uint32_t n_atoms,
                                const double *depths,
                                const double *multiplicity,
                                uint32_t n_groups,
                                double *out_mean, double *out_variance) {
    BeamError err;

    if (!p || !out_mean || !out_variance)
        return BEAM_ENULL(!p ? "probabilities"
                            : (!out_mean ? "mean output" : "variance output"));
    if (n_atoms == 0) return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT, "no probabilities");

    err = publicness_check_groups(depths, multiplicity, n_groups);
    if (err != BEAM_OK) return err;

    for (uint32_t a = 0; a < n_atoms; a++) {
        double mean = 0.0, variance = 0.0;

        err = publicness_check_probability(p[a], a);
        if (err != BEAM_OK) return err;

        for (uint32_t b = 0; b < n_groups; b++) {
            double pi = publicness_detection(p[a], depths[b]);
            mean += multiplicity[b] * pi;
            variance += multiplicity[b] * pi * (1.0 - pi);
        }

        out_mean[a] = mean;
        out_variance[a] = variance < 0.0 ? 0.0 : variance;
    }

    return BEAM_OK;
}
