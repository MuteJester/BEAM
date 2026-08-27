/**
 * @file occupancy_internal.h
 * @brief Shared pieces of the Poisson-binomial occupancy model.
 *
 * Internal to src/native/occupancy/.
 */
#ifndef BEAM_INTERNAL_OCCUPANCY_H
#define BEAM_INTERNAL_OCCUPANCY_H

#include <math.h>

#include "beam/common.h"
#include "beam/publicness.h"

BeamError publicness_check_groups(const double *depths,
                                        const double *multiplicity,
                                        uint32_t n_groups);
BeamError publicness_check_probability(double p, uint32_t index);

/* BEAM_INLINE is always_inline. This runs once per (atom, depth bin), so the
 * whole accumulation is a loop around it; it belongs in a header, not behind
 * a call. */
/**
 * Detection probability 1 - (1 - p)^N, evaluated through log1p/expm1 so that
 * it keeps its relative accuracy for the p ~ 1e-80 atoms that dominate the
 * counting spectrum. The guards matter: a zero-depth group would otherwise
 * reach 0 * -inf = NaN at p = 1.
 */
BEAM_INLINE double publicness_detection(double p, double depth) {
    double pi;
    if (!(p > 0.0) || !(depth > 0.0)) return 0.0;
    if (p >= 1.0) return 1.0;
    pi = -expm1(depth * log1p(-p));
    if (pi < 0.0) return 0.0;
    return pi > 1.0 ? 1.0 : pi;
}

#endif /* BEAM_INTERNAL_OCCUPANCY_H */
