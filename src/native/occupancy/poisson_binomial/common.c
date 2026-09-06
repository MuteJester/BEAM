/**
 * @file common.c
 * @brief Argument validation shared by every entry point.
 */
#include <math.h>
#include <stdlib.h>

#include "beam/publicness.h"
#include "beam/errors.h"
#include "beam/occupancy_internal.h"

/* 2*pi to the last representable bit; the codebase spells such constants out
 * rather than relying on M_PI, which is not in C11 proper. */

/* ── Shared validation ─────────────────────────────────────── */

BeamError publicness_check_groups(const double *depths,
                                        const double *multiplicity,
                                        uint32_t n_groups) {
    if (!depths || !multiplicity)
        return BEAM_ENULL(!depths ? "depths" : "multiplicity");
    if (n_groups == 0)
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT, "no depth groups given");

    for (uint32_t b = 0; b < n_groups; b++) {
        if (!(depths[b] >= 0.0) || !isfinite(depths[b]))
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "depth group %u has depth %g; must be finite "
                            "and non-negative", b, depths[b]);
        if (!(multiplicity[b] >= 0.0) ||
            multiplicity[b] != floor(multiplicity[b]) ||
            multiplicity[b] > 4294967295.0)
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "depth group %u has multiplicity %g; must be a "
                            "non-negative integer below 2^32", b,
                            multiplicity[b]);
    }
    return BEAM_OK;
}

BeamError publicness_check_probability(double p, uint32_t index) {
    if (!(p >= 0.0) || !(p <= 1.0))
        return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                        "probability %g at index %u is outside [0, 1]",
                        p, index);
    return BEAM_OK;
}
