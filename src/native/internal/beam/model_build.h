/**
 * @file model_build.h
 * @brief Helpers shared between the model's build-time translation units.
 *
 * These were static functions inside a single build.c. Splitting that file by
 * responsibility left six of them with callers in another unit, so they are
 * declared here rather than duplicated.
 *
 * `beam_ipow_u32` stays `static inline` deliberately. It is small, it is called
 * while computing table dimensions, and giving it external linkage would stop
 * it being inlined at those call sites. The rest are large enough that their
 * call overhead is irrelevant, and they are called once per table.
 */
#ifndef BEAM_INTERNAL_MODEL_BUILD_H
#define BEAM_INTERNAL_MODEL_BUILD_H

#include "beam/model.h"
#include "beam/model_state.h"

/** Integer power, for table dimensions. */
static inline uint32_t beam_ipow_u32(uint32_t base, uint32_t exp) {
    uint32_t r = 1;
    while (exp--) r *= base;
    return r;
}

/** True if the configuration is self-consistent and within its bounds. */
bool beam_config_valid(const BeamModelConfig *c);

/** Release the count tables, dense or sparse. */
void beam_free_counts(BeamModel *m);

/** Release the finalized log-probability tables, dense or sparse. */
void beam_free_tables(BeamModel *m);

/**
 * Build the backoff hierarchy for one table.
 *
 * Writes `out[0..order]`, level 0 being the base and level `order` the full
 * context. The arithmetic is unchanged from when this lived in build.c; do
 * not reorder its accumulations.
 */
BeamError beam_build_hierarchy(const double *counts, uint32_t n_depth,
                               uint32_t order, const double *kappa,
                               double **out, bool strict);

/** Convert probabilities to logs in place. Returns false on a dead table. */
bool beam_to_log_inplace(double *p, size_t n, bool strict);

#endif /* BEAM_INTERNAL_MODEL_BUILD_H */
