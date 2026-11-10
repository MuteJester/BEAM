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

/**
 * The same, in 64 bits, for computing a bound rather than a dimension.
 *
 * `beam_ipow_u32` is a `uint32_t` multiply loop that WRAPS silently, which is
 * why each alphabet's order ceiling is chosen to keep `radix^max_order` in
 * range (`alphabet.h`). A bound on the row count is the one place that cannot
 * use it: the question being asked is whether a product exceeds what 32 bits
 * hold, and a helper that wraps would answer no every time it mattered.
 */
static inline uint64_t beam_ipow_u64(uint64_t base, uint32_t exp) {
    uint64_t r = 1;
    while (exp--) r *= base;
    return r;
}

/** True if the configuration is self-consistent and within its bounds. */
/** Room for the longest sentence `beam_config_invalid_reason` can produce. */
#define BEAM_CONFIG_REASON_MAX 320

/**
 * NULL when `c` is valid; otherwise `buf`, filled with the reason.
 *
 * One sentence naming the field that failed, the value it was given, and what
 * was required. The caller supplies the buffer so this needs no state and can
 * be called from anywhere.
 *
 * It exists because the refusal used to be assembled by the caller from a
 * fixed list of fields, which could not name a rule that was not on the list:
 * a config rejected for asking sparse storage without the closed world was
 * reported by printing order, max_length, the anchor weights and
 * the anchor weights, all of them valid.
 */
const char *beam_config_invalid_reason(const BeamModelConfig *c,
                                       char *buf, size_t cap);

/** `beam_config_invalid_reason` with the sentence discarded. */
bool beam_config_valid(const BeamModelConfig *c);

/** Release the count tables, dense or sparse. */
void beam_free_counts(BeamModel *m);

/** Release the finalized log-probability tables, dense or sparse. */
void beam_free_tables(BeamModel *m);

#endif /* BEAM_INTERNAL_MODEL_BUILD_H */
