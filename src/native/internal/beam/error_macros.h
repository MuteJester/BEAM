/**
 * @file error_macros.h
 * @brief Diagnosable failure macros.
 *
 * Named error_macros.h, not errors.h, because the public
 * include/beam/errors.h owns that path: -Iinclude precedes
 * -Isrc/native/internal, so an internal header of the same name would be
 * silently shadowed and every macro here would vanish into an implicit
 * declaration.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_ERROR_MACROS_H
#define BEAM_INTERNAL_ERROR_MACROS_H

#include "beam/model.h"
#include "beam/model_state.h"

/* ── Diagnosable failures ────────────────────────────────────── */

/*
 * Every BEAM entry point returns an error CODE, and a bare code reaches Python
 * as "invalid argument" with nothing to act on. These attach the message that
 * `beam_error_message()` reports, so a caller learns which argument, which
 * length, or which table ran out.
 *
 * `__func__` names the function that refused, which for a static helper is the
 * helper: that is the more useful half of the answer, since the public entry
 * point is already obvious from the call site.
 */
#define BEAM_ENULL(what)                                                  \
    BEAM_FAIL(BEAM_ERR_NULL_ARG, "%s: %s must not be NULL", __func__, (what))

#define BEAM_ENOTBUILT()                                                  \
    BEAM_FAIL(BEAM_ERR_NOT_BUILT,                                               \
             "%s: the model is not finalized; observe sequences and call "    \
             "finalize before scoring or analysing it", __func__)

#define BEAM_ECOMPACTED(what)                                             \
    BEAM_FAIL(BEAM_ERR_NOT_BUILT,                                               \
             "%s: the model is compacted, so %s", __func__, (what))

#define BEAM_EALLOC(what)                                                 \
    BEAM_FAIL(BEAM_ERR_ALLOC, "%s: out of memory allocating %s", __func__, (what))

#define BEAM_EBADLEN(mdl, L)                                              \
    BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,                                      \
             "%s: length %u is not one this model generates (lengths 1..%u "  \
             "with a finite length prior are)", __func__, (unsigned)(L),      \
             (unsigned)(mdl)->max_len)

#define BEAM_ERANGE(...)                                                  \
    BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE, __VA_ARGS__)

#define BEAM_ENOPATHS(what)                                               \
    BEAM_FAIL(BEAM_ERR_NO_LIVE_PATHS, "%s: %s", __func__, (what))

/**
 * CRC-32C, hardware-accelerated where the running CPU offers it.
 *
 * Identical in value to `beam_crc32c_update`, which it falls back to; the only
 * difference is that a gigabyte of counts checksums in a fifth of the time.
 * Exposed here rather than kept static in io.c so the equivalence can be
 * tested directly.
 */
uint32_t beam_model_crc32c(uint32_t crc, const void *data, size_t len);

#endif /* BEAM_INTERNAL_ERROR_MACROS_H */
