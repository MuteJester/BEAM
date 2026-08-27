/**
 * @file math_internal.h
 * @brief Helpers shared between the named mathematical operations.
 *
 * Internal to src/native/math/. These were static functions inside one
 * moments.c; splitting that file by operation left them with callers in
 * another unit.
 */
#ifndef BEAM_INTERNAL_MATH_H
#define BEAM_INTERNAL_MATH_H

#include "beam/model.h"
#include "beam/model_state.h"

/**
 * Per-length Mellin terms at `q`, and the largest of them.
 *
 * The shared front half of every q-tilted quantity: the moments, the
 * per-length jets, and the depth attribution all start here.
 */
BeamError beam_mellin_terms(const BeamModel *m, double q, double *terms,
                             double *out_max);

#endif /* BEAM_INTERNAL_MATH_H */
