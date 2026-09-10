/**
 * @file reduction.h
 * @brief How a parallel sum is split so its answer does not depend on cores.
 *
 * Internal to src/native/. Not a public header and not installed.
 *
 * One constant, and it is here rather than in `model_state.h` because it has
 * nothing to do with a model's layout. It was in that header, which meant
 * `query_internal.h` included two hundred lines publishing `struct BeamModel_`
 * -- and with it the whole sparse-row and frozen-count representations -- to
 * get one integer.
 */
#ifndef BEAM_INTERNAL_REDUCTION_H
#define BEAM_INTERNAL_REDUCTION_H

/**
 * Chunks a parallel SUM is split into, fixed and independent of the thread
 * count.
 *
 * A sum is the one reduction whose answer depends on the order it is taken
 * in, so `reduction(+:)` would make a result depend on how many cores
 * happened to be free. Splitting the range into a fixed number of chunks,
 * summing each serially and then adding the chunk totals in index order gives
 * one answer for every thread count — the property the tests pin — at the
 * cost of differing from a single serial sweep in the last bit or two.
 */
#define BEAM_SUM_CHUNKS 256u

#endif /* BEAM_INTERNAL_REDUCTION_H */
