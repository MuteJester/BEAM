/**
 * @file provenance.h
 * @brief What a model knows about its own construction.
 *
 * Internal to src/native/. Not a public header and not installed.
 *
 * Three facts that cannot be recomputed from a finished model: when it was
 * first built, and how much wall time went into observing and into
 * finalizing. Everything else a caller might want to know about a model --
 * its size, its occupancy, what it can spell -- is derivable from the tables
 * and is computed on demand rather than stored.
 *
 * Wall time rather than CPU time, and MONOTONIC rather than realtime. Wall
 * time because "how long did this take" is a question about the clock on the
 * wall, and because the counting pass is already parallel so CPU time would
 * report a number that depends on the thread count. Monotonic because a
 * duration measured across a clock adjustment is not a duration.
 *
 * The creation timestamp is the one place realtime is right: it is a date,
 * not an interval.
 */
#ifndef BEAM_INTERNAL_PROVENANCE_H
#define BEAM_INTERNAL_PROVENANCE_H

#include <stdint.h>
#include <time.h>

/** Monotonic nanoseconds, for measuring an interval. */
static inline int64_t beam_now_nanos(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
#endif
    return 0;
}

/** Seconds since the epoch, for recording a date. */
static inline int64_t beam_now_unix(void) {
    return (int64_t)time(NULL);
}

#endif /* BEAM_INTERNAL_PROVENANCE_H */
