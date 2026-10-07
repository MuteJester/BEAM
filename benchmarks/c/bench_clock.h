#ifndef BENCH_CLOCK_H
#define BENCH_CLOCK_H

#include <stdint.h>
#include <time.h>

/*
 * Wall-clock nanoseconds from CLOCK_MONOTONIC_RAW, which NTP never slews.
 * Wall time rather than CPU time on purpose: most BEAM analytics are OpenMP
 * parallel, and the cost a caller pays is the elapsed time of the region.
 * The harness measures what one empty timed window costs at startup and
 * reports it, so results near that floor can be recognised.
 */
static inline uint64_t bench_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#endif /* BENCH_CLOCK_H */
