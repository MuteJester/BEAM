/**
 * @file rng.h
 * @brief xoshiro256++ fast PRNG.
 */
#ifndef BEAM_RNG_H
#define BEAM_RNG_H

#include "beam/common.h"

typedef struct {
    uint64_t s[4];
} BeamRng;

/** Seed from a single 64-bit value (splitmix64 expansion). */
void beam_rng_seed(BeamRng *rng, uint64_t seed);

/** Next random uint64. */
BEAM_INLINE uint64_t beam_rng_next(BeamRng *rng) {
    const uint64_t s0 = rng->s[0], s3 = rng->s[3];
    const uint64_t result = ((s0 + s3) << 23 | (s0 + s3) >> 41) + s0;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= s0;
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = (rng->s[3] << 45) | (rng->s[3] >> 19);
    return result;
}

/** Uniform double in [0, 1). */
BEAM_INLINE double beam_rng_double(BeamRng *rng) {
    return (beam_rng_next(rng) >> 11) * 0x1.0p-53;
}

/** Uniform uint32 in [0, n). */
BEAM_INLINE uint32_t beam_rng_bounded(BeamRng *rng, uint32_t n) {
    return (uint32_t)((beam_rng_next(rng) >> 33) * (uint64_t)n >> 31);
}

#endif /* BEAM_RNG_H */
