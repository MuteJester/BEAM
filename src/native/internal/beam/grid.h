/**
 * @file grid.h
 * @brief Moving mass onto a surprisal grid, shared by everything that bins.
 *
 * Internal to src/native/. Not a public header and not installed.
 *
 * A surprisal grid is a histogram over `-ln P(s)` in nats, `bins` wide with a
 * fixed spacing. Every dynamic program that bins rather than sums moves mass
 * along that axis the same way: a path's surprisal is the sum of its edges'
 * surprisals, so extending a path by an edge SHIFTS its whole histogram by
 * that edge's contribution. The shift is not an integer number of bins, so
 * the mass is split linearly between the two adjacent bins.
 *
 * This lived in `math/spectrum.c` as two static functions. It is here because
 * `math/positional_banded.c` runs the same transport in both directions to
 * condition a positional marginal on a surprisal band, and a second copy of a
 * splitting rule is exactly the kind of duplicate that drifts: the two halves
 * of a forward-backward pair MUST split mass identically or the recombined
 * total is not the distribution either half describes.
 *
 * They are `static inline` in a header rather than a separate translation
 * unit, and that is a performance decision, not a filing one. The transport
 * is called once per edge from the most expensive loop in the library, and
 * moving it across a translation-unit boundary would turn an inlined body
 * into a real call there. `budget.h` carries its helpers the same way for the
 * same reason.
 */
#ifndef BEAM_INTERNAL_GRID_H
#define BEAM_INTERNAL_GRID_H

#include <math.h>
#include <stdint.h>

/**
 * Occupied bin range of one node's histogram, as a half-open interval.
 *
 * Carrying this is what makes a grid DP affordable. Early layers occupy a
 * handful of bins out of a thousand, and without bounds every edge would
 * sweep the whole grid: the production FlashBack transport
 * (`lib/flashback/graph_analytics.c:pseq_grid_transport`) keeps the same
 * bookkeeping for the same reason.
 */
typedef struct { uint32_t lo, hi; } BeamBinRange;

/**
 * Add `src` shifted up by `u` bins into `dst`, splitting linearly between the
 * two adjacent bins so total mass is conserved and the mean is unbiased.
 * Source bins that fall outside the grid contribute nothing; that must hold at
 * BOTH ends, because the sink shift can be negative.
 *
 * `sr` bounds the source's occupied bins; `dr` is widened to cover whatever
 * this call writes.
 */
static inline void beam_grid_transport(const double *src, double *dst,
                                       uint32_t bins, double u, double factor,
                                       BeamBinRange sr, BeamBinRange *dr) {
    const int64_t k = (int64_t)floor(u);
    const double  f = u - (double)k;
    const double  g = 1.0 - f;
    for (uint32_t b = sr.lo; b < sr.hi; ++b) {
        const double v = src[b];
        if (v == 0.0) continue;
        const int64_t lo = (int64_t)b + k;
        if (lo >= 0 && lo < (int64_t)bins) {
            dst[lo] += v * g * factor;
            if ((uint32_t)lo < dr->lo) dr->lo = (uint32_t)lo;
            if ((uint32_t)lo + 1u > dr->hi) dr->hi = (uint32_t)lo + 1u;
        }
        const int64_t hi = lo + 1;
        if (hi >= 0 && hi < (int64_t)bins) {
            dst[hi] += v * f * factor;
            if ((uint32_t)hi < dr->lo) dr->lo = (uint32_t)hi;
            if ((uint32_t)hi + 1u > dr->hi) dr->hi = (uint32_t)hi + 1u;
        }
    }
}

/**
 * Add `src` shifted up by `u` bins into `dst` for TWO measures at once,
 * splitting linearly between the two adjacent bins exactly as
 * `beam_grid_transport` does for one.
 *
 * The point is not to save a pass over the source bins — it is that the shift
 * `u`, and therefore the integer bin `k` and the split fractions `f` and `g`,
 * are computed ONCE and used for both. Two separate transports would compute
 * them twice from the same inputs and get the same answer today, but nothing
 * would hold them together, and the ratio of the two measures per bin is only
 * meaningful if the mass was split identically. That ratio is the exact
 * mass-weighted mean probability of the sequences in the bin, which is what
 * the discovery-curve arithmetic is built on.
 */
static inline void beam_grid_transport2(const double *src_a, const double *src_b,
                                        double *dst_a, double *dst_b,
                                        uint32_t bins, double u,
                                        double factor_a, double factor_b,
                                        BeamBinRange sr, BeamBinRange *dr) {
    const int64_t k = (int64_t)floor(u);
    const double  f = u - (double)k;
    const double  g = 1.0 - f;
    for (uint32_t b = sr.lo; b < sr.hi; ++b) {
        const double va = src_a[b], vb = src_b[b];
        if (va == 0.0 && vb == 0.0) continue;
        const int64_t lo = (int64_t)b + k;
        if (lo >= 0 && lo < (int64_t)bins) {
            dst_a[lo] += va * g * factor_a;
            dst_b[lo] += vb * g * factor_b;
            if ((uint32_t)lo < dr->lo) dr->lo = (uint32_t)lo;
            if ((uint32_t)lo + 1u > dr->hi) dr->hi = (uint32_t)lo + 1u;
        }
        const int64_t hi = lo + 1;
        if (hi >= 0 && hi < (int64_t)bins) {
            dst_a[hi] += va * f * factor_a;
            dst_b[hi] += vb * f * factor_b;
            if ((uint32_t)hi < dr->lo) dr->lo = (uint32_t)hi;
            if ((uint32_t)hi + 1u > dr->hi) dr->hi = (uint32_t)hi + 1u;
        }
    }
}

#endif /* BEAM_INTERNAL_GRID_H */
