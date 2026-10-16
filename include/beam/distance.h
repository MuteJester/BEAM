/** Exact local-window Hellinger metric and additive squared-distance terms.
 * Distinct coordinates may describe overlapping sequence events. Their terms
 * add by definition; their probabilities do not. Positions are zero-based.
 */
#ifndef BEAM_DISTANCE_H
#define BEAM_DISTANCE_H
#include "beam/model.h"

typedef struct {
    uint32_t width; /* 0 selects order+1; otherwise 1..order+1. */
    uint32_t top_k; /* retain at most this many strictly positive terms. */
} BeamLocalDistanceConfig;

typedef struct {
    uint32_t length;
    double p, q; /* length probabilities */
    double length_contribution, pattern_contribution;
} BeamDistanceLength;

typedef struct {
    uint32_t length, position, width;
    double contribution;
} BeamDistanceWindow;

typedef struct {
    uint32_t length, position, width;
    uint64_t word_index; /* base-R, alphabet order, oldest residue first */
    double p, q; /* conditional on length */
    double contribution;
} BeamDistancePattern;

typedef struct {
    uint32_t width, n_lengths, n_windows, n_patterns;
    double squared_distance, length_squared_distance, pattern_squared_distance;
    double reported_pattern_squared_distance, remainder;
    uint64_t scratch_bytes; /* conservative peak including native outputs */
    uint64_t output_bytes;  /* native allocation retained in this result */
    BeamDistanceLength *lengths;
    BeamDistanceWindow *windows;
    BeamDistancePattern *patterns;
} BeamLocalDistanceResult;

/** Both models must be finalized and have the same alphabet and order.
 * Maximum lengths may differ; all lengths supported by either model enter.
 * At length L the actual width is min(config.width,L). The metric is
 *
 * D² = sum_L (sqrt(pi_P)-sqrt(pi_Q))²/2
 *    + sum_{L,i,u} sqrt(pi_P*pi_Q)/(2*n_windows(L))
 *                   * (sqrt(P(u at i|L))-sqrt(Q(u at i|L)))².
 *
 * Lengths and windows are complete; patterns are global top-K positive terms
 * sorted by decreasing contribution, then length, position and word index.
 * Their sum plus remainder equals pattern_squared_distance. A missing length
 * has zero within-length contribution; no undefined conditional is returned.
 * Width/top-K do not depend on model values. Changing width changes the metric.
 * top_k=0 skips pattern retention, not exact metric computation.
 *
 * Streams lengths and windows with compact cached backward vectors; no model
 * caches are mutated. Refuses before allocating if operation scratch+outputs
 * exceed the configured memory budget. On failure *out remains NULL.
 */
BeamError beam_local_distance(const BeamModel *p, const BeamModel *q,
                              const BeamLocalDistanceConfig *config,
                              BeamLocalDistanceResult **out);
void beam_local_distance_free(BeamLocalDistanceResult *result);
#endif
