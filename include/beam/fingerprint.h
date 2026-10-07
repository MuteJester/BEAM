/** Interpretable, bounded contrast search over exact positional-mask events.
 * The search is heuristic. Returned probabilities are conditional on length.
 * A tree partitions the entire fixed-length space, including remainder leaves.
 */
#ifndef BEAM_FINGERPRINT_H
#define BEAM_FINGERPRINT_H
#include "beam/model.h"

#define BEAM_FINGERPRINT_LOOKAHEAD 3u

typedef struct {
    uint32_t max_leaves;   /* >=1; includes remainder categories */
    uint32_t max_depth;    /* maximum binary questions on a path */
    uint32_t lookahead;    /* 1..3; positive-literal chain proposals */
    uint32_t beam_width;   /* retained probes at each look-ahead depth */
    uint32_t max_queries;  /* pairs of conditional sweeps, including root */
    double min_mass;       /* prune search when mean conditional mass is below */
    double min_gain;       /* minimum total squared-Hellinger split gain */
} BeamFingerprintConfig;

typedef struct {
    int32_t parent, yes, no; /* -1 for absent; root parent -1 */
    int32_t position, residue; /* question at internal nodes, -1 at leaves */
    uint32_t depth;
    double p, q;           /* P(event|L), Q(event|L) */
} BeamFingerprintNode;

typedef struct {
    uint32_t length, n_nodes, n_leaves, queries;
    bool budget_exhausted;
    uint64_t scratch_bytes; /* conservative peak operation allocation */
    double squared_distance;
    BeamFingerprintNode *nodes;
} BeamFingerprintResult;

/** Does not mutate either model. Alphabet must agree; orders may differ.
 * Both models must have live support at length. Refuses before allocation
 * when the cache + workspace + search exceeds the process memory budget.
 * On failure *out remains NULL. Free success with beam_fingerprint_free.
 */
BeamError beam_fingerprint_search(const BeamModel *p, const BeamModel *q,
                                  uint32_t length,
                                  const BeamFingerprintConfig *config,
                                  BeamFingerprintResult **out);
void beam_fingerprint_free(BeamFingerprintResult *result);
#endif
