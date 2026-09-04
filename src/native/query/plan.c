/**
 * @file plan.c
 * @brief What a query will cost, computed before anything is allocated.
 *
 * The difference between a query that returns in two milliseconds and one that
 * returns in twenty seconds is four orders of magnitude of state space, and
 * the caller cannot see that from the question it asked. A mutual-information
 * matrix is the same sentence at order 3 and at order 6.
 *
 * So the cost is answerable separately from the answer. This file allocates
 * nothing, acquires no liveness frontier and runs no sweep; it is arithmetic
 * over the model's dimensions and the constraint's masks, and it is cheap
 * enough to call before every query.
 *
 * `bytes` comes from `beam_query_sweep_bytes`, the same expression the sweep
 * allocates from. A guard computed from a different formula than the
 * allocation it guards is not a guard, and the discrepancy surfaces only on
 * the machine that runs out of memory.
 */
#include "beam/internal.h"
#include "beam/query.h"
#include "beam/query_internal.h"

BeamError beam_query_estimate(const BeamModel *m, const BeamConstraint *c,
                              BeamQueryCost *out) {
    if (!out) return BEAM_ENULL("output");

    const BeamResidueMask *mask = NULL;
    uint32_t length = 0u;
    BeamError err = beam_query_resolve_masks_only(m, c, &mask, &length);
    if (err != BEAM_OK) return err;

    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const BeamResidueMask all = BEAM_RESIDUES_ALL(R);

    /*
     * An upper bound on the states, not the live count. Reading the real
     * frontier means acquiring it, which allocates and may evict a cache this
     * call has no business touching -- and the bound is what the sweep
     * allocates for regardless, since the buffers are sized by `n_states`.
     */
    out->states = ((uint64_t)length + 1u) * S;

    /*
     * Both sweeps relax `n_states * |allowed|` edges per layer: the backward
     * one visits every state and the residues the layer admits, and the
     * forward one visits the `S/A * |allowed|` targets those residues can
     * reach, each gathering from `A` sources.
     */
    uint64_t edges = 0u;
    for (uint32_t i = 0; i < length; ++i) {
        const BeamResidueMask allowed = mask ? mask[i] : all;
        edges += 2ull * S * (uint32_t)__builtin_popcount(allowed);
    }
    out->edges = edges;
    /* The marginals figure, which is the larger of the two: a probability
     * query needs no forward buffers and is charged less when it runs. An
     * estimate that under-reported the biggest query would be the one number
     * here that could mislead. */
    out->bytes = beam_query_sweep_bytes(S, R, length, true);
    return BEAM_OK;
}
