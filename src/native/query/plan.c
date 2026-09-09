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
    BeamError err = beam_query_resolve(m, c, &mask, &length);
    if (err != BEAM_OK) return err;

    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t Q = beam_query_automaton_states(c->automaton);
    const uint32_t n_aut = Q ? Q : 1u;
    const BeamResidueMask all = BEAM_RESIDUES_ALL(R);

    /*
     * An upper bound on the states, not the live count. Reading the real
     * frontier means acquiring it, which allocates and may evict a cache this
     * call has no business touching -- and the bound is what the sweep
     * allocates for regardless, since the buffers are sized by `n_states`.
     */
    out->states = ((uint64_t)length + 1u) * S * n_aut;

    /*
     * Both sweeps relax `n_states * |allowed|` edges per layer: the backward
     * one visits every state and the residues the layer admits, and the
     * forward one visits the `S/A * |allowed|` targets those residues can
     * reach, each gathering from `A` sources.
     */
    uint64_t edges = 0u;
    for (uint32_t i = 0; i < length; ++i) {
        const BeamResidueMask allowed = mask ? mask[i] : all;
        edges += 2ull * S * n_aut * (uint32_t)__builtin_popcount(allowed);
    }
    out->edges = edges;
    /*
     * The largest figure any query here will be guarded against, not the
     * posterior's alone. `beam_query_expect` is guarded by
     * `beam_query_moment_bytes`, which has no length term, so at length 1 or 2
     * it exceeds the sweep figure -- and a caller following this header's
     * advice, comparing the estimate against the budget before asking, was
     * refused anyway. Found by review, along with the test that started its
     * loop at length 8 and so never saw it.
     */
    const uint64_t sweep  = beam_query_sweep_bytes(S, n_aut, R, length, true);
    const uint64_t moment = beam_query_moment_bytes(S, n_aut, R);
    out->bytes = sweep > moment ? sweep : moment;
    return BEAM_OK;
}
