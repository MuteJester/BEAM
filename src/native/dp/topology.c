/**
 * @file topology.c
 * @brief Node and edge counts of the trellis at one length.
 */
#include "beam/internal.h"


/* ── Topology ────────────────────────────────────────────────── */

BeamError beam_model_topology(const BeamModel *m, uint32_t length,
                                 double *out_nodes, double *out_edges,
                                 uint32_t *out_layer_nodes) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);

    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    LiveHandle h;
    BeamError err = beam_live_acquire(m, length, &h);
    if (err != BEAM_OK) return err;
    const BeamLive *L = h.L;

    uint32_t *bits = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    if (!bits) {
        beam_live_release(&h);
        return BEAM_EALLOC("the topology sweep buffers");
    }

    /*
     * A node counts only if it is BOTH forward-reachable and able to reach a
     * live sink, which is what makes this the size of the graph a walk can
     * actually traverse rather than the size of the state space -- and that
     * intersection is precisely what the frontier's `live` list already is.
     *
     * The dense form of this counted `cur[s] > 0.0 && alive[i][s]` over all
     * 194,481 states per layer, where `cur > 0` marked "reachable along a
     * path of live states". The two agree state for state: a state that is
     * forward-reachable and backward-live has a witness path on which every
     * state is itself forward-reachable (a prefix of the witness) and
     * backward-live (it reaches the state, which reaches a sink), so the
     * dense sweep marks it; and `cur > 0` cannot hold for anything the
     * frontier failed to reach. `nodes_by_layer` is therefore unchanged, and
     * that is worth checking against a known model rather than believing --
     * `topology(13)` on the reference 14,615-sequence donor gives
     * 1,1,7,30,155,552,1281,2022,2534,2222,1423,631,237,88.
     */
    double nodes = 0.0;
    for (uint32_t i = 0; i <= length; ++i) {
        if (out_layer_nodes) out_layer_nodes[i] = L->n_live[i];
        nodes += (double)L->n_live[i];
    }

    /*
     * Edges, counted by target for the same reason path counting is: the 21
     * sources of a target are the target's private business, so no two
     * threads write the same accumulator. Both totals are counts, so summing
     * them in any order is exact and the reduction is safe.
     *
     * Restricting the targets to `live[i + 1]` drops nothing: a target with
     * at least one qualifying source is forward-reached by construction, and
     * the dense loop already required it to be backward-live, so the pairs
     * counted are the same pairs.
     */
    int64_t edges = 0;
    for (uint32_t i = 0; i < length; ++i) {
        {
            const uint32_t *rd = L->reached[i];
            const uint32_t *eb = L->edge_bits[i];
            const uint32_t nr = L->n_reached[i];
            for (uint32_t k = 0; k < nr; ++k) bits[rd[k]] = eb[k];
        }
        const uint64_t *mcur = L->mask[i];
        const uint32_t nlive = L->n_live[i + 1];
        const uint32_t *tidx = L->live[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:edges) \
                num_threads(beam_threads()) if(nlive > 4096)
#endif
        for (int64_t k = 0; k < (int64_t)nlive; ++k) {
            const uint32_t t = tidx[k];
            const uint32_t x = t % A;
            if (x >= R) continue;
            const uint32_t rest = t / A;
            int64_t hits = 0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                /* Liveness first: only a live state is guaranteed to have had
                 * its mask scattered into `bits` this layer. */
                if (!live_bit_get(mcur, src)) continue;
                if (!((bits[src] >> x) & 1u)) continue;
                hits++;
            }
            edges += hits;
        }
    }

    if (out_nodes) *out_nodes = nodes;
    if (out_edges) *out_edges = (double)edges;
    free(bits);
    beam_live_release(&h);
    return BEAM_OK;
}
