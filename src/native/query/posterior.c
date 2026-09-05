/**
 * @file posterior.c
 * @brief Per-position marginals under a constraint, from the edge posterior.
 *
 * `gamma_i(s, x) = alpha_i(s) w_i(s, x) beta_{i+1}(s -> x) / Z`, summed over
 * states to give `P(x_i = a | length, C)`. The constraint enters twice, in the
 * backward values and in the residues each layer is allowed to emit, and
 * nowhere else: conditioning is edge deletion, so the same recursion that
 * computes a marginal computes a conditional.
 *
 * Partials are accumulated in a fixed number of chunks and combined in chunk
 * order, so the answer does not depend on how many threads happened to be
 * free. That is the same discipline the unconstrained sweep uses and it is why
 * the two agree bit for bit rather than merely closely.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/query.h"
#include "beam/query_internal.h"

BeamError beam_query_marginals(const BeamModel *m, const BeamConstraint *c,
                               double *out, double *out_log_z) {
    if (!out) return BEAM_ENULL("output buffer");

    const BeamResidueMask *mask = NULL;
    uint32_t length = 0u;
    BeamError err = beam_query_resolve(m, c, &mask, &length);
    if (err != BEAM_OK) return err;
    const BeamAutomaton *aut = c->automaton;
    const uint32_t Q = aut ? aut->n_states : 1u;

    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    for (uint32_t i = 0; i < length; ++i)
        for (uint32_t x = 0; x < R; ++x) out[(size_t)i * R + x] = 0.0;
    if (out_log_z) *out_log_z = -INFINITY;

    /* An empty mask admits nothing, so the conditional does not exist. Said
     * here rather than after a sweep that would find the same thing slower. */
    const uint32_t dead = beam_query_empty_position(c);
    if (dead < length)
        return BEAM_ENOPATHS("the constraint permits no residue at a "
                            "position, so it admits no sequence and the "
                            "conditional it asks for does not exist");

    BeamQueryWork k;
    err = beam_query_work_alloc(m, length, Q, true, &k);
    if (err != BEAM_OK) return err;

    double *chunk = (double *)malloc((size_t)BEAM_SUM_CHUNKS * R
                                     * sizeof(double));
    if (!chunk) {
        beam_query_work_free(&k);
        return BEAM_EALLOC("the marginal partials");
    }

    err = aut ? beam_query_betas_product(m, length, mask, aut, k.w, k.tail,
                                         k.beta)
              : beam_query_betas(m, length, mask, k.w, k.beta);
    if (err != BEAM_OK) goto done;

    {
    /*
     * Z from the backward pass rather than from `m->log_z`. Under a constraint
     * `m->log_z` is the wrong normaliser, and even unconstrained, taking Z from
     * the sweep the numerator came from is what makes each row sum to one to
     * the last bits rather than to within the difference between a forward and
     * a backward evaluation of the same quantity.
     */
    /* The all-sentinel trellis state paired with the automaton's start. */
    const double logz = k.beta[0][(size_t)(S - 1u) * Q];
    if (!isfinite(logz)) {
        err = (mask || aut)
            ? BEAM_ENOPATHS("the constraint describes a configuration this "
                           "model cannot produce at this length, so the "
                           "conditional it asks for does not exist")
            : BEAM_ENOPATHS("the trellis carries no path at this length");
        goto done;
    }
    if (out_log_z) *out_log_z = logz;

    k.cur[(size_t)(S - 1u) * Q] = 1.0;    /* all-sentinel, automaton start */
    double log_scale = 0.0;

    for (uint32_t i = 0; i < length; ++i) {
        const BeamResidueMask allowed =
            mask ? mask[i] : BEAM_RESIDUES_ALL(R);
        beam_layer_weights(m, i, length, k.w, NULL);
        const double *bn = k.beta[i + 1];
        /*
         * The rescale is taken over edges a live path can USE, always, and
         * not over the layer as a whole.
         *
         * An earlier version used the layer's own maximum whenever the mask
         * was full, on the reasoning that an automaton removes (state, q)
         * combinations rather than edges from the layer. That was wrong. A
         * heavy edge into a state the automaton REJECTS still sets the layer
         * maximum, and every accepted path then underflows to zero -- so a row
         * of a distribution came back summing to nothing while the call
         * reported success. Found by review.
         */
        double wmax = beam_query_layer_max_live(m, aut, i, k.w, k.cur, bn, Q,
                                                allowed);
        if (wmax == -INFINITY) {
            err = BEAM_ENOPATHS("the layer has no edge that both a reached "
                               "state can take and a completion can leave, "
                               "although the trellis has a path");
            goto done;
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads())
#endif
        for (int64_t ch = 0; ch < (int64_t)BEAM_SUM_CHUNKS; ++ch) {
            double *part = chunk + (size_t)ch * R;
            for (uint32_t x = 0; x < R; ++x) part[x] = 0.0;
            const uint32_t slo =
                (uint32_t)(((uint64_t)S * ch) / BEAM_SUM_CHUNKS);
            const uint32_t shi =
                (uint32_t)(((uint64_t)S * (ch + 1)) / BEAM_SUM_CHUNKS);
            for (uint32_t s = slo; s < shi; ++s) {
                const double *cs = k.cur + (size_t)s * Q;
                const double *row = k.w + (size_t)s * R;
                const uint32_t base = (s % SK) * A;
                /*
                 * With no automaton `Q` is 1 and `qn` is 0, so the index
                 * arithmetic collapses to `bn[base + x]` and the floating
                 * point performed is exactly the mask sweep's, in exactly its
                 * order. That is what lets one loop serve both tiers.
                 */
                for (uint32_t q = 0; q < Q; ++q) {
                    if (cs[q] == 0.0) continue;
                    const double la = log(cs[q]) + log_scale - logz;
                    for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u) {
                        const uint32_t x = (uint32_t)__builtin_ctz(mm);
                        if (row[x] == -INFINITY) continue;
                        const uint32_t qn =
                            aut ? beam_query_automaton_step(aut, i, q, x) : 0u;
                        const double nb = bn[(size_t)(base + x) * Q + qn];
                        if (nb == -INFINITY) continue;
                        part[x] += exp(la + row[x] + nb);
                    }
                }
            }
        }
        double *row_out = out + (size_t)i * R;
        for (uint32_t ch = 0; ch < BEAM_SUM_CHUNKS; ++ch) {
            const double *part = chunk + (size_t)ch * R;
            for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u) {
                const uint32_t x = (uint32_t)__builtin_ctz(mm);
                row_out[x] += part[x];
            }
        }

        const bool moved = aut
            ? beam_query_advance_product(m, aut, i, k.w, wmax, k.cur, k.next,
                                         allowed, k.beta[i + 1], &log_scale)
            : beam_query_advance(m, k.w, wmax, k.cur, k.next, allowed,
                                 k.beta[i + 1], &log_scale);
        if (!moved) {
            err = BEAM_ENOPATHS("the forward sweep reached a layer with no "
                               "mass although the trellis has a path");
            goto done;
        }
        double *tmp = k.cur; k.cur = k.next; k.next = tmp;
    }
    }

done:
    free(chunk);
    beam_query_work_free(&k);
    return err;
}
