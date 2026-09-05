/**
 * @file information.c
 * @brief Entropy per position, pairwise joints, and mutual information.
 *
 * Everything here is built from one primitive: conditioning on a single
 * position and asking for the marginals. That one sweep answers
 * `P(x_j = b | x_i = a)` for EVERY `j` at once, so multiplying by
 * `P(x_i = a)` gives a whole row of joints, and `length * n_residues` such
 * sweeps give every pair.
 *
 * The alternative -- an automaton that records `x_i` in the product state, so
 * that one sweep per position replaces `n_residues` of them -- was measured
 * and rejected. It does the same work to within 5% because its backward field
 * is `(n_residues + 1)` times taller, and it costs twenty-one times the peak
 * memory; at nucleotide order 9 that is 22 GB against 3.7 GB, and only one of
 * those fits the default budget. Clamping stays on the free tier.
 *
 * Mutual information is assembled from entropies rather than from
 * `sum p log (p / (p_i p_j))` directly, because the entropies are wanted in
 * their own right and the difference form reuses them. Both are exact; the
 * difference form can round to a very slightly negative number where two
 * positions are independent, which is handled where it arises rather than by
 * pretending it cannot happen.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/query.h"
#include "beam/query_internal.h"

/** Shannon entropy of a distribution, in nats. Zero probabilities contribute
 *  nothing, which is the limit of `p ln p` and not a special case. */
static double entropy_of(const double *p, uint32_t n) {
    double h = 0.0;
    for (uint32_t k = 0; k < n; ++k)
        if (p[k] > 0.0) h -= p[k] * log(p[k]);
    return h;
}

/**
 * Narrow one position of a working copy to a single residue.
 *
 * The copy is made once per position and re-pinned for each residue in turn,
 * rather than rebuilt for each of the `length * n_residues` pins: cloning an
 * automaton per residue would allocate a transition table twenty times over
 * to change one word of one mask.
 *
 * The caller's constraint is never touched, and the copy carries the
 * automaton, because conditioning inside a language means conditioning inside
 * that language and not merely alongside it.
 *
 * Intersecting with the source mask rather than overwriting it cannot
 * currently be observed: both callers skip residues whose marginal is zero,
 * and a residue the mask excludes has marginal zero, so `a` is always already
 * in `src->mask[i]`. It is written this way so that `repin` is correct on its
 * own terms rather than on its callers', which is the difference between a
 * helper and a fragment.
 */
static inline void repin(BeamConstraint *d, const BeamConstraint *src,
                         uint32_t i, uint32_t a) {
    d->mask[i] = src->mask[i] & ((BeamResidueMask)1u << a);
}

/* ── Per-position entropy ────────────────────────────────────── */

BeamError beam_query_position_entropy(const BeamModel *m,
                                      const BeamConstraint *c, double *out) {
    if (!out) return BEAM_ENULL("output");
    const BeamResidueMask *mask = NULL;
    uint32_t length = 0u;
    BeamError err = beam_query_resolve(m, c, &mask, &length);
    if (err != BEAM_OK) return err;

    const uint32_t R = m->n_residues;
    double *marg = (double *)malloc((size_t)length * R * sizeof(double));
    if (!marg) return BEAM_EALLOC("the positional marginals");
    err = beam_query_marginals(m, c, marg, NULL);
    if (err == BEAM_OK)
        for (uint32_t i = 0; i < length; ++i)
            out[i] = entropy_of(marg + (size_t)i * R, R);
    free(marg);
    return err;
}

/* ── Pairwise joints ─────────────────────────────────────────── */

BeamError beam_query_pair_joint(const BeamModel *m, const BeamConstraint *c,
                                uint32_t i, uint32_t j, double *out) {
    if (!out) return BEAM_ENULL("output");
    const BeamResidueMask *mask = NULL;
    uint32_t length = 0u;
    BeamError err = beam_query_resolve(m, c, &mask, &length);
    if (err != BEAM_OK) return err;
    if (i >= length || j >= length)
        return BEAM_ERANGE("beam_query_pair_joint: positions %u and %u must "
                          "both be below the constraint's length of %u",
                          (unsigned)i, (unsigned)j, (unsigned)length);

    const uint32_t R = m->n_residues;
    for (uint32_t k = 0; k < R * R; ++k) out[k] = 0.0;

    double *marg = (double *)malloc((size_t)length * R * sizeof(double));
    double *cond = (double *)malloc((size_t)length * R * sizeof(double));
    if (!marg || !cond) { free(marg); free(cond);
        return BEAM_EALLOC("the joint's working marginals"); }

    BeamConstraint *d = beam_query_constraint_clone(c);
    if (!d) { free(marg); free(cond);
        return BEAM_EALLOC("the working constraint"); }

    err = beam_query_marginals(m, c, marg, NULL);
    for (uint32_t a = 0; a < R && err == BEAM_OK; ++a) {
        const double pa = marg[(size_t)i * R + a];
        if (pa <= 0.0) continue;      /* contributes nothing, and cannot be
                                       * conditioned on */
        repin(d, c, i, a);
        err = beam_query_marginals(m, d, cond, NULL);
        if (err != BEAM_OK) break;
        for (uint32_t b = 0; b < R; ++b)
            out[(size_t)a * R + b] = pa * cond[(size_t)j * R + b];
    }
    beam_query_constraint_free(d);
    free(marg); free(cond);
    return err;
}

/* ── Mutual information ──────────────────────────────────────── */

BeamError beam_query_information_matrix(const BeamModel *m,
                                        const BeamConstraint *c, double *out) {
    if (!out) return BEAM_ENULL("output");
    const BeamResidueMask *mask = NULL;
    uint32_t length = 0u;
    BeamError err = beam_query_resolve(m, c, &mask, &length);
    if (err != BEAM_OK) return err;

    const uint32_t R = m->n_residues;
    const size_t LL = (size_t)length * length;

    double *marg = (double *)malloc((size_t)length * R * sizeof(double));
    double *cond = (double *)malloc((size_t)length * R * sizeof(double));
    double *hj   = (double *)calloc(LL, sizeof(double));  /* H(X_i, X_j) */
    double *h    = (double *)malloc((size_t)length * sizeof(double));
    if (!marg || !cond || !hj || !h) {
        free(marg); free(cond); free(hj); free(h);
        return BEAM_EALLOC("the information matrix accumulators");
    }

    err = beam_query_marginals(m, c, marg, NULL);
    if (err == BEAM_OK)
        for (uint32_t i = 0; i < length; ++i)
            h[i] = entropy_of(marg + (size_t)i * R, R);

    /*
     * One sweep per (position, residue) that carries mass. Each yields a whole
     * row of the joint -- `P(x_j = b | x_i = a)` for every `j` -- so the joint
     * entropies accumulate a residue at a time and no joint is ever stored.
     */
    BeamConstraint *d = beam_query_constraint_clone(c);
    if (!d) {
        free(marg); free(cond); free(hj); free(h);
        return BEAM_EALLOC("the working constraint");
    }
    for (uint32_t i = 0; i < length && err == BEAM_OK; ++i) {
        for (uint32_t a = 0; a < R; ++a) {
            const double pa = marg[(size_t)i * R + a];
            if (pa <= 0.0) continue;
            repin(d, c, i, a);
            err = beam_query_marginals(m, d, cond, NULL);
            if (err != BEAM_OK) break;
            for (uint32_t j = 0; j < length; ++j) {
                double acc = 0.0;
                for (uint32_t b = 0; b < R; ++b) {
                    const double pab = pa * cond[(size_t)j * R + b];
                    if (pab > 0.0) acc -= pab * log(pab);
                }
                hj[(size_t)i * length + j] += acc;
            }
        }
        /* Restore before moving on, or position i stays pinned for i + 1. */
        d->mask[i] = c->mask[i];
    }
    beam_query_constraint_free(d);

    if (err == BEAM_OK) {
        for (uint32_t i = 0; i < length; ++i) {
            for (uint32_t j = 0; j < length; ++j) {
                const double mi = h[i] + h[j] - hj[(size_t)i * length + j];
                /*
                 * Mutual information is provably non-negative, so a negative
                 * value here is either rounding or a defect, and the two are
                 * told apart by size. Rounding on a difference of entropies
                 * of order 1 lands within a few ulps -- the smallest positive
                 * entry measured on an independent-position model is 1e-31 --
                 * while anything larger is not rounding.
                 *
                 * Neither branch is reachable from a working engine, so no
                 * test exercises them and none can: this is a defect detector,
                 * and the alternative to having it is a silently negative
                 * information reported as fact.
                 */
                if (mi < -1e-9) {
                    free(marg); free(cond); free(hj); free(h);
                    return BEAM_FAIL(BEAM_ERR_INTERNAL,
                        "mutual information between positions %u and %u came "
                        "out as %.17g, and it cannot be negative; this is a "
                        "defect in the engine rather than a property of the "
                        "model", (unsigned)i, (unsigned)j, mi);
                }
                out[(size_t)i * length + j] = mi > 0.0 ? mi : 0.0;
            }
        }
    }
    free(marg); free(cond); free(hj); free(h);
    return err;
}

BeamError beam_query_information_cost(const BeamModel *m,
                                      const BeamConstraint *c,
                                      BeamQueryCost *out) {
    BeamError err = beam_query_estimate(m, c, out);
    if (err != BEAM_OK) return err;
    /* One sweep per (position, residue), plus the unconstrained one that
     * finds out which of them carry mass. */
    const uint64_t L = beam_query_constraint_length(c);
    const uint64_t R = m->n_residues;
    const uint64_t sweeps = L * R + 1u;
    out->edges  *= sweeps;
    out->states *= sweeps;

    /*
     * Held ACROSS every one of those sweeps, so it is peak memory the sweep's
     * own figure does not include: two marginal blocks, the joint-entropy
     * accumulator, the per-position entropies, and a working copy of the
     * constraint with its automaton. Reporting one sweep's bytes let a caller
     * set the budget to exactly the reported cost and then exceed it. Found
     * by review.
     */
    uint64_t held = 2u * L * R * sizeof(double)      /* marg and cond      */
                  + L * L * sizeof(double)           /* joint entropies    */
                  + L * sizeof(double)               /* position entropies */
                  + L * sizeof(BeamResidueMask);     /* the cloned masks   */
    const BeamAutomaton *aut = beam_query_constraint_automaton(c);
    if (aut) {
        const uint64_t Q = beam_query_automaton_states(aut);
        held += (uint64_t)beam_query_automaton_layers(aut) * Q * R
                * sizeof(uint32_t) + Q;
    }
    out->bytes += held;
    return BEAM_OK;
}
