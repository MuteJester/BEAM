/**
 * @file engine.c
 * @brief The constrained sum-product sweep: partition functions under a mask.
 *
 * This is `math/positional.c`'s forward-backward with the residue RANGE of a
 * clamp generalised to a residue SET. That generalisation is deliberately the
 * only difference, and the two agree BIT FOR BIT: a singleton set visits the
 * residue the range `lo..lo+1` would, a full set visits `0..R-1` in the same
 * ascending order, and both halves of both sweeps now carry logarithms.
 *
 * The forward half did not always. It rescaled each layer by a single factor,
 * and no such factor can hold a trellis whose live states span more than 745
 * nats -- at the C API's own defaults a length-60 nucleotide sweep does, and
 * the states that underflowed included pinned positions whose row must be
 * one-hot and came back summing to 2e-54.
 *
 * Sets are iterated by clearing the lowest set bit, which is ascending order
 * and keeps the saving that motivated the range: a layer pinned to one residue
 * does a twentieth of the work of a free one rather than the same work and
 * nineteen twentieths of it discarded.
 *
 * Everything here is log-space sums. The other semirings reuse this file's
 * traversal through engine_kernels.h and do not reimplement it.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/query.h"
#include "beam/query_internal.h"

/* ── Work block ──────────────────────────────────────────────── */

void beam_query_work_free(BeamQueryWork *k) {
    if (!k) return;
    if (k->beta) {
        for (uint32_t i = 0; i < k->n_beta; ++i) free(k->beta[i]);
        free(k->beta);
    }
    free(k->w); free(k->cur); free(k->next); free(k->acc); free(k->tail);
    memset(k, 0, sizeof(*k));
}

BeamError beam_query_work_alloc(const BeamModel *m, uint32_t length,
                                uint32_t n_aut, bool want_forward,
                                BeamQueryWork *k) {
    memset(k, 0, sizeof(*k));
    k->n_aut = n_aut;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const size_t   P = (size_t)S * n_aut;

    /*
     * Refused before allocating, not after. `calloc` on Linux grants an absurd
     * request lazily and fails later on first write, with the OOM killer
     * choosing a victim from the whole machine rather than from this process.
     */
    const uint64_t need   = beam_query_sweep_bytes(S, n_aut, R, length,
                                                   want_forward);
    const uint64_t budget = beam_model_memory_budget();
    if (need > budget)
        return BEAM_FAIL(BEAM_ERR_ALLOC,
                        "a constrained sweep at length %u needs about %.1f GB "
                        "for the backward field of %u trellis states times %u "
                        "automaton states, and the memory budget is %.1f GB; "
                        "raise it with BEAM.set_memory_budget, ask at a lower "
                        "order, or use a simpler constraint",
                        (unsigned)length, (double)need / 1e9, (unsigned)S,
                        (unsigned)n_aut, (double)budget / 1e9);

    k->beta = (double **)calloc((size_t)length + 1u, sizeof(double *));
    if (!k->beta) return BEAM_EALLOC("the backward-value layer table");
    k->n_beta = length + 1u;
    for (uint32_t i = 0; i <= length; ++i) {
        k->beta[i] = (double *)malloc(P * sizeof(double));
        if (!k->beta[i]) {
            beam_query_work_free(k);
            return BEAM_EALLOC("a backward layer");
        }
    }
    k->w = (double *)malloc((size_t)S * R * sizeof(double));
    if (!k->w) {
        beam_query_work_free(k);
        return BEAM_EALLOC("one layer of edge weights");
    }
    /*
     * Allocated for every sweep, not only the product one. The mask sweep
     * writes the sink straight into its last backward layer and never touches
     * this, but `beam_query_betas_product` writes through it unconditionally,
     * and gating the allocation on `n_aut > 1` left a null dereference waiting
     * for the first one-state automaton. Found by review.
     */
    k->tail = (double *)malloc((size_t)S * sizeof(double));
    if (!k->tail) {
        beam_query_work_free(k);
        return BEAM_EALLOC("the sink weights");
    }
    if (!want_forward) return BEAM_OK;

    k->cur  = (double *)malloc(P * sizeof(double));
    k->next = (double *)malloc(P * sizeof(double));
    k->acc  = (double *)malloc(P * sizeof(double));
    if (!k->cur || !k->next || !k->acc) {
        beam_query_work_free(k);
        return BEAM_EALLOC("the constrained sweep buffers");
    }
    return BEAM_OK;
}

/* ── Resolving a constraint against a model ──────────────────── */

BeamError beam_query_resolve(const BeamModel *m, const BeamConstraint *c,
                             const BeamResidueMask **out_mask,
                             uint32_t *out_length) {
    if (!m) return BEAM_ENULL("model");
    if (!c) return BEAM_ENULL("constraint");
    if (!m->finalized) return BEAM_ENOTBUILT();

    if (c->n_residues != m->n_residues)
        return BEAM_ERANGE("beam_query: the constraint is expressed over %u "
                          "residues and this model has %u, so its residue "
                          "indices would mean different letters",
                          (unsigned)c->n_residues, (unsigned)m->n_residues);
    if (!beam_length_live(m, c->length)) return BEAM_EBADLEN(m, c->length);

    *out_length = c->length;
    *out_mask   = beam_query_all_free(c) ? NULL : c->mask;
    return BEAM_OK;
}



/**
 * The residues layer `i` may emit, as a set.
 *
 * A NULL mask means unconstrained, which is the full set. Keeping that in one
 * place is what lets every loop below impose the constraint by the set it
 * iterates rather than by a test in its body.
 */
static inline BeamResidueMask layer_set(const BeamResidueMask *mask,
                                        uint32_t i, uint32_t R) {
    return mask ? mask[i] : BEAM_RESIDUES_ALL(R);
}

/* ── Backward values under a constraint ──────────────────────── */

BeamError beam_query_betas(const BeamModel *m, uint32_t length,
                           const BeamResidueMask *mask, double *w,
                           double **beta) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    beam_tail_weights(m, length, beta[length]);

    for (int32_t i = (int32_t)length - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, length, w, NULL);
        const BeamResidueMask allowed = layer_set(mask, (uint32_t)i, R);
        const double *nb = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
            const uint32_t s = (uint32_t)s64;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u) {
                const uint32_t x = (uint32_t)__builtin_ctz(mm);
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            if (mx == -INFINITY) { beta[i][s] = -INFINITY; continue; }
            double acc = 0.0;
            for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u) {
                const uint32_t x = (uint32_t)__builtin_ctz(mm);
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                acc += exp(row[x] + nb[base + x] - mx);
            }
            beta[i][s] = mx + log(acc);
        }
    }
    return BEAM_OK;
}

/* ── The forward sweep ───────────────────────────────────────── */

bool beam_query_advance(const BeamModel *m, const double *w,
                        const double *cur, double *next, double *acc,
                        BeamResidueMask allowed, const double *beta_next) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    bool any = false;
    /* Accumulated by TARGET: a target's sources are its private business, so
     * no two threads write the same accumulator. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:any) \
            num_threads(beam_threads())
#endif
    for (int64_t t = 0; t < (int64_t)S; ++t) {
        const uint32_t x = (uint32_t)t % A;
        /*
         * The sentinel sits at index R and no mask has a bit there, so the
         * first test excludes both a forbidden residue and a non-residue.
         *
         * The second is now only PRUNING. A state with no completion
         * contributes to no posterior term either way, and in log space it can
         * no longer distort a rescale, because there is no rescale. It was
         * load-bearing while there was one, and saying so plainly is better
         * than a comment implying the answer still depends on it.
         */
        if (!((allowed >> x) & 1u) || (beta_next && beta_next[t] == -INFINITY)) {
            next[t] = -INFINITY;
            continue;
        }
        const uint32_t rest = (uint32_t)t / A;
        double mx = -INFINITY, sum = 0.0;
        for (uint32_t a = 0; a < A; ++a) {
            const uint32_t src = a * SK + rest;
            if (cur[src] == -INFINITY) continue;
            const double wv = w[(size_t)src * R + x];
            if (wv == -INFINITY) continue;
            lse_add(&mx, &sum, cur[src] + wv);
        }
        next[t] = (mx == -INFINITY) ? -INFINITY : mx + log(sum);
        if (next[t] != -INFINITY) any = true;
    }
    (void)acc;
    return any;
}

/* ── Constrained probability ─────────────────────────────────── */

BeamError beam_query_log_probability(const BeamModel *m,
                                     const BeamConstraint *c,
                                     double *out_log_p) {
    if (!out_log_p) return BEAM_ENULL("output");

    const BeamResidueMask *mask = NULL;
    uint32_t length = 0u;
    BeamError err = beam_query_resolve(m, c, &mask, &length);
    if (err != BEAM_OK) return err;
    const BeamAutomaton *aut = c->automaton;

    /*
     * Two answers are exact without a sweep, and taking them here is not an
     * optimisation. A constraint that admits everything has probability one,
     * and `ln 1` is 0.0 exactly -- whereas subtracting the forward-evaluated
     * `m->log_z` from the backward-evaluated `beta[0][S-1]` gives the last
     * bits of the difference between two routes to the same number, which is
     * a probability fractionally above or below one. `log_clamp_probability`
     * already short-circuits the same case for the same reason.
     */
    if (!mask && !aut) { *out_log_p = 0.0; return BEAM_OK; }
    /* And no sequence draws a residue from the empty set. */
    if (beam_query_empty_position(c) < length) {
        *out_log_p = -INFINITY;
        return BEAM_OK;
    }

    /* No forward buffers: a partition function is a backward sweep. */
    const uint32_t Q = aut ? aut->n_states : 1u;
    BeamQueryWork k;
    err = beam_query_work_alloc(m, length, Q, false, &k);
    if (err != BEAM_OK) return err;

    err = aut ? beam_query_betas_product(m, length, mask, aut, k.w, k.tail,
                                         k.beta)
              : beam_query_betas(m, length, mask, k.w, k.beta);
    if (err == BEAM_OK) {
        /*
         * The numerator is the constrained backward sweep and the denominator
         * is the model's own partition function, which is what
         * `log_clamp_probability` does today and what this must agree with.
         * A constraint the trellis cannot satisfy leaves `-inf`, which is the
         * answer to the question rather than a failure to answer it; it is
         * CONDITIONING on such an event that has no answer, and that refusal
         * lives in beam_query_marginals.
         */
        /* The all-sentinel trellis state paired with the automaton's start,
         * which is state 0 by construction. */
        const double logz_c = k.beta[0][(size_t)(m->n_states - 1u) * Q];
        *out_log_p = isfinite(logz_c)
                   ? logz_c - beam_model_log_partition(m, length)
                   : -INFINITY;
    }
    beam_query_work_free(&k);
    return err;
}
