/**
 * @file engine.c
 * @brief The constrained sum-product sweep: partition functions under a mask.
 *
 * This is `math/positional.c`'s forward-backward with the residue RANGE of a
 * clamp generalised to a residue SET. That generalisation is deliberately the
 * only difference, because the equivalence it preserves is the whole
 * correctness argument for this subsystem: a singleton set visits the residue
 * the range `lo..lo+1` would visit, a full set visits `0..R-1` in the same
 * ascending order, so on those two inputs this file performs the same
 * operations on the same values in the same order and agrees BIT FOR BIT.
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
    free(k->w); free(k->cur); free(k->next); free(k->tail);
    memset(k, 0, sizeof(*k));
}

BeamError beam_query_work_alloc(const BeamModel *m, uint32_t length,
                                bool want_forward, BeamQueryWork *k) {
    memset(k, 0, sizeof(*k));
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;

    /*
     * Refused before allocating, not after. `calloc` on Linux grants an absurd
     * request lazily and fails later on first write, with the OOM killer
     * choosing a victim from the whole machine rather than from this process.
     */
    const uint64_t need   = beam_query_sweep_bytes(S, R, length, want_forward);
    const uint64_t budget = beam_model_memory_budget();
    if (need > budget)
        return BEAM_FAIL(BEAM_ERR_ALLOC,
                        "a constrained sweep at length %u needs about %.1f GB "
                        "for the backward field of %u states, and the memory "
                        "budget is %.1f GB; raise it with "
                        "BEAM.set_memory_budget, or ask at a lower order",
                        (unsigned)length, (double)need / 1e9, (unsigned)S,
                        (double)budget / 1e9);

    k->beta = (double **)calloc((size_t)length + 1u, sizeof(double *));
    if (!k->beta) return BEAM_EALLOC("the backward-value layer table");
    k->n_beta = length + 1u;
    for (uint32_t i = 0; i <= length; ++i) {
        k->beta[i] = (double *)malloc((size_t)S * sizeof(double));
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
    if (!want_forward) return BEAM_OK;

    k->cur  = (double *)calloc(S, sizeof(double));
    k->next = (double *)malloc((size_t)S * sizeof(double));
    k->tail = (double *)malloc((size_t)S * sizeof(double));
    if (!k->cur || !k->next || !k->tail) {
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

double beam_query_layer_max(const double *w, const double *cur,
                            uint32_t n_states, uint32_t n_residues,
                            BeamResidueMask allowed) {
    double mx = -INFINITY;
    for (uint32_t s = 0; s < n_states; ++s) {
        if (cur[s] == 0.0) continue;
        const double *row = w + (size_t)s * n_residues;
        for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u) {
            const uint32_t x = (uint32_t)__builtin_ctz(mm);
            if (row[x] > mx) mx = row[x];
        }
    }
    return mx;
}

bool beam_query_advance(const BeamModel *m, const double *w, double wmax,
                        const double *cur, double *next,
                        BeamResidueMask allowed, double *log_scale) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double nmax = 0.0;
    /* Accumulated by TARGET: a target's sources are its private business, so
     * no two threads write the same accumulator. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:nmax) \
            num_threads(beam_threads())
#endif
    for (int64_t t = 0; t < (int64_t)S; ++t) {
        const uint32_t x = (uint32_t)t % A;
        /* The sentinel sits at index R and no mask has a bit there, so this
         * one test excludes both a forbidden residue and a non-residue. */
        if (!((allowed >> x) & 1u)) { next[t] = 0.0; continue; }
        const uint32_t rest = (uint32_t)t / A;
        double sum = 0.0;
        for (uint32_t a = 0; a < A; ++a) {
            const uint32_t src = a * SK + rest;
            if (cur[src] == 0.0) continue;
            const double wv = w[(size_t)src * R + x];
            if (wv == -INFINITY) continue;
            sum += cur[src] * exp(wv - wmax);
        }
        next[t] = sum;
        if (sum > nmax) nmax = sum;
    }
    *log_scale += wmax;
    if (!(nmax > 0.0)) return false;
    for (uint32_t s = 0; s < S; ++s) next[s] /= nmax;
    *log_scale += log(nmax);
    return true;
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

    /*
     * Two answers are exact without a sweep, and taking them here is not an
     * optimisation. A constraint that admits everything has probability one,
     * and `ln 1` is 0.0 exactly -- whereas subtracting the forward-evaluated
     * `m->log_z` from the backward-evaluated `beta[0][S-1]` gives the last
     * bits of the difference between two routes to the same number, which is
     * a probability fractionally above or below one. `log_clamp_probability`
     * already short-circuits the same case for the same reason.
     */
    if (!mask) { *out_log_p = 0.0; return BEAM_OK; }
    /* And no sequence draws a residue from the empty set. */
    if (beam_query_empty_position(c) < length) {
        *out_log_p = -INFINITY;
        return BEAM_OK;
    }

    /* No forward buffers: a partition function is a backward sweep. */
    BeamQueryWork k;
    err = beam_query_work_alloc(m, length, false, &k);
    if (err != BEAM_OK) return err;

    err = beam_query_betas(m, length, mask, k.w, k.beta);
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
        const double logz_c = k.beta[0][m->n_states - 1u];
        *out_log_p = isfinite(logz_c)
                   ? logz_c - beam_model_log_partition(m, length)
                   : -INFINITY;
    }
    beam_query_work_free(&k);
    return err;
}
