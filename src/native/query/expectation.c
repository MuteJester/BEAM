/**
 * @file expectation.c
 * @brief Exact first two moments of an additive functional, in one forward pass.
 *
 * For `f(s) = sum_i g(i, x_i)` the moments propagate along the trellis, so
 * this file needs no backward field: it carries three numbers per state and
 * sweeps once. That makes it dramatically cheaper than the posterior --
 * `6 * n_states * |Q|` doubles rather than `(length + 1) * n_states * |Q|` --
 * and it is why entropy at a length the marginals cannot afford is still
 * answerable.
 *
 * The variance is accumulated CENTRED. Each state carries a running weight, a
 * running mean and a running sum of squared deviations, and contributions are
 * merged by the parallel-axis identity:
 *
 *   d = m_k - M;  P' = P + p_k;  M' = M + d p_k / P';
 *   S' = S + s_k + d^2 P p_k / P'.
 *
 * The obvious alternative, accumulating `E[f]` and `E[f^2]` and subtracting,
 * loses all significance precisely where a variance is worth asking for --
 * when the spread is small beside the mean. For a surprisal of about 30 nats
 * with a spread of 1e-7, `E[f^2] - E[f]^2` has nothing left to say.
 *
 * A constraint enters exactly as it does in the sweep next door: masks bound
 * the residues a layer may emit, and an automaton adds a dimension whose
 * accepting states are the only ones the final aggregation counts.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/query.h"
#include "beam/query_internal.h"

/* ── Running (weight, mean, sum of squared deviations) ───────── */

/**
 * Merge one contribution into an accumulator.
 *
 * `P == 0` is the first-contribution case and falls out without a branch: the
 * deviation term carries a factor of `P`, so it vanishes, and `M` becomes
 * `m_k` exactly.
 */
static inline void merge(double *P, double *M, double *S,
                         double p_k, double m_k, double s_k) {
    if (p_k == 0.0) return;
    const double pn = *P + p_k;
    const double d  = m_k - *M;
    *S += s_k + d * d * (*P) * p_k / pn;
    *M += d * p_k / pn;
    *P  = pn;
}

/* ── Scratch ─────────────────────────────────────────────────── */

typedef struct {
    double *a[2];     /* rescaled weight per product state      */
    double *mu[2];    /* mean of f over prefixes through it     */
    double *s2[2];    /* weight times variance, same rescale    */
    double *w;        /* one layer of edge weights              */
    double *tail;     /* sink weights                           */
} MomentWork;

static void mw_free(MomentWork *k) {
    for (int i = 0; i < 2; ++i) {
        free(k->a[i]); free(k->mu[i]); free(k->s2[i]);
    }
    free(k->w); free(k->tail);
    memset(k, 0, sizeof(*k));
}

static BeamError mw_alloc(const BeamModel *m, uint32_t n_aut, MomentWork *k) {
    memset(k, 0, sizeof(*k));
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const size_t   P = (size_t)S * n_aut;

    const uint64_t need   = beam_query_moment_bytes(S, n_aut, R);
    const uint64_t budget = beam_model_memory_budget();
    if (need > budget)
        return BEAM_FAIL(BEAM_ERR_ALLOC,
                        "a moment sweep over %u trellis states times %u "
                        "automaton states needs about %.1f GB, and the memory "
                        "budget is %.1f GB; raise it with "
                        "BEAM.set_memory_budget, ask at a lower order, or use "
                        "a simpler constraint",
                        (unsigned)S, (unsigned)n_aut, (double)need / 1e9,
                        (double)budget / 1e9);

    for (int i = 0; i < 2; ++i) {
        k->a[i]  = (double *)calloc(P, sizeof(double));
        k->mu[i] = (double *)calloc(P, sizeof(double));
        k->s2[i] = (double *)calloc(P, sizeof(double));
        if (!k->a[i] || !k->mu[i] || !k->s2[i]) {
            mw_free(k);
            return BEAM_EALLOC("the moment accumulators");
        }
    }
    k->w    = (double *)malloc((size_t)S * R * sizeof(double));
    k->tail = (double *)malloc((size_t)S * sizeof(double));
    if (!k->w || !k->tail) {
        mw_free(k);
        return BEAM_EALLOC("the moment sweep buffers");
    }
    return BEAM_OK;
}

/* ── The sweep ───────────────────────────────────────────────── */

BeamError beam_query_expect(const BeamModel *m, const BeamConstraint *c,
                            const BeamFunctional *f, BeamQueryMoments *out) {
    if (!out) return BEAM_ENULL("output");
    if (!f)   return BEAM_ENULL("functional");

    const BeamResidueMask *mask = NULL;
    uint32_t length = 0u;
    BeamError err = beam_query_resolve(m, c, &mask, &length);
    if (err != BEAM_OK) return err;

    if (f->kind != BEAM_FUNCTIONAL_TABLE
        && f->kind != BEAM_FUNCTIONAL_LOG_WEIGHT)
        return BEAM_ERANGE("beam_query_expect: functional kind %d is not one "
                          "this engine computes; only a per-position table and "
                          "the edge log weight decompose over the trellis",
                          (int)f->kind);
    if (f->kind == BEAM_FUNCTIONAL_TABLE && !f->table)
        return BEAM_ENULL("functional table");

    const BeamAutomaton *aut = c->automaton;
    const uint32_t Q  = aut ? aut->n_states : 1u;
    const uint32_t A  = m->n_alphabet;
    const uint32_t R  = m->n_residues;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;

    out->mean = 0.0; out->variance = 0.0; out->log_z = -INFINITY;

    if (beam_query_empty_position(c) < length)
        return BEAM_ENOPATHS("the constraint permits no residue at a "
                            "position, so it admits no sequence and has no "
                            "moments");

    MomentWork k;
    err = mw_alloc(m, Q, &k);
    if (err != BEAM_OK) return err;

    const size_t n_prod = (size_t)S * Q;
    int cur = 0;
    k.a[cur][(size_t)(S - 1u) * Q] = 1.0;   /* all-sentinel, automaton start */
    double log_scale = 0.0;

    for (uint32_t i = 0; i < length; ++i) {
        const BeamResidueMask allowed = mask ? mask[i] : BEAM_RESIDUES_ALL(R);
        double wmax = -INFINITY;
        beam_layer_weights(m, i, length, k.w, &wmax);
        if (wmax == -INFINITY) {
            mw_free(&k);
            return BEAM_ENOPATHS("the layer has no live edge at all");
        }
        if (allowed != BEAM_RESIDUES_ALL(R)) {
            wmax = aut
                 ? beam_query_layer_max_product(k.w, k.a[cur], S, R, Q, allowed)
                 : beam_query_layer_max(k.w, k.a[cur], S, R, allowed);
            if (wmax == -INFINITY) {
                mw_free(&k);
                return BEAM_ENOPATHS("the constraint leaves this layer no "
                                    "live edge, so the sequences it describes "
                                    "do not exist at this length");
            }
        }

        const int nxt = cur ^ 1;
        double *na = k.a[nxt], *nm = k.mu[nxt], *ns = k.s2[nxt];
        memset(na, 0, n_prod * sizeof(double));
        memset(nm, 0, n_prod * sizeof(double));
        memset(ns, 0, n_prod * sizeof(double));

        /*
         * Accumulated by TARGET, as every sweep in this subsystem is, so no
         * two threads write one accumulator -- and here that matters twice
         * over, because a merge reads the running mean it is about to update.
         */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < (int64_t)S; ++t) {
            const uint32_t x = (uint32_t)t % A;
            if (!((allowed >> x) & 1u)) continue;
            const uint32_t rest = (uint32_t)t / A;
            const double g = (f->kind == BEAM_FUNCTIONAL_TABLE)
                           ? f->table[(size_t)i * R + x] : 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                const double wv = k.w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                const double e = exp(wv - wmax);
                /* LOG_WEIGHT's increment is the edge's own weight. */
                const double inc = (f->kind == BEAM_FUNCTIONAL_LOG_WEIGHT)
                                 ? wv : g;
                for (uint32_t q = 0; q < Q; ++q) {
                    const double pa = k.a[cur][(size_t)src * Q + q];
                    if (pa == 0.0) continue;
                    const uint32_t qn = aut
                        ? beam_query_automaton_step(aut, i, q, x) : 0u;
                    const size_t d = (size_t)t * Q + qn;
                    merge(&na[d], &nm[d], &ns[d],
                          pa * e,
                          k.mu[cur][(size_t)src * Q + q] + inc,
                          k.s2[cur][(size_t)src * Q + q] * e);
                }
            }
        }

        double nmax = 0.0;
        for (size_t p = 0; p < n_prod; ++p) if (na[p] > nmax) nmax = na[p];
        if (!(nmax > 0.0)) {
            mw_free(&k);
            return BEAM_ENOPATHS("the forward sweep reached a layer with no "
                                "mass, so the constraint admits no sequence "
                                "of this length");
        }
        for (size_t p = 0; p < n_prod; ++p) { na[p] /= nmax; ns[p] /= nmax; }
        log_scale += wmax + log(nmax);
        cur = nxt;
    }

    /*
     * The sink settles the last residues' backward terms, so it is a weight on
     * every path and -- for LOG_WEIGHT -- the last increment of the functional
     * too. Acceptance enters here and only here.
     */
    beam_tail_weights(m, length, k.tail);
    double tmax = -INFINITY;
    for (uint32_t s = 0; s < S; ++s) if (k.tail[s] > tmax) tmax = k.tail[s];
    if (tmax == -INFINITY) {
        mw_free(&k);
        return BEAM_ENOPATHS("no state of the final layer can terminate");
    }

    double P = 0.0, M = 0.0, Sacc = 0.0;
    for (uint32_t s = 0; s < S; ++s) {
        if (k.tail[s] == -INFINITY) continue;
        const double te = exp(k.tail[s] - tmax);
        const double inc = (f->kind == BEAM_FUNCTIONAL_LOG_WEIGHT)
                         ? k.tail[s] : 0.0;
        for (uint32_t q = 0; q < Q; ++q) {
            if (aut && !aut->accept[q]) continue;
            const size_t p = (size_t)s * Q + q;
            if (k.a[cur][p] == 0.0) continue;
            merge(&P, &M, &Sacc, k.a[cur][p] * te, k.mu[cur][p] + inc,
                  k.s2[cur][p] * te);
        }
    }
    mw_free(&k);

    if (!(P > 0.0))
        return BEAM_ENOPATHS("the constraint describes a configuration this "
                            "model cannot produce at this length, so its "
                            "moments do not exist");

    out->log_z    = log(P) + log_scale + tmax;
    out->mean     = M;
    /* Never negative: a sum of squared deviations divided by a positive
     * weight. Clamped anyway, because a caller taking a square root of a
     * -1e-19 deserves better than a NaN. */
    out->variance = Sacc > 0.0 ? Sacc / P : 0.0;
    return BEAM_OK;
}

/* ── Entropy ─────────────────────────────────────────────────── */

BeamError beam_query_entropy(const BeamModel *m, const BeamConstraint *c,
                             double *out_entropy, double *out_variance) {
    if (!out_entropy) return BEAM_ENULL("output");
    const BeamFunctional f = { BEAM_FUNCTIONAL_LOG_WEIGHT, NULL };
    BeamQueryMoments mo;
    const BeamError e = beam_query_expect(m, c, &f, &mo);
    if (e != BEAM_OK) return e;
    /* H = ln Z - E[ln w]. The surprisal -ln P = ln Z - ln w differs from
     * -ln w by a constant, so their variances are the same number. */
    *out_entropy = mo.log_z - mo.mean;
    if (out_variance) *out_variance = mo.variance;
    return BEAM_OK;
}
