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

/* ── Running (log weight, mean, variance) ────────────────────── */

/**
 * Merge one contribution into an accumulator, in log space.
 *
 * The weight is carried as a logarithm and the merge uses only NORMALISED
 * ratios, so nothing here can overflow or underflow however far the mass sits
 * below the mode. That is why there is no rescaling pass in this file at all.
 *
 * The rescaled-linear form this replaces had two defects that review found.
 * Its first-contribution case formed `d * d` where `d` was a whole mean rather
 * than a difference, so a functional valued around 1e155 overflowed to
 * infinity and `infinity * 0` gave a NaN that was later reported as a variance
 * of zero. And its per-layer rescale was chosen over states that could not
 * reach an accepting end, so a dominant rejected path drove the surviving mass
 * below the smallest double and a producible event came back as
 * NO_LIVE_PATHS. Neither failure is possible without a rescale.
 *
 * The variance is a mixture, by the law of total variance:
 *
 *   v = wP v_P + wk v_k + wP wk (m_k - M)^2,
 *
 * with `wP + wk = 1`. Both weights come from one `logaddexp`, so they are
 * exact complements and lie in [0, 1] whatever the inputs.
 */
static inline void merge(double *logP, double *M, double *V,
                         double log_p, double m_k, double v_k) {
    if (log_p == -INFINITY) return;
    if (*logP == -INFINITY) { *logP = log_p; *M = m_k; *V = v_k; return; }

    const double hi = *logP > log_p ? *logP : log_p;
    const double ln = hi + log(exp(*logP - hi) + exp(log_p - hi));
    const double wk = exp(log_p - ln);
    const double wP = exp(*logP - ln);
    const double d  = m_k - *M;

    *V    = wP * (*V) + wk * v_k + (wP * wk) * d * d;
    *M   += wk * d;
    *logP = ln;
}

/* ── Scratch ─────────────────────────────────────────────────── */

typedef struct {
    double *la[2];    /* LOG weight per product state           */
    double *mu[2];    /* mean of f over prefixes through it     */
    double *v[2];     /* variance of f over those prefixes      */
    double *w;        /* one layer of edge weights              */
    double *tail;     /* sink weights                           */
} MomentWork;

static void mw_free(MomentWork *k) {
    for (int i = 0; i < 2; ++i) {
        free(k->la[i]); free(k->mu[i]); free(k->v[i]);
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
    if (need > budget) {
        const char *nu, *bu;
        const double nv = beam_budget_scaled(need, &nu);
        const double bv = beam_budget_scaled(budget, &bu);
        return BEAM_FAIL(BEAM_ERR_ALLOC,
                        "a moment sweep over %u trellis states times %u "
                        "automaton states needs about %.2f %s, and the memory "
                        "budget is %.2f %s; raise it with "
                        "BEAM.set_memory_budget, ask at a lower order, or use "
                        "a simpler constraint",
                        (unsigned)S, (unsigned)n_aut, nv, nu, bv, bu);
    }

    for (int i = 0; i < 2; ++i) {
        k->la[i] = (double *)malloc(P * sizeof(double));
        k->mu[i] = (double *)calloc(P, sizeof(double));
        k->v[i]  = (double *)calloc(P, sizeof(double));
        if (!k->la[i] || !k->mu[i] || !k->v[i]) {
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
    const uint32_t Rv = m->n_residues;

    /*
     * Every table entry must be finite, and the check is not pedantry.
     * `np.log(p)` is the obvious way to build a functional and gives `-inf`
     * wherever `p` is zero; the sweep then forms `m_k - M` between two
     * infinities, and the NaN that follows reached the end and was reported as
     * a VARIANCE OF ZERO, because `NaN > 0` is false. Found by review.
     *
     * Refused rather than propagated. A functional taking a non-finite value
     * on a live path has an expectation of plus or minus infinity and no
     * variance at all, and this engine's promise is an exact number or a
     * refusal that says why -- not a number that happens to be zero.
     */
    if (f->kind == BEAM_FUNCTIONAL_TABLE) {
        for (uint32_t i = 0; i < c->length; ++i)
            for (uint32_t x = 0; x < Rv; ++x)
                if (!isfinite(f->table[(size_t)i * Rv + x]))
                    return BEAM_ERANGE(
                        "beam_query_expect: the functional is %g at position "
                        "%u residue %u, and a non-finite value has no variance "
                        "and an infinite expectation; a probability of zero "
                        "wants a weight of zero, not a logarithm of one",
                        f->table[(size_t)i * Rv + x], (unsigned)i, (unsigned)x);
    }
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
    for (size_t p = 0; p < n_prod; ++p) k.la[cur][p] = -INFINITY;
    k.la[cur][(size_t)(S - 1u) * Q] = 0.0;   /* ln 1: the start state */

    for (uint32_t i = 0; i < length; ++i) {
        const BeamResidueMask allowed = mask ? mask[i] : BEAM_RESIDUES_ALL(R);
        beam_layer_weights(m, i, length, k.w, NULL);

        const int nxt = cur ^ 1;
        double *nl = k.la[nxt], *nm = k.mu[nxt], *nv = k.v[nxt];
        for (size_t p = 0; p < n_prod; ++p) {
            nl[p] = -INFINITY; nm[p] = 0.0; nv[p] = 0.0;
        }

        /*
         * Accumulated by TARGET, as every sweep here is, so no two threads
         * write one accumulator -- and here that matters twice over, because
         * a merge reads the running mean it is about to update.
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
                /* LOG_WEIGHT's increment is the edge's own weight. */
                const double inc = (f->kind == BEAM_FUNCTIONAL_LOG_WEIGHT)
                                 ? wv : g;
                for (uint32_t q = 0; q < Q; ++q) {
                    const double pl = k.la[cur][(size_t)src * Q + q];
                    if (pl == -INFINITY) continue;
                    const uint32_t qn = aut
                        ? beam_query_automaton_step(aut, i, q, x) : 0u;
                    const size_t d = (size_t)t * Q + qn;
                    merge(&nl[d], &nm[d], &nv[d],
                          pl + wv,
                          k.mu[cur][(size_t)src * Q + q] + inc,
                          k.v[cur][(size_t)src * Q + q]);
                }
            }
        }
        cur = nxt;
    }

    /*
     * The sink settles the last residues' backward terms, so it is a weight on
     * every path and -- for LOG_WEIGHT -- the last increment of the functional
     * too. Acceptance enters here and only here.
     */
    beam_tail_weights(m, length, k.tail);

    double logP = -INFINITY, M = 0.0, V = 0.0;
    for (uint32_t s = 0; s < S; ++s) {
        if (k.tail[s] == -INFINITY) continue;
        const double inc = (f->kind == BEAM_FUNCTIONAL_LOG_WEIGHT)
                         ? k.tail[s] : 0.0;
        for (uint32_t q = 0; q < Q; ++q) {
            if (aut && !aut->accept[q]) continue;
            const size_t p = (size_t)s * Q + q;
            if (k.la[cur][p] == -INFINITY) continue;
            merge(&logP, &M, &V, k.la[cur][p] + k.tail[s],
                  k.mu[cur][p] + inc, k.v[cur][p]);
        }
    }
    mw_free(&k);

    if (logP == -INFINITY)
        return BEAM_ENOPATHS("the constraint describes a configuration this "
                            "model cannot produce at this length, so its "
                            "moments do not exist");

    out->log_z    = logP;
    out->mean     = M;
    /*
     * A mixture of non-negative variances and a squared deviation, so never
     * negative; clamped anyway, because a caller taking a square root of a
     * -1e-19 deserves better than a NaN.
     *
     * A NaN is passed THROUGH rather than clamped. `NaN > 0.0` is false, so
     * the bare clamp turned one into a confident zero -- the defect above, and
     * the shape of one an earlier review found on the overflow path. No
     * arithmetic here can produce a NaN now, which is exactly why this must
     * stay: if one ever appears it has to be visible.
     */
    out->variance = isnan(V) ? V : (V > 0.0 ? V : 0.0);
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
