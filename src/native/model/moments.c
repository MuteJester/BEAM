/**
 * @file moments.c
 * @brief Mellin transform, surprisal moments, and the surprisal range.
 *
 * Everything here is one idea applied three ways: run the forward sweep over
 * the DAG in a richer semiring than plain addition.
 *
 *   - carrying mass alone gives the tilted partition function, hence the
 *     Mellin transform and every Hill number;
 *   - carrying mass together with the first `n` powers of the accumulated
 *     path weight (a "jet") gives the moments of the surprisal, hence its
 *     cumulants and the whole moment-generating apparatus;
 *   - carrying a running maximum or minimum instead of a sum gives the exact
 *     extremes of the surprisal, hence the dynamic range.
 *
 * All exact. Nothing here samples.
 */
#include "beam/internal.h"

/* ── Shared scratch for a forward sweep ──────────────────────── */

typedef struct {
    double *w;      /* [n_states * 20] edge weights of the current layer */
    double *tail;   /* [n_states]      sink weights                      */
    double *cur;    /* [(order+1) * n_states] jet, moment-major          */
    double *next;
} Sweep;

static void sweep_free(Sweep *s) {
    free(s->w); free(s->tail); free(s->cur); free(s->next);
}

static bool sweep_alloc(Sweep *s, uint32_t n_states, uint32_t n_moments) {
    const size_t jet = (size_t)n_moments * n_states;
    s->w    = (double *)malloc((size_t)n_states * BEAM_RESIDUES * sizeof(double));
    s->tail = (double *)malloc((size_t)n_states * sizeof(double));
    s->cur  = (double *)calloc(jet, sizeof(double));
    s->next = (double *)calloc(jet, sizeof(double));
    if (!s->w || !s->tail || !s->cur || !s->next) { sweep_free(s); return false; }
    return true;
}

/** Binomial coefficients up to order 16, enough for any sane moment order. */
#define BEAM_MAX_MOMENT 16u

static void binomials(double c[][BEAM_MAX_MOMENT + 1], uint32_t n) {
    for (uint32_t i = 0; i <= n; ++i) {
        c[i][0] = 1.0;
        for (uint32_t j = 1; j <= i; ++j)
            c[i][j] = (j == i) ? 1.0 : c[i - 1][j - 1] + c[i - 1][j];
        for (uint32_t j = i + 1; j <= n; ++j) c[i][j] = 0.0;
    }
}

/**
 * Forward jet sweep over one length.
 *
 * On return `out_moments[j]` is `sum over length-L paths of P(s)^q * raw(s)^j`
 * divided by the same sum at `j = 0`, i.e. `E_q[raw^j | L]`, and `*out_log_z`
 * is `ln sum over paths of exp(q * raw)`.
 *
 * `raw` is the accumulated edge weight, so `ln P(s) = c_L + raw(s)` with
 * `c_L = ln pi(L) - ln Z(L)`; the caller shifts by that constant.
 *
 * Numerics: the whole jet is rescaled by the same factor each layer, so the
 * moment ratios are untouched while the mass stays in range. That is why the
 * scale never has to be tracked for the moments, only for the partition
 * function.
 */
static BeamError jet_sweep(const BeamModel *m, uint32_t len, double q,
                          uint32_t order, double *out_moments,
                          double *out_log_z) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t NM = order + 1u;

    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    binomials(bc, order);

    Sweep sw = {0};
    if (!sweep_alloc(&sw, S, NM))
        return BEAM_EALLOC("the jet-sweep scratch buffers");

    sw.cur[(size_t)0 * S + (S - 1)] = 1.0;    /* moment 0 of the start state */
    double log_scale = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_layer_weights(m, i, len, sw.w, NULL);

        /*
         * Rescale by the best edge THAT CARRIES MASS, not the best edge
         * anywhere in the layer. At a strong tilt the two can be far apart,
         * and scaling by an unreachable edge underflows every reachable one to
         * zero, which looks exactly like a length with no paths. That is what
         * made the saddlepoint fail near the ends of the support.
         */
        double wmax = -INFINITY;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:wmax) \
                num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) {
            if (sw.cur[s] == 0.0) continue;
            const double *row = sw.w + (size_t)s * R;
            for (uint32_t x = 0; x < R; ++x)
                if (row[x] > wmax) wmax = row[x];
        }
        if (wmax == -INFINITY) { sweep_free(&sw); *out_log_z = -INFINITY; return BEAM_OK; }
        const double qmax = q * wmax;

        memset(sw.next, 0, (size_t)NM * S * sizeof(double));
        double nmax = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < (int64_t)S; ++t) {
            const uint32_t x = (uint32_t)t % A;
            if (x >= R) continue;
            const uint32_t rest = (uint32_t)t / A;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                if (sw.cur[src] == 0.0) continue;          /* no mass here */
                const double wv = sw.w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                const double e = exp(q * wv - qmax);
                /* (raw + w)^j expands binomially over the carried moments. */
                for (uint32_t j = 0; j < NM; ++j) {
                    double acc = 0.0;
                    double wpow = 1.0;
                    for (uint32_t u = 0; u <= j; ++u) {
                        acc += bc[j][j - u] * wpow
                             * sw.cur[(size_t)(j - u) * S + src];
                        wpow *= wv;
                    }
                    sw.next[(size_t)j * S + t] += e * acc;
                }
            }
            if (sw.next[t] > nmax) nmax = sw.next[t];
        }
        log_scale += qmax;

        if (!(nmax > 0.0)) { sweep_free(&sw); *out_log_z = -INFINITY; return BEAM_OK; }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < (int64_t)NM * S; ++t) sw.next[t] /= nmax;
        log_scale += log(nmax);

        double *tmp = sw.cur; sw.cur = sw.next; sw.next = tmp;
    }

    /* Sink: one more edge whose weight is the tail. */
    beam_tail_weights(m, len, sw.tail);
    double tmax = -INFINITY;
    for (uint32_t s = 0; s < S; ++s)
        if (sw.cur[s] > 0.0 && sw.tail[s] != -INFINITY && q * sw.tail[s] > tmax)
            tmax = q * sw.tail[s];
    if (tmax == -INFINITY) { sweep_free(&sw); *out_log_z = -INFINITY; return BEAM_OK; }

    double *fin = (double *)calloc(NM, sizeof(double));
    if (!fin) { sweep_free(&sw); return BEAM_EALLOC("the sink jet"); }
    for (uint32_t s = 0; s < S; ++s) {
        if (sw.cur[s] == 0.0 || sw.tail[s] == -INFINITY) continue;
        const double tv = sw.tail[s];
        const double e = exp(q * tv - tmax);
        for (uint32_t j = 0; j < NM; ++j) {
            double acc = 0.0, wpow = 1.0;
            for (uint32_t u = 0; u <= j; ++u) {
                acc += bc[j][j - u] * wpow * sw.cur[(size_t)(j - u) * S + s];
                wpow *= tv;
            }
            fin[j] += e * acc;
        }
    }
    if (!(fin[0] > 0.0)) {
        free(fin); sweep_free(&sw);
        *out_log_z = -INFINITY;
        return BEAM_OK;
    }
    for (uint32_t j = 0; j < NM; ++j) out_moments[j] = fin[j] / fin[0];
    *out_log_z = log_scale + tmax + log(fin[0]);

    free(fin);
    sweep_free(&sw);
    return BEAM_OK;
}

/* ── Mellin transform ────────────────────────────────────────── */

/**
 * `ln sum_s P(s)^q`, assembled from the per-length tilted partition functions.
 *
 * `sum over length-L sequences of P^q = exp(q * c_L) * Z_q(L)` with
 * `c_L = ln pi(L) - ln Z(L)`, so the lengths combine as a log-sum-exp. Doing
 * that in the log domain rather than by summing exponentials is what keeps a
 * long tail of rare lengths from underflowing out of the total.
 */
/**
 * Live lengths, longest first: the per-length sweeps are the unit of parallel
 * work and cost grows with the length, so the longest must be handed out
 * first or fifteen cores wait on it at the end.
 */
static uint32_t live_lengths_desc(const BeamModel *m, uint32_t *out) {
    uint32_t n = 0;
    for (uint32_t L = m->n_len_slots; L-- > 1; )
        if (beam_length_live(m, L)) out[n++] = L;
    return n;
}

static BeamError mellin_terms(const BeamModel *m, double q, double *terms,
                             double *out_max) {
    const uint32_t n = m->n_len_slots;
    for (uint32_t L = 0; L < n; ++L) terms[L] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *lzs = (double *)malloc((size_t)n * sizeof(double));
    if (!work || !lzs) {
        free(work); free(lzs);
        return BEAM_EALLOC("the per-length Mellin work list");
    }
    const uint32_t nw = live_lengths_desc(m, work);
    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double mom[1], lz = 0.0;
        const BeamError e = jet_sweep(m, work[k], q, 0u, mom, &lz);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
        else lzs[work[k]] = lz;
    }
    free(work);
    if (failed != BEAM_OK) {
        free(lzs);
        return BEAM_FAIL((BeamError)failed,
                        "a per-length tilted sweep could not be computed");
    }
    /* Scanned in ascending length, so the maximum and the terms are exactly
     * what the serial loop produced. */
    double mx = -INFINITY;
    for (uint32_t L = 1; L < n; ++L) {
        if (!beam_length_live(m, L) || lzs[L] == -INFINITY) continue;
        terms[L] = q * (m->log_pi[L] - m->log_z[L]) + lzs[L];
        if (terms[L] > mx) mx = terms[L];
    }
    free(lzs);
    *out_max = mx;
    return BEAM_OK;
}

BeamError beam_model_log_mellin(const BeamModel *m, double q, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    if (!terms) return BEAM_EALLOC("the per-length Mellin terms");
    double mx = 0.0;
    BeamError e = mellin_terms(m, q, terms, &mx);
    if (e != BEAM_OK) { free(terms); return e; }
    if (mx == -INFINITY) { free(terms); *out = -INFINITY; return BEAM_OK; }

    double acc = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        if (terms[L] != -INFINITY) acc += exp(terms[L] - mx);
    free(terms);
    *out = mx + log(acc);
    return BEAM_OK;
}

/* ── Surprisal moments ───────────────────────────────────────── */

BeamError beam_model_surprisal_moments(const BeamModel *m, double q,
                                          uint32_t order, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (order > BEAM_MAX_MOMENT)
        return BEAM_ERANGE("%s: moment order %u exceeds the supported "
                               "maximum of %u", __func__, (unsigned)order,
                               (unsigned)BEAM_MAX_MOMENT);

    const uint32_t NM = order + 1u;
    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    binomials(bc, order);

    double *mom   = (double *)calloc(NM, sizeof(double));
    double *acc   = (double *)calloc(NM, sizeof(double));
    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    if (!mom || !acc || !terms) {
        free(mom); free(acc); free(terms);
        return BEAM_EALLOC("the per-length moment accumulators");
    }

    /* Length weights under the q-tilted distribution. */
    double mx = 0.0;
    BeamError e = mellin_terms(m, q, terms, &mx);
    if (e != BEAM_OK || mx == -INFINITY) {
        free(mom); free(acc); free(terms);
        return e != BEAM_OK ? e
             : BEAM_ENOPATHS("no length carries tilted mass at this q");
    }
    double wsum = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        if (terms[L] != -INFINITY) wsum += exp(terms[L] - mx);

    /* One jet per length, all of them independent. They are computed up front
     * so the mixing below still runs in ascending length and the raw moments
     * accumulate in exactly the serial order. */
    double *jets = (double *)calloc((size_t)m->n_len_slots * NM, sizeof(double));
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!jets || !work) {
        free(jets); free(work); free(mom); free(acc); free(terms);
        return BEAM_EALLOC("the per-length jets");
    }
    uint32_t nw = 0;
    for (uint32_t L = m->n_len_slots; L-- > 1; )
        if (terms[L] != -INFINITY) work[nw++] = L;
    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double lz = 0.0;
        const BeamError je = jet_sweep(m, work[k], q, order,
                                      jets + (size_t)work[k] * NM, &lz);
        if (je != BEAM_OK) { if ((int)je > failed) failed = (int)je; }
    }
    if (failed != BEAM_OK) {
        free(jets); free(work); free(mom); free(acc); free(terms);
        return BEAM_FAIL((BeamError)failed,
                        "surprisal_moments: a per-length jet sweep failed");
    }
    free(work);

    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (terms[L] == -INFINITY) continue;
        const double weight = exp(terms[L] - mx) / wsum;
        memcpy(mom, jets + (size_t)L * NM, NM * sizeof(double));

        /*
         * The jet gives moments of `raw`; the surprisal is
         * `u = -(c_L + raw)`, so shift and negate:
         *   E[u^k] = sum_j C(k,j) * (-c_L)^(k-j) * (-1)^j * E[raw^j].
         *
         * A mixture's RAW moments are the weighted average of its components'
         * raw moments, which is why this is summed here and converted to
         * cumulants only at the very end. Averaging cumulants across lengths
         * instead would be quietly wrong.
         */
        const double cL = m->log_pi[L] - m->log_z[L];
        double pow_negc[BEAM_MAX_MOMENT + 1];
        pow_negc[0] = 1.0;
        for (uint32_t t = 1; t <= order; ++t) pow_negc[t] = pow_negc[t - 1] * (-cL);

        for (uint32_t k = 0; k < NM; ++k) {
            double v = 0.0;
            for (uint32_t j = 0; j <= k; ++j) {
                double term = bc[k][j] * mom[j] * pow_negc[k - j];
                v += (j & 1u) ? -term : term;
            }
            acc[k] += weight * v;
        }
    }

    for (uint32_t k = 0; k < NM; ++k) out[k] = acc[k];
    out[0] = 1.0;                                   /* exact by construction */
    free(jets); free(mom); free(acc); free(terms);
    return BEAM_OK;
}

/* ── Per-length moments, unmixed ─────────────────────────────── */

BeamError beam_model_length_derivatives(const BeamModel *m, double q,
                                           uint32_t order, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (order > BEAM_MAX_MOMENT)
        return BEAM_ERANGE("%s: moment order %u exceeds the supported "
                               "maximum of %u", __func__, (unsigned)order,
                               (unsigned)BEAM_MAX_MOMENT);

    const uint32_t n  = m->n_len_slots;
    const uint32_t NM = order + 1u;
    memset(out, 0, (size_t)n * NM * sizeof(double));

    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    binomials(bc, order);

    double *terms = (double *)malloc((size_t)n * sizeof(double));
    double *jets  = (double *)calloc((size_t)n * NM, sizeof(double));
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!terms || !jets || !work) {
        free(terms); free(jets); free(work);
        return BEAM_EALLOC("the per-length moment buffers");
    }

    double mx = 0.0;
    BeamError e = mellin_terms(m, q, terms, &mx);
    if (e != BEAM_OK || mx == -INFINITY) {
        free(terms); free(jets); free(work);
        return e != BEAM_OK ? e
             : BEAM_ENOPATHS("no length carries tilted mass at this q");
    }
    double wsum = 0.0;
    for (uint32_t L = 0; L < n; ++L)
        if (terms[L] != -INFINITY) wsum += exp(terms[L] - mx);

    uint32_t nw = 0;
    for (uint32_t L = n; L-- > 1; ) if (terms[L] != -INFINITY) work[nw++] = L;

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double lz = 0.0;
        const BeamError je = jet_sweep(m, work[k], q, order,
                                      jets + (size_t)work[k] * NM, &lz);
        if (je != BEAM_OK) { if ((int)je > failed) failed = (int)je; }
    }
    if (failed != BEAM_OK) {
        free(terms); free(jets); free(work);
        return BEAM_FAIL((BeamError)failed,
                        "length_derivatives: a per-length jet sweep failed");
    }
    free(work);

    for (uint32_t L = 1; L < n; ++L) {
        if (terms[L] == -INFINITY) continue;
        const double *mom = jets + (size_t)L * NM;
        double *row = out + (size_t)L * NM;

        /* Slot 0 is the length's own share of the tilted law, not the
         * constant 1 the jet carries there: with the share in hand the
         * mixture is `sum_L row[0] * row[k]`, which is exactly what
         * `beam_model_surprisal_moments` returns, and the caller can
         * check that rather than take it on trust. */
        row[0] = exp(terms[L] - mx) / wsum;

        /* The jet gives moments of `raw`; the surprisal is `u = -(c_L + raw)`,
         * so shift and negate, exactly as the pooled form does before mixing. */
        const double cL = m->log_pi[L] - m->log_z[L];
        double pow_negc[BEAM_MAX_MOMENT + 1];
        pow_negc[0] = 1.0;
        for (uint32_t t = 1; t <= order; ++t) pow_negc[t] = pow_negc[t - 1] * (-cL);
        for (uint32_t k = 1; k < NM; ++k) {
            double v = 0.0;
            for (uint32_t j = 0; j <= k; ++j) {
                const double term = bc[k][j] * mom[j] * pow_negc[k - j];
                v += (j & 1u) ? -term : term;
            }
            row[k] = v;
        }
    }
    free(terms); free(jets);
    return BEAM_OK;
}

BeamError beam_model_surprisal_cumulants(const BeamModel *m, double q,
                                            uint32_t order, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (order < 1 || order > BEAM_MAX_MOMENT)
        return BEAM_ERANGE("%s: cumulant order %u is outside 1..%u",
                               __func__, (unsigned)order,
                               (unsigned)BEAM_MAX_MOMENT);

    double *mu = (double *)calloc(order + 1u, sizeof(double));
    if (!mu) return BEAM_EALLOC("the raw-moment buffer");
    BeamError e = beam_model_surprisal_moments(m, q, order, mu);
    if (e != BEAM_OK) { free(mu); return e; }

    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    binomials(bc, order);

    /* k_n = mu_n - sum_{i=1}^{n-1} C(n-1, i-1) k_i mu_{n-i} */
    for (uint32_t n = 1; n <= order; ++n) {
        double v = mu[n];
        for (uint32_t i = 1; i < n; ++i)
            v -= bc[n - 1][i - 1] * out[i - 1] * mu[n - i];
        out[n - 1] = v;
    }
    free(mu);
    return BEAM_OK;
}

/* ── One-sweep tilt summary, for saddlepoint stepping ────────── */

BeamError beam_model_surprisal_tilt(const BeamModel *m, double q,
                                       double *out_log_mellin,
                                       double *out_mean, double *out_var) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();

    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    double *m1    = (double *)calloc(m->n_len_slots, sizeof(double));
    double *m2    = (double *)calloc(m->n_len_slots, sizeof(double));
    if (!terms || !m1 || !m2) {
        free(terms); free(m1); free(m2);
        return BEAM_EALLOC("the per-length tilt summaries");
    }

    /* One jet sweep per length gives that length's tilted mass and its first
     * two moments of the surprisal at the same time. */
    double mx = -INFINITY;
    for (uint32_t L = 0; L < m->n_len_slots; ++L) terms[L] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    double *jets = (double *)calloc((size_t)m->n_len_slots * 4u, sizeof(double));
    if (!work || !jets) {
        free(work); free(jets); free(terms); free(m1); free(m2);
        return BEAM_EALLOC("the per-length tilt jets");
    }
    const uint32_t nw = live_lengths_desc(m, work);
    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double mom[3] = {0}, lz = 0.0;
        const BeamError e = jet_sweep(m, work[k], q, 2u, mom, &lz);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; continue; }
        double *j = jets + (size_t)work[k] * 4u;
        j[0] = lz; j[1] = mom[1]; j[2] = mom[2];
    }
    free(work);
    if (failed != BEAM_OK) {
        free(jets); free(terms); free(m1); free(m2);
        return BEAM_FAIL((BeamError)failed,
                        "surprisal_tilt: a per-length jet sweep failed");
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        const double *j = jets + (size_t)L * 4u;
        if (j[0] == -INFINITY) continue;
        terms[L] = q * (m->log_pi[L] - m->log_z[L]) + j[0];
        if (terms[L] > mx) mx = terms[L];

        /* u = -(c_L + raw), so E[u] = -(c_L + E[raw]) and
         * E[u^2] = c_L^2 + 2 c_L E[raw] + E[raw^2]. */
        const double cL = m->log_pi[L] - m->log_z[L];
        m1[L] = -(cL + j[1]);
        m2[L] = cL * cL + 2.0 * cL * j[1] + j[2];
    }
    free(jets);
    if (mx == -INFINITY) {
        free(terms); free(m1); free(m2);
        return BEAM_ENOPATHS("no length carries tilted mass at this q");
    }

    double wsum = 0.0, e1 = 0.0, e2 = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L) {
        if (terms[L] == -INFINITY) continue;
        const double w = exp(terms[L] - mx);
        wsum += w;
        e1 += w * m1[L];
        e2 += w * m2[L];
    }
    e1 /= wsum; e2 /= wsum;

    if (out_log_mellin) *out_log_mellin = mx + log(wsum);
    if (out_mean)       *out_mean = e1;
    if (out_var)        *out_var  = e2 - e1 * e1;

    free(terms); free(m1); free(m2);
    return BEAM_OK;
}

/* ── Surprisal by depth ──────────────────────────────────────── */
/**
 * One length's depth profile: `out[i]` receives the expected weight settled at
 * layer `i` under the length-`L` q-tilted conditional, UNWEIGHTED by that
 * length's share of the mixture.
 *
 * Split out of the pooled entry point below so the lengths can run
 * concurrently: each writes its own row, and the rows are mixed afterwards in
 * ascending length, so the pooled answer is the one the serial loop produced.
 */
static BeamError by_depth_one(const BeamModel *m, uint32_t L, double q,
                             double *out) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    for (uint32_t d = 0; d < L; ++d) out[d] = 0.0;

    /* Backward values of the q-tilted trellis. */
    double **beta = (double **)calloc(L + 1u, sizeof(double *));
    if (!beta) return BEAM_EALLOC("the backward-value layer table");
    bool ok = true;
    for (uint32_t i = 0; i <= L && ok; ++i) {
        beta[i] = (double *)malloc((size_t)S * sizeof(double));
        ok = beta[i] != NULL;
    }
    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)calloc(S, sizeof(double));
    double *next = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    double *chunk = (double *)malloc(BEAM_SUM_CHUNKS * sizeof(double));
    BeamError err = BEAM_OK;
    if (!ok || !w || !cur || !next || !tail || !chunk) {
        err = BEAM_EALLOC("the tilted forward-backward buffers");
        goto done;
    }
    err = beam_betas_tilted(m, L, q, beta);
    if (err != BEAM_OK) goto done;

    {
    const double logz = beta[0][S - 1];
    cur[S - 1] = 1.0;
    double log_scale = 0.0;
    for (uint32_t i = 0; i < L; ++i) {
        double wmax = -INFINITY;
        beam_layer_weights(m, i, L, w, &wmax);
        if (wmax == -INFINITY) break;
        const double qmax = q * wmax;

        /* Posterior weight of each edge, and the surprisal it settles.
         * Summed by source state in ascending order, so the parallel form
         * needs per-thread partials combined in thread order to reproduce it
         * exactly; a bare `reduction(+:)` would not. */
        const double *b_next = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads())
#endif
        for (int64_t c = 0; c < (int64_t)BEAM_SUM_CHUNKS; ++c) {
            const uint32_t lo = (uint32_t)(((uint64_t)S * c) / BEAM_SUM_CHUNKS);
            const uint32_t hi = (uint32_t)(((uint64_t)S * (c + 1)) / BEAM_SUM_CHUNKS);
            double part = 0.0;
            for (uint32_t s = lo; s < hi; ++s) {
                if (cur[s] == 0.0) continue;
                const double *row = w + (size_t)s * R;
                const uint32_t base = (s % SK) * A;
                for (uint32_t x = 0; x < R; ++x) {
                    if (row[x] == -INFINITY) continue;
                    const double nb = b_next[base + x];
                    if (nb == -INFINITY) continue;
                    const double lp = log(cur[s]) + log_scale + q * row[x]
                                    + nb - logz;
                    part += exp(lp) * (-row[x]);
                }
            }
            chunk[c] = part;
        }
        double acc = 0.0;
        for (uint32_t c = 0; c < BEAM_SUM_CHUNKS; ++c) acc += chunk[c];
        /* The edge emitted at layer i settles the forward score of
         * residue i and the backward score of residue i - state_order,
         * so its weight belongs to the sequence at large rather than to
         * one position. Attributing it to the layer is the honest
         * granularity: it is where the cost is incurred. */
        out[i] += acc;

        double nmax = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < (int64_t)S; ++t) {
            const uint32_t x = (uint32_t)t % A;
            if (x >= R) { next[t] = 0.0; continue; }
            const uint32_t rest = (uint32_t)t / A;
            double sum = 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                if (cur[src] == 0.0) continue;
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                sum += cur[src] * exp(q * wv - qmax);
            }
            next[t] = sum;
            if (sum > nmax) nmax = sum;
        }
        log_scale += qmax;
        if (!(nmax > 0.0)) break;
        for (uint32_t s = 0; s < S; ++s) next[s] /= nmax;
        log_scale += log(nmax);
        double *tmp = cur; cur = next; next = tmp;
    }

    /*
     * The sink settles the backward scores of the final `state_order`
     * residues, which never get a full right-hand context during the
     * walk. Leaving it out would make the decomposition quietly fail to
     * add up, which is exactly the kind of error a per-part breakdown is
     * supposed to make visible.
     */
    beam_tail_weights(m, L, tail);
    {
        double acc = 0.0;
        for (uint32_t s = 0; s < S; ++s) {
            if (cur[s] == 0.0 || tail[s] == -INFINITY) continue;
            const double lp = log(cur[s]) + log_scale + q * tail[s] - logz;
            acc += exp(lp) * (-tail[s]);
        }
        const uint32_t last = (L >= 1) ? L - 1u : 0u;
        out[last] += acc;
    }
    }

done:
    for (uint32_t i = 0; i <= L; ++i) free(beta[i]);
    free(beta); free(w); free(cur); free(next); free(tail); free(chunk);
    return err;
}

/**
 * Expected per-depth surprisal contribution.
 *
 * The edge emitted at layer `i` carries the forward score of residue `i` and
 * the backward score of residue `i - state_order`, so a layer's weight is not
 * one residue's cost. What IS well defined, and what this reports, is the
 * expected weight settled at each depth of the sequence, obtained by running
 * the forward-backward posterior over edges and attributing each edge's weight
 * to the position being emitted.
 */
BeamError beam_model_surprisal_by_depth(const BeamModel *m, double q,
                                           double *out,
                                           double *out_length_constant) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    const uint32_t n = m->n_len_slots;
    const uint32_t W = m->max_len;
    for (uint32_t d = 0; d < W; ++d) out[d] = 0.0;
    if (out_length_constant) *out_length_constant = 0.0;

    double *terms = (double *)malloc((size_t)n * sizeof(double));
    if (!terms) return BEAM_EALLOC("the per-length Mellin terms");
    double mx = 0.0;
    BeamError err = mellin_terms(m, q, terms, &mx);
    if (err != BEAM_OK) { free(terms); return err; }
    if (mx == -INFINITY) {
        free(terms);
        return BEAM_ENOPATHS("no length carries tilted mass at this q");
    }
    double wsum = 0.0;
    for (uint32_t L = 0; L < n; ++L)
        if (terms[L] != -INFINITY) wsum += exp(terms[L] - mx);

    /* One row per length, so the lengths never touch each other's memory. */
    double *rows = (double *)calloc((size_t)n * W, sizeof(double));
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!rows || !work) {
        free(rows); free(work); free(terms);
        return BEAM_EALLOC("the per-length depth profiles");
    }
    uint32_t nw = 0;
    for (uint32_t L = n; L-- > 1; ) if (terms[L] != -INFINITY) work[nw++] = L;

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        const BeamError e = by_depth_one(m, work[k], q,
                                        rows + (size_t)work[k] * W);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
    }
    free(work);
    if (failed != BEAM_OK) {
        free(rows); free(terms);
        return BEAM_FAIL((BeamError)failed,
                        "surprisal_by_depth: a per-length sweep failed");
    }

    /* Mixed in ascending length, exactly as the serial loop did. */
    for (uint32_t L = 1; L < n; ++L) {
        if (terms[L] == -INFINITY) continue;
        const double lw = exp(terms[L] - mx) / wsum;
        const double *row = rows + (size_t)L * W;
        for (uint32_t d = 0; d < L && d < W; ++d) out[d] += lw * row[d];
        /* The per-length normalisation belongs to the choice of length, not
         * to any position, so it is reported separately rather than smeared
         * across depths. */
        if (out_length_constant)
            *out_length_constant += lw * (m->log_z[L] - m->log_pi[L]);
    }
    free(rows); free(terms);
    return BEAM_OK;
}

/* ── Surprisal range ─────────────────────────────────────────── */

/**
 * Min-plus and max-plus sweeps over one length, giving the exact extremes of
 * `-ln P` among producible sequences of that length.
 */
static BeamError range_at(const BeamModel *m, uint32_t len,
                         double *out_min, double *out_max) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *lo   = (double *)malloc((size_t)S * sizeof(double));
    double *hi   = (double *)malloc((size_t)S * sizeof(double));
    double *nlo  = (double *)malloc((size_t)S * sizeof(double));
    double *nhi  = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !lo || !hi || !nlo || !nhi || !tail) {
        free(w); free(lo); free(hi); free(nlo); free(nhi); free(tail);
        return BEAM_EALLOC("the min-plus / max-plus sweep buffers");
    }
    for (uint32_t s = 0; s < S; ++s) { lo[s] = INFINITY; hi[s] = -INFINITY; }
    lo[S - 1] = 0.0; hi[S - 1] = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_layer_weights(m, i, len, w, NULL);
        for (uint32_t s = 0; s < S; ++s) { nlo[s] = INFINITY; nhi[s] = -INFINITY; }
        for (uint32_t s = 0; s < S; ++s) {
            if (hi[s] == -INFINITY) continue;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY) continue;
                const double a = lo[s] - row[x];      /* surprisal accumulates */
                const double b = hi[s] - row[x];
                if (a < nlo[base + x]) nlo[base + x] = a;
                if (b > nhi[base + x]) nhi[base + x] = b;
            }
        }
        memcpy(lo, nlo, (size_t)S * sizeof(double));
        memcpy(hi, nhi, (size_t)S * sizeof(double));
    }

    beam_tail_weights(m, len, tail);
    const double off = m->log_z[len] - m->log_pi[len];
    double best = INFINITY, worst = -INFINITY;
    for (uint32_t s = 0; s < S; ++s) {
        if (tail[s] == -INFINITY || hi[s] == -INFINITY) continue;
        const double a = lo[s] - tail[s] + off;
        const double b = hi[s] - tail[s] + off;
        if (a < best) best = a;
        if (b > worst) worst = b;
    }
    free(w); free(lo); free(hi); free(nlo); free(nhi); free(tail);
    *out_min = best;
    *out_max = worst;
    return BEAM_OK;
}

BeamError beam_model_surprisal_range(const BeamModel *m,
                                        double *out_min, double *out_max) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!work) return BEAM_EALLOC("the per-length work list");
    const uint32_t nw = live_lengths_desc(m, work);
    double best = INFINITY, worst = -INFINITY;
    int failed = BEAM_OK;
    /* Minimum and maximum are order-independent, so reducing them across
     * threads gives the same answer the serial scan did, bit for bit. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(min:best) \
            reduction(max:worst) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double a = 0.0, b = 0.0;
        const BeamError e = range_at(m, work[k], &a, &b);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; continue; }
        if (a < best) best = a;
        if (b > worst) worst = b;
    }
    free(work);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                        "surprisal_range: a per-length sweep failed");
    if (best == INFINITY)
        return BEAM_ENOPATHS("no length has a producible sequence, so "
                                 "the surprisal has no range");
    if (out_min) *out_min = best;
    if (out_max) *out_max = worst;
    return BEAM_OK;
}

/* ── Per-length marginals ────────────────────────────────────── */

BeamError beam_model_length_marginals(const BeamModel *m,
                                         double *counting, double *generated,
                                         double *mean_surprisal,
                                         double *var_surprisal) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();

    for (uint32_t L = 0; L < m->n_len_slots; ++L) {
        if (counting)       counting[L] = 0.0;
        if (generated)      generated[L] = 0.0;
        if (mean_surprisal) mean_surprisal[L] = 0.0;
        if (var_surprisal)  var_surprisal[L] = 0.0;
    }
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!work) return BEAM_EALLOC("the per-length work list");
    const uint32_t nw = live_lengths_desc(m, work);
    int failed = BEAM_OK;
    /* Every length is an independent pair of sweeps writing its own slot, so
     * this needs no reduction at all. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        const uint32_t L = work[k];
        if (counting)  counting[L]  = beam_model_path_count(m, L);
        if (generated) generated[L] = exp(m->log_pi[L]);
        if (mean_surprisal || var_surprisal) {
            double mom[3] = {0}, lz = 0.0;
            BeamError e = jet_sweep(m, L, 1.0, 2u, mom, &lz);
            if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; continue; }
            /* u = -(c_L + raw), so the shift is the same for both moments. */
            const double cL = m->log_pi[L] - m->log_z[L];
            const double e1 = -(cL + mom[1]);
            const double e2 = cL * cL + 2.0 * cL * mom[1] + mom[2];
            if (mean_surprisal) mean_surprisal[L] = e1;
            if (var_surprisal) {
                const double v = e2 - e1 * e1;
                var_surprisal[L] = v > 0.0 ? v : 0.0;
            }
        }
    }
    free(work);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                        "length_marginals: a per-length sweep failed");
    return BEAM_OK;
}

/* ── Count addition ──────────────────────────────────────────── */

