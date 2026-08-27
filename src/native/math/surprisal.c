/**
 * @file surprisal.c
 * @brief Surprisal attributed by depth.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"



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
    BeamError err = beam_mellin_terms(m, q, terms, &mx);
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
