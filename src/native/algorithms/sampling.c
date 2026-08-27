/**
 * @file sampling.c
 * @brief Exact generation from the BEAM DAG.
 *
 * Draws are exactly proportional to sequence probability, not approximately.
 * Walking the trellis with each step weighted by the backward value of the
 * state it leads to is the standard forward-filtering backward-sampling
 * argument: the product of the local choices telescopes into the path's own
 * probability.
 *
 * Reference: `.private/edge/src/edge.py::EDGE.sample`.
 */
#include "beam/internal.h"

static uint32_t draw_length(const BeamModel *m, BeamRng *rng) {
    double u = beam_rng_double(rng);
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        const double p = exp(m->log_pi[L]);
        if (u < p) return L;
        u -= p;
    }
    /* Round-off can leave a sliver: fall back to the longest live length. */
    for (uint32_t L = m->n_len_slots; L-- > 1; )
        if (beam_length_live(m, L)) return L;
    return 0;
}

/** Draw `n` sequences of one fixed length, sharing one backward pass. */
static BeamError sample_fixed(const BeamModel *m, uint32_t len, uint32_t n,
                             BeamRng *rng, char *out, uint32_t *out_lengths,
                             const uint32_t *slots) {
    const uint32_t A  = BEAM_ALPHABET;
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t stride = m->max_len + 1u;

    double **beta = (double **)calloc(len + 1u, sizeof(double *));
    if (!beta) return BEAM_EALLOC("the backward-value layer table");
    for (uint32_t i = 0; i <= len; ++i) {
        beta[i] = (double *)malloc((size_t)S * sizeof(double));
        if (!beta[i]) {
            for (uint32_t j = 0; j <= i; ++j) free(beta[j]);
            free(beta);
            return BEAM_EALLOC("one layer of backward values");
        }
    }
    BeamError err = beam_betas(m, len, beta);
    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    double *p = (double *)malloc((size_t)R * sizeof(double));
    uint32_t *state = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (err != BEAM_OK || !w || !p || !state) {
        if (err == BEAM_OK) err = BEAM_EALLOC("the sampler scratch buffers");
        goto done;
    }
    if (beta[0][S - 1] == -INFINITY) {
        err = BEAM_ENOPATHS("the trellis at this length has no path from "
                                "the start state, so nothing can be drawn");
        goto done;
    }

    for (uint32_t k = 0; k < n; ++k) state[k] = S - 1u;

    for (uint32_t i = 0; i < len; ++i) {
        beam_layer_weights(m, i, len, w, NULL);
        const double *nb = beta[i + 1];
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t s = state[k];
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;

            double mx = -INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                p[x] = (row[x] == -INFINITY || nb[base + x] == -INFINITY)
                     ? -INFINITY : row[x] + nb[base + x];
                if (p[x] > mx) mx = p[x];
            }
            /* beta[i][s] was finite when we stepped here, so some x survives. */
            double tot = 0.0;
            for (uint32_t x = 0; x < R; ++x) {
                p[x] = (p[x] == -INFINITY) ? 0.0 : exp(p[x] - mx);
                tot += p[x];
            }
            double u = beam_rng_double(rng) * tot;
            uint32_t pick = R - 1u;
            for (uint32_t x = 0; x < R; ++x) {
                u -= p[x];
                if (u <= 0.0) { pick = x; break; }
            }
            while (pick > 0 && p[pick] == 0.0) pick--;   /* never emit a dead x */

            out[(size_t)slots[k] * stride + i] = beam_idx_to_char[pick];
            state[k] = base + pick;
        }
    }
    for (uint32_t k = 0; k < n; ++k) {
        out[(size_t)slots[k] * stride + len] = '\0';
        if (out_lengths) out_lengths[slots[k]] = len;
    }

done:
    for (uint32_t i = 0; i <= len; ++i) free(beta[i]);
    free(beta); free(w); free(p); free(state);
    return err;
}

BeamError beam_model_sample(const BeamModel *m, uint32_t n,
                               uint32_t length, BeamRng *rng,
                               char *out, uint32_t *out_lengths) {
    if (!m || !rng || !out)
        return BEAM_ENULL(!m ? "model" : (!rng ? "rng" : "output buffer"));
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (n == 0) return BEAM_OK;

    /* Group the draws by length so each length pays for its backward pass
     * once, however many sequences land on it. */
    uint32_t *len_of = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    uint32_t *slots  = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!len_of || !slots) {
        free(len_of); free(slots);
        return BEAM_EALLOC("the per-draw length assignment");
    }

    if (length != 0) {
        if (!beam_length_live(m, length)) {
            free(len_of); free(slots);
            return BEAM_EBADLEN(m, length);
        }
        for (uint32_t k = 0; k < n; ++k) len_of[k] = length;
    } else {
        for (uint32_t k = 0; k < n; ++k) {
            len_of[k] = draw_length(m, rng);
            /* Unreachable after a successful finalize, which guarantees at
             * least one live length. Refusing anyway is what keeps a zero out
             * of the grouping loop below, where it would silently leave that
             * draw's output slot and its `out_lengths` entry unwritten. */
            if (len_of[k] == 0) {
                free(len_of); free(slots);
                return BEAM_ENOPATHS("the length prior has no live length "
                                         "to draw from");
            }
        }
    }

    /*
     * Group the draws by length, then run the lengths concurrently.
     *
     * Nearly all of a draw's cost is the backward pass its length needs, and
     * those passes are independent, so the length is the axis worth
     * parallelising. What is NOT shareable is the random stream: threads would
     * consume it in whatever order they happened to run, and the same seed
     * would stop giving the same sequences. Each length therefore gets its own
     * generator, seeded from the caller's in ascending length before any
     * thread starts. Reproducibility survives; the actual draws for a given
     * seed differ from the single-stream version, which no caller can rely on
     * anyway.
     */
    uint32_t *cnt   = (uint32_t *)calloc(m->n_len_slots, sizeof(uint32_t));
    uint32_t *start = (uint32_t *)calloc(m->n_len_slots, sizeof(uint32_t));
    BeamRng   *sub   = (BeamRng *)calloc(m->n_len_slots, sizeof(BeamRng));
    uint32_t *work  = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!cnt || !start || !sub || !work) {
        free(cnt); free(start); free(sub); free(work);
        free(len_of); free(slots);
        return BEAM_EALLOC("the per-length draw grouping");
    }
    for (uint32_t k = 0; k < n; ++k) cnt[len_of[k]]++;
    uint32_t at = 0, nw = 0;
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        start[L] = at; at += cnt[L];
        if (cnt[L]) beam_rng_seed(&sub[L], beam_rng_next(rng));
    }
    /* Slots stay grouped by length and ascending within a length, so a draw
     * still lands in the output row it was assigned. */
    {
        uint32_t *fill = (uint32_t *)calloc(m->n_len_slots, sizeof(uint32_t));
        if (!fill) {
            free(cnt); free(start); free(sub); free(work);
            free(len_of); free(slots);
            return BEAM_EALLOC("the per-length draw grouping");
        }
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t L = len_of[k];
            slots[start[L] + fill[L]++] = k;
        }
        free(fill);
    }
    /* Longest first: the backward pass costs one layer per residue. */
    for (uint32_t L = m->n_len_slots; L-- > 1; ) if (cnt[L]) work[nw++] = L;

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        const uint32_t L = work[k];
        const BeamError e = sample_fixed(m, L, cnt[L], &sub[L], out, out_lengths,
                                        slots + start[L]);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
    }
    free(cnt); free(start); free(sub); free(work);
    free(len_of); free(slots);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed, "sample: a per-length draw failed");
    return BEAM_OK;
}
