/**
 * @file diversity.c
 * @brief Entropy and Hill numbers.
 */
#include "beam/internal.h"


/* ── Entropy and Hill numbers ────────────────────────────────── */


/**
 * Tilted partition function `sum over length-L paths of exp(alpha * raw)`.
 *
 * `alpha = 0` counts paths, `alpha = 1` is the ordinary partition function,
 * and general `alpha` gives the power sums the Hill numbers are built from.
 */
/* Sizing for the sweeps below is `beam_tilted_sweep_bytes` in dp.h: `n_vec`
 * is 4 for `tilted_log_z` and 6 for `expected_raw`. It is shared with
 * `math/extrema.c`, which carried a byte-identical copy of it. */
static BeamError tilted_log_z(const BeamModel *m, uint32_t len, double alpha,
                             double *out) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    /* Sweep the states the corpus can reach, not the whole state space. On a
     * per-repertoire model that is a several-hundred-fold smaller iteration
     * space; on the foundation the two nearly coincide and this costs nothing.
     *
     * REACHED, not LIVE, and the distinction is load-bearing. `nmax` rescales
     * every layer and `log_scale` accumulates `log(nmax)`, so the maximum must
     * be taken over the same set as before or the rescaling moves and the last
     * bits with it. `reached` holds every forward-reachable state, so its
     * maximum equals the maximum over all states, the rest being 0.0. `live`
     * additionally demands reaching the sink, and a forward-reachable dead end
     * can hold the layer maximum: mathematically irrelevant, numerically not.
     *
     * Both index arrays are ascending, so every sum receives its terms in the
     * order it did before. That is what makes this bit-identical rather than
     * merely equal. */
    LiveHandle h;
    if (beam_live_acquire(m, len, &h) != BEAM_OK)
        return BEAM_EALLOC("the forward frontier");
    const BeamLive *L = h.L;

    BEAM_BUDGET_OR_REFUSE("a tilted forward sweep", len, beam_tilted_sweep_bytes(S, R, 4u));

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)calloc(S, sizeof(double));
    double *next = (double *)calloc(S, sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    uint8_t *alive = (uint8_t *)malloc(S);
    if (!w || !cur || !next || !tail || !alive) {
        free(w); free(cur); free(next); free(tail); free(alive);
        beam_live_release(&h);
        return BEAM_EALLOC("the tilted forward-sweep buffers");
    }
    cur[S - 1] = 1.0;
    double log_scale = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        double wmax = -INFINITY;
        /* The frontier shortcut is only sound where the layer maximum can be
         * had without building the whole layer, and `beam_layer_weights_live`
         * documents that a caller asking for `out_max` does not qualify: its
         * fused maximum scans only the rows it built. Taking the max over the
         * reached set alone changes the per-layer rescale and therefore the
         * last bits, which is exactly what happened here -- the open-world
         * Hill numbers moved at every order, because in the open world a
         * state can carry the layer maximum and still be unreachable.
         *
         * On a sparse table `beam_layer_max` walks the occupied rows and is
         * cheap, and its answer IS the maximum over all states: a missing row
         * reads -inf and cannot win a maximum. On a dense table it would have
         * to build the layer anyway, so there the original full build stays,
         * unchanged and trivially identical. Sparse is the closed-world
         * default and is what the per-repertoire work uses. */
        if (m->cfg.sparse) {
            wmax = beam_layer_max(m, i, len);
            beam_reached_mask(L, i, S, alive);
            beam_layer_weights_live(m, i, len, w, NULL, alive);
        } else {
            beam_layer_weights(m, i, len, w, &wmax);
        }
        if (wmax == -INFINITY) goto dead;
        const double amax = beam_tilted_layer_max(w, cur, S, R, alpha,
                                                  wmax);
        if (amax == -INFINITY) goto dead;

        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        double nmax = 0.0;
        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            const uint32_t t = tgt[k];
            const uint32_t x = t % A;
            if (x >= R) { next[t] = 0.0; continue; }
            const uint32_t rest = t / A;
            double acc = 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                const double v = cur[src];
                if (v == 0.0) continue;
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                acc += v * exp(alpha * wv - amax);
            }
            next[t] = acc;
            if (acc > nmax) nmax = acc;
        }
        log_scale += amax;
        if (!(nmax > 0.0)) goto dead;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k)
            next[tgt[k]] = beam_rescale(next[tgt[k]], nmax);
        log_scale += log(nmax);
        /* `next` becomes `cur`, and the old `cur` is reused as the next
         * layer's `next`. Only its reached entries will be written, so its
         * values from two layers back have to be cleared first. */
        {
            const uint32_t  np = L->n_reached[i];
            const uint32_t *pv = L->reached[i];
            for (uint32_t q = 0; q < np; ++q) cur[pv[q]] = 0.0;
        }
        double *tmp = cur; cur = next; next = tmp;
    }

    beam_tail_weights(m, len, tail);
    {
        const uint32_t  ns = L->n_reached[len];
        const uint32_t *sv = L->reached[len];
        double tmax = -INFINITY;
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            if (cur[s] > 0.0 && tail[s] != -INFINITY && alpha * tail[s] > tmax)
                tmax = alpha * tail[s];
        }
        if (tmax == -INFINITY) goto dead;
        double acc = 0.0;
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            if (cur[s] == 0.0 || tail[s] == -INFINITY) continue;
            acc += cur[s] * exp(alpha * tail[s] - tmax);
        }
        *out = log_scale + tmax + log(acc);
    }
    free(w); free(cur); free(next); free(tail); free(alive);
    beam_live_release(&h);
    return BEAM_OK;
dead:
    free(w); free(cur); free(next); free(tail); free(alive);
    beam_live_release(&h);
    *out = -INFINITY;
    return BEAM_OK;
}

/**
 * Expected accumulated edge weight under the length-`L` conditional, by the
 * expectation semiring: carry both the path mass and the mass-weighted sum of
 * accumulated weight, and divide at the sink. Exact, one forward sweep.
 */
static BeamError expected_raw(const BeamModel *m, uint32_t len, double *out) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    BEAM_BUDGET_OR_REFUSE("an expected-weight sweep", len,
                          beam_tilted_sweep_bytes(S, R, 6u));

    double *w  = (double *)malloc((size_t)S * R * sizeof(double));
    double *a  = (double *)calloc(S, sizeof(double));
    double *b  = (double *)calloc(S, sizeof(double));
    double *na = (double *)malloc((size_t)S * sizeof(double));
    double *nb = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    uint8_t *alive = (uint8_t *)malloc(S);
    /* Same frontier argument as tilted_log_z above: reached, not live, and
     * ascending order, so every sum sees its terms unchanged. */
    LiveHandle h;
    if (beam_live_acquire(m, len, &h) != BEAM_OK) {
        free(w); free(a); free(b); free(na); free(nb); free(tail); free(alive);
        return BEAM_EALLOC("the forward frontier");
    }
    const BeamLive *L = h.L;
    if (!w || !a || !b || !na || !nb || !tail || !alive) {
        free(w); free(a); free(b); free(na); free(nb); free(tail); free(alive);
        beam_live_release(&h);
        return BEAM_EALLOC("the expectation-semiring buffers");
    }
    memset(na, 0, (size_t)S * sizeof(double));
    memset(nb, 0, (size_t)S * sizeof(double));
    a[S - 1] = 1.0;

    for (uint32_t i = 0; i < len; ++i) {
        double wmax = -INFINITY;
        /* The frontier shortcut is only sound where the layer maximum can be
         * had without building the whole layer, and `beam_layer_weights_live`
         * documents that a caller asking for `out_max` does not qualify: its
         * fused maximum scans only the rows it built. Taking the max over the
         * reached set alone changes the per-layer rescale and therefore the
         * last bits, which is exactly what happened here -- the open-world
         * Hill numbers moved at every order, because in the open world a
         * state can carry the layer maximum and still be unreachable.
         *
         * On a sparse table `beam_layer_max` walks the occupied rows and is
         * cheap, and its answer IS the maximum over all states: a missing row
         * reads -inf and cannot win a maximum. On a dense table it would have
         * to build the layer anyway, so there the original full build stays,
         * unchanged and trivially identical. Sparse is the closed-world
         * default and is what the per-repertoire work uses. */
        if (m->cfg.sparse) {
            wmax = beam_layer_max(m, i, len);
            beam_reached_mask(L, i, S, alive);
            beam_layer_weights_live(m, i, len, w, NULL, alive);
        } else {
            beam_layer_weights(m, i, len, w, &wmax);
        }
        if (wmax == -INFINITY) { *out = 0.0; goto out; }

        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        double nmax = 0.0;
        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t q = 0; q < (int64_t)nt; ++q) {
            const uint32_t t = tgt[q];
            const uint32_t x = t % A;
            if (x >= R) { na[t] = 0.0; nb[t] = 0.0; continue; }
            const uint32_t rest = t / A;
            double sa = 0.0, sb = 0.0;
            for (uint32_t k = 0; k < A; ++k) {
                const uint32_t src = k * SK + rest;
                if (a[src] == 0.0 && b[src] == 0.0) continue;
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                const double e = exp(wv - wmax);
                sa += a[src] * e;
                sb += (b[src] + a[src] * wv) * e;
            }
            na[t] = sa; nb[t] = sb;
            if (sa > nmax) nmax = sa;
        }
        /* Rescale both accumulators by the same factor: the ratio is what we
         * want, so the scale cancels and never has to be tracked. */
        if (!(nmax > 0.0)) { *out = 0.0; goto out; }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t q = 0; q < (int64_t)nt; ++q) {
            const uint32_t s = tgt[q];
            na[s] = beam_rescale(na[s], nmax);   /* the mass */
            nb[s] /= nmax;                       /* may be negative */
        }
        /* Clear what becomes the next layer's scratch: only reached entries
         * are written, so values from two layers back would survive. */
        {
            const uint32_t  np = L->n_reached[i];
            const uint32_t *pv = L->reached[i];
            for (uint32_t q = 0; q < np; ++q) { a[pv[q]] = 0.0; b[pv[q]] = 0.0; }
        }
        { double *t; t = a; a = na; na = t; t = b; b = nb; nb = t; }
    }

    beam_tail_weights(m, len, tail);
    {
        const uint32_t  ns = L->n_reached[len];
        const uint32_t *sv = L->reached[len];
        double tmax = -INFINITY;
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            if (a[s] > 0.0 && tail[s] != -INFINITY && tail[s] > tmax) tmax = tail[s];
        }
        if (tmax == -INFINITY) { *out = 0.0; goto out; }
        double A_ = 0.0, B_ = 0.0;
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            if (tail[s] == -INFINITY || (a[s] == 0.0 && b[s] == 0.0)) continue;
            const double e = exp(tail[s] - tmax);
            A_ += a[s] * e;
            B_ += (b[s] + a[s] * tail[s]) * e;
        }
        *out = (A_ > 0.0) ? B_ / A_ : 0.0;
    }
out:
    free(w); free(a); free(b); free(na); free(nb); free(tail); free(alive);
    beam_live_release(&h);
    return BEAM_OK;
}

BeamError beam_model_entropy(const BeamModel *m, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    const uint32_t n = m->n_len_slots;
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *per = (double *)calloc(n, sizeof(double));
    if (!work || !per) {
        free(work); free(per);
        return BEAM_EALLOC("the per-length expected weights");
    }
    uint32_t nw = 0;
    for (uint32_t L = n; L-- > 1; )
        if (beam_length_live(m, L) && exp(m->log_pi[L]) > 0.0) work[nw++] = L;

    /* `BEAM_OK` is zero and every failure code is positive, so a max-reduction
     * both avoids a data race on the flag and picks the same code however the
     * iterations are distributed. */
    /* Six vectors per sweep, and the sweeps run concurrently, so they divide
     * the budget rather than multiplying it. The size does not depend on the
     * length here, only on the state space. */
    const uint32_t conc_e = beam_budget_concurrency(
        beam_tilted_sweep_bytes(m->n_states, m->n_residues, 6u), nw ? nw : 1u);
    if (nw && conc_e == 0u) {
        const uint32_t at = work[0];
        const uint64_t need = beam_tilted_sweep_bytes(m->n_states, m->n_residues, 6u);
        free(work); free(per);
        return beam_budget_refuse("an expected-weight sweep", at, need);
    }

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc_e)
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double e = 0.0;
        const BeamError err = expected_raw(m, work[k], &e);
        if (err != BEAM_OK) { if ((int)err > failed) failed = (int)err; }
        else per[work[k]] = e;
    }
    if (failed != BEAM_OK) {
        free(work); free(per);
        return BEAM_FAIL((BeamError)failed,
                        "entropy: a per-length sweep could not be computed");
    }
    /* Accumulated in ascending length, exactly as the serial loop did. */
    double h = 0.0;
    for (uint32_t L = 1; L < n; ++L) {
        if (!beam_length_live(m, L)) continue;
        const double pi = exp(m->log_pi[L]);
        if (!(pi > 0.0)) continue;
        h -= pi * (m->log_pi[L] - m->log_z[L] + per[L]);
    }
    free(work); free(per);
    *out = h;
    return BEAM_OK;
}

BeamError beam_model_hill_number(const BeamModel *m, double alpha,
                                    double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    if (fabs(alpha - 1.0) < 1e-12) {
        double h = 0.0;
        BeamError e = beam_model_entropy(m, &h);
        if (e != BEAM_OK) return e;
        *out = exp(h);
        return BEAM_OK;
    }
    if (alpha == 0.0) {
        *out = beam_model_path_count_total(m);
        return BEAM_OK;
    }

    /* sum_s P(s)^alpha = sum_L (pi(L)/Z(L))^alpha * Z_alpha(L), summed in the
     * log domain so a long tail of tiny lengths cannot underflow the total. */
    double mx = -INFINITY;
    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    if (!terms) return BEAM_EALLOC("the per-length Hill terms");
    for (uint32_t L = 0; L < m->n_len_slots; ++L) terms[L] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    double *za = (double *)malloc((size_t)m->n_len_slots * sizeof(double));
    if (!work || !za) {
        free(work); free(za); free(terms);
        return BEAM_EALLOC("the per-length tilted partition functions");
    }
    const uint32_t nw = beam_live_lengths_desc(m, work);
    const uint32_t conc_z = beam_budget_concurrency(
        beam_tilted_sweep_bytes(m->n_states, m->n_residues, 4u), nw ? nw : 1u);
    if (nw && conc_z == 0u) {
        const uint32_t at = work[0];
        const uint64_t need = beam_tilted_sweep_bytes(m->n_states, m->n_residues, 4u);
        free(terms); free(work); free(za);
        return beam_budget_refuse("a tilted forward sweep", at, need);
    }

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc_z)
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double v = 0.0;
        const BeamError e = tilted_log_z(m, work[k], alpha, &v);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
        else za[work[k]] = v;
    }
    if (failed != BEAM_OK) {
        free(work); free(za); free(terms);
        return BEAM_FAIL((BeamError)failed,
                        "hill_number: a per-length tilted sweep failed");
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        if (za[L] == -INFINITY) continue;
        terms[L] = alpha * (m->log_pi[L] - m->log_z[L]) + za[L];
        if (terms[L] > mx) mx = terms[L];
    }
    free(work); free(za);
    if (mx == -INFINITY) { free(terms); *out = 0.0; return BEAM_OK; }
    double acc = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        if (terms[L] != -INFINITY) acc += exp(terms[L] - mx);
    free(terms);

    const double log_sum = mx + log(acc);
    *out = exp(log_sum / (1.0 - alpha));
    return BEAM_OK;
}
