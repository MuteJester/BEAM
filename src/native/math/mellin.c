/**
 * @file mellin.c
 * @brief The Mellin transform and its per-length terms.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"



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
 * The widest stretch of real orders, a whole number of vectors, whose sweep
 * fits the budget -- all of them wherever they fit. Each stretch is swept by
 * every thread, so there is nothing to balance and no reason to cut a line
 * of orders that fits.
 */
BeamLinePlan beam_real_plan(uint32_t S, uint32_t R, uint32_t A, uint32_t N,
                            uint32_t nth, uint64_t budget) {
    BeamLinePlan p = {0u, 0u};
    if (N == 0u) return p;
    if (nth == 0u) nth = 1u;
    const uint32_t V = BEAM_REAL_LANES;
    const uint32_t whole = (N + V - 1u) / V * V;
    if (beam_jet_real_bytes(S, R, A, whole, nth) <= budget) {
        p.width = N;
        p.n_stretches = 1u;
        return p;
    }
    uint32_t lo = 0u, hi = whole / V;             /* in vectors; hi does not fit */
    while (hi - lo > 1u) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (beam_jet_real_bytes(S, R, A, mid * V, nth) <= budget) lo = mid;
        else hi = mid;
    }
    const uint32_t fit = lo * V;
    if (fit == 0u) return p;                       /* nothing fits */
    const uint32_t nb = (N + fit - 1u) / fit;
    uint32_t w = (N + nb - 1u) / nb;
    w = (w + V - 1u) / V * V;                      /* within fit: a whole vector count */
    p.width = w;
    p.n_stretches = (N + w - 1u) / w;
    return p;
}

/**
 * `ln Z_q(L)` terms for the real orders qs[0 .. nq), combined per order into
 * `terms[j * n_len_slots + L] = q c_L + ln Z_q(L)` and its maximum over the
 * lengths, `out_max[j]`. One routine for one order and for many, so the two
 * agree to the bit: an order's value depends on q alone (see jet_sweep_real).
 *
 * One (length, stretch) at a time, each swept by every thread, in one
 * workspace; every length's live frontier is built first, in parallel.
 */
static BeamError real_terms(const BeamModel *m, const double *qs, uint32_t nq,
                            double *terms, double *out_max) {
    const uint32_t n = m->n_len_slots;
    for (size_t k = 0; k < (size_t)nq * n; ++k) terms[k] = -INFINITY;
    for (uint32_t j = 0; j < nq; ++j) out_max[j] = -INFINITY;
    if (nq == 0u) return BEAM_OK;

    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *lzs = (double *)malloc((size_t)nq * n * sizeof(double));
    if (!work || !lzs) {
        free(work); free(lzs);
        return BEAM_EALLOC("the per-length Mellin work list");
    }
    for (size_t k = 0; k < (size_t)nq * n; ++k) lzs[k] = -INFINITY;
    const uint32_t nw = beam_live_lengths_desc(m, work);
    const int th = beam_model_effective_threads();
    const uint32_t nth = th > 0 ? (uint32_t)th : 1u;
    const BeamLinePlan plan = beam_real_plan(m->n_states, m->n_residues,
                                             m->n_alphabet, nq, nth,
                                             beam_model_memory_budget());
    if (nw && plan.width == 0u) {
        const uint32_t at = work[0];
        free(work); free(lzs);
        return beam_budget_refuse("a tilted per-length sweep", at,
                                  beam_jet_real_bytes(m->n_states, m->n_residues,
                                                      m->n_alphabet,
                                                      BEAM_REAL_LANES, nth));
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads((int)nth)
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        LiveHandle h;
        if (beam_live_acquire(m, work[k], &h) == BEAM_OK) beam_live_release(&h);
    }
    int failed = BEAM_OK;
    const uint32_t width = plan.width, nb = plan.n_stretches;
    const uint64_t items = (uint64_t)nw * nb;
    BeamRealWork *wk = NULL;
    if (items) {
        const uint64_t need = beam_jet_real_bytes(m->n_states, m->n_residues,
                                                  m->n_alphabet, width, nth);
        if (!beam_budget_fits(need)) {
            const uint32_t at = work[0];
            free(work); free(lzs);
            return beam_budget_refuse("a tilted per-length sweep", at, need);
        }
        wk = beam_real_work_new(m, width, nth);
        if (!wk) {
            free(work); free(lzs);
            return BEAM_EALLOC("the real-order sweep workspace");
        }
    }
    for (uint64_t k = 0; k < items && failed == BEAM_OK; ++k) {
        const uint32_t L  = work[k / nb];
        const uint32_t q0 = (uint32_t)(k % nb) * width;
        const uint32_t wq = nq - q0 < width ? nq - q0 : width;
        const BeamError e = jet_sweep_real(m, wk, L, qs + q0, wq,
                                           lzs + (size_t)q0 * n + L, n);
        if (e != BEAM_OK) failed = (int)e;
    }
    beam_real_work_free(wk);
    free(work);
    if (failed != BEAM_OK) {
        free(lzs);
        return BEAM_FAIL((BeamError)failed,
                        "a per-length tilted sweep could not be computed");
    }
    /* Per order, scanned in ascending length. */
    for (uint32_t j = 0; j < nq; ++j) {
        const double q = qs[j];
        const double *lz = lzs + (size_t)j * n;
        double *tj = terms + (size_t)j * n;
        double mx = -INFINITY;
        for (uint32_t L = 1; L < n; ++L) {
            if (!beam_length_live(m, L) || lz[L] == -INFINITY) continue;
            tj[L] = q * (m->log_pi[L] - m->log_z[L]) + lz[L];
            if (tj[L] > mx) mx = tj[L];
        }
        out_max[j] = mx;
    }
    free(lzs);
    return BEAM_OK;
}

BeamError beam_mellin_terms(const BeamModel *m, double q, double *terms,
                             double *out_max) {
    return real_terms(m, &q, 1u, terms, out_max);
}

BeamError beam_model_log_mellin(const BeamModel *m, double q, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    if (!terms) return BEAM_EALLOC("the per-length Mellin terms");
    double mx = 0.0;
    BeamError e = beam_mellin_terms(m, q, terms, &mx);
    if (e != BEAM_OK) { free(terms); return e; }
    if (mx == -INFINITY) { free(terms); *out = -INFINITY; return BEAM_OK; }

    double acc = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        if (terms[L] != -INFINITY) acc += exp(terms[L] - mx);
    free(terms);
    *out = mx + log(acc);
    return BEAM_OK;
}


/* ── Many orders in one pass ─────────────────────────────────── */

/**
 * `beam_mellin_terms` at `nq` orders from one sweep of each length: the same
 * routine, so `terms[j * n_len_slots + L]` and `out_max[j]` are bit-identical
 * to what `beam_mellin_terms(m, qs[j], ...)` writes, whatever the budget, the
 * thread count, or the company `qs[j]` keeps.
 */
static BeamError mellin_terms_many(const BeamModel *m, const double *qs,
                                   uint32_t nq, double *terms,
                                   double *out_max) {
    return real_terms(m, qs, nq, terms, out_max);
}

BeamError beam_model_log_mellin_many(const BeamModel *m, const double *qs,
                                     size_t nq, double *out) {
    if (!m) return BEAM_ENULL("model");
    if (nq && (!qs || !out)) return BEAM_ENULL(!qs ? "orders" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (nq == 0u) return BEAM_OK;
    if (nq > UINT32_MAX)
        return BEAM_ERANGE("%s: %zu orders in one call; the limit is %u",
                           __func__, nq, (unsigned)UINT32_MAX);

    const uint32_t n = m->n_len_slots;
    double *terms = (double *)malloc(nq * n * sizeof(double));
    double *mx = (double *)malloc(nq * sizeof(double));
    if (!terms || !mx) {
        free(terms); free(mx);
        return BEAM_EALLOC("the per-length many-order Mellin terms");
    }
    const BeamError e = mellin_terms_many(m, qs, (uint32_t)nq, terms, mx);
    if (e != BEAM_OK) { free(terms); free(mx); return e; }

    /* `beam_model_log_mellin`'s tail, per order. */
    for (size_t j = 0; j < nq; ++j) {
        if (mx[j] == -INFINITY) { out[j] = -INFINITY; continue; }
        const double *tj = terms + j * n;
        double acc = 0.0;
        for (uint32_t L = 0; L < n; ++L)
            if (tj[L] != -INFINITY) acc += exp(tj[L] - mx[j]);
        out[j] = mx[j] + log(acc);
    }
    free(terms); free(mx);
    return BEAM_OK;
}


/* ── Complex orders ──────────────────────────────────────────── */

BeamError beam_model_log_mellin_complex(const BeamModel *m, const double *q_re,
                                        const double *q_im, size_t nq,
                                        double *out_re, double *out_im) {
    if (!m) return BEAM_ENULL("model");
    if (nq && (!q_re || !q_im || !out_re || !out_im))
        return BEAM_ENULL(!q_re ? "real parts" : !q_im ? "imaginary parts"
                                               : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (nq == 0u) return BEAM_OK;
    if (nq > UINT32_MAX)
        return BEAM_ERANGE("%s: %zu orders in one call; the limit is %u",
                           __func__, nq, (unsigned)UINT32_MAX);
    for (size_t j = 0; j < nq; ++j)
        if (!isfinite(q_re[j]) || !isfinite(q_im[j]))
            return BEAM_ERANGE("%s: order %zu is %g%+gi; orders must be finite",
                               __func__, j, q_re[j], q_im[j]);

    const uint32_t n = m->n_len_slots;
    const uint32_t N = (uint32_t)nq;
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *lmod = (double *)malloc((size_t)N * n * sizeof(double));
    double *lph  = (double *)malloc((size_t)N * n * sizeof(double));
    if (!work || !lmod || !lph) {
        free(work); free(lmod); free(lph);
        return BEAM_EALLOC("the per-length complex Mellin terms");
    }
    for (size_t k = 0; k < (size_t)N * n; ++k) { lmod[k] = -INFINITY; lph[k] = 0.0; }
    const uint32_t nw = beam_live_lengths_desc(m, work);

    /* Budgeted exactly as `mellin_terms_many`: the widest fusion that leaves
     * room for one sweep per thread, equal blocks when the orders do not fit
     * one sweep, and a refusal only when a single order does not fit. */
    const uint32_t S = m->n_states, R = m->n_residues;
    const uint64_t base = beam_jet_cfan_bytes(S, R, 0u);
    const uint64_t per_tilt = beam_jet_cfan_bytes(S, R, 1u) - base;
    const int th = beam_model_effective_threads();
    uint32_t want = th > 0 ? (uint32_t)th : 1u;
    if (nw && want > nw) want = nw;
    const uint64_t room = beam_model_memory_budget() / want;
    const uint64_t fit = room > base ? (room - base) / per_tilt : 0u;
    uint32_t width = fit >= N ? N : (fit > 0u ? (uint32_t)fit : 1u);
    const uint32_t nb = (N + width - 1u) / width;
    width = (N + nb - 1u) / nb;

    const uint64_t items = (uint64_t)nw * nb;
    const uint64_t need = beam_jet_cfan_bytes(S, R, width);
    const uint32_t conc = beam_budget_concurrency(
        need, items == 0u ? 1u : (items > UINT32_MAX ? UINT32_MAX
                                                     : (uint32_t)items));
    if (items && conc == 0u) {
        const uint32_t at = work[0];
        free(work); free(lmod); free(lph);
        return beam_budget_refuse("a complex tilted per-length sweep", at, need);
    }

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc)
#endif
    for (int64_t k = 0; k < (int64_t)items; ++k) {
        const uint32_t L  = work[(uint64_t)k / nb];
        const uint32_t q0 = (uint32_t)((uint64_t)k % nb) * width;
        const uint32_t wq = N - q0 < width ? N - q0 : width;
        const BeamError e = jet_sweep_cfan(m, L, q_re + q0, q_im + q0, wq,
                                           lmod + (size_t)q0 * n + L,
                                           lph + (size_t)q0 * n + L, n);
        if (e != BEAM_OK && (int)e > failed) failed = (int)e;
    }
    free(work);
    if (failed != BEAM_OK) {
        free(lmod); free(lph);
        return BEAM_FAIL((BeamError)failed,
                        "a complex per-length tilted sweep could not be computed");
    }

    /*
     * Combine the lengths. Length L contributes
     *     exp(q c_L) Z_q(L),   c_L = ln pi(L) - ln Z(L),
     * whose log-modulus is `re(q) c_L + lmod[L]` and whose phase is
     * `im(q) c_L + lph[L]`. Summed in ascending length after shifting by the
     * largest log-modulus, as the real transform's log-sum-exp is.
     */
    for (uint32_t j = 0; j < N; ++j) {
        const double *lm = lmod + (size_t)j * n;
        const double *lp = lph + (size_t)j * n;
        double mx = -INFINITY;
        for (uint32_t L = 1; L < n; ++L) {
            if (!beam_length_live(m, L) || lm[L] == -INFINITY) continue;
            const double t = q_re[j] * (m->log_pi[L] - m->log_z[L]) + lm[L];
            if (t > mx) mx = t;
        }
        if (mx == -INFINITY) { out_re[j] = -INFINITY; out_im[j] = 0.0; continue; }
        double ar = 0.0, ai = 0.0;
        for (uint32_t L = 1; L < n; ++L) {
            if (!beam_length_live(m, L) || lm[L] == -INFINITY) continue;
            const double c = m->log_pi[L] - m->log_z[L];
            const double mod = exp(q_re[j] * c + lm[L] - mx);
            const double ph = q_im[j] * c + lp[L];
            ar += mod * cos(ph);
            ai += mod * sin(ph);
        }
        const double a = hypot(ar, ai);
        if (!(a > 0.0)) { out_re[j] = -INFINITY; out_im[j] = 0.0; continue; }
        out_re[j] = mx + log(a);
        out_im[j] = atan2(ai, ar);
    }
    free(lmod); free(lph);
    return BEAM_OK;
}


/* ── A vertical line ─────────────────────────────────────────── */

/* The narrowest stretch the balance above will make; see there. */
/* Budgeted as the line sweep itself. Where the orders are split into
 * stretches does not change any order's bits (see jet_sweep_line), so the
 * width is chosen for speed within the memory budget.
 *
 * AS WIDE AS THE BUDGET ALLOWS. Each stretch of each length is swept by all
 * the threads at once, which share out a layer's targets, so a stretch needs
 * no company to keep them busy and every extra stretch only repeats work that
 * is per edge rather than per order: the layer's weights, each edge's modulus
 * and fixed rotations. The line is therefore one stretch wherever it fits,
 * and otherwise the fewest stretches of whole blocks that do.
 *
 * It used to be the other way round -- many narrow stretches, one thread
 * each -- which left threads idle whenever there were fewer stretches than
 * threads: a 128-order line on the 27-length foundation ran 13 of 40. */
BeamLinePlan beam_line_plan(uint32_t S, uint32_t R, uint32_t A, uint32_t N,
                            uint32_t nth, uint64_t budget) {
    BeamLinePlan p = {0u, 0u};
    if (N == 0u) return p;
    if (nth == 0u) nth = 1u;
    const uint32_t L = BEAM_LINE_LANES;
    const uint32_t whole = (N + L - 1u) / L * L;
    if (beam_jet_line_bytes(S, R, A, whole, nth) <= budget) {
        p.width = N;
        p.n_stretches = 1u;
        return p;
    }
    /* The widest whole number of chains that fits, by bisection: the bytes
     * grow with the width. */
    uint32_t lo = 0u, hi = whole / L;              /* in chains; hi does not fit */
    while (hi - lo > 1u) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (beam_jet_line_bytes(S, R, A, mid * L, nth) <= budget) lo = mid;
        else hi = mid;
    }
    uint32_t fit = lo * L;
    if (fit == 0u) return p;                        /* nothing fits */
    const uint32_t grain = fit >= BEAM_LINE_BLOCK ? BEAM_LINE_BLOCK : L;
    fit -= fit % grain;
    /* Even the stretches out, in whole grains: still within `fit`, which is
     * a whole number of them. */
    uint32_t nb = (N + fit - 1u) / fit;
    uint32_t w = (N + nb - 1u) / nb;
    w = (w + grain - 1u) / grain * grain;
    p.width = w;
    p.n_stretches = (N + w - 1u) / w;
    return p;
}

BeamError beam_model_log_mellin_line(const BeamModel *m, double re, double im0,
                                     double dim, size_t nq, double *out_re,
                                     double *out_im) {
    if (!m) return BEAM_ENULL("model");
    if (nq && (!out_re || !out_im)) return BEAM_ENULL("output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (!isfinite(re) || !isfinite(im0) || !isfinite(dim))
        return BEAM_ERANGE("%s: the line %g + i(%g + k %g) must be finite",
                           __func__, re, im0, dim);
    if (nq == 0u) return BEAM_OK;
    if (nq > UINT32_MAX)
        return BEAM_ERANGE("%s: %zu orders in one call; the limit is %u",
                           __func__, nq, (unsigned)UINT32_MAX);

    const uint32_t n = m->n_len_slots;
    const uint32_t N = (uint32_t)nq;
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *lmod = (double *)malloc((size_t)N * n * sizeof(double));
    double *lph  = (double *)malloc((size_t)N * n * sizeof(double));
    if (!work || !lmod || !lph) {
        free(work); free(lmod); free(lph);
        return BEAM_EALLOC("the per-length line Mellin terms");
    }
    for (size_t k = 0; k < (size_t)N * n; ++k) { lmod[k] = -INFINITY; lph[k] = 0.0; }
    const uint32_t nw = beam_live_lengths_desc(m, work);
    const int th = beam_model_effective_threads();
    const uint32_t nth = th > 0 ? (uint32_t)th : 1u;
    const BeamLinePlan plan = beam_line_plan(m->n_states, m->n_residues,
                                             m->n_alphabet, N, nth,
                                             beam_model_memory_budget());
    const uint32_t width = plan.width, nb = plan.n_stretches;
    if (nw && width == 0u) {
        const uint32_t at = work[0];
        free(work); free(lmod); free(lph);
        return beam_budget_refuse("a complex tilted line sweep", at,
                                  beam_jet_line_bytes(m->n_states, m->n_residues,
                                                      m->n_alphabet,
                                                      BEAM_LINE_LANES, nth));
    }

    /* One (length, stretch) at a time; each uses every thread, and all of
     * them one workspace. */
    int failed = BEAM_OK;
    const uint64_t items = (uint64_t)nw * nb;
    BeamLineWork *wk = NULL;
    /* Each length's live frontier first, all at once, into the model's
     * cache: the sweeps below take one length at a time, so a frontier they
     * built for themselves would be built with every other thread waiting --
     * 11.7 s of a 20.7 s first line on the foundation. Where the cache cannot
     * hold them this builds nothing lasting and the sweeps build their own. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads((int)nth)
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        LiveHandle h;
        if (beam_live_acquire(m, work[k], &h) == BEAM_OK) beam_live_release(&h);
    }
    if (items) {
        const uint64_t need = beam_jet_line_bytes(m->n_states, m->n_residues,
                                                  m->n_alphabet, width, nth);
        if (!beam_budget_fits(need)) {
            const uint32_t at = work[0];
            free(work); free(lmod); free(lph);
            return beam_budget_refuse("a complex tilted line sweep", at, need);
        }
        wk = beam_line_work_new(m, width, nth);
        if (!wk) {
            free(work); free(lmod); free(lph);
            return BEAM_EALLOC("the complex line-sweep workspace");
        }
    }
    for (uint64_t k = 0; k < items && failed == BEAM_OK; ++k) {
        const uint32_t L  = work[k / nb];
        const uint32_t q0 = (uint32_t)(k % nb) * width;
        const uint32_t wq = N - q0 < width ? N - q0 : width;
        const BeamError e = jet_sweep_line(m, wk, L, re, im0, dim, q0, wq,
                                           lmod + (size_t)q0 * n + L,
                                           lph + (size_t)q0 * n + L, n);
        if (e != BEAM_OK) failed = (int)e;
    }
    beam_line_work_free(wk);
    free(work);
    if (failed != BEAM_OK) {
        free(lmod); free(lph);
        return BEAM_FAIL((BeamError)failed,
                        "a complex per-length line sweep could not be computed");
    }

    /* The lengths combined exactly as in `beam_model_log_mellin_complex`. */
    for (uint32_t j = 0; j < N; ++j) {
        const double qi = im0 + (double)j * dim;
        const double *lm = lmod + (size_t)j * n;
        const double *lp = lph + (size_t)j * n;
        double mx = -INFINITY;
        for (uint32_t L = 1; L < n; ++L) {
            if (!beam_length_live(m, L) || lm[L] == -INFINITY) continue;
            const double t = re * (m->log_pi[L] - m->log_z[L]) + lm[L];
            if (t > mx) mx = t;
        }
        if (mx == -INFINITY) { out_re[j] = -INFINITY; out_im[j] = 0.0; continue; }
        double ar = 0.0, ai = 0.0;
        for (uint32_t L = 1; L < n; ++L) {
            if (!beam_length_live(m, L) || lm[L] == -INFINITY) continue;
            const double c = m->log_pi[L] - m->log_z[L];
            const double mod = exp(re * c + lm[L] - mx);
            const double ph = qi * c + lp[L];
            ar += mod * cos(ph);
            ai += mod * sin(ph);
        }
        const double a = hypot(ar, ai);
        if (!(a > 0.0)) { out_re[j] = -INFINITY; out_im[j] = 0.0; continue; }
        out_re[j] = mx + log(a);
        out_im[j] = atan2(ai, ar);
    }
    free(lmod); free(lph);
    return BEAM_OK;
}
