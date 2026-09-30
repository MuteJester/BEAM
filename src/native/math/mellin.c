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
 * Live lengths, longest first: the per-length sweeps are the unit of parallel
 * work and cost grows with the length, so the longest must be handed out
 * first or fifteen cores wait on it at the end.
 */
BeamError beam_mellin_terms(const BeamModel *m, double q, double *terms,
                             double *out_max) {
    const uint32_t n = m->n_len_slots;
    for (uint32_t L = 0; L < n; ++L) terms[L] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *lzs = (double *)malloc((size_t)n * sizeof(double));
    if (!work || !lzs) {
        free(work); free(lzs);
        return BEAM_EALLOC("the per-length Mellin work list");
    }
    const uint32_t nw = beam_live_lengths_desc(m, work);
    int failed = BEAM_OK;
    /* Concurrent sweeps DIVIDE the budget rather than multiplying it. A
     * concurrency of 0 means one sweep does not fit at all, which the sweep
     * itself refuses with the better message, so run one and report that. */
    const uint32_t conc = beam_budget_concurrency(
        beam_jet_sweep_bytes(m->n_states, m->n_residues, 1u), nw ? nw : 1u);
    if (nw && conc == 0u) {
        const uint32_t at = work[0];
        const uint64_t need = beam_jet_sweep_bytes(m->n_states, m->n_residues, 1u);
        free(work); free(lzs);
        return beam_budget_refuse("a tilted per-length sweep", at, need);
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc)
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
 * `beam_mellin_terms` at `nq` orders, from one traversal of each length where
 * there were `nq`.
 *
 * `terms[j * n_len_slots + L]` and `out_max[j]` are bit-identical to what
 * `beam_mellin_terms(m, qs[j], ...)` writes, whatever the budget, the thread
 * count, or the company `qs[j]` keeps. Each order's per-length value comes
 * from `jet_sweep_fan`, whose contract is exactly that, and is combined here
 * by the same expression, in the same ascending-length order.
 *
 * THE BUDGET is shared between two kinds of parallelism, and both cost memory.
 * Fusing `W` orders into one sweep multiplies its mass planes by `W`; running
 * `conc` lengths at once multiplies the whole sweep by `conc`. The lengths are
 * the unit of parallel work and the reason the single-order form is fast, so
 * they are budgeted first: the widest fusion that still lets one sweep per
 * thread run is taken, and only when not even one order per sweep allows that
 * does concurrency give way instead -- to what the single-order form would
 * have run. At the default 4 GB it never binds on a default amino-acid model:
 * 31 orders at order 4 is 128 MB a sweep, and 27 lengths at most 3.5 GB. It
 * binds on a nucleotide model near the ceiling order, where one plane is 80 MB.
 *
 * When the orders do not fit one sweep they are split into equal blocks, and
 * each (length, block) pair is one item of work. Blocks change which orders
 * share a traversal, never what any one of them computes.
 */
static BeamError mellin_terms_many(const BeamModel *m, const double *qs,
                                   uint32_t nq, double *terms,
                                   double *out_max) {
    const uint32_t n = m->n_len_slots;
    for (size_t k = 0; k < (size_t)nq * n; ++k) terms[k] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *lzs = (double *)malloc((size_t)nq * n * sizeof(double));
    if (!work || !lzs) {
        free(work); free(lzs);
        return BEAM_EALLOC("the per-length many-order Mellin work list");
    }
    const uint32_t nw = beam_live_lengths_desc(m, work);

    /* Width first: the widest fusion that leaves room for one sweep per
     * thread. `per_tilt` is what one more order adds to a sweep, taken as a
     * difference of the sizing function so it cannot disagree with it. */
    const uint32_t S = m->n_states, R = m->n_residues;
    const uint64_t base = beam_jet_fan_bytes(S, R, 0u);
    const uint64_t per_tilt = beam_jet_fan_bytes(S, R, 1u) - base;
    const int th = beam_model_effective_threads();
    uint32_t want = th > 0 ? (uint32_t)th : 1u;
    if (nw && want > nw) want = nw;
    const uint64_t room = beam_model_memory_budget() / want;
    const uint64_t fit = room > base ? (room - base) / per_tilt : 0u;
    uint32_t width = fit >= nq ? nq : (fit > 0u ? (uint32_t)fit : 1u);
    const uint32_t nb = (nq + width - 1u) / width;
    width = (nq + nb - 1u) / nb;             /* equal blocks, no straggler */

    const uint64_t items = (uint64_t)nw * nb;
    const uint64_t need = beam_jet_fan_bytes(S, R, width);
    /* Concurrent sweeps DIVIDE the budget. A concurrency of 0 means one sweep
     * of a single order does not fit at all, which is exactly where the
     * single-order form refuses too. */
    const uint32_t conc = beam_budget_concurrency(
        need, items == 0u ? 1u : (items > UINT32_MAX ? UINT32_MAX
                                                     : (uint32_t)items));
    if (items && conc == 0u) {
        const uint32_t at = work[0];
        free(work); free(lzs);
        return beam_budget_refuse("a fused tilted per-length sweep", at, need);
    }

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc)
#endif
    for (int64_t k = 0; k < (int64_t)items; ++k) {
        /* Longest length first, as in the single-order form; within a
         * length, its blocks in order. */
        const uint32_t L  = work[(uint64_t)k / nb];
        const uint32_t q0 = (uint32_t)((uint64_t)k % nb) * width;
        const uint32_t wq = nq - q0 < width ? nq - q0 : width;
        const BeamError e = jet_sweep_fan(m, L, qs + q0, wq,
                                          lzs + (size_t)q0 * n + L, n);
        if (e != BEAM_OK && (int)e > failed) failed = (int)e;
    }
    free(work);
    if (failed != BEAM_OK) {
        free(lzs);
        return BEAM_FAIL((BeamError)failed,
                        "a fused per-length tilted sweep could not be computed");
    }
    /* Per order, scanned in ascending length: the same expression and the
     * same order as `beam_mellin_terms`, so the terms and the maximum are its
     * bits. */
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
#define BEAM_LINE_MIN_WIDTH 128u

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

    /* Budgeted as the complex form. Where the orders are split into stretches
     * does not change any order's bits (see jet_sweep_line), so the width is
     * chosen for speed within the memory budget.
     *
     * BALANCE. The work items are (length, stretch) pairs and a length-L
     * sweep costs L layers, so with one stretch per length the longest length
     * alone bounds the line's time: the lengths' sum over their maximum, 14
     * at max_length 27, is all the parallelism there is, whatever the thread
     * count. Stretches are therefore narrowed until there are about three
     * items per thread -- but kept at least BEAM_LINE_MIN_WIDTH orders wide,
     * since each edge pays a few cosines, sines and an exp per stretch. */
    const uint32_t S = m->n_states, R = m->n_residues;
    const uint64_t base = beam_jet_cfan_bytes(S, R, 0u);
    const uint64_t per_tilt = beam_jet_cfan_bytes(S, R, 1u) - base;
    const int th = beam_model_effective_threads();
    const uint32_t nth = th > 0 ? (uint32_t)th : 1u;
    uint32_t want = nth;
    if (nw && want > nw) want = nw;
    const uint64_t room = beam_model_memory_budget() / want;
    const uint64_t fit = room > base ? (room - base) / per_tilt : 0u;
    uint32_t width = fit >= N ? N : (fit > 0u ? (uint32_t)fit : 1u);
    if (nw) {
        const uint64_t per_len = ((uint64_t)3u * nth + nw - 1u) / nw;
        uint64_t bal = ((uint64_t)N + per_len - 1u) / per_len;
        if (bal < BEAM_LINE_MIN_WIDTH) bal = BEAM_LINE_MIN_WIDTH;
        if (bal < width) width = (uint32_t)bal;
    }
    uint32_t nb = (N + width - 1u) / width;
    width = (N + nb - 1u) / nb;
    /* The sweep pads a stretch to whole vectors of four orders: stretches
     * are whole vectors wide, so only the last one pads, and the budget is
     * asked for the padded width. */
    if (width > 4u) width -= width % 4u;
    nb = (N + width - 1u) / width;

    const uint64_t items = (uint64_t)nw * nb;
    const uint64_t need = beam_jet_cfan_bytes(S, R, (width + 3u) / 4u * 4u);
    const uint32_t conc = beam_budget_concurrency(
        need, items == 0u ? 1u : (items > UINT32_MAX ? UINT32_MAX
                                                     : (uint32_t)items));
    if (items && conc == 0u) {
        const uint32_t at = work[0];
        free(work); free(lmod); free(lph);
        return beam_budget_refuse("a complex tilted line sweep", at, need);
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
        const BeamError e = jet_sweep_line(m, L, re, im0, dim, q0, wq,
                                           lmod + (size_t)q0 * n + L,
                                           lph + (size_t)q0 * n + L, n);
        if (e != BEAM_OK && (int)e > failed) failed = (int)e;
    }
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
