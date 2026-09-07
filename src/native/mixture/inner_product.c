/**
 * @file inner_product.c
 * @brief `sum_s prod_f P_f(s)`: one forward sweep over a shared trellis.
 *
 * The whole of the integer-q mixture algebra rests on this one quantity. For
 * `pbar = sum_i w_i P_i`,
 *
 *     sum_s pbar(s)^q = sum over q-tuples (f1..fq) of
 *                       w_f1 .. w_fq * <P_f1, .., P_fq>,
 *
 * so the q = 2 collision probability -- and with it D2 of the mixture, and
 * with that beta diversity between individuals -- is M(M+1)/2 calls to this
 * file and nothing else.
 *
 * ## Why one sweep suffices
 *
 * `ln P_f(s) = ln pi_f(L) - ln Z_f(L) + raw_f(s)`, where `raw_f` is the sum of
 * the edge weights `f` assigns along the path plus its sink term. The per-path
 * constant comes out of the sum, and what is left,
 *
 *     ln sum_{paths of length L} exp( sum_f raw_f(s) ),
 *
 * is an ordinary forward partition function over a trellis whose edge weight
 * is the SUM of the members' edge weights. A path no member spells has a
 * -INFINITY somewhere in that sum and drops out on its own, which is how the
 * closed world survives: nothing is floored, nothing is smoothed, and a
 * disjoint pair returns -INFINITY rather than a small positive number.
 *
 * ## Two traps, both load-bearing
 *
 * `beam_layer_weights_live` leaves the rows of dead states UNTOUCHED rather
 * than filling them with -INFINITY -- see its own comment, which measured the
 * alternative and rejected it. So the `alive` mask handed to the k weight
 * builds must be the INTERSECTION of the k reached sets, not one model's set
 * and not their union: a state model A reaches and model B does not would
 * leave B's buffer holding a row from a previous LAYER, and the sweep would
 * read it as a live edge. The intersection is also exactly where the product
 * can carry mass, so this costs nothing and is not a compromise.
 *
 * And the weights are summed rather than the models' layers being held at
 * once. k layers of edge weights is 31 MB per member at amino-acid order 4;
 * summing into one accumulator as each is built holds two layers whatever k
 * is, which is what makes a 64-member mixture the same footprint as a pair.
 */
#include "beam/internal.h"
#include "beam/mixture.h"


/**
 * Peak bytes one sweep holds.
 *
 * Independent of k, which is the point of accumulating the weights rather than
 * holding one layer per member. Beside the allocation it guards so the two
 * cannot drift apart, and shared with the driver's concurrency bound so what
 * is refused and what is scheduled come from one expression.
 *
 * The per-length live frontiers are NOT counted, matching every other sweep in
 * the library: they are a cache on the model rather than scratch for the
 * program. A mixture acquires k of them at once instead of one, so a caller
 * holding many members should expect the cache, not this figure, to dominate.
 */
static inline uint64_t product_sweep_bytes(uint32_t S, uint32_t R) {
    return 2ull * S * R * sizeof(double)      /* summed layer, plus scratch  */
         + 4ull * S * sizeof(double)          /* cur, next, tail, tail tmp   */
         + 2ull * S                           /* the two intersection masks  */
         + 2ull * S * sizeof(uint32_t);       /* the two index lists         */
}

/**
 * States reached at `layer` by EVERY member, as an ascending list and a mask.
 *
 * Counts up rather than intersecting pairwise: member 0 marks its reached
 * states with 1, and each later member promotes a mark it also reaches, so a
 * state carries `k` exactly when all k reached it. Walking member 0's list at
 * the end both collects the survivors in ascending order and clears the
 * partial marks, so the mask is left holding 1 on the intersection and 0
 * everywhere else.
 *
 * Ascending order matters beyond tidiness: it is what makes every sum in the
 * sweep receive its terms in a fixed sequence, so the answer does not depend
 * on how the frontier happened to be built.
 */
static uint32_t intersect_reached(const BeamLive *const *L, uint32_t k,
                                  uint32_t layer, uint32_t S,
                                  uint8_t *mask, uint32_t *idx) {
    memset(mask, 0, S);
    const uint32_t  n0 = L[0]->n_reached[layer];
    const uint32_t *v0 = L[0]->reached[layer];
    for (uint32_t j = 0; j < n0; ++j) mask[v0[j]] = 1u;
    for (uint32_t f = 1; f < k; ++f) {
        const uint32_t  n = L[f]->n_reached[layer];
        const uint32_t *v = L[f]->reached[layer];
        for (uint32_t j = 0; j < n; ++j)
            if (mask[v[j]] == (uint8_t)f) mask[v[j]] = (uint8_t)(f + 1u);
    }
    uint32_t n_out = 0;
    for (uint32_t j = 0; j < n0; ++j) {
        const uint32_t s = v0[j];
        if (mask[s] == (uint8_t)k) { idx[n_out++] = s; mask[s] = 1u; }
        else                        { mask[s] = 0u; }
    }
    (void)S;
    return n_out;
}


BeamError beam_mixture_log_inner_product(const BeamModel *const *models,
                                         uint32_t k, uint32_t len,
                                         double *out) {
    if (!out) return BEAM_ENULL("output buffer");
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;

    const BeamModel *m0 = models[0];
    if (len < 1u || len > m0->max_len) return BEAM_EBADLEN(m0, len);

    /* A length one member never produces contributes nothing to a PRODUCT,
     * however much mass the others put there. Checked before any allocation
     * because it is the common case in a cohort: donors differ in which
     * lengths they carry at all. */
    double konst = 0.0;
    for (uint32_t f = 0; f < k; ++f) {
        if (!beam_length_live(models[f], len)) { *out = -INFINITY; return BEAM_OK; }
        konst += models[f]->log_pi[len] - models[f]->log_z[len];
    }

    const uint32_t A  = m0->n_alphabet;
    const uint32_t R  = m0->n_residues;
    const uint32_t S  = m0->n_states;
    const uint32_t SK = m0->state_shift;

    BEAM_BUDGET_OR_REFUSE("a mixture inner-product sweep", len,
                          product_sweep_bytes(S, R));

    LiveHandle h[BEAM_MIXTURE_MAX_MEMBERS];
    const BeamLive *L[BEAM_MIXTURE_MAX_MEMBERS];
    uint32_t got = 0;
    for (uint32_t f = 0; f < k; ++f) {
        if (beam_live_acquire(models[f], len, &h[f]) != BEAM_OK) {
            for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
            return BEAM_EALLOC("the forward frontier");
        }
        L[f] = h[f].L;
        ++got;
    }

    double  *w     = (double *)malloc((size_t)S * R * sizeof(double));
    double  *wt    = (double *)malloc((size_t)S * R * sizeof(double));
    double  *cur   = (double *)calloc(S, sizeof(double));
    double  *next  = (double *)calloc(S, sizeof(double));
    double  *tail  = (double *)malloc((size_t)S * sizeof(double));
    double  *tt    = (double *)malloc((size_t)S * sizeof(double));
    uint8_t *am    = (uint8_t *)malloc(S);
    uint8_t *am_n  = (uint8_t *)malloc(S);
    uint32_t *ix   = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t *ix_n = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    if (!w || !wt || !cur || !next || !tail || !tt || !am || !am_n
        || !ix || !ix_n) {
        free(w); free(wt); free(cur); free(next); free(tail); free(tt);
        free(am); free(am_n); free(ix); free(ix_n);
        for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
        return BEAM_EALLOC("the mixture inner-product buffers");
    }

    cur[S - 1] = 1.0;
    double log_scale = 0.0;
    uint32_t n_cur = intersect_reached(L, k, 0u, S, am, ix);

    for (uint32_t i = 0; i < len; ++i) {
        const uint32_t n_nxt = intersect_reached(L, k, i + 1u, S, am_n, ix_n);
        if (n_nxt == 0u) goto dead;

        /*
         * The summed layer. -INFINITY propagates through the addition on its
         * own, which is the closed world doing its own bookkeeping: an edge
         * one member cannot spell removes the whole path from the product. No
         * weight is ever +INFINITY -- they are mixed log probabilities, finite
         * or -INFINITY -- so the sum cannot produce a NaN.
         */
        beam_layer_weights_live(models[0], i, len, w, NULL, am);
        for (uint32_t f = 1; f < k; ++f) {
            beam_layer_weights_live(models[f], i, len, wt, NULL, am);
            for (uint32_t j = 0; j < n_cur; ++j) {
                const size_t off = (size_t)ix[j] * R;
                double *dst = w + off;
                const double *src = wt + off;
                for (uint32_t x = 0; x < R; ++x) dst[x] += src[x];
            }
        }

        /* Rescale by the best edge THAT CARRIES MASS, for the reason
         * `jet_sweep` sets out: scaling by an unreachable edge underflows
         * every reachable one to zero, which is indistinguishable from a
         * length with no paths. */
        double wmax = -INFINITY;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:wmax) \
                num_threads(beam_threads())
#endif
        for (int64_t j = 0; j < (int64_t)n_cur; ++j) {
            const uint32_t s = ix[j];
            if (cur[s] == 0.0) continue;
            const double *row = w + (size_t)s * R;
            for (uint32_t x = 0; x < R; ++x)
                if (row[x] > wmax) wmax = row[x];
        }
        if (wmax == -INFINITY) goto dead;

        /* By TARGET, so each accumulator is the private property of one
         * thread; the scatter by source has A sources per target and races. */
        double nmax = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t j = 0; j < (int64_t)n_nxt; ++j) {
            const uint32_t t = ix_n[j];
            const uint32_t x = t % A;
            if (x >= R) { next[t] = 0.0; continue; }
            const uint32_t rest = t / A;
            double acc = 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                const double v = cur[src];
                if (v == 0.0) continue;      /* implies src is in the mask */
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                acc += v * exp(wv - wmax);
            }
            next[t] = acc;
            if (acc > nmax) nmax = acc;
        }
        log_scale += wmax;
        if (!(nmax > 0.0)) goto dead;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t j = 0; j < (int64_t)n_nxt; ++j) next[ix_n[j]] /= nmax;
        log_scale += log(nmax);

        /* `cur` is about to become the next layer's `next`, which is written
         * only at the target states; its values from two layers back have to
         * go, and they live exactly on this layer's intersection. */
        for (uint32_t j = 0; j < n_cur; ++j) cur[ix[j]] = 0.0;
        { double *t2 = cur; cur = next; next = t2; }
        { uint8_t *t2 = am; am = am_n; am_n = t2; }
        { uint32_t *t2 = ix; ix = ix_n; ix_n = t2; }
        n_cur = n_nxt;
    }

    /* The sink settles the backward terms of the last residues: one more edge,
     * whose weight is again the sum over members. `beam_tail_weights` writes
     * every state, so there are no stale rows to avoid here. */
    beam_tail_weights(models[0], len, tail);
    for (uint32_t f = 1; f < k; ++f) {
        beam_tail_weights(models[f], len, tt);
        for (uint32_t j = 0; j < n_cur; ++j) tail[ix[j]] += tt[ix[j]];
    }
    {
        double tmax = -INFINITY;
        for (uint32_t j = 0; j < n_cur; ++j) {
            const uint32_t s = ix[j];
            if (cur[s] > 0.0 && tail[s] != -INFINITY && tail[s] > tmax)
                tmax = tail[s];
        }
        if (tmax == -INFINITY) goto dead;
        double acc = 0.0;
        for (uint32_t j = 0; j < n_cur; ++j) {
            const uint32_t s = ix[j];
            if (cur[s] == 0.0 || tail[s] == -INFINITY) continue;
            acc += cur[s] * exp(tail[s] - tmax);
        }
        *out = konst + log_scale + tmax + log(acc);
    }
    free(w); free(wt); free(cur); free(next); free(tail); free(tt);
    free(am); free(am_n); free(ix); free(ix_n);
    for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
    return BEAM_OK;
dead:
    free(w); free(wt); free(cur); free(next); free(tail); free(tt);
    free(am); free(am_n); free(ix); free(ix_n);
    for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
    *out = -INFINITY;                 /* an answer: the supports never meet */
    return BEAM_OK;
}


BeamError beam_mixture_log_inner_product_by_length(
        const BeamModel *const *models, uint32_t k,
        double *out_by_length, double *out_total) {
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    if (!out_total) return BEAM_ENULL("output buffer");

    const BeamModel *m0 = models[0];
    const uint32_t n = m0->n_len_slots;
    double *val = (double *)malloc((size_t)n * sizeof(double));
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!val || !work) {
        free(val); free(work);
        return BEAM_EALLOC("the per-length inner products");
    }
    for (uint32_t Lx = 0; Lx < n; ++Lx) val[Lx] = -INFINITY;

    /* Longest first, so the expensive lengths are handed out before the cheap
     * ones and the tail of the schedule is short. A length any member fails to
     * produce is skipped outright: the product there is zero. */
    uint32_t nw = 0;
    for (uint32_t Lx = n; Lx-- > 1; ) {
        bool all = true;
        for (uint32_t f = 0; f < k && all; ++f)
            if (!beam_length_live(models[f], Lx)) all = false;
        if (all) work[nw++] = Lx;
    }

    /* The sweeps run concurrently, so they divide the budget rather than
     * multiplying it. 0 means one does not fit at all, which has to become a
     * refusal here on the calling thread: `beam_set_error` is thread-local, so
     * a refusal raised inside a worker would reach the caller as a bare code
     * with its message lost. */
    const uint64_t per = product_sweep_bytes(m0->n_states, m0->n_residues);
    const uint32_t conc = beam_budget_concurrency(per, nw ? nw : 1u);
    if (nw && conc == 0u) {
        const uint32_t at = work[0];
        free(val); free(work);
        return beam_budget_refuse("a mixture inner-product sweep", at, per);
    }

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc)
#endif
    for (int64_t j = 0; j < (int64_t)nw; ++j) {
        double v = 0.0;
        const BeamError e = beam_mixture_log_inner_product(models, k,
                                                           work[j], &v);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
        else val[work[j]] = v;
    }
    if (failed != BEAM_OK) {
        free(val); free(work);
        return BEAM_FAIL((BeamError)failed,
                         "mixture inner product: a per-length sweep failed");
    }

    /* Summed in the log domain, ascending in length, so a long tail of tiny
     * lengths neither underflows the total nor reorders it run to run. */
    double mx = -INFINITY;
    for (uint32_t Lx = 1; Lx < n; ++Lx) if (val[Lx] > mx) mx = val[Lx];
    if (mx == -INFINITY) *out_total = -INFINITY;
    else {
        double acc = 0.0;
        for (uint32_t Lx = 1; Lx < n; ++Lx)
            if (val[Lx] != -INFINITY) acc += exp(val[Lx] - mx);
        *out_total = mx + log(acc);
    }
    if (out_by_length)
        for (uint32_t Lx = 0; Lx < n; ++Lx) out_by_length[Lx] = val[Lx];

    free(val); free(work);
    return BEAM_OK;
}
