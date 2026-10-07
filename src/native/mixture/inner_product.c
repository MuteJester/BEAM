/**
 * @file inner_product.c
 * @brief `sum_s prod_f P_f(s)^a_f`: one forward sweep over a shared trellis.
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
 * ## Per-member EXPONENTS, and why they are free
 *
 * `P_f(s)^a_f` raises a per-path constant and a sum of edge weights to a
 * power, and a power of a product is the product of the powers, so the whole
 * derivation above survives with every member's contribution SCALED:
 *
 *     ln sum_s prod_f P_f(s)^a_f
 *       = sum_f a_f (ln pi_f(L) - ln Z_f(L))
 *         + ln sum_{paths} exp( sum_f a_f raw_f(s) ).
 *
 * The sweep is therefore identical except that the edge weight is the WEIGHTED
 * sum. Nothing about the recursion, the rescaling or the sink changes, and the
 * cost is one multiply per edge. That is what makes the whole Chernoff family
 * computable here: `a = (1/2, 1/2)` is the Bhattacharyya coefficient, whose
 * complement is the squared Hellinger distance between two models, and
 * `(a, 1 - a)` is the Chernoff coefficient at that tilt.
 *
 * An exponent of exactly 0 means `P_f(s)^0 = 1` AT EVERY SEQUENCE, including
 * ones `f` cannot spell -- so that member is DROPPED from the sweep rather
 * than multiplied in with a zeroed weight. The two are different answers:
 * zeroing the weight would leave `f` still intersecting the support, and
 * `<P_A^1, P_B^0>` would come back as the mass `A` puts on `B`'s support
 * instead of as 1. Dropping is also what makes the identity
 * `sum_s P_A(s) = 1` hold, which is the test that pins it.
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
 * The weight buffers are independent of k -- that is the point of accumulating
 * into one layer rather than holding one per member. The two handle arrays are
 * not, and this counted only the first group for as long as they were on the
 * stack. When they moved to the heap the formula did not follow, and the
 * comment here still said "independent of k, beside the allocation it guards
 * so the two cannot drift apart" while they had already drifted: measured, the
 * shortfall is exactly 80 bytes per member on x86-64, 5.24 MB at the 65,536
 * ceiling, and a caller who set the budget to precisely the figure a refusal
 * printed was admitted and then exceeded it.
 *
 * Shared with the driver's concurrency bound, so what is refused and what is
 * scheduled come from one expression. `k` is the members ONE sweep holds, which
 * is not the caller's member count everywhere: the Gram driver sweeps pairs and
 * must pass 2.
 *
 * The per-length live frontiers are NOT counted, matching every other sweep in
 * the library: they are a cache on the model rather than scratch for the
 * program. A mixture acquires k of them at once instead of one, so a caller
 * holding many members should expect the cache, not this figure, to dominate.
 */
static inline uint64_t product_sweep_bytes(uint32_t S, uint32_t R, uint32_t k) {
    return 2ull * S * R * sizeof(double)      /* summed layer, plus scratch  */
         + 4ull * S * sizeof(double)          /* cur, next, tail, tail tmp   */
         + 2ull * S                           /* the two intersection masks  */
         + 2ull * S * sizeof(uint32_t)        /* the two index lists         */
         + (uint64_t)k * (sizeof(LiveHandle)  /* the per-member handles      */
                          + sizeof(const BeamLive *));
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


/**
 * Refuse an exponent vector that is not one.
 *
 * NULL means "every exponent is 1", which is the plain inner product and needs
 * no vector. Otherwise every entry must be finite and non-negative -- a
 * negative exponent asks for `sum_s P(s)^-a`, which diverges over a support
 * this large and would come back as `inf` with no indication why -- and at
 * least one must be positive, since an all-zero vector asks for `sum_s 1` over
 * the whole space, a support count rather than a product.
 */
static BeamError check_exponents(const double *expo, uint32_t k) {
    if (!expo) return BEAM_OK;
    uint32_t live = 0;
    for (uint32_t f = 0; f < k; ++f) {
        if (!isfinite(expo[f]))
            return BEAM_ERANGE("%s: exponent %u is %g; exponents must be "
                               "finite", __func__, f, expo[f]);
        if (expo[f] < 0.0)
            return BEAM_ERANGE("%s: exponent %u is %g; exponents must be "
                               "non-negative, because sum_s P(s)^-a diverges "
                               "over this support", __func__, f, expo[f]);
        if (expo[f] > 0.0) ++live;
    }
    if (live == 0u)
        return BEAM_ERANGE("%s: every exponent is 0, so the product is 1 at "
                           "every sequence and the sum is the size of the "
                           "whole space; at least one exponent must be "
                           "positive", __func__);
    return BEAM_OK;
}

/**
 * Members with a positive exponent, compacted, and their exponents with them.
 *
 * A zero exponent contributes a factor of 1 everywhere, so the member leaves
 * the sweep entirely -- it must not appear in the reached-set intersection or
 * in `konst`. Compacting here keeps `intersect_reached` and the sweep itself
 * ignorant of exponents being optional at all.
 *
 * `mm` and `aa` must hold `k` entries. Returns the surviving count, which
 * `check_exponents` has already guaranteed is at least 1.
 */
static uint32_t compact_members(const BeamModel *const *models,
                                const double *expo, uint32_t k,
                                const BeamModel **mm, double *aa) {
    if (!expo) {
        for (uint32_t f = 0; f < k; ++f) { mm[f] = models[f]; aa[f] = 1.0; }
        return k;
    }
    uint32_t n = 0;
    for (uint32_t f = 0; f < k; ++f)
        if (expo[f] > 0.0) { mm[n] = models[f]; aa[n] = expo[f]; ++n; }
    return n;
}


/**
 * The sweep. `k >= 1`, every `expo[f] > 0`, shape already checked.
 *
 * `expo == NULL` is the all-ones case and takes the same path: the scaling is
 * written as a multiply by 1.0, which is exact in binary floating point, so
 * the two agree to the last bit and the plain inner product is not a separate
 * implementation that could drift from this one.
 */
static BeamError power_sweep(const BeamModel *const *models,
                             const double *expo, uint32_t k, uint32_t len,
                             double *out, double *by_pos, double *sink_out,
                             double *const_out) {
    const BeamModel *m0 = models[0];
    if (len < 1u || len > m0->max_len) return BEAM_EBADLEN(m0, len);

    /* The decomposition's outputs are filled as the sweep goes and have to be
     * well defined on every exit, including the two that leave early. */
    if (by_pos) for (uint32_t i = 0; i < len; ++i) by_pos[i] = -INFINITY;
    if (sink_out) *sink_out = -INFINITY;
    if (const_out) *const_out = -INFINITY;

    /* A length one member never produces contributes nothing to a PRODUCT,
     * however much mass the others put there. Checked before any allocation
     * because it is the common case in a cohort: donors differ in which
     * lengths they carry at all. */
    double konst = 0.0;
    for (uint32_t f = 0; f < k; ++f) {
        if (!beam_length_live(models[f], len)) { *out = -INFINITY; return BEAM_OK; }
        const double a = expo ? expo[f] : 1.0;
        konst += a * (models[f]->log_pi[len] - models[f]->log_z[len]);
    }
    if (const_out) *const_out = konst;

    const uint32_t A  = m0->n_alphabet;
    const uint32_t R  = m0->n_residues;
    const uint32_t S  = m0->n_states;
    const uint32_t SK = m0->state_shift;

    BEAM_BUDGET_OR_REFUSE("a mixture inner-product sweep", len,
                          product_sweep_bytes(S, R, k));

    /*
     * On the heap, not the stack. These were fixed arrays sized by the union's
     * 64-member ceiling, which was fine while the inner product shared that
     * ceiling and is not now that it does not: at a few thousand members the
     * same arrays would be megabytes of automatic storage, and a stack
     * overflow is the one failure this library cannot turn into an error code.
     */
    LiveHandle *h = (LiveHandle *)malloc((size_t)k * sizeof(LiveHandle));
    const BeamLive **L = (const BeamLive **)malloc((size_t)k * sizeof(*L));
    if (!h || !L) {
        free(h); free((void *)L);
        return BEAM_EALLOC("the per-member frontier handles");
    }
    uint32_t got = 0;
    for (uint32_t f = 0; f < k; ++f) {
        if (beam_live_acquire(models[f], len, &h[f]) != BEAM_OK) {
            for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
            free(h); free((void *)L);
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
        free(h); free((void *)L);
        return BEAM_EALLOC("the mixture inner-product buffers");
    }

    cur[S - 1] = 1.0;
    double log_scale = 0.0;
    uint32_t n_cur = intersect_reached(L, k, 0u, S, am, ix);

    /* The frontier's TOTAL mass, which is what the positional terms are ratios
     * of. Read off the array rather than assumed to be the 1.0 just written,
     * so a start state outside the intersection gives -INFINITY here too
     * instead of a term computed against a mass the sweep is not carrying. */
    double msum = 0.0;
    if (by_pos || sink_out)
        for (uint32_t j = 0; j < n_cur; ++j) msum += cur[ix[j]];

    for (uint32_t i = 0; i < len; ++i) {
        const uint32_t n_nxt = intersect_reached(L, k, i + 1u, S, am_n, ix_n);
        if (n_nxt == 0u) goto dead;

        /*
         * The summed layer, each member's weights scaled by its exponent.
         * -INFINITY propagates through both the scaling and the addition on
         * its own -- every exponent here is strictly positive, so no term can
         * become `0 * -INFINITY` -- which is the closed world doing its own
         * bookkeeping: an edge one member cannot spell removes the whole path
         * from the product. No weight is ever +INFINITY, they being mixed log
         * probabilities, so the sum cannot produce a NaN either.
         *
         * An exponent of exactly 1 skips its multiply rather than performing
         * it. `1.0 * x` is exact, so this is not for the arithmetic's sake;
         * it is so the all-ones case touches the same bytes the kernel touched
         * before exponents existed, and the two can be asserted equal to the
         * last bit instead of to a tolerance.
         */
        beam_layer_weights_live(models[0], i, len, w, NULL, am);
        const double a0 = expo ? expo[0] : 1.0;
        if (a0 != 1.0) {
            for (uint32_t j = 0; j < n_cur; ++j) {
                double *dst = w + (size_t)ix[j] * R;
                for (uint32_t x = 0; x < R; ++x) dst[x] *= a0;
            }
        }
        for (uint32_t f = 1; f < k; ++f) {
            const double af = expo ? expo[f] : 1.0;
            beam_layer_weights_live(models[f], i, len, wt, NULL, am);
            for (uint32_t j = 0; j < n_cur; ++j) {
                const size_t off = (size_t)ix[j] * R;
                double *dst = w + off;
                const double *src = wt + off;
                if (af == 1.0)
                    for (uint32_t x = 0; x < R; ++x) dst[x] += src[x];
                else
                    for (uint32_t x = 0; x < R; ++x) dst[x] += af * src[x];
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
        for (int64_t j = 0; j < (int64_t)n_nxt; ++j)
            next[ix_n[j]] = beam_rescale(next[ix_n[j]], nmax);
        log_scale += log(nmax);

        /*
         * Position `i`'s term: the log ratio of the frontier masses either
         * side of this layer. The two rescalings just applied are exactly the
         * difference between the stored vectors and the true masses, so adding
         * them back recovers a quantity that depends on the model alone:
         *
         *     log(m_{i+1}/m_i) = wmax + log nmax + log sum(next) - log sum(cur)
         *
         * Summed serially in index order, not by a reduction, so the terms do
         * not shift with the thread count -- the total must not move and
         * neither should its decomposition. One pass over the frontier against
         * the scatter's `A` passes, so it is not where the time goes.
         */
        if (by_pos || sink_out) {
            double snext = 0.0;
            for (uint32_t j = 0; j < n_nxt; ++j) snext += next[ix_n[j]];
            if (by_pos)
                by_pos[i] = (snext > 0.0 && msum > 0.0)
                          ? wmax + log(nmax) + log(snext) - log(msum)
                          : -INFINITY;
            msum = snext;
        }

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
    {
        const double a0 = expo ? expo[0] : 1.0;
        if (a0 != 1.0)
            for (uint32_t j = 0; j < n_cur; ++j) tail[ix[j]] *= a0;
    }
    for (uint32_t f = 1; f < k; ++f) {
        const double af = expo ? expo[f] : 1.0;
        beam_tail_weights(models[f], len, tt);
        if (af == 1.0)
            for (uint32_t j = 0; j < n_cur; ++j) tail[ix[j]] += tt[ix[j]];
        else
            for (uint32_t j = 0; j < n_cur; ++j) tail[ix[j]] += af * tt[ix[j]];
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
        /* The sink carries one more edge than any position does, so it is not
         * commensurable with them and is reported on its own. Same gauge: a
         * ratio against the frontier mass it consumes. */
        if (sink_out)
            *sink_out = (acc > 0.0 && msum > 0.0)
                      ? tmax + log(acc) - log(msum) : -INFINITY;
    }
    free(w); free(wt); free(cur); free(next); free(tail); free(tt);
    free(am); free(am_n); free(ix); free(ix_n);
    for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
    free(h); free((void *)L);
    return BEAM_OK;
dead:
    free(w); free(wt); free(cur); free(next); free(tail); free(tt);
    free(am); free(am_n); free(ix); free(ix_n);
    for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
    free(h); free((void *)L);
    *out = -INFINITY;                 /* an answer: the supports never meet */
    /*
     * And the decomposition goes with it. Reaching here part-way through the
     * sweep leaves `by_pos` holding finite terms for the layers already
     * walked and `const_out` holding a real constant, against a total of
     * -INFINITY. The sum identity survives that -- the -INFINITY sink absorbs
     * it -- so no test caught it, but a caller would be handed a plottable
     * profile for a length that has no mass, which is the header's contract
     * broken in the direction that misleads.
     *
     * GUARDED, NOT TESTED: under the hierarchical prior every member gives
     * every string positive mass, so two finalized models' supports always
     * meet and this path is not reachable from the Python API. No test is
     * claimed for it.
     */
    if (by_pos) for (uint32_t i = 0; i < len; ++i) by_pos[i] = -INFINITY;
    if (sink_out) *sink_out = -INFINITY;
    if (const_out) *const_out = -INFINITY;
    return BEAM_OK;
}


BeamError beam_mixture_log_inner_product(const BeamModel *const *models,
                                         uint32_t k, uint32_t len,
                                         double *out) {
    if (!out) return BEAM_ENULL("output buffer");
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    return power_sweep(models, NULL, k, len, out, NULL, NULL, NULL);
}


BeamError beam_mixture_log_power_product(const BeamModel *const *models,
                                         const double *expo, uint32_t k,
                                         uint32_t len, double *out) {
    if (!out) return BEAM_ENULL("output buffer");
    if (!expo) return beam_mixture_log_inner_product(models, k, len, out);
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    const BeamError bad = check_exponents(expo, k);
    if (bad != BEAM_OK) return bad;

    /* Shape is checked on EVERY member, including the ones about to be
     * dropped: an unfinalized model handed in with exponent 0 is a mistake
     * worth refusing, not a member that happens not to matter. */
    const BeamModel **mm = (const BeamModel **)malloc((size_t)k * sizeof(*mm));
    double *aa = (double *)malloc((size_t)k * sizeof(double));
    if (!mm || !aa) {
        free((void *)mm); free(aa);
        return BEAM_EALLOC("the compacted member list");
    }
    const uint32_t kp = compact_members(models, expo, k, mm, aa);
    const BeamError e = power_sweep(mm, aa, kp, len, out, NULL, NULL, NULL);
    free((void *)mm); free(aa);
    return e;
}


/**
 * The by-length driver. Shape checked, exponents validated and compacted.
 */
static BeamError by_length_core(
        const BeamModel *const *models, const double *expo, uint32_t k,
        double *out_by_length, double *out_total) {
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
    const uint64_t per = product_sweep_bytes(m0->n_states, m0->n_residues, k);
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
        const BeamError e = power_sweep(models, expo, k, work[j], &v, NULL, NULL, NULL);
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


BeamError beam_mixture_log_inner_product_by_length(
        const BeamModel *const *models, uint32_t k,
        double *out_by_length, double *out_total) {
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    return by_length_core(models, NULL, k, out_by_length, out_total);
}


BeamError beam_mixture_log_power_product_by_length(
        const BeamModel *const *models, const double *expo, uint32_t k,
        double *out_by_length, double *out_total) {
    if (!expo)
        return beam_mixture_log_inner_product_by_length(models, k,
                                                        out_by_length,
                                                        out_total);
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    const BeamError bad = check_exponents(expo, k);
    if (bad != BEAM_OK) return bad;

    const BeamModel **mm = (const BeamModel **)malloc((size_t)k * sizeof(*mm));
    double *aa = (double *)malloc((size_t)k * sizeof(double));
    if (!mm || !aa) {
        free((void *)mm); free(aa);
        return BEAM_EALLOC("the compacted member list");
    }
    const uint32_t kp = compact_members(models, expo, k, mm, aa);
    const BeamError e = by_length_core(mm, aa, kp, out_by_length, out_total);
    free((void *)mm); free(aa);
    return e;
}


BeamError beam_mixture_log_power_product_by_position(
        const BeamModel *const *models, const double *expo, uint32_t k,
        uint32_t len, double *out_by_position, double *out_sink,
        double *out_const, double *out_total) {
    if (!out_total) return BEAM_ENULL("output buffer");
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    if (!expo)
        return power_sweep(models, NULL, k, len, out_total, out_by_position,
                           out_sink, out_const);
    const BeamError bad = check_exponents(expo, k);
    if (bad != BEAM_OK) return bad;

    const BeamModel **mm = (const BeamModel **)malloc((size_t)k * sizeof(*mm));
    double *aa = (double *)malloc((size_t)k * sizeof(double));
    if (!mm || !aa) {
        free((void *)mm); free(aa);
        return BEAM_EALLOC("the compacted member list");
    }
    const uint32_t kp = compact_members(models, expo, k, mm, aa);
    const BeamError e = power_sweep(mm, aa, kp, len, out_total,
                                    out_by_position, out_sink, out_const);
    free((void *)mm); free(aa);
    return e;
}


/**
 * Every pairwise log inner product at once, ONE PAIR PER THREAD.
 *
 * `out` receives `k * k` doubles, row-major, with `out[i * k + j]` the natural
 * log of `<P_i, P_j>`. The matrix is symmetric by construction: one triangle
 * is computed and mirrored, because the sweep sums the members' edge weights
 * in the order it is given them and computing both triangles independently
 * could differ in the last bit.
 *
 * ## Why the parallelism is here and not inside the sweep
 *
 * A single inner product parallelises over the ~25 lengths, and that axis runs
 * out almost immediately. Measured on donor-sized models at amino-acid order
 * 4: 110 ms of core time per pair at one thread, 107 ms at four -- and 399 ms
 * at sixteen, which is 3.6x the work to get 4.4x the wall clock. Past four
 * threads the length loop is buying nothing and paying for it.
 *
 * Pairs are a far better axis: there are `k(k+1)/2` of them, they share
 * nothing, and each is the same size. So this driver hands one pair to each
 * thread and lets the sweep inside run serial, which is what OpenMP does on
 * its own -- nested parallelism is off by default, so the inner regions
 * collapse when they open inside this one.
 *
 * That distinction is the difference between a Gram matrix that scales with
 * the machine and one that stops at four cores. For 512 members it is 131,328
 * sweeps: about four hours one pair at a time, about fifteen minutes with the
 * cores of an ordinary host behind it.
 *
 * ## What it costs
 *
 * `k(k+1)/2` sweeps, and no less. `sum_i w_i P_i(s)` does not factor over
 * edges, so there is no reformulation that avoids the pairs; carrying a `k x k`
 * accumulator per trellis state would fuse them into one sweep with exactly
 * the same operation count and a state vector nobody can afford. Budget the
 * time from a single `beam_mixture_log_inner_product` on your own models.
 */
static BeamError gram_core(const BeamModel *const *models, uint32_t k,
                           double a, double *out) {
    if (!out) return BEAM_ENULL("output buffer");
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    if (!(isfinite(a) && a > 0.0))
        return BEAM_ERANGE("%s: the per-member exponent is %g; it must be "
                           "finite and strictly positive, since an exponent "
                           "of 0 would drop both members of every pair",
                           __func__, a);

    const BeamModel *m0 = models[0];
    /*
     * Refused here, on the calling thread, before any worker opens. A sweep
     * that refuses inside the region reaches this function as a bare code with
     * its message lost, `beam_set_error` being thread-local -- the same reason
     * every by-length driver in the library checks its budget up front.
     */
    /* Pairs: the Gram driver hands `beam_mixture_log_inner_product_by_length`
     * two models at a time, so a sweep here holds two handles, not k. */
    const uint64_t per = product_sweep_bytes(m0->n_states, m0->n_residues, 2u);
    const uint32_t conc = beam_budget_concurrency(
        per, k ? (uint32_t)((uint64_t)k * (k + 1u) / 2u) : 1u);
    if (conc == 0u)
        return beam_budget_refuse("a mixture inner-product sweep", 0u, per);

    int failed = BEAM_OK;
    const int64_t n_pairs = (int64_t)k * ((int64_t)k + 1) / 2;
    /*
     * By PAIR, not by row. Rows were the obvious unit and they are the wrong
     * one: row 0 holds `k` pairs and the last holds 1, so the longest row is
     * the critical path however many threads are free. At 16 members that
     * capped the speedup at 1.67x on 16 threads -- measured, before this was
     * changed -- because one thread was still walking 16 sweeps while the rest
     * had finished. Every pair is the same size, so handing them out
     * individually is what actually fills the machine.
     *
     * The flat index is inverted by binary search rather than by the closed
     * form, which needs a square root and starts disagreeing with itself near
     * the row boundaries once `k` is large. Ten comparisons against a 110 ms
     * sweep is not a cost worth being clever about.
     */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:failed) \
            num_threads((int)conc)
#endif
    for (int64_t idx = 0; idx < n_pairs; ++idx) {
        /* Pairs before row i number `i*k - i*(i-1)/2`, which rises with i. */
        int64_t lo = 0, hi = (int64_t)k - 1, i = 0;
        while (lo <= hi) {
            const int64_t mid = (lo + hi) / 2;
            const int64_t off = mid * (int64_t)k - mid * (mid - 1) / 2;
            if (off <= idx) { i = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        const int64_t off_i = i * (int64_t)k - i * (i - 1) / 2;
        const int64_t j = i + (idx - off_i);

        const BeamModel *pair[2] = {models[i], models[j]};
        const double aa[2] = {a, a};
        double v = 0.0;
        const BeamError e = by_length_core(pair, a == 1.0 ? NULL : aa, 2u,
                                           NULL, &v);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; v = 0.0; }
        out[(size_t)i * k + (size_t)j] = v;
        out[(size_t)j * k + (size_t)i] = v;
    }
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                         "mixture gram: a pairwise sweep failed");
    return BEAM_OK;
}


BeamError beam_mixture_gram(const BeamModel *const *models, uint32_t k,
                            double *out) {
    return gram_core(models, k, 1.0, out);
}


/**
 * Every pairwise `ln sum_s (P_i P_j)^a` at once, one pair per thread.
 *
 * `a = 1/2` is the pairwise Bhattacharyya coefficient, which is what this
 * exists for: comparing every repertoire in a cohort against every other
 * needs `k(k+1)/2` sweeps whichever exponent is wanted, and the pairs are the
 * axis that scales. See `beam_mixture_gram` for why the parallelism sits here.
 *
 * The DIAGONAL is `ln sum_s P_i(s)^{2a}`, which at `a = 1/2` is exactly
 * `ln 1 = 0` and is therefore a free correctness check on every row.
 */
BeamError beam_mixture_power_gram(const BeamModel *const *models, uint32_t k,
                                  double a, double *out) {
    return gram_core(models, k, a, out);
}


/**
 * `ln sum_s (P_i P_j)^a` for a NAMED LIST of pairs, one pair per thread.
 *
 * The Gram matrix is the wrong shape for a cohort. Its cost is `k(k+1)/2`
 * sweeps, so doubling the cohort quadruples the bill, and a study that needs
 * a thinning null plus a sample of between-donor pairs wants a few hundred
 * specific pairs out of the tens of thousands available. Naming them costs
 * `n_pairs` sweeps and nothing else, which is what makes a cohort of hundreds
 * affordable at all.
 *
 * `pairs` holds `2 * n_pairs` member indices, `i` then `j`; `out` receives
 * `n_pairs` values in the same order. A pair may name one member twice, which
 * gives `ln sum_s P_i(s)^{2a}` -- exactly `ln 1 = 0` at `a = 1/2`, and so a
 * free check that the caller can put anywhere in the list.
 *
 * The parallelism is here for the same reason it is in `beam_mixture_gram`:
 * one sweep stops scaling at about four threads and the pairs are independent
 * and equal-sized. Nested parallelism being off, the sweeps inside collapse
 * to serial on their own.
 */
BeamError beam_mixture_power_pairs(const BeamModel *const *models, uint32_t k,
                                   double a, const uint32_t *pairs,
                                   uint64_t n_pairs, double *out) {
    if (!out) return BEAM_ENULL("output buffer");
    if (!pairs && n_pairs) return BEAM_ENULL("the pair list");
    const BeamError shape = beam_mixture_check_shape(models, k);
    if (shape != BEAM_OK) return shape;
    if (!(isfinite(a) && a > 0.0))
        return BEAM_ERANGE("%s: the per-member exponent is %g; it must be "
                           "finite and strictly positive", __func__, a);

    /* Every index is checked BEFORE any sweep opens. An out-of-range member
     * read inside the parallel region is a wild pointer, not an error code. */
    for (uint64_t p = 0; p < n_pairs; ++p)
        for (uint32_t t = 0; t < 2u; ++t)
            if (pairs[2 * p + t] >= k)
                return BEAM_ERANGE("%s: pair %llu names member %u, which is "
                                   "outside 0..%u", __func__,
                                   (unsigned long long)p, pairs[2 * p + t],
                                   k - 1u);
    if (n_pairs == 0) return BEAM_OK;

    const BeamModel *m0 = models[0];
    const uint64_t per = product_sweep_bytes(m0->n_states, m0->n_residues, 2u);
    const uint32_t conc = beam_budget_concurrency(
        per, n_pairs > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)n_pairs);
    if (conc == 0u)
        return beam_budget_refuse("a mixture inner-product sweep", 0u, per);

    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads((int)conc)
#endif
    for (int64_t p = 0; p < (int64_t)n_pairs; ++p) {
        const BeamModel *pair[2] = {models[pairs[2 * p]],
                                    models[pairs[2 * p + 1]]};
        const double aa[2] = {a, a};
        double v = 0.0;
        const BeamError e = by_length_core(pair, a == 1.0 ? NULL : aa, 2u,
                                           NULL, &v);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; v = 0.0; }
        out[p] = v;
    }
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                         "mixture power pairs: a pairwise sweep failed");
    return BEAM_OK;
}
