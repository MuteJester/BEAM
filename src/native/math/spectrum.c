/**
 * @file spectrum.c
 * @brief Surprisal spectrum: grid transport and the sweeps over it.
 */
#include "beam/internal.h"


/*
 * This file used to carry its own `trim`, computing BACKWARD liveness into a
 * byte per state per layer. It is gone: `beam_live_acquire` in liveness.c
 * answers the same question, intersects it with forward reachability so the
 * set is the small one, and caches the result for every other analytic
 * instead of rebuilding it per spectrum.
 */

/* ── Surprisal spectrum ──────────────────────────────────────── */

/**
 * Occupied bin range of one node's histogram, as a half-open interval.
 *
 * Carrying this is what makes the spectrum DP affordable. Early layers occupy
 * a handful of bins out of a thousand, and without bounds every edge would
 * sweep the whole grid: the production FlashBack transport
 * (`lib/flashback/graph_analytics.c:pseq_grid_transport`) keeps the same
 * bookkeeping for the same reason.
 */
typedef struct { uint32_t lo, hi; } BinRange;

/**
 * Add `src` shifted up by `u` bins into `dst`, splitting linearly between the
 * two adjacent bins so total mass is conserved and the mean is unbiased.
 * Source bins that fall outside the grid contribute nothing; that must hold at
 * BOTH ends, because the sink shift can be negative.
 *
 * `sr` bounds the source's occupied bins; `dr` is widened to cover whatever
 * this call writes.
 */
static void transport(const double *src, double *dst, uint32_t bins,
                      double u, double factor, BinRange sr, BinRange *dr) {
    const int64_t k = (int64_t)floor(u);
    const double  f = u - (double)k;
    const double  g = 1.0 - f;
    for (uint32_t b = sr.lo; b < sr.hi; ++b) {
        const double v = src[b];
        if (v == 0.0) continue;
        const int64_t lo = (int64_t)b + k;
        if (lo >= 0 && lo < (int64_t)bins) {
            dst[lo] += v * g * factor;
            if ((uint32_t)lo < dr->lo) dr->lo = (uint32_t)lo;
            if ((uint32_t)lo + 1u > dr->hi) dr->hi = (uint32_t)lo + 1u;
        }
        const int64_t hi = lo + 1;
        if (hi >= 0 && hi < (int64_t)bins) {
            dst[hi] += v * f * factor;
            if ((uint32_t)hi < dr->lo) dr->lo = (uint32_t)hi;
            if ((uint32_t)hi + 1u > dr->hi) dr->hi = (uint32_t)hi + 1u;
        }
    }
}

/**
 * Add `src` shifted up by `u` bins into `dst` for TWO measures at once,
 * splitting linearly between the two adjacent bins exactly as `transport`
 * does for one.
 *
 * The point is not to save a pass over the source bins — it is that the shift
 * `u`, and therefore the integer bin `k` and the split fractions `f` and `g`,
 * are computed ONCE and used for both. Two separate transports would compute
 * them twice from the same inputs and get the same answer today, but nothing
 * would hold them together, and the ratio of the two measures per bin is only
 * meaningful if the mass was split identically. That ratio is the exact
 * mass-weighted mean probability of the sequences in the bin, which is what
 * the discovery-curve arithmetic is built on.
 */
static void transport2(const double *src_a, const double *src_b,
                       double *dst_a, double *dst_b, uint32_t bins,
                       double u, double factor_a, double factor_b,
                       BinRange sr, BinRange *dr) {
    const int64_t k = (int64_t)floor(u);
    const double  f = u - (double)k;
    const double  g = 1.0 - f;
    for (uint32_t b = sr.lo; b < sr.hi; ++b) {
        const double va = src_a[b], vb = src_b[b];
        if (va == 0.0 && vb == 0.0) continue;
        const int64_t lo = (int64_t)b + k;
        if (lo >= 0 && lo < (int64_t)bins) {
            dst_a[lo] += va * g * factor_a;
            dst_b[lo] += vb * g * factor_b;
            if ((uint32_t)lo < dr->lo) dr->lo = (uint32_t)lo;
            if ((uint32_t)lo + 1u > dr->hi) dr->hi = (uint32_t)lo + 1u;
        }
        const int64_t hi = lo + 1;
        if (hi >= 0 && hi < (int64_t)bins) {
            dst_a[hi] += va * f * factor_a;
            dst_b[hi] += vb * f * factor_b;
            if ((uint32_t)hi < dr->lo) dr->lo = (uint32_t)hi;
            if ((uint32_t)hi + 1u > dr->hi) dr->hi = (uint32_t)hi + 1u;
        }
    }
}

/**
 * One length's surprisal spectrum, carrying either one measure or both.
 *
 * `out_b` NULL means one measure, governed by `counting`. Non-NULL means both
 * are carried through the same transport: `out_a` receives the counting
 * measure and `out_b` the generated one, and `counting` is ignored. Both
 * accumulate into whatever `out_a` / `out_b` already hold, so pooling over
 * lengths is a matter of calling this repeatedly.
 */
static BeamError spectrum_one(const BeamModel *m, bool counting,
                             uint32_t len, uint32_t bins, double ds,
                             double *out_a, double *out_b) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t nm = out_b ? 2u : 1u;      /* measures carried per state */

    /*
     * THE HISTOGRAMS ARE INDEXED BY LIVE SLOT, NOT BY STATE, and that is the
     * whole of this function's memory footprint.
     *
     * A per-state grid is `n_states * measures * bins` doubles: 796 MB at
     * order 4 with 512 bins, twice over, which is why `spectra_by_length` has
     * to cap its concurrency on a memory budget instead of on cores. The live
     * set is nothing like that big -- on a 101,373-sequence repertoire the
     * widest layer of a length-15 trellis holds 11,099 states of 194,481, and
     * the mean layer holds 3,502 -- so sizing the grids to the widest layer
     * instead of to the state space is a 17x cut.
     *
     * The frontier also replaces this file's own `trim`. That pass computed
     * BACKWARD liveness only, which on a smoothed model is very nearly every
     * state; `beam_live_acquire` intersects it with forward reachability and
     * caches the result for every other analytic. Restricting to it is
     * bit-identical: a state the walk cannot reach carries no mass, so the
     * terms dropped are exactly zero.
     */
    LiveHandle lh;
    BeamError err = beam_live_acquire(m, len, &lh);
    if (err != BEAM_OK) return err;
    const BeamLive *L = lh.L;

    uint32_t maxlive = 0;
    for (uint32_t i = 0; i <= len; ++i)
        if (L->n_live[i] > maxlive) maxlive = L->n_live[i];
    if (maxlive == 0) { beam_live_release(&lh); return BEAM_OK; }

    const size_t hist = (size_t)nm * bins;    /* doubles per live slot */
    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    double *cur  = (double *)calloc((size_t)maxlive * hist, sizeof(double));
    double *next = (double *)calloc((size_t)maxlive * hist, sizeof(double));
    BinRange *rc = (BinRange *)malloc((size_t)maxlive * sizeof(BinRange));
    BinRange *rn = (BinRange *)malloc((size_t)maxlive * sizeof(BinRange));
    /* state -> slot within its own layer's live list, one map per side */
    uint32_t *sc = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t *sn = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint8_t  *av = (uint8_t *)malloc((size_t)S);   /* byte mask for the build */
    if (!w || !tail || !cur || !next || !rc || !rn || !sc || !sn || !av) {
        free(w); free(tail); free(cur); free(next); free(rc); free(rn);
        free(sc); free(sn); free(av);
        beam_live_release(&lh);
        return BEAM_EALLOC("the per-state surprisal histograms");
    }

    for (uint32_t k = 0; k < maxlive; ++k) { rc[k].lo = bins; rc[k].hi = 0; }
    /* Layer 0 is the root alone, so its live list has one entry and the walk
     * starts in that slot. */
    for (uint32_t k = 0; k < L->n_live[0]; ++k) sc[L->live[0][k]] = k;
    {
        const uint32_t root = S - 1u;
        uint32_t rk = 0;
        bool have = false;
        for (uint32_t k = 0; k < L->n_live[0]; ++k)
            if (L->live[0][k] == root) { rk = k; have = true; break; }
        if (!have) {          /* the root cannot reach a sink: no paths */
            free(w); free(tail); free(cur); free(next); free(rc); free(rn);
            free(sc); free(sn); free(av);
            beam_live_release(&lh);
            return BEAM_OK;
        }
        for (uint32_t mi = 0; mi < nm; ++mi)
            cur[(size_t)rk * hist + (size_t)mi * bins + 0] = 1.0;
        rc[rk].lo = 0; rc[rk].hi = 1;
    }

    for (uint32_t i = 0; i < len; ++i) {
        /* Pass the mask this function already computed. The transport loop
         * below tests `av_cur[s]` / `av_next[tt]` before reading a row, so the
         * "dead rows are never read" contract holds and skipping them is
         * bit-identical -- it was simply never wired up. Worth 1.36x on a
         * per-repertoire spectrum. */
        /* `beam_layer_weights_live` takes a byte per state; the frontier
         * stores membership as an ascending list, so paint the mask from it.
         * A memset over the state space per layer is far less than building
         * weights for states no path reaches. */
        memset(av, 0, S);
        for (uint32_t k = 0; k < L->n_live[i]; ++k) av[L->live[i][k]] = 1;
        beam_layer_weights_live(m, i, len, w, NULL, av);
        /*
         * Accumulate by TARGET rather than by source, for the same reason the
         * forward sweeps do: 21 sources land on every target, so a
         * source-major scatter races. Gathering makes each target histogram
         * one thread's private property.
         *
         * It is also bit-identical rather than merely equivalent. The sources
         * of target `rest * A + x` are `a * state_shift + rest` for ascending
         * `a`, which is ascending in the source state too, so the gather adds
         * them in exactly the order the scatter did.
         *
         * This is the single most expensive loop in the library — a whole
         * histogram is moved per edge, not a number — and it was the one loop
         * of its kind still running on one core.
         */
        const uint64_t *mk_cur = L->mask[i];
        const uint32_t *lv_next = L->live[i + 1];
        const uint32_t  n_next  = L->n_live[i + 1];
        for (uint32_t k = 0; k < n_next; ++k) sn[lv_next[k]] = k;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(beam_threads())
#endif
        for (int64_t kk = 0; kk < (int64_t)n_next; ++kk) {
            rn[kk].lo = bins; rn[kk].hi = 0;
            const uint32_t tt = lv_next[kk];
            const uint32_t x = tt % A;
            if (x >= R) continue;
            const uint32_t rest = tt / A;
            double *dst = next + (size_t)kk * hist;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t s = a * SK + rest;
                if (!live_bit_get(mk_cur, s)) continue;
                const uint32_t ks = sc[s];
                if (rc[ks].lo >= rc[ks].hi) continue;   /* no mass here */
                const double wv = w[(size_t)s * R + x];
                if (wv == -INFINITY) continue;
                const double *src = cur + (size_t)ks * hist;
                if (nm == 1u) {
                    transport(src, dst, bins, -wv / ds,
                              counting ? 1.0 : exp(wv), rc[ks], &rn[kk]);
                } else {
                    transport2(src, src + bins, dst, dst + bins, bins,
                               -wv / ds, 1.0, exp(wv), rc[ks], &rn[kk]);
                }
            }
        }
        /* Clear only what was written, so the sweep costs occupied bins
         * rather than the whole grid. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)L->n_live[i]; ++k) {
            if (rc[k].lo >= rc[k].hi) continue;
            for (uint32_t mi = 0; mi < nm; ++mi)
                memset(cur + (size_t)k * hist + (size_t)mi * bins + rc[k].lo, 0,
                       (size_t)(rc[k].hi - rc[k].lo) * sizeof(double));
        }
        { double *t = cur; cur = next; next = t; }
        { BinRange *t = rc; rc = rn; rn = t; }
        { uint32_t *t = sc; sc = sn; sn = t; }
    }

    /* Sink: the backward tail, plus the per-length constant that turns the
     * accumulated weight into a surprisal. */
    beam_tail_weights(m, len, tail);
    {
        const double off = m->log_z[len] - m->log_pi[len];
        BinRange sink = { bins, 0 };
        for (uint32_t k = 0; k < L->n_live[len]; ++k) {
            const uint32_t s = L->live[len][k];
            if (tail[s] == -INFINITY || rc[k].lo >= rc[k].hi) continue;
            const double *src = cur + (size_t)k * hist;
            const double u = (-tail[s] + off) / ds;
            if (nm == 1u) {
                transport(src, out_a, bins, u,
                          counting ? 1.0 : exp(tail[s] - off), rc[k], &sink);
            } else {
                transport2(src, src + bins, out_a, out_b, bins, u,
                           1.0, exp(tail[s] - off), rc[k], &sink);
            }
        }
    }

    free(w); free(tail); free(cur); free(next); free(rc); free(rn);
    free(sc); free(sn); free(av);
    beam_live_release(&lh);
    return BEAM_OK;
}

/**
 * Grid spacing for a spectrum, sized automatically when `spacing` is not
 * positive. Shared by every spectrum entry point so they cannot disagree
 * about what grid they are on — which for the paired and per-length forms is
 * not a tidiness point but the property that makes their outputs comparable.
 */
static BeamError spectrum_spacing(const BeamModel *m, uint32_t bins,
                                 double spacing, double *out_ds) {
    if (bins < 2)
        return BEAM_ERANGE("%s: a spectrum needs at least 2 bins, got %u",
                               __func__, (unsigned)bins);
    if (spacing > 0.0) { *out_ds = spacing; return BEAM_OK; }

    const double hi = beam_model_accumulator_max(m);
    if (!(hi > 0.0))
        return BEAM_ENOPATHS("the accumulator bound is zero, so the "
                                 "model generates nothing to bin");
    /*
     * Headroom above the exact bound, and it has to scale with length.
     *
     * Linear splitting is a smoothing operation: each layer can push a sliver
     * of mass one bin past where the exact arithmetic would put it, so after L
     * layers plus the sink the occupied range can run up to L+1 bins beyond
     * `hi`. Reserving a fixed handful of bins is enough for short sequences
     * and silently truncates long ones.
     */
    const uint32_t reserve = m->max_len + 9u;
    if (bins <= reserve)
        return BEAM_ERANGE("%s: %u bins is not enough to size a grid "
                               "automatically at max_length %u; %u are "
                               "reserved for splitting headroom, or pass an "
                               "explicit spacing", __func__, (unsigned)bins,
                               (unsigned)m->max_len, (unsigned)reserve);
    *out_ds = hi * 1.02 / (double)(bins - reserve);
    return BEAM_OK;
}

BeamError beam_model_spectrum(const BeamModel *m, bool counting,
                                 uint32_t length, uint32_t bins, double spacing,
                                 double *weights, double *out_spacing) {
    if (!m || !weights) return BEAM_ENULL(!m ? "model" : "weights buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    double ds = 0.0;
    BeamError e = spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(weights, 0, (size_t)bins * sizeof(double));
    if (length != 0) {
        if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
        return spectrum_one(m, counting, length, bins, ds, weights, NULL);
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        e = spectrum_one(m, counting, L, bins, ds, weights, NULL);
        if (e != BEAM_OK) return e;
    }
    return BEAM_OK;
}

BeamError beam_model_spectrum_pair(const BeamModel *m, uint32_t length,
                                      uint32_t bins, double spacing,
                                      double *counting_weights,
                                      double *generated_weights,
                                      double *out_spacing) {
    if (!m || !counting_weights || !generated_weights)
        return BEAM_ENULL(!m ? "model"
                                 : (!counting_weights ? "counting weights"
                                                      : "generated weights"));
    if (!m->finalized) return BEAM_ENOTBUILT();

    double ds = 0.0;
    BeamError e = spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(counting_weights,  0, (size_t)bins * sizeof(double));
    memset(generated_weights, 0, (size_t)bins * sizeof(double));
    if (length != 0) {
        if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
        return spectrum_one(m, false, length, bins, ds,
                            counting_weights, generated_weights);
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        e = spectrum_one(m, false, L, bins, ds,
                         counting_weights, generated_weights);
        if (e != BEAM_OK) return e;
    }
    return BEAM_OK;
}

/**
 * Scratch one length of the spectrum DP needs, in bytes.
 *
 * Dominated by the two per-state histogram grids, which at order 4 with 512
 * bins are 800 MB each. That is why the per-length form caps its concurrency
 * by memory rather than by core count: sixteen lengths at once would ask for
 * twenty-five gigabytes.
 */
static uint64_t spectrum_bytes(const BeamModel *m, uint32_t bins,
                               uint32_t nm, uint32_t maxlive) {
    const uint64_t S = m->n_states;
    if (maxlive == 0u || maxlive > S) maxlive = (uint32_t)S;
    return 2ull * maxlive * nm * bins * sizeof(double)  /* cur + next   */
         + 2ull * maxlive * sizeof(BinRange)
         + S * BEAM_RESIDUES * sizeof(double)           /* edge weights */
         + S * sizeof(double)                           /* tail         */
         + 2ull * S * sizeof(uint32_t)                  /* slot maps    */
         + S;                                           /* byte mask    */
}

BeamError beam_model_spectra_by_length(const BeamModel *m, bool counting,
                                          uint32_t max_length, uint32_t bins,
                                          double spacing, double *weights,
                                          double *out_spacing) {
    if (!m || !weights) return BEAM_ENULL(!m ? "model" : "weights buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (max_length > m->max_len)
        return BEAM_ERANGE("%s: length %u is beyond this model's "
                               "max_length of %u", __func__,
                               (unsigned)max_length, (unsigned)m->max_len);
    if (max_length == 0) max_length = m->max_len;

    double ds = 0.0;
    BeamError e = spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(weights, 0, (size_t)(max_length + 1u) * bins * sizeof(double));

    uint32_t *work = (uint32_t *)malloc((size_t)(max_length + 1u) * sizeof(uint32_t));
    if (!work) return BEAM_EALLOC("the per-length work list");
    uint32_t nw = 0;
    for (uint32_t L = max_length + 1u; L-- > 1; )
        if (beam_length_live(m, L)) work[nw++] = L;
    if (nw == 0) { free(work); return BEAM_OK; }

    /*
     * Which axis to parallelise, decided by MEMORY.
     *
     * Each length needs its own pair of per-state histogram grids, and at
     * order 4 with 512 bins that is 1.6 GB apiece: sixteen at once would ask
     * for twenty-five gigabytes and find the swap instead of the speed-up.
     *
     * When the budget cannot hold one length per thread, the lengths run one
     * at a time and each gets every thread INSIDE its own sweeps, which are
     * parallel in their own right. That is not a compromise, it is the faster
     * of the two: OpenMP does not nest by default, so an outer loop of two
     * threads would put the whole job on two cores. Running the lengths
     * concurrently is worth it only when every thread can have one.
     *
     * Either way the answers are identical — the rows are written
     * independently — so this decides only speed and peak memory.
     */
    /*
     * Sized from the live sets, not from the state space. The grids are now
     * `maxlive * bins` rather than `n_states * bins`, and on a per-repertoire
     * model that is a seventeenth of the memory -- which is the difference
     * between running the lengths one at a time and running one per core.
     *
     * Measuring it means acquiring each length's frontier, which is not extra
     * work: every sweep below acquires the same frontier, and the cache hands
     * back the one built here. The widest layer is NOT the longest length --
     * on a real repertoire length 15 peaks at 11,099 live states and length 21
     * at 289 -- so this takes the max over the work list rather than guessing
     * from one.
     */
    uint32_t maxlive = 0;
    for (uint32_t k = 0; k < nw; ++k) {
        LiveHandle h;
        if (beam_live_acquire(m, work[k], &h) != BEAM_OK) continue;
        for (uint32_t i = 0; i <= h.L->len; ++i)
            if (h.L->n_live[i] > maxlive) maxlive = h.L->n_live[i];
        beam_live_release(&h);
    }

    const uint64_t need = spectrum_bytes(m, bins, 1u, maxlive);
    const int th = beam_threads();
    uint32_t conc = (uint32_t)(need ? beam_model_memory_budget() / need
                                    : (uint64_t)nw);
    if (conc > nw) conc = nw;
    if (th > 0 && conc > (uint32_t)th) conc = (uint32_t)th;
    if (th > 1 && conc < (uint32_t)th && conc < nw) conc = 1u;
    if (conc < 1u) conc = 1u;

    int failed = BEAM_OK;
    if (conc <= 1u) {
        /* No enclosing parallel region at all: one with a single thread is
         * still an ACTIVE region, and would serialise every sweep inside it. */
        for (uint32_t k = 0; k < nw; ++k) {
            const BeamError err = spectrum_one(m, counting, work[k], bins, ds,
                                              weights + (size_t)work[k] * bins,
                                              NULL);
            if (err != BEAM_OK && (int)err > failed) failed = (int)err;
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
                num_threads((int)conc)
#endif
        for (int64_t k = 0; k < (int64_t)nw; ++k) {
            const uint32_t L = work[k];
            const BeamError err = spectrum_one(m, counting, L, bins, ds,
                                              weights + (size_t)L * bins, NULL);
            if (err != BEAM_OK) { if ((int)err > failed) failed = (int)err; }
        }
    }
    free(work);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                        "spectra_by_length: a per-length sweep failed");
    return BEAM_OK;
}
