/**
 * @file spectrum.c
 * @brief Surprisal spectrum: grid transport and the sweeps over it.
 */
#include "beam/internal.h"


/* ── Live-state trimming ─────────────────────────────────────── */

/**
 * `alive[i][s]` is true iff a live path runs from state `s` at layer `i` to a
 * live sink.
 *
 * On the models this library builds, that is every state: hierarchical backoff
 * leaves every cell positive, so no transition is dead. The pass is kept
 * anyway, because the alternative is to ASSUME saturation rather than measure
 * it — and because the DPs downstream are written against a trimmed trellis,
 * so a smoother that ever did leave a cell at zero would be handled correctly
 * instead of silently counting sequences the model cannot produce.
 *
 * `alive` must be `len + 1` arrays of `n_states` bytes.
 */
static BeamError trim(const BeamModel *m, uint32_t len, uint8_t **alive) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *t = (double *)malloc((size_t)S * sizeof(double));
    if (!t) return BEAM_EALLOC("the live-state trimming buffers");
    beam_tail_weights(m, len, t);
    for (uint32_t s = 0; s < S; ++s) alive[len][s] = (t[s] != -INFINITY);
    free(t);

    /*
     * When no table cell is -inf, no BEAM is -inf either, so which states are
     * live stops depending on the weights at all and becomes pure
     * reachability. That is the common case — hierarchical backoff leaves
     * every cell positive — and skipping the weights here is what stops
     * `path_count`, `topology` and the spectrum from building every layer
     * twice. The flag is measured at finalize, not assumed; the general path
     * below still runs when it is false.
     */
    if (m->tables_all_finite) {
        for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
            const uint8_t *nb = alive[i + 1];
            uint8_t *cu = alive[i];
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
            for (int64_t s = 0; s < (int64_t)S; ++s) {
                const uint32_t base = ((uint32_t)s % SK) * A;
                uint8_t ok = 0;
                for (uint32_t x = 0; x < R && !ok; ++x) ok = nb[base + x];
                cu[s] = ok;
            }
        }
        return BEAM_OK;
    }

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    if (!w) return BEAM_EALLOC("the live-state trimming buffers");
    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, len, w, NULL);
        const uint8_t *nb = alive[i + 1];
        uint8_t *cu = alive[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) {
            const double *row = w + (size_t)s * R;
            const uint32_t base = ((uint32_t)s % SK) * A;
            uint8_t ok = 0;
            for (uint32_t x = 0; x < R && !ok; ++x)
                ok = (row[x] != -INFINITY) && nb[base + x];
            cu[s] = ok;
        }
    }
    free(w);
    return BEAM_OK;
}


static uint8_t **alloc_alive(uint32_t len, uint32_t n_states) {
    uint8_t **a = (uint8_t **)calloc(len + 1u, sizeof(uint8_t *));
    if (!a) return NULL;
    for (uint32_t i = 0; i <= len; ++i) {
        a[i] = (uint8_t *)calloc(n_states, 1);
        if (!a[i]) {
            for (uint32_t j = 0; j <= i; ++j) free(a[j]);
            free(a);
            return NULL;
        }
    }
    return a;
}

static void free_alive(uint8_t **a, uint32_t len) {
    if (!a) return;
    for (uint32_t i = 0; i <= len; ++i) free(a[i]);
    free(a);
}


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

    uint8_t **alive = alloc_alive(len, S);
    if (!alive) return BEAM_EALLOC("the live-state map");
    BeamError err = trim(m, len, alive);
    if (err != BEAM_OK) { free_alive(alive, len); return err; }

    const size_t hist = (size_t)nm * bins;    /* doubles per state */
    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    double *cur  = (double *)calloc((size_t)S * hist, sizeof(double));
    double *next = (double *)calloc((size_t)S * hist, sizeof(double));
    BinRange *rc = (BinRange *)malloc((size_t)S * sizeof(BinRange));
    BinRange *rn = (BinRange *)malloc((size_t)S * sizeof(BinRange));
    if (!w || !tail || !cur || !next || !rc || !rn) {
        free(w); free(tail); free(cur); free(next); free(rc); free(rn);
        free_alive(alive, len);
        return BEAM_EALLOC("the per-state surprisal histograms");
    }

    for (uint32_t s = 0; s < S; ++s) { rc[s].lo = bins; rc[s].hi = 0; }
    for (uint32_t mi = 0; mi < nm; ++mi)
        cur[(size_t)(S - 1) * hist + (size_t)mi * bins + 0] = 1.0;
    rc[S - 1].lo = 0; rc[S - 1].hi = 1;

    for (uint32_t i = 0; i < len; ++i) {
        /* Pass the mask this function already computed. The transport loop
         * below tests `av_cur[s]` / `av_next[tt]` before reading a row, so the
         * "dead rows are never read" contract holds and skipping them is
         * bit-identical -- it was simply never wired up. Worth 1.36x on a
         * per-repertoire spectrum. */
        beam_layer_weights_live(m, i, len, w, NULL, alive[i]);
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
        const uint8_t *av_cur = alive[i], *av_next = alive[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(beam_threads())
#endif
        for (int64_t tt = 0; tt < (int64_t)S; ++tt) {
            rn[tt].lo = bins; rn[tt].hi = 0;
            const uint32_t x = (uint32_t)tt % A;
            if (x >= R || !av_next[tt]) continue;
            const uint32_t rest = (uint32_t)tt / A;
            double *dst = next + (size_t)tt * hist;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t s = a * SK + rest;
                if (rc[s].lo >= rc[s].hi) continue;    /* no mass here */
                if (!av_cur[s]) continue;
                const double wv = w[(size_t)s * R + x];
                if (wv == -INFINITY) continue;
                const double *src = cur + (size_t)s * hist;
                if (nm == 1u) {
                    transport(src, dst, bins, -wv / ds,
                              counting ? 1.0 : exp(wv), rc[s], &rn[tt]);
                } else {
                    transport2(src, src + bins, dst, dst + bins, bins,
                               -wv / ds, 1.0, exp(wv), rc[s], &rn[tt]);
                }
            }
        }
        /* Clear only what was written, so the sweep costs occupied bins
         * rather than the whole grid. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) {
            if (rc[s].lo >= rc[s].hi) continue;
            for (uint32_t mi = 0; mi < nm; ++mi)
                memset(cur + (size_t)s * hist + (size_t)mi * bins + rc[s].lo, 0,
                       (size_t)(rc[s].hi - rc[s].lo) * sizeof(double));
        }
        { double *t = cur; cur = next; next = t; }
        { BinRange *t = rc; rc = rn; rn = t; }
    }

    /* Sink: the backward tail, plus the per-length constant that turns the
     * accumulated weight into a surprisal. */
    beam_tail_weights(m, len, tail);
    {
        const double off = m->log_z[len] - m->log_pi[len];
        BinRange sink = { bins, 0 };
        for (uint32_t s = 0; s < S; ++s) {
            if (tail[s] == -INFINITY || rc[s].lo >= rc[s].hi) continue;
            const double *src = cur + (size_t)s * hist;
            const double u = (-tail[s] + off) / ds;
            if (nm == 1u) {
                transport(src, out_a, bins, u,
                          counting ? 1.0 : exp(tail[s] - off), rc[s], &sink);
            } else {
                transport2(src, src + bins, out_a, out_b, bins, u,
                           1.0, exp(tail[s] - off), rc[s], &sink);
            }
        }
    }

    free(w); free(tail); free(cur); free(next); free(rc); free(rn);
    free_alive(alive, len);
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
                               uint32_t nm) {
    const uint64_t S = m->n_states;
    return 2ull * S * nm * bins * sizeof(double)      /* cur + next   */
         + S * BEAM_RESIDUES * sizeof(double)     /* edge weights */
         + S * (2ull * sizeof(BinRange) + 2ull * sizeof(double))
         + (uint64_t)(m->max_len + 1u) * S;           /* live-state map */
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
    const uint64_t need = spectrum_bytes(m, bins, 1u);
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
