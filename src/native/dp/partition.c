/**
 * @file partition.c
 * @brief The forward partition function, log Z(L).
 */
#include "beam/internal.h"
#include "beam/mixing.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/** One rest-group of the target-major transition. */
static inline double logz_accumulate_rest(
        uint32_t rest, uint32_t A, uint32_t R, uint32_t SK,
        const uint32_t *off, const uint32_t *by_rest,
        const double *w, const double *cur, double *next, double wmax) {
    const uint32_t lo = off[rest], hi = off[rest + 1u];
    double local_max = 0.0;
    for (uint32_t x = 0; x < R; ++x) {
        double acc = 0.0;
        for (uint32_t j = lo; j < hi; ++j) {
            const uint32_t s = by_rest[j] * SK + rest;
            const double wv = w[(size_t)s * R + x];
            if (wv == -INFINITY) continue;
            acc += cur[s] * exp(wv - wmax);
        }
        next[(size_t)rest * A + x] = acc;
        if (acc > local_max) local_max = acc;
    }
    return local_max;
}


/* ── Partition function ──────────────────────────────────────── */

/**
 * Forward sweep over the trellis, in linear space with per-layer rescaling.
 *
 * Log space would cost a `log` per target state and an `exp` per predecessor;
 * rescaling costs one `exp` per edge and one `log` per layer, which is an
 * order of magnitude cheaper on a 194k-state trellis. Underflow within a layer
 * is harmless: a path 700 nats below the layer's best contributes less than
 * 1e-300 of the total.
 */
BeamError beam_compute_log_z(BeamModel *m, uint32_t len, double *out,
                             double *layer_logmass) {
    const uint32_t A  = m->n_alphabet;
    const uint32_t R  = m->n_residues;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;

    /* Refused before allocating, from the sizing function `beam_model_finalize`
     * also divides the budget by, so the two cannot disagree about what a
     * length costs. */
    const uint64_t need = beam_logz_sweep_bytes(S, R, SK);
    BEAM_BUDGET_OR_REFUSE("the partition-function trellis", len, need);

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)malloc((size_t)S * sizeof(double));
    double *next = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));

    /*
     * The frontier: which states the sweep has actually reached, not just how
     * many. `src` holds them ASCENDING; `by_rest` holds the same set regrouped
     * by `rest = s % SK`, which is the axis the accumulation runs along.
     *
     * See the accumulation loop for why the regrouping is what makes this
     * bit-identical rather than merely equivalent.
     */
    uint32_t *src     = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t *nxt     = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t *by_rest = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t *off     = (uint32_t *)malloc(((size_t)SK + 1u) * sizeof(uint32_t));
    uint32_t *pos     = (uint32_t *)malloc((size_t)SK * sizeof(uint32_t));
    uint32_t *rests   = (uint32_t *)malloc((size_t)SK * sizeof(uint32_t));
    uint8_t  *needed  = (uint8_t *)calloc((size_t)S, sizeof(uint8_t));

#define BEAM_LOGZ_FREE() do {                                                 \
        free(w); free(cur); free(next); free(tail);                           \
        free(src); free(nxt); free(by_rest); free(off); free(pos);             \
        free(rests); free(needed);                                            \
    } while (0)

    if (!w || !cur || !next || !tail || !src || !nxt || !by_rest || !off ||
        !pos || !rests || !needed) {
        BEAM_LOGZ_FREE();
        return BEAM_EALLOC("the partition-function trellis buffers");
    }

    /* `layer_logmass[i]` is the running log of the trellis mass once residue
     * i has been emitted, so the caller differences it to get the per-layer
     * factor that `log Z` is a product of. It telescopes: the last entry IS
     * `*out`, and the differences sum to it exactly. NULL to skip. */
    if (layer_logmass)
        for (uint32_t i = 0; i < len; ++i) layer_logmass[i] = -INFINITY;

    /* The walk starts in the all-sentinel state. */
    for (uint32_t s = 0; s < S; ++s) cur[s] = 0.0;
    cur[S - 1] = 1.0;
    double log_scale = 0.0;
    uint32_t nsrc = 1u;
    src[0] = S - 1u;
    /* How many states the sweep can currently reach, carried from the layer
     * that produced `cur` rather than recounted. Counting it costs one pass
     * over S per layer, which is the very thing the count is used to avoid. */
    uint32_t reach = 1u;
    /*
     * WHICH LAYERS BUILD WEIGHTS THINLY IS NOW A PER-LAYER QUESTION ONLY.
     *
     * It used to be gated on the model's occupancy as well, because the thin
     * path had to FIND the reached states by testing `cur[s] != 0` for all
     * 194,481 of them, and on a model dense enough that most rows were wanted
     * that scan cost more than it saved -- measured at 0.85x on a
     * 200,000-sequence model. The frontier names the reached states outright,
     * so the scan is gone and the gate with it.
     *
     * Removing it is worth 3.15x on a 101,373-sequence model, which failed the
     * old occupancy test and so paid the batched build for a trellis that was
     * 5% reached. Re-measured at the other end of the range: on the 927M
     * foundation, thin and batched are 4.77s and 4.84s, so the `reach` guard
     * below already falls back on its own where the batched build belongs.
     */

    for (uint32_t i = 0; i < len; ++i) {
        /*
         * Rescale by the exact all-state maximum, but materialise only weights
         * the frontier will consume. The sparse thin-frontier kernel does both
         * in one directory traversal; densely reached layers use the batched
         * full-layer build and obtain its fused maximum.
         *
         * This sweep is the partition function, run once per live length at
         * the end of finalize, and it was the 1.6s floor on every build -- and
         * on every load, since the file stores counts and loading refinalizes.
         */
        double wmax;
        if (m->cfg.sparse && (uint64_t)reach * 8u < (uint64_t)S) {
            /*
             * Thinly reached: mark only the rows the accumulation consumes,
             * but take the maximum over every occupied state.  The combined
             * kernel writes marked rows while max-only-scoring the rest, so a
             * frontier state is never scored once for the maximum and again
             * for the transition.
             */
            for (uint32_t k = 0; k < nsrc; ++k) {
                const uint32_t s = src[k];
                needed[s] = 1u;
            }
            wmax = -INFINITY;
            beam_layer_weights_live(m, i, len, w, &wmax, needed);
            for (uint32_t k = 0; k < nsrc; ++k) needed[src[k]] = 0u;
        } else {
            /* Densely reached, or a dense model: the batched build wins. Its
             * loop order turns the wide table into one sequential scan rather
             * than millions of scattered lookups -- worth 2-3x -- and that
             * beats skipping rows once most of them are wanted. `reach * 8 < S`
             * is where the two cross; on the foundation the late layers land
             * here and the early ones do not. */
            wmax = -INFINITY;
            beam_layer_weights(m, i, len, w, &wmax);
        }
        if (wmax == -INFINITY) {           /* no live edge: no length-L path */
            BEAM_LOGZ_FREE();
            *out = -INFINITY;
            return BEAM_OK;
        }

        /*
         * Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread.
         *
         * Which target a source feeds depends only on `rest = s % SK`: the
         * twenty targets of `rest` are `rest * A + x`, and the twenty-one
         * candidate sources of each are `a * SK + rest` for `a = 0..A-1`. So
         * regrouping the frontier by `rest` -- a counting sort, `nsrc + SK`
         * work -- turns the sweep from one over every possible target into one
         * over the targets that can actually receive anything.
         *
         * BIT-IDENTICAL, not merely equivalent, and that is the whole reason
         * for the regroup. `by_rest` is filled by walking the ASCENDING
         * frontier, so within one `rest` group the `a` values come out
         * ascending -- the exact order the dense loop added them in. The only
         * terms dropped are those with `cur[src] == 0`, which the dense loop
         * already skipped and which contribute exactly zero anyway.
         */
        for (uint32_t r = 0; r <= SK; ++r) off[r] = 0u;
        for (uint32_t k = 0; k < nsrc; ++k) off[(src[k] % SK) + 1u]++;
        uint32_t nrest = 0;
        for (uint32_t r = 0; r < SK; ++r) {
            if (off[r + 1u] != 0u) rests[nrest++] = r;
            off[r + 1u] += off[r];
        }
        for (uint32_t r = 0; r < SK; ++r) pos[r] = off[r];
        for (uint32_t k = 0; k < nsrc; ++k)
            by_rest[pos[src[k] % SK]++] = src[k] / SK;       /* store `a` */

        double nmax = 0.0;
#ifdef _OPENMP
        if (!omp_in_parallel() && nrest > 1u && beam_threads() > 1) {
            #pragma omp parallel for schedule(static) reduction(max:nmax) \
                    num_threads(beam_threads())
            for (int64_t q = 0; q < (int64_t)nrest; ++q) {
                const double v = logz_accumulate_rest(
                    rests[q], A, R, SK, off, by_rest, w, cur, next, wmax);
                if (v > nmax) nmax = v;
            }
        } else
#endif
        {
            for (uint32_t q = 0; q < nrest; ++q) {
                const double v = logz_accumulate_rest(
                    rests[q], A, R, SK, off, by_rest, w, cur, next, wmax);
                if (v > nmax) nmax = v;
            }
        }
        log_scale += wmax;

        if (!(nmax > 0.0)) {
            BEAM_LOGZ_FREE();
            *out = -INFINITY;
            return BEAM_OK;
        }

        /* Rescale and collect the next frontier in one pass. Ascending in
         * `rest` and then in `x` is ascending in the target index, so the new
         * frontier comes out sorted without a sort. Targets outside it are
         * never read again: the next layer's accumulation reaches `cur` only
         * through `by_rest`, and the tail sweep only through `src`. */
        uint32_t nnew = 0;
        double layer_sum = 0.0;
        for (uint32_t q = 0; q < nrest; ++q) {
            const size_t base = (size_t)rests[q] * A;
            for (uint32_t x = 0; x < R; ++x) {
                const size_t t = base + x;
                if (next[t] == 0.0) continue;
                next[t] = beam_rescale(next[t], nmax);
                layer_sum += next[t];
                nxt[nnew++] = (uint32_t)t;
            }
        }
        log_scale += log(nmax);
        /* `layer_sum` is rescaled mass and `log_scale` is what it was divided
         * by, so their combination is the true running mass. One add inside a
         * pass that already rescales and collects every surviving state. */
        if (layer_logmass) layer_logmass[i] = log(layer_sum) + log_scale;

        double *tmp = cur; cur = next; next = tmp;
        uint32_t *tmpi = src; src = nxt; nxt = tmpi;
        nsrc = nnew;
        reach = nnew;
    }

    /* The sink. Only states the walk actually reached can carry mass, and the
     * frontier is ascending, so both sweeps below visit exactly the states the
     * dense `s = 0 .. S` loops visited non-trivially, in the same order. */
    beam_tail_weights(m, len, tail);
    double tmax = -INFINITY;
    for (uint32_t k = 0; k < nsrc; ++k) {
        const uint32_t s = src[k];
        if (cur[s] > 0.0 && tail[s] > tmax) tmax = tail[s];
    }
    if (tmax == -INFINITY) {
        BEAM_LOGZ_FREE();
        *out = -INFINITY;
        return BEAM_OK;
    }
    double acc = 0.0;
    for (uint32_t k = 0; k < nsrc; ++k) {
        const uint32_t s = src[k];
        if (cur[s] == 0.0 || tail[s] == -INFINITY) continue;
        acc += cur[s] * exp(tail[s] - tmax);
    }

    *out = log_scale + tmax + log(acc);
    /* The sink settles the backward scores of the final min(order, len)
     * residues, which never got a full right-hand context during the walk.
     * It is attributed to the LAST layer, the same convention
     * `beam_model_explain` uses for `by_depth[-1]`, which is what makes the
     * last entry equal `*out` and the differences telescope to it. */
    if (layer_logmass && len > 0u) layer_logmass[len - 1u] = *out;
    BEAM_LOGZ_FREE();
    return BEAM_OK;
}
#undef BEAM_LOGZ_FREE
