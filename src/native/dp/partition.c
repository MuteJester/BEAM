/**
 * @file partition.c
 * @brief The forward partition function, log Z(L).
 */
#include "beam/internal.h"
#include "beam/mixing.h"

#ifdef _OPENMP
#include <omp.h>
#endif


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
BeamError beam_compute_log_z(BeamModel *m, uint32_t len, double *out) {
    const uint32_t A  = m->n_alphabet;
    const uint32_t R  = m->n_residues;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;

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

#define BEAM_LOGZ_FREE() do {                                                 \
        free(w); free(cur); free(next); free(tail);                           \
        free(src); free(nxt); free(by_rest); free(off); free(pos);             \
        free(rests);                                                          \
    } while (0)

    if (!w || !cur || !next || !tail || !src || !nxt || !by_rest || !off ||
        !pos || !rests) {
        BEAM_LOGZ_FREE();
        return BEAM_EALLOC("the partition-function trellis buffers");
    }

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
        /* Rescale the layer's weights by their maximum so the exponentials
         * stay in range regardless of how deep the tables go. The maximum
         * comes out of the layer build itself; taking a second pass over the
         * 31 MB of weights to find it cost a fifth of the sweep. */
        /*
         * The maximum comes from `beam_layer_max`, which reaches it
         * through the occupied forward rows instead of building all 194,481
         * states. Identical by construction -- a state with no forward row
         * scores -inf and cannot carry a maximum -- and verified layer by
         * layer against the full scan on the foundation.
         *
         * The weights themselves are then built only for states the sweep can
         * reach, which the frontier now names outright instead of rediscovering
         * with a scan over all 194,481 states.
         *
         * This sweep is the partition function, run once per live length at
         * the end of finalize, and it was the 1.6s floor on every build -- and
         * on every load, since the file stores counts and loading refinalizes.
         */
        double wmax;
        if ((uint64_t)reach * 8u < (uint64_t)S) {
            /* Thinly reached: build only the rows the accumulation reads,
             * walking the frontier rather than testing every state. */
            wmax = beam_layer_max(m, i, len);
            if (wmax != -INFINITY)
                for (uint32_t k = 0; k < nsrc; ++k)
                    beam_state_weights(m, i, len, src[k],
                                       w + (size_t)src[k] * R);
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
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t q = 0; q < (int64_t)nrest; ++q) {
            const uint32_t rest = rests[q];
            const uint32_t lo = off[rest], hi = off[rest + 1u];
            for (uint32_t x = 0; x < R; ++x) {
                double acc = 0.0;
                for (uint32_t j = lo; j < hi; ++j) {
                    const uint32_t s = by_rest[j] * SK + rest;
                    const double wv = w[(size_t)s * R + x];
                    if (wv == -INFINITY) continue;
                    acc += cur[s] * exp(wv - wmax);
                }
                next[(size_t)rest * A + x] = acc;
                if (acc > nmax) nmax = acc;
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
        for (uint32_t q = 0; q < nrest; ++q) {
            const size_t base = (size_t)rests[q] * A;
            for (uint32_t x = 0; x < R; ++x) {
                const size_t t = base + x;
                if (next[t] == 0.0) continue;
                next[t] /= nmax;
                nxt[nnew++] = (uint32_t)t;
            }
        }
        log_scale += log(nmax);

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
    BEAM_LOGZ_FREE();
    return BEAM_OK;
}
#undef BEAM_LOGZ_FREE
