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
    const uint32_t A  = BEAM_ALPHABET;
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)malloc((size_t)S * sizeof(double));
    double *next = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !cur || !next || !tail) {
        free(w); free(cur); free(next); free(tail);
        return BEAM_EALLOC("the partition-function trellis buffers");
    }

    /* The walk starts in the all-sentinel state. */
    for (uint32_t s = 0; s < S; ++s) cur[s] = 0.0;
    cur[S - 1] = 1.0;
    double log_scale = 0.0;
    /* How many states the sweep can currently reach, carried from the layer
     * that produced `cur` rather than recounted. Counting it costs one pass
     * over S per layer, which is the very thing the count is used to avoid. */
    uint32_t reach = 1u;
    /*
     * Decided once, from the model's own occupancy, rather than per layer.
     * The per-state build wins only when the sweep reaches a small slice of
     * the trellis, and that tracks how full the tables are: a per-repertoire
     * model fills 0.2% of its cells, the 927M foundation fills 42.7%. Below
     * the cut the thin path is 4x; above it the batched build's sequential
     * scan wins and even COUNTING the reached states costs more than it
     * saves -- measured at 0.85x on a 200,000-sequence model when the count
     * ran unconditionally.
     */
    bool try_thin = false;
    if (m->cfg.sparse) {
        /* Occupancy read off the sparse tables' own row counts, which is a
         * handful of loads. `beam_model_n_nonzero_cells` would answer the
         * same question by counting all 147 million cells -- once per length,
         * which is worse than anything it could save. */
        uint32_t nw = 0, nm = 0, nl = 0;
        (void)beam_sparse_rows(&m->sp_lp_wide[BEAM_FWD], 0u, &nw);
        (void)beam_sparse_rows(&m->sp_lp_mid[BEAM_FWD],  0u, &nm);
        (void)beam_sparse_rows(&m->sp_lp_len[BEAM_FWD],  0u, &nl);
        const uint64_t occupied = (uint64_t)nw + nm + nl;
        const uint64_t total = (uint64_t)m->n_depth_wide * m->n_ctx_wide
                             + (uint64_t)m->max_len * m->n_ctx_mid
                             + (uint64_t)m->n_len_slots * m->max_len * m->n_ctx_len;
        try_thin = occupied * 20u < total;
    }

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
         * reach: `cur[src] != 0` is exactly the set the accumulation below
         * reads, so every other row would be computed and never looked at.
         * On a per-repertoire model that is about 1% of the trellis.
         *
         * This sweep is the partition function, run once per live length at
         * the end of finalize, and it was the 1.6s floor on every build -- and
         * on every load, since the file stores counts and loading refinalizes.
         */
        double wmax;
        if (try_thin && (uint64_t)reach * 8u < (uint64_t)S) {
            /* Sparse and thinly reached: build only the rows the accumulation
             * reads. `cur[src] != 0` is exactly that set. */
            wmax = beam_layer_max(m, i, len);
            if (wmax != -INFINITY)
                for (uint32_t s = 0; s < S; ++s)
                    if (cur[s] != 0.0)
                        beam_state_weights(m, i, len, s, w + (size_t)s * R);
        } else {
            /* Densely reached, or a dense model: the batched build wins. Its
             * loop order turns the wide table into one sequential scan rather
             * than millions of scattered lookups -- worth 2-3x -- and that
             * beats skipping rows once most of them are wanted. Measured:
             * per-state was 2.4x FASTER at 100 sequences and 1.2x SLOWER at
             * 200,000, which is what the threshold is for. */
            wmax = -INFINITY;
            beam_layer_weights(m, i, len, w, &wmax);
        }
        if (wmax == -INFINITY) {           /* no live edge: no length-L path */
            free(w); free(cur); free(next); free(tail);
            *out = -INFINITY;
            return BEAM_OK;
        }

        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        double nmax = 0.0;
        int64_t nreach = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                reduction(+:nreach) num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < (int64_t)S; ++t) {
            const uint32_t x = (uint32_t)t % A;
            if (x >= R) { next[t] = 0.0; continue; }
            const uint32_t rest = (uint32_t)t / A;
            double acc = 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                const double v = cur[src];
                if (v == 0.0) continue;
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                acc += v * exp(wv - wmax);
            }
            next[t] = acc;
            if (try_thin && acc != 0.0) nreach++;
            if (acc > nmax) nmax = acc;
        }
        if (try_thin) reach = (uint32_t)nreach;
        log_scale += wmax;

        if (!(nmax > 0.0)) {
            free(w); free(cur); free(next); free(tail);
            *out = -INFINITY;
            return BEAM_OK;
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) next[s] /= nmax;
        log_scale += log(nmax);

        double *tmp = cur; cur = next; next = tmp;
    }

    beam_tail_weights(m, len, tail);
    double tmax = -INFINITY;
    for (uint32_t s = 0; s < S; ++s)
        if (cur[s] > 0.0 && tail[s] > tmax) tmax = tail[s];
    if (tmax == -INFINITY) {
        free(w); free(cur); free(next); free(tail);
        *out = -INFINITY;
        return BEAM_OK;
    }
    double acc = 0.0;
    for (uint32_t s = 0; s < S; ++s) {
        if (cur[s] == 0.0 || tail[s] == -INFINITY) continue;
        acc += cur[s] * exp(tail[s] - tmax);
    }

    *out = log_scale + tmax + log(acc);
    free(w); free(cur); free(next); free(tail);
    return BEAM_OK;
}
