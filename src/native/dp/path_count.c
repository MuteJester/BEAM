/**
 * @file path_count.c
 * @brief Support size: how many sequences the model can spell.
 */
#include "beam/internal.h"


/* ── Support size ────────────────────────────────────────────── */

/**
 * Support size at one length.
 *
 * Returns 0.0 for three different reasons — an unusable model, a length the
 * model does not generate, and an allocation failure — which a `double` cannot
 * distinguish. Each sets the thread-local error context, so a caller that
 * needs to tell them apart reads `beam_last_error()` after a zero:
 * `BEAM_OK` means the length genuinely has no paths.
 */

double beam_model_path_count(const BeamModel *m, uint32_t length) {
    if (!m || !m->finalized) {
        if (!m) beam_set_error(BEAM_ERR_NULL_ARG,
                              "path_count: model must not be NULL");
        else    (void)BEAM_ENOTBUILT();
        return 0.0;
    }
    beam_clear_error();
    if (!beam_length_live(m, length)) { (void)BEAM_EBADLEN(m, length); return 0.0; }

    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    LiveHandle h;
    if (beam_live_acquire(m, length, &h) != BEAM_OK) return 0.0;
    const BeamLive *L = h.L;

    /*
     * No weights, anywhere in this sweep.
     *
     * Path counting never asks what an edge WEIGHS, only whether it exists,
     * and the frontier already reduced that to one bit per (state, residue)
     * while it was discovering reachability. `bits[s]` is the twenty-bit
     * residue mask of state `s` at the current layer, scattered out of the
     * frontier's `reached`-parallel list once per layer. On the 927M
     * foundation this took the sweep from 36.1 s to 0.97 s; the arithmetic is
     * untouched, because `w[x] == -INFINITY` and `bit x clear` are the same
     * predicate over the same weights.
     */
    uint32_t *bits = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    double *cur  = (double *)calloc(S, sizeof(double));
    /* calloc, then cleared only where it was last written: see the loop.
     * A full memset of S doubles per layer is 1.5 MB at order 4, which over
     * every layer of every length moves more memory than the sweep it serves. */
    double *next = (double *)calloc(S, sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!bits || !cur || !next || !tail) {
        free(bits); free(cur); free(next); free(tail);
        beam_live_release(&h);
        (void)BEAM_EALLOC("the path-counting sweep buffers");
        return 0.0;
    }

    cur[S - 1] = live_bit_get(L->mask[0], S - 1) ? 1.0 : 0.0;
    for (uint32_t i = 0; i < length; ++i) {
        /* Scatter this layer's edge masks so the sweep can index them by
         * state. Only `reached` entries are written, and only a state that
         * carries mass is ever read -- mass implies live implies reached --
         * so the stale entries left from earlier layers are unreachable. */
        {
            const uint32_t *rd = L->reached[i];
            const uint32_t *eb = L->edge_bits[i];
            const uint32_t nr = L->n_reached[i];
            for (uint32_t k = 0; k < nr; ++k) bits[rd[k]] = eb[k];
        }
        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        /* `next` is the buffer that held layer i-1 before the last swap, so the
         * only entries that can be stale are the ones live at i-1. Clearing
         * exactly those keeps the invariant "everything not written this layer
         * is zero" at O(live) rather than O(S). */
        if (i >= 1) {
            const uint32_t np = L->n_live[i - 1];
            const uint32_t *pidx = L->live[i - 1];
            for (uint32_t k = 0; k < np; ++k) next[pidx[k]] = 0.0;
        }
        const uint32_t nlive = L->n_live[i + 1];
        const uint32_t *tidx = L->live[i + 1];
#ifdef _OPENMP
        /* `num_threads` is not optional, and `if` is not an optimisation.
         * Without the first this ignores beam_set_threads() and takes every
         * core the machine has. Without the second a team is created and torn
         * down once per layer even when the layer holds a single live state:
         * on a per-repertoire model that overhead WAS the whole runtime --
         * 0.600s of a 0.628s path_count, for about fifty microseconds of
         * arithmetic. The threshold is deliberately generous; below it the
         * serial loop is faster than the fork/join. */
        #pragma omp parallel for schedule(static) \
                num_threads(beam_threads()) if(nlive > 4096)
#endif
        for (int64_t k = 0; k < (int64_t)nlive; ++k) {
            const uint32_t t = tidx[k];
            const uint32_t x = t % A;
            if (x >= R) continue;
            const uint32_t rest = t / A;
            double acc = 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                /* cur[src] != 0 implies src was live at this layer, so
                 * bits[src] was written above and holds its residue mask. */
                if (cur[src] == 0.0) continue;
                if (!((bits[src] >> x) & 1u)) continue;
                acc += cur[src];
            }
            next[t] = acc;
        }
        double *tmp = cur; cur = next; next = tmp;
    }
    beam_tail_weights(m, length, tail);
    double total = 0.0;
    for (uint32_t s = 0; s < S; ++s)
        if (tail[s] != -INFINITY) total += cur[s];

    free(bits); free(cur); free(next); free(tail);
    beam_live_release(&h);
    return total;
}

/**
 * Live lengths, longest first, written into `out`. Returns how many there are.
 *
 * Longest first because the per-length dynamic programs are the unit of
 * parallel work and their cost is proportional to the length: handing out the
 * 25-layer job last leaves fifteen idle cores waiting for it, and handing it
 * out first hides it behind everything else. `out` must hold `n_len_slots`.
 */
uint32_t beam_live_lengths_desc(const BeamModel *m, uint32_t *out) {
    uint32_t n = 0;
    for (uint32_t L = m->n_len_slots; L-- > 1; )
        if (beam_length_live(m, L)) out[n++] = L;
    return n;
}

double beam_model_path_count_total(const BeamModel *m) {
    if (!m || !m->finalized) return 0.0;
    const uint32_t n = m->n_len_slots;
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *per = (double *)calloc(n, sizeof(double));
    if (!work || !per) {
        free(work); free(per);
        (void)BEAM_EALLOC("the per-length path counts");
        return 0.0;
    }
    const uint32_t nw = beam_live_lengths_desc(m, work);
    /* The lengths are independent dynamic programs and this is where the
     * parallelism belongs: one whole DP per thread beats sixteen threads
     * inside one DP's layer, which has to synchronise once per layer. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k)
        per[work[k]] = beam_model_path_count(m, work[k]);
    /* Summed in ascending length, exactly as the serial loop did. */
    double t = 0.0;
    for (uint32_t L = 1; L < n; ++L) t += per[L];
    free(work); free(per);
    return t;
}
