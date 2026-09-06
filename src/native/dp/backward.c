/**
 * @file backward.c
 * @brief Backward values, plain and q-tilted.
 */
#include "beam/internal.h"
#include "beam/mixing.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* ── Backward values ─────────────────────────────────────────── */

/**
 * `beta[i][s]` is the natural log of the total weight of every completion of a
 * length-`len` path from state `s` at layer `i`, including the sink term.
 *
 * Kept in log space rather than rescaled: the sampler indexes it randomly
 * rather than sweeping it, so a per-layer scale factor would have to be
 * carried alongside and applied at every draw.
 */
BeamError beam_betas(const BeamModel *m, uint32_t len, double **beta) {
    return beam_betas_tilted(m, len, 1.0, beta);
}

BeamError beam_betas_tilted(const BeamModel *m, uint32_t len, double q,
                               double **beta) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    /*
     * One layer, `S * R` doubles: 31 MB on an order-4 amino-acid model. The
     * backward FIELD is the caller's -- it is `(len + 1) * S` doubles and
     * dwarfs this -- so it is budgeted where it is allocated, not here.
     */
    BEAM_BUDGET_OR_REFUSE("a backward sweep", len,
                          (uint64_t)S * R * sizeof(double));

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    if (!w) return BEAM_EALLOC("one layer of edge weights");

    beam_tail_weights(m, len, beta[len]);
    if (q != 1.0)
        for (uint32_t s = 0; s < S; ++s)
            if (beta[len][s] != -INFINITY) beta[len][s] *= q;

    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, len, w, NULL);
        const double *nb = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
            const uint32_t s = (uint32_t)s64;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = q * row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            if (mx == -INFINITY) { beta[i][s] = -INFINITY; continue; }
            double acc = 0.0;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                acc += exp(q * row[x] + nb[base + x] - mx);
            }
            beta[i][s] = mx + log(acc);
        }
    }
    free(w);
    return BEAM_OK;
}
