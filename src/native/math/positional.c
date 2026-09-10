/**
 * @file positional.c
 * @brief Per-position residue marginals, exact and optionally conditioned.
 *
 * ## The quantity
 *
 * `P(x_i = a | L)`, the probability that position `i` of a length-`L`
 * sequence is residue `a`, obtained by summing the per-layer edge posterior
 *
 *     gamma_i(s, x) = alpha_i(s) * w_i(s, x) * beta_{i+1}(s -> x) / Z(L)
 *
 * over states. The posterior itself is built by `math/fb_sweep.c`, which
 * `math/expected_counts.c` also uses to answer a different question about the
 * same object.
 *
 * ## Why this is exact and not an approximation of anything
 *
 * The DAG is layered by position and every path weight factorises over its
 * edges plus one sink term, so the sum above is a finite sum a dynamic program
 * evaluates in full. Nothing is sampled, nothing is binned, and no state is
 * pruned: the sweeps run over all `n_states` at every layer. The only error is
 * floating point, and it is at the last bits.
 *
 * That is a stronger claim than `beam_model_spectrum` can make, and the
 * difference is worth naming. The spectrum transports mass onto a GRID and
 * splits it linearly between adjacent bins at every layer, so its bins are
 * approximate even though its measure is exact. Marginals have no grid: a
 * residue is one of twenty, the axis is already discrete, and the answer lands
 * on it exactly.
 *
 * ## Clamping
 *
 * A clamp pins chosen positions to chosen residues, and the result is the
 * exact conditional `P(x_i = a | L, clamp)`. It is imposed by DELETING the
 * edges that disagree with it -- the inner loops run over `[lo, hi)` rather
 * than over all twenty residues -- in the backward pass, the forward pass and
 * the posterior alike. Deleting edges is exactly conditioning here, because
 * the set of length-`L` paths agreeing with the clamp is precisely the set of
 * paths that avoid those edges, and the weight of a path is the product over
 * the edges it uses. So the clamped `Z` is the unnormalised probability of the
 * clamped event, and dividing by it normalises the conditional.
 *
 * A clamp is NOT applied by reweighting or by masking the answer afterwards.
 * Masking would leave the other positions' marginals as their unconditional
 * values, which is a different and wrong quantity.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"
#include "beam/fb_sweep.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/** True when any position is pinned. */
static bool clamp_active(const uint8_t *clamp, uint32_t len) {
    if (!clamp) return false;
    for (uint32_t i = 0; i < len; ++i)
        if (clamp[i] != BEAM_CLAMP_FREE) return true;
    return false;
}


/* ── Positional marginals ────────────────────────────────────── */

BeamError beam_model_positional_marginals(const BeamModel *m, uint32_t length,
                                          const uint8_t *clamp,
                                          double *out, double *out_log_z) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);

    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    for (uint32_t i = 0; i < length; ++i)
        for (uint32_t x = 0; x < R; ++x) out[(size_t)i * R + x] = 0.0;
    if (out_log_z) *out_log_z = -INFINITY;

    for (uint32_t i = 0; i < length; ++i)
        if (clamp && clamp[i] != BEAM_CLAMP_FREE && clamp[i] >= R)
            return BEAM_ERANGE("positional_marginals: position %u is clamped "
                              "to residue index %u, which is not one of the "
                              "%u residues of this model's alphabet",
                              (unsigned)i, (unsigned)clamp[i], (unsigned)R);

    BeamMemAccount pool;
    memset(&pool, 0, sizeof(pool));
    BeamFBWork k;
    BeamError err = beam_fb_alloc(m, length, &k, &pool);
    if (err != BEAM_OK) return err;

    /* Per-chunk partial rows, combined in chunk order so the answer does not
     * depend on how many threads happened to be free. See BEAM_SUM_CHUNKS. */
    const uint64_t partials = (uint64_t)BEAM_SUM_CHUNKS * R * sizeof(double);
    if (!beam_account_take(&pool, partials)) {
        beam_fb_free(&k);
        return beam_budget_refuse("a positional-marginal sweep", length,
                                  pool.held + partials);
    }
    double *chunk = (double *)malloc((size_t)partials);
    if (!chunk) { beam_fb_free(&k); return BEAM_EALLOC("the marginal partials"); }

    err = beam_fb_betas(m, length, clamp, k.beta, k.w);
    if (err != BEAM_OK) goto done;

    {
    /*
     * Z from the backward pass rather than from `m->log_z`, and that is not a
     * convenience. Under a clamp `m->log_z` is the WRONG normaliser -- it is
     * the unclamped partition function -- and even unclamped, taking Z from
     * the same sweep the numerator comes from is what makes each row sum to
     * one to the last bits instead of to within the difference between a
     * forward and a backward evaluation of the same quantity.
     */
    const double logz = k.beta[0][S - 1u];
    if (!isfinite(logz)) {
        err = clamp_active(clamp, length)
            ? BEAM_ENOPATHS("the clamped positions are a configuration this "
                           "model cannot produce at this length, so the "
                           "conditional it asks for does not exist")
            : BEAM_ENOPATHS("the trellis carries no path at this length");
        goto done;
    }
    if (out_log_z) *out_log_z = logz;

    for (uint32_t s = 0; s < S; ++s) k.cur[s] = -INFINITY;
    k.cur[S - 1u] = 0.0;         /* ln 1: the all-sentinel start state */

    for (uint32_t i = 0; i < length; ++i) {
        uint32_t lo, hi;
        beam_clamp_range(clamp, i, R, &lo, &hi);
        beam_layer_weights(m, i, length, k.w, NULL);
        const double *bn = k.beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads())
#endif
        for (int64_t c = 0; c < (int64_t)BEAM_SUM_CHUNKS; ++c) {
            double *part = chunk + (size_t)c * R;
            for (uint32_t x = 0; x < R; ++x) part[x] = 0.0;
            const uint32_t slo = (uint32_t)(((uint64_t)S * c) / BEAM_SUM_CHUNKS);
            const uint32_t shi =
                (uint32_t)(((uint64_t)S * (c + 1)) / BEAM_SUM_CHUNKS);
            for (uint32_t s = slo; s < shi; ++s) {
                if (k.cur[s] == -INFINITY) continue;
                const double la = k.cur[s] - logz;
                const double *row = k.w + (size_t)s * R;
                const uint32_t base = (s % SK) * A;
                for (uint32_t x = lo; x < hi; ++x) {
                    if (row[x] == -INFINITY) continue;
                    const double nb = bn[base + x];
                    if (nb == -INFINITY) continue;
                    part[x] += exp(la + row[x] + nb);
                }
            }
        }
        double *row_out = out + (size_t)i * R;
        for (uint32_t c = 0; c < BEAM_SUM_CHUNKS; ++c) {
            const double *part = chunk + (size_t)c * R;
            for (uint32_t x = lo; x < hi; ++x) row_out[x] += part[x];
        }

        if (!beam_fb_advance(m, k.w, k.cur, k.next, lo, hi)) {
            err = BEAM_ENOPATHS("the forward sweep reached a layer with no "
                               "mass although the trellis has a path");
            goto done;
        }
        double *tmp = k.cur; k.cur = k.next; k.next = tmp;
    }
    }

done:
    free(chunk);
    beam_fb_free(&k);
    return err;
}
