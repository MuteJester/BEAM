/**
 * @file smoothing.c
 * @brief Hierarchical backoff and the conversion to log space.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"


/* ── Hierarchical smoothing ──────────────────────────────────── */

/**
 * One level of the backoff recurrence, in probability space:
 *
 *     p_j[d][ctx][x] = (c_j[d][ctx][x] + kappa * p_(j-1)[d][ctx >> A][x])
 *                      / rowsum
 *
 * where `ctx >> A` drops the MOST DISTANT context residue, so a higher-order
 * estimate falls back on the shorter context it refines. A row whose total is
 * zero cannot be produced at all and is written as exactly zero, which the
 * caller turns into -inf; that only happens when kappa is 0.
 *
 * @param counts   [n_depth][n_ctx][R] counts at this order.
 * @param parent   [n_depth][n_ctx / A][R] probabilities at the order below.
 * @param out      [n_depth][n_ctx][R] probabilities at this order.
 */
static void backoff_level(const double *counts, const double *parent,
                          double kappa, uint32_t n_depth, uint32_t n_ctx,
                          uint32_t A, uint32_t R, double *out) {
    const uint32_t n_parent_ctx = n_ctx / A;
    const int64_t n_rows = (int64_t)n_depth * n_ctx;
    /* Every row is independent of every other, so the only thing the loop
     * order decides is speed. Flattening depth and context into one index
     * keeps the largest table (the wide expert's, three million rows) from
     * being split across a dimension with sixteen values. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t row = 0; row < n_rows; ++row) {
        const uint32_t d = (uint32_t)(row / n_ctx);
        const uint32_t c = (uint32_t)(row % n_ctx);
        const double *cr = counts + (size_t)row * R;
        const double *pr = parent + ((size_t)d * n_parent_ctx
                                     + (c % n_parent_ctx)) * R;
        double *orow = out + (size_t)row * R;
        double tot = 0.0;
        for (uint32_t x = 0; x < R; ++x) {
            orow[x] = cr[x] + kappa * pr[x];
            tot += orow[x];
        }
        if (tot > 0.0) {
            for (uint32_t x = 0; x < R; ++x) orow[x] /= tot;
        } else {
            for (uint32_t x = 0; x < R; ++x) orow[x] = 0.0;
        }
    }
}

/** Sum out the most distant context residue: [n_ctx] -> [n_ctx / A]. */
static void marginalise(const double *in, uint32_t n_depth, uint32_t n_ctx,
                        uint32_t A, uint32_t R, double *out) {
    const uint32_t n_out = n_ctx / A;
    const int64_t n_rows = (int64_t)n_depth * n_out;
    /* Inverted to accumulate BY TARGET: the source form scatters `A` rows onto
     * every output row, so parallelising it would race. Gathering instead
     * makes each output row one thread's private property, and it visits the
     * sources in the same ascending order the scatter did, so the sums are
     * bit-identical rather than merely equivalent. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t row = 0; row < n_rows; ++row) {
        const uint32_t d = (uint32_t)(row / n_out);
        const uint32_t c = (uint32_t)(row % n_out);
        double *orow = out + (size_t)row * R;
        for (uint32_t x = 0; x < R; ++x) orow[x] = 0.0;
        for (uint32_t a = 0; a < A; ++a) {
            const double *ir = in + ((size_t)d * n_ctx + a * n_out + c)
                                    * R;
            for (uint32_t x = 0; x < R; ++x) orow[x] += ir[x];
        }
    }
}

/** Order-0 level: Laplace alpha = 1 over the R residues, per depth. */
static void base_level(const double *c0, uint32_t n_depth, uint32_t R,
                       double *out, bool strict) {
    for (uint32_t d = 0; d < n_depth; ++d) {
        const double *cr = c0 + (size_t)d * R;
        double *orow = out + (size_t)d * R;
        double tot = 0.0;
        for (uint32_t x = 0; x < R; ++x) {
            /*
             * The add-one is what makes zero unreachable: it is the last rung
             * of the backoff ladder, so a residue unseen everywhere still
             * leaves with 1/(total+20). Under `strict` the row is the raw MLE
             * instead, and an unseen residue stays at exactly zero so the
             * -inf survives the log.
             */
            orow[x] = strict ? cr[x] : cr[x] + 1.0;
            tot += orow[x];
        }
        if (tot <= 0.0) {
            /*
             * Strict mode only. An empty row has no observation to normalise,
             * and 0/0 would be NaN, which propagates silently through the DP
             * and is far worse than the -inf it would have replaced. Leave the
             * row at zero: every residue is then impossible from this context,
             * which is exactly what "nothing was ever seen here" means.
             */
            continue;
        }
        for (uint32_t x = 0; x < R; ++x) orow[x] /= tot;
    }
}

/**
 * Full hierarchy over an order-`k` count table, returning probabilities at
 * every order 0..k. `out[j]` must have room for `n_depth * A^j * R` doubles.
 * Caller owns the buffers.
 */
BeamError beam_build_hierarchy(const double *counts, uint32_t n_depth,
                                uint32_t order, uint32_t A, uint32_t R,
                                const double *kappa,
                                double **out, bool strict) {
    /* Marginals, computed top down into scratch. */
    const double *level_counts[BEAM_MAX_ORDER + 1];
    double *scratch[BEAM_MAX_ORDER + 1];
    memset(scratch, 0, sizeof(scratch));

    level_counts[order] = counts;
    for (int32_t j = (int32_t)order - 1; j >= 0; --j) {
        uint32_t n_out = beam_ipow_u32(A, (uint32_t)j);
        scratch[j] = (double *)malloc((size_t)n_depth * n_out
                                      * R * sizeof(double));
        if (!scratch[j]) {
            for (uint32_t t = 0; t <= order; ++t) free(scratch[t]);
            return BEAM_EALLOC("a backoff-hierarchy marginal");
        }
        marginalise(level_counts[j + 1], n_depth,
                    beam_ipow_u32(A, (uint32_t)j + 1u), A, R, scratch[j]);
        level_counts[j] = scratch[j];
    }

    base_level(level_counts[0], n_depth, R, out[0], strict);
    for (uint32_t j = 1; j <= order; ++j) {
        backoff_level(level_counts[j], out[j - 1], kappa[j - 1], n_depth,
                      beam_ipow_u32(A, j), A, R, out[j]);
    }
    for (uint32_t t = 0; t <= order; ++t) free(scratch[t]);
    return BEAM_OK;
}

/**
 * Probabilities to natural logs, reporting whether any cell came out -inf.
 *
 * The floor makes that impossible today — `log(1e-300)` is a perfectly finite
 * -690.8, which is the whole point of flooring rather than letting a zero
 * through. It is still MEASURED rather than assumed, because what depends on
 * it is not a cosmetic detail: with no -inf cell there is no dead edge, so the
 * live-state trim in analytics.c collapses to a reachability sweep and stops
 * building every layer's weights a second time. A future change that removed
 * the floor would switch that off automatically instead of silently counting
 * sequences the model cannot produce.
 */
bool beam_to_log_inplace(double *p, size_t n, bool strict) {
    bool nonfinite = false;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:nonfinite) \
            num_threads(beam_threads())
#endif
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        const double v = p[i];
        /*
         * The floor is the second reason zero is unreachable by default: it
         * turns a genuine 0 into log(1e-300) = -690.8, finite and therefore
         * still counted by D0. Under `strict` the value goes through
         * untouched, so 0 becomes -inf, the cell is a dead edge, and the
         * live-state trim in analytics.c switches to its general path on the
         * `tables_all_finite` flag the caller sets from this return.
         */
        const double lv = strict
            ? log(v)
            : log(v > BEAM_PROB_FLOOR ? v : BEAM_PROB_FLOOR);
        p[i] = lv;
        if (!isfinite(lv)) nonfinite = true;
    }
    return nonfinite;
}
