/**
 * @file smoothing.c
 * @brief The conversion to log space.
 *
 * This held the hierarchical backoff too, five functions of it, until the
 * three-expert mixture went and took the backoff strengths with it. What is
 * left is one function, and it is kept in its own translation unit rather
 * than folded into finalize.c because the reason it measures rather than
 * asserts is a property of the whole library, not of finalizing.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"


/**
 * Probabilities to natural logs, reporting whether any cell came out -inf.
 *
 * In the closed world a zero cell becomes -inf, which is the support. In the
 * open world the floor makes that impossible: `log(1e-300)` is a perfectly
 * finite -690.8.
 *
 * That floor no longer fires at all. The open world is a Laplace add-one at
 * one table, so the smallest a cell can be is 1/(n + R) and nothing reaches
 * 1e-300. It is kept because it costs one comparison and it is the thing
 * standing between a future change and a silent -inf; it is worth knowing
 * that it is currently unreachable rather than assuming it protects
 * something.
 *
 * The return value is still MEASURED rather than assumed, because what
 * depends on
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
