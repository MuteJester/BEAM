/**
 * @file math_internal.h
 * @brief Helpers shared between the named mathematical operations.
 *
 * Internal to src/native/math/. These were static functions inside one
 * moments.c; splitting that file by operation left them with callers in
 * another unit.
 */
#ifndef BEAM_INTERNAL_MATH_H
#define BEAM_INTERNAL_MATH_H

#include <math.h>

#include "beam/model.h"
/* NOT model_state.h: every declaration below takes `const BeamModel *`, and
 * a pointer to an opaque type needs only the public header. Including the
 * layout here put it into `query_internal.h`, and from there into
 * `query/automaton.c` and `query/constraint.c`, neither of which mentions a
 * model at all. */

/**
 * Per-length Mellin terms at `q`, and the largest of them.
 *
 * The shared front half of every q-tilted quantity: the moments, the
 * per-length jets, and the depth attribution all start here.
 */
BeamError beam_mellin_terms(const BeamModel *m, double q, double *terms,
                             double *out_max);

/**
 * Grid spacing for a surprisal histogram, sized automatically when `spacing`
 * is not positive.
 *
 * Shared by every spectrum entry point so they cannot disagree about what
 * grid they are on — which for the paired and per-length forms is not a
 * tidiness point but the property that makes their outputs comparable.
 *
 * `math/positional_banded.c` is here for exactly that reason. Its band masses
 * are only checkable against `beam_model_spectrum` summed over the same bins
 * if both landed on the same grid, and a second copy of this arithmetic would
 * agree today and drift later.
 */
BeamError beam_spectrum_spacing(const BeamModel *m, uint32_t bins,
                                double spacing, double *out_ds);

/**
 * Exact extremes of `-ln P` among the producible sequences of ONE length.
 *
 * Two sweeps, min-plus and max-plus, and nothing else -- which is why it is
 * shared. `insights/logo.c` needs it to refuse a surprisal band that lies
 * outside what a length can attain, and it must refuse BEFORE allocating: the
 * banded sweep it would otherwise run holds gigabytes, where this holds a
 * handful of arrays over the state space.
 *
 * `beam_model_surprisal_range` is the public form over every length, and is
 * too loose for that check -- a band can sit inside the model's whole range
 * and outside this length's.
 */
BeamError beam_surprisal_range_at(const BeamModel *m, uint32_t len,
                                  double *out_min, double *out_max);

/**
 * The figure `beam_model_positional_banded` refuses on, for the same length,
 * grid and number of edges, exposed so that `beam_insights_logo_cost`
 * reports that number rather than a second derivation of it. `budget.h`
 * requires a guard and its allocation to come from one expression; an
 * estimator that disagrees with the guard is the same defect wearing a
 * different hat.
 *
 * It builds the sweep's own plan to get it -- the sink weights, the backward
 * origin and an integer pass over the edges that sizes every backward row --
 * because the rows are stored at their own widths and those are not known
 * until then. That is one layer-weight evaluation and one pass of integer
 * work per layer, against a sweep that moves every occupied bin twice.
 */
BeamError beam_positional_banded_cost(const BeamModel *m, uint32_t length,
                                      uint32_t bins, double spacing,
                                      uint32_t n_edges, uint64_t *out_bytes);

/* ── Log-space accumulation ──────────────────────────────────── */

/**
 * Add one term to a running (max, sum) pair holding `sum * exp(max)`.
 *
 * Streaming log-sum-exp: the running maximum is raised when a larger term
 * arrives and the accumulated sum is rescaled to match. One `exp` per term and
 * no second pass, so a target can be finished in a single sweep of its sources
 * while never leaving log space.
 *
 * This lives here rather than beside either caller because both the query
 * engine's forward sweeps and `math/positional.c`'s use it, for the same
 * reason: a single rescale per layer cannot hold a trellis whose live states
 * span more than 745 nats, and the states that underflow are not marginal
 * ones.
 */
static inline void lse_add(double *mx, double *sum, double v) {
    if (v == -INFINITY) return;
    if (v > *mx) { *sum = *sum * exp(*mx - v) + 1.0; *mx = v; }
    else         { *sum += exp(v - *mx); }
}

#endif /* BEAM_INTERNAL_MATH_H */
