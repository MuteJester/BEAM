/**
 * @file dp.h
 * @brief Dynamic-programming entry points shared across files.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_DP_H
#define BEAM_INTERNAL_DP_H

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/table_access.h"

/* ── Cross-file helpers ──────────────────────────────────────── */

/**
 * Mixed log-weight of one residue in one direction, before the direction
 * weight is applied. `ctx_wide`, `ctx_mid` and `ctx_len` are pre-packed
 * context indices of order `order`, 3 and 2 respectively.
 */
double beam_mix(const BeamModel *m, uint32_t dir, uint32_t len,
                    uint32_t depth, uint32_t ctx_wide, uint32_t ctx_mid,
                    uint32_t ctx_len, uint32_t x);

/**
 * Edge weights of one DAG layer: `out[state * 20 + x]`, the full weight of
 * emitting `x` from `state` at position `i` of a length-`len` sequence.
 * Includes the forward term for the residue being emitted and the backward
 * term for the residue emitted `order` steps ago, which only now has its
 * complete right-hand context.
 *
 * `out` must hold `n_states * 20` doubles.
 *
 * `out_max`, if given, receives the largest weight in the layer. Every sweep
 * needs it to rescale, and computing it here costs nothing while the values
 * are still in register: computing it afterwards means a second pass over 31
 * MB per layer, which measured at a fifth of the sweep. A maximum is
 * order-independent, so the fused value is bit-identical to the separate one.
 * `-INFINITY` means the layer has no live edge at all.
 */
/** The layer's maximum edge weight, computed without building the layer. */
double beam_layer_max(const BeamModel *m, uint32_t i, uint32_t len);

void beam_layer_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            double *out, double *out_max);

/**
 * As above, but skipping states `alive` marks dead and filling their rows with
 * -inf. See the definition in score.c for when this is safe: only where the
 * caller provably never reads a dead row, which excludes every caller that
 * asks for `out_max`.
 */
void beam_layer_weights_live(const BeamModel *m, uint32_t i, uint32_t len,
                                 double *out, double *out_max,
                                 const uint8_t *alive);

/**
 * Weights of the 20 edges leaving ONE state, identical to the corresponding
 * row of `beam_layer_weights`.
 *
 * For a consumer that visits states in an unpredictable order — the
 * best-first search in topk.c is the only one — building the whole layer to
 * read one row of it is 3.9 million wasted edges per visit.
 */
void beam_state_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            uint32_t state, double *out);

/**
 * Sink weights: `out[state]`, settling the backward terms of the final
 * `min(order, len)` residues, which never get a full right-hand context
 * during the walk. `out` must hold `n_states` doubles.
 */
void beam_tail_weights(const BeamModel *m, uint32_t len, double *out);

/** Compute `log_z[len]`. Returns BEAM_OK or an allocation error. */
BeamError beam_compute_log_z(BeamModel *m, uint32_t len, double *out);

/**
 * Backward values of the trellis: `beta[i][state]`, the log of the total
 * weight of every completion from `state` at layer `i`. `beta` must be an
 * array of `len + 1` pointers, each to `n_states` doubles.
 */
BeamError beam_betas(const BeamModel *m, uint32_t len, double **beta);

/**
 * Backward values of the `q`-TILTED trellis, i.e. with every edge weight
 * scaled by `q` before exponentiating. `q = 1` reproduces `beam_betas`.
 *
 * Needed wherever a posterior under a tilted law is required, such as
 * attributing surprisal to depths at `q != 1`.
 */
BeamError beam_betas_tilted(const BeamModel *m, uint32_t len, double q,
                               double **beta);

/**
 * Lengths the model actually generates.
 *
 * `finalize` zeroes the prior of any length whose trellis has no surviving
 * path, so a finite prior implies a finite partition function and this single
 * test is sufficient. Before that renormalisation the two could disagree, and
 * a length with prior mass but no support would poison every mean it entered.
 */
static inline bool beam_length_live(const BeamModel *m, uint32_t len) {
    return len >= 1 && len <= m->max_len && isfinite(m->log_pi[len]);
}


/* ── Moment jet ──────────────────────────────────────────────── */

/** Highest moment order the jet sweep carries. */
#define BEAM_MAX_MOMENT 16u

/**
 * One forward pass carrying `n_moments + 1` moments of the q-tilted law.
 *
 * The arithmetic here is a separate numerical path from `tilted_log_z` in
 * math/diversity.c even though both compute tilted quantities. They are not
 * to be consolidated on the grounds that they look alike: each has its own
 * regression value.
 */
/** Binomial coefficients up to `n`, for the moment recurrence. */
void beam_binomials(double c[][BEAM_MAX_MOMENT + 1], uint32_t n);

BeamError jet_sweep(const BeamModel *m, uint32_t len, double q,
                          uint32_t order, double *out_moments,
                          double *out_log_z);

#endif /* BEAM_INTERNAL_DP_H */
