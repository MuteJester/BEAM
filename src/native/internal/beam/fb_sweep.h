/**
 * @file fb_sweep.h
 * @brief The forward-backward workspace both exact per-position sweeps share.
 *
 * Internal to src/native/. Not a public header and not installed.
 *
 * There are two exact quantities built from the same per-layer edge posterior
 *
 *     gamma_i(s, x) = alpha_i(s) * w_i(s, x) * beta_{i+1}(s -> x) / Z(L)
 *
 * and they are computed in different files because they are different
 * answers: `math/positional.c` sums gamma over states to get
 * `P(x_i = a | L)`, and `math/expected_counts.c` sums it into the table cell
 * each edge would have incremented during training. What they SHARE is the
 * machinery that produces gamma -- the backward values, the forward step, and
 * the workspace all of it runs in -- and that lives here.
 *
 * It was one file. The two halves have different memory behaviour (one holds
 * a fixed workspace, the other holds accumulators that grow with the answer)
 * and the file was eleven hundred lines, so nothing about the sweep was
 * findable by someone looking for either quantity. Splitting the answers apart
 * and leaving the shared sweep in the middle is what makes each of the three
 * readable on its own.
 */
#ifndef BEAM_INTERNAL_FB_SWEEP_H
#define BEAM_INTERNAL_FB_SWEEP_H

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/memory.h"

/**
 * "This position is not pinned."
 *
 * Deliberately `BEAM_RESIDUE_INVALID`, so a caller can `memset` a clamp array
 * to 0xFF and mean "free everywhere", and so the one value that can never be a
 * residue index is the one that means "no residue".
 */
#define BEAM_CLAMP_FREE BEAM_RESIDUE_INVALID

/**
 * The half-open range of residues layer `i` may emit.
 *
 * One value when the layer is pinned and all of them otherwise, which is what
 * lets every loop that takes a clamp impose it by its bounds rather than by a
 * test in its body. A pinned layer then costs a twentieth of an unpinned one
 * instead of the same and discarding nineteen twentieths.
 *
 * Inline and in the header because it is called once per layer per sweep and
 * is four instructions.
 */
static inline void beam_clamp_range(const uint8_t *clamp, uint32_t i,
                                    uint32_t R, uint32_t *lo, uint32_t *hi) {
    if (clamp && clamp[i] != BEAM_CLAMP_FREE) {
        *lo = clamp[i];
        *hi = clamp[i] + 1u;
    } else {
        *lo = 0u;
        *hi = R;
    }
}

/**
 * Scratch for one length's forward-backward sweep.
 *
 * Allocated and released as a unit, and CHARGED as a unit: `pool` is the
 * account the whole operation shares and `held` is what this workspace took
 * from it, so a sweep running beside accumulators cannot pass a budget check
 * the accumulators have already spent most of.
 */
typedef struct {
    double **beta;          /**< [len + 1][n_states], log space          */
    double  *w;             /**< [n_states * n_residues], one layer      */
    double  *cur;           /**< [n_states], LOG forward vector          */
    double  *next;          /**< [n_states]                              */
    double  *tail;          /**< [n_states], sink weights                */
    uint32_t n_beta;        /**< rows actually allocated, for the free   */
    BeamMemAccount *pool;   /**< the account charged                     */
    uint64_t held;          /**< what was charged to it, for the release */
} BeamFBWork;

/**
 * Allocate the workspace for one length, charging `pool`, or refuse.
 *
 * The whole backward field is held at once because the posterior needs layer
 * `i + 1` while the forward sweep is at layer `i`, and the forward sweep only
 * runs once the backward one has finished. That is `(len + 1) * n_states`
 * doubles, 44 MB on the default amino-acid model and 6.6 GB on a nucleotide
 * model at the alphabet's ceiling order, so the bound is a real one at the top
 * of the range rather than a formality.
 */
BeamError beam_fb_alloc(const BeamModel *m, uint32_t len, BeamFBWork *k,
                        BeamMemAccount *pool);

/** Release the workspace and give its bytes back to the account. */
void beam_fb_free(BeamFBWork *k);

/**
 * `beta[i][s]`, the log total weight of every completion from state `s` at
 * layer `i` that AGREES WITH THE CLAMP, sink term included.
 *
 * This is `beam_betas` with the residue loops bounded by `beam_clamp_range`,
 * and with no clamp it performs the same operations in the same order on the
 * same values, so the two agree bit for bit. It is a separate function rather
 * than a parameter added to `beam_betas` because that function is on the
 * sampler's and the top-k search's hot path, and neither has a clamp to pass.
 *
 * `w` is one layer of edge weights, BORROWED from the caller's workspace
 * rather than allocated here. This runs to completion before the forward
 * sweep touches that buffer, so there is nothing to share, and allocating a
 * second one meant 31 MB on an order-4 amino-acid model that no budget knew
 * about -- `beam_fb_alloc` charges for exactly one layer and this made two.
 */
BeamError beam_fb_betas(const BeamModel *m, uint32_t len, const uint8_t *clamp,
                        double **beta, double *w);

/**
 * Advance the LOG forward vector across one layer.
 *
 * Returns false when the layer leaves no mass at all, which under a clamp is a
 * legitimate answer. There is no rescale and no scale factor: see the note at
 * the definition for what was tried before it was removed.
 */
bool beam_fb_advance(const BeamModel *m, const double *w, const double *cur,
                     double *next, uint32_t lo, uint32_t hi);

#endif /* BEAM_INTERNAL_FB_SWEEP_H */
