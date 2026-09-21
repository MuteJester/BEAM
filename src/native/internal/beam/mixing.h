/**
 * @file mixing.h
 * @brief One direction's per-residue log term.
 *
 * `static inline` deliberately, and it must stay that way. These are the
 * innermost arithmetic in the model: they run once per (state, residue) per
 * layer, so every layer sweep calls them millions of times. Giving them
 * external linkage would turn that into a call, and would also change which
 * expressions the compiler is free to contract into fused multiply-adds --
 * which changes results in the last bit, silently.
 */
#ifndef BEAM_INTERNAL_MIXING_H
#define BEAM_INTERNAL_MIXING_H

#include <math.h>

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/indexing.h"
#include "beam/table_access.h"

/*
 * The per-residue log term of ONE direction: a single table lookup.
 *
 * There is no mixture. A weighted geometric mean of several experts used to
 * live here, and with it the rule that any dead expert killed the edge. What
 * survives of that rule is the closed world itself: an absent row, or an
 * absent cell within a present row, is -INFINITY, and a caller that scales it
 * must not turn `0 * -inf` into a NaN. Every caller does that test itself,
 * because each one has a cheaper place to put it than here.
 */
static inline double beam_cell_dense(const BeamModel *m, uint32_t dir,
                                     uint32_t depth, uint32_t ctx,
                                     uint32_t x) {
    return m->lp[dir][beam_ix(m, depth, ctx, x)];
}

static inline double beam_cell_sparse(const BeamModel *m, uint32_t dir,
                                      uint32_t depth, uint32_t ctx,
                                      uint32_t x) {
    const BeamLogRow r = beam_lp_row(m, dir, depth, ctx);
    if (!r.present) return -INFINITY;
    return beam_log_cell(&r, x);
}

/**
 * Sum of both directions' per-residue log terms for an ENCODED sequence, before the
 * length prior and the partition function are applied. `-INFINITY` as soon as
 * any residue is unreachable, which is what makes the closed world cheap: a
 * sequence the model cannot spell costs one position, not the whole length.
 *
 * `model/scoring.c` is the only caller today, so this could fold back into it.
 * It stays declared here because the reason it was shared is still the right
 * reason: a second caller that scores sequences must compute exactly the same
 * quantity as `beam_model_log_pseq`, and two copies would eventually disagree.
 *
 * The comment here used to name `repair.c` as that second caller. There was no
 * `repair.c`: it had been renamed to `model/prism.c` and the signpost was left
 * behind, which is how it read for long enough to be found by an audit rather
 * than by a compiler. `prism.c` has since been removed altogether.
 */
double beam_raw_log_weight(const BeamModel *m, const uint8_t *s, uint32_t len);

#endif /* BEAM_INTERNAL_MIXING_H */
