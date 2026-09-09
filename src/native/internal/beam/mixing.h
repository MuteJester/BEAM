/**
 * @file mixing.h
 * @brief Log-linear mixing of the three experts into one edge weight.
 *
 * `static inline` deliberately, and it must stay that way. `mix_values` is the
 * innermost arithmetic in the model: it runs once per (state, residue) per
 * layer, so every layer sweep calls it millions of times. Giving it external
 * linkage would turn that into a call, and would also change which expressions
 * the compiler is free to contract into fused multiply-adds -- which changes
 * results in the last bit, silently.
 */
#ifndef BEAM_INTERNAL_MIXING_H
#define BEAM_INTERNAL_MIXING_H

#include <math.h>

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/indexing.h"
#include "beam/table_access.h"

static inline double mix_values(const BeamModel *m, double a, double b,
                                double c) {
    /* Any dead expert kills the edge: a geometric product with a zero factor
     * is zero, and -inf * a positive coefficient must not become NaN. */
    if (a == -INFINITY || b == -INFINITY || c == -INFINITY) return -INFINITY;
    return m->cfg.coef_wide * a + m->cfg.coef_mid * b + m->cfg.coef_len * c;
}

static inline double mix_dense(const BeamModel *m, uint32_t dir,
                               uint32_t len, uint32_t depth,
                               uint32_t ctx_wide, uint32_t ctx_mid,
                               uint32_t ctx_len, uint32_t x) {
    const double a = m->lp_wide[dir][beam_ix_wide(m, depth, ctx_wide, x)];
    const double b = m->lp_mid[dir][beam_ix_mid(m, depth, ctx_mid, x)];
    const double c = m->lp_len[dir][beam_ix_len(m, len, depth, ctx_len, x)];
    return mix_values(m, a, b, c);
}

static inline double mix_sparse(const BeamModel *m, uint32_t dir,
                                uint32_t len, uint32_t depth,
                                uint32_t ctx_wide, uint32_t ctx_mid,
                                uint32_t ctx_len, uint32_t x) {
    const BeamLogRow a = beam_lp_wide_row(m, dir, depth, ctx_wide);
    const BeamLogRow b = beam_lp_mid_row(m, dir, depth, ctx_mid);
    const BeamLogRow c = beam_lp_len_row(m, dir, len, depth, ctx_len);
    if (!a.present || !b.present || !c.present) return -INFINITY;
    return mix_values(m, beam_log_cell(&a, x), beam_log_cell(&b, x),
                      beam_log_cell(&c, x));
}


/**
 * Sum of the mixed per-residue log terms for an ENCODED sequence, before the
 * length prior and the partition function are applied. `-INFINITY` as soon as
 * any residue is unreachable, which is what makes the closed world cheap: a
 * sequence the model cannot spell costs one position, not the whole length.
 *
 * Shared between scoring.c and repair.c. The repair factor scores 40L+20
 * one-edit neighbours of every sequence, so it must compute exactly the same
 * quantity as `beam_model_log_pseq`; two copies of this would eventually
 * disagree and the Bayes factor would stop being a ratio of like for like.
 */
double beam_raw_log_weight(const BeamModel *m, const uint8_t *s, uint32_t len);

#endif /* BEAM_INTERNAL_MIXING_H */
