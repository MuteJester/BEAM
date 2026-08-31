/**
 * @file algebra.c
 * @brief Model arithmetic: add and subtract.
 */
#include <stdio.h>

#include "beam/internal.h"


BeamError beam_model_subtract(BeamModel *m, const BeamModel *o) {
    if (!m || !o)
        return BEAM_FAIL(BEAM_ERR_NULL_ARG, "subtract: %s must not be NULL",
                        !m ? "model" : "operand");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (m->compacted || o->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "subtract: the %s model is compacted, so its counts "
                        "are gone", m->compacted ? "target" : "operand");
    /* The alphabet belongs in the shape, not beside it: two models can agree
     * on order, length and depth and still have tables of different widths,
     * and the loops below would then read the operand off the end. */
    if (m->cfg.alphabet != o->cfg.alphabet
        || m->order != o->order || m->max_len != o->max_len
        || m->n_depth_wide != o->n_depth_wide)
        return BEAM_FAIL(BEAM_ERR_VARIANT_MISMATCH,
                        "subtract: shape mismatch, target is %s "
                        "order=%u max_length=%u depth_cap=%u and the operand "
                        "is %s order=%u max_length=%u depth_cap=%u",
                        m->alpha->name,
                        (unsigned)m->order, (unsigned)m->max_len,
                        (unsigned)m->n_depth_wide - 1u,
                        o->alpha->name,
                        (unsigned)o->order, (unsigned)o->max_len,
                        (unsigned)o->n_depth_wide - 1u);

    /*
     * Refuse a subtraction that would leave nothing behind, BEFORE touching a
     * single cell.
     *
     * The counts are the only copy there is: once they are decremented there
     * is nothing to roll back to, and the caller is left holding a model whose
     * `finalize` can only fail. Checking the length histogram first is cheap
     * (max_length + 1 cells) and turns an unrecoverable state into an error
     * code the caller can act on.
     */
    {
        double remaining = 0.0;
        for (uint32_t L = 0; L < m->n_len_slots; ++L) {
            const double v = m->cnt_length[L] - o->cnt_length[L];
            if (v > 0.0) remaining += v;
        }
        if (!(remaining > 0.0) || m->n_observed <= o->n_observed)
            return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT,
                            "subtract: the operand accounts for every "
                            "sequence in the target (%llu of %llu observed), "
                            "so the subtraction would empty the model; build a "
                            "fresh model instead",
                            (unsigned long long)o->n_observed,
                            (unsigned long long)m->n_observed);
    }

    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        for (uint32_t kind = BEAM_TABLE_WIDE;
             kind <= BEAM_TABLE_LEN; ++kind) {
            const BeamTableKind k = (BeamTableKind)kind;
            const uint32_t nr = beam_count_n_rows(m, k);
            for (uint32_t row = 0; row < nr; ++row) {
                double *dst = beam_count_row_mut(m, d, k, row, false);
                if (!dst) continue;
                const double *src = beam_count_row(o, d, k, row);
                if (!src) continue;
                bool any = false;
                for (uint32_t x = 0; x < m->n_residues; ++x) {
                    const double v = dst[x] - src[x];
                    dst[x] = v > 0.0 ? v : 0.0;
                    if (dst[x] > 0.0) any = true;
                }
                if (!any) (void)beam_count_erase(m, d, k, row);
            }
        }
    }
    for (uint32_t L = 0; L < m->n_len_slots; ++L) {
        const double v = m->cnt_length[L] - o->cnt_length[L];
        m->cnt_length[L] = v > 0.0 ? v : 0.0;
    }
    /* The sequence counters are part of the count state, so they have to move
     * with it: `add` adjusts all three, and an inverse that adjusted only one
     * would leave every leave-one-cohort-out workflow reporting a corpus size
     * that never shrinks. Floored at zero because the cells are. */
    m->total_weight = m->total_weight > o->total_weight
                    ? m->total_weight - o->total_weight : 0.0;
    m->n_observed   = m->n_observed > o->n_observed
                    ? m->n_observed - o->n_observed : 0u;
    m->n_skipped    = m->n_skipped > o->n_skipped
                    ? m->n_skipped - o->n_skipped : 0u;
    m->finalized = false;
    return BEAM_OK;
}

BeamError beam_model_add(BeamModel *m, const BeamModel *o) {
    if (!m || !o) return BEAM_ENULL(!m ? "model" : "operand");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (m->compacted || o->compacted)
        return BEAM_ECOMPACTED("its counts are gone");
    if (m->cfg.alphabet != o->cfg.alphabet
        || m->order != o->order || m->max_len != o->max_len
        || m->n_depth_wide != o->n_depth_wide)
        return BEAM_FAIL(BEAM_ERR_VARIANT_MISMATCH,
                        "add: shape mismatch, target is %s order=%u "
                        "max_length=%u depth_cap=%u and the operand is "
                        "%s order=%u max_length=%u depth_cap=%u",
                        m->alpha->name,
                        (unsigned)m->order, (unsigned)m->max_len,
                        (unsigned)m->n_depth_wide - 1u,
                        o->alpha->name,
                        (unsigned)o->order, (unsigned)o->max_len,
                        (unsigned)o->n_depth_wide - 1u);

    if (!m->cfg.sparse && !o->cfg.sparse) {
        /* Preserve the original dense/open-world traversal exactly. */
        #define ADD(dst, src, count)                                    \
            do { size_t _n = (count);                                   \
                 for (size_t _i = 0; _i < _n; ++_i)                     \
                     (dst)[_i] += (src)[_i];                             \
            } while (0)
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            ADD(m->cnt_wide[d], o->cnt_wide[d], beam_n_wide(m));
            ADD(m->cnt_mid[d],  o->cnt_mid[d],  beam_n_mid(m));
            ADD(m->cnt_len[d],  o->cnt_len[d],  beam_n_len(m));
        }
        #undef ADD
    } else {
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            for (uint32_t kind = BEAM_TABLE_WIDE;
                 kind <= BEAM_TABLE_LEN; ++kind) {
                const BeamTableKind k = (BeamTableKind)kind;
                const uint32_t nr = beam_count_n_rows(m, k);
                for (uint32_t row = 0; row < nr; ++row) {
                    const double *src = beam_count_row(o, d, k, row);
                    if (!src) continue;
                    bool any = false;
                    for (uint32_t x = 0; x < m->n_residues; ++x)
                        if (src[x] != 0.0) { any = true; break; }
                    if (!any) continue;
                    double *dst = beam_count_row_mut(m, d, k, row, true);
                    if (!dst) {
                        m->finalized = false;
                        return BEAM_EALLOC("a sparse count row during add");
                    }
                    for (uint32_t x = 0; x < m->n_residues; ++x)
                        dst[x] += src[x];
                }
            }
        }
    }
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        m->cnt_length[L] += o->cnt_length[L];
    m->total_weight += o->total_weight;
    m->n_observed   += o->n_observed;
    m->n_skipped    += o->n_skipped;
    m->finalized = false;
    return BEAM_OK;
}
