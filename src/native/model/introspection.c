/**
 * @file introspection.c
 * @brief Configuration, dimensions, memory and per-length getters.
 */
#include <stdio.h>

#include "beam/internal.h"


/* ── Introspection ───────────────────────────────────────────── */

const BeamModelConfig *beam_model_config(const BeamModel *m) {
    return m ? &m->cfg : NULL;
}
double   beam_model_total_weight(const BeamModel *m) {
    return m ? m->total_weight : 0.0;
}
uint64_t beam_model_n_observed(const BeamModel *m) {
    return m ? m->n_observed : 0;
}
uint64_t beam_model_n_skipped(const BeamModel *m) {
    return m ? m->n_skipped : 0;
}
BeamAlphabet beam_model_alphabet(const BeamModel *m) {
    /* Amino acid for a NULL model, matching every other getter here, which
     * reports the zero of its type rather than refusing. */
    return m ? m->cfg.alphabet : BEAM_ALPHABET_AA;
}
uint32_t beam_model_n_residues(const BeamModel *m) {
    return m ? m->n_residues : 0;
}
uint32_t beam_model_radix(const BeamModel *m) {
    return m ? m->n_alphabet : 0;
}
uint32_t beam_model_n_states(const BeamModel *m) {
    return m ? m->n_states : 0;
}
uint64_t beam_model_n_cells(const BeamModel *m) {
    if (!m) return 0;
    return (uint64_t)BEAM_DIRS
         * beam_n_table(m);
}
uint64_t beam_model_n_nonzero_cells(const BeamModel *m) {
    if (!m || m->compacted) return 0;
    uint64_t n = 0;
    if (m->cfg.sparse) {
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            {
                /* Frozen counts already know this: a cell is stored only
                 * when it is present, so the answer is the set bits of the
                 * masks. Counting them beats thawing 95 MB to look. */
                if (m->counts_frozen) {
                    const BeamFrozenCounts *f =
                        beam_model_frozen_table(m, d);
                    for (uint32_t r = 0; r < f->n_rows; ++r)
                        n += (uint64_t)__builtin_popcount(
                            f->mask_width[r]
                            & ((1u << BEAM_FROZEN_MASK_BITS) - 1u));
                    continue;
                }
                const BeamSparseRows *t = &m->sp_cnt[d];
                /*
                 * Through the accessor rather than over `values[p]` directly.
                 * This walked the payload as doubles, which is right for a
                 * mutable count table and wrong for a packed one -- where
                 * those doubles are dictionary indices -- and this function is
                 * public, so the wrong number would reach Python looking
                 * exactly like a right one. Nothing packs a count table today.
                 * The refusal is what makes that a fact the code checks rather
                 * than one a reader has to know.
                 */
                uint64_t here = 0;
                if (!beam_sparse_count_positive_cells(t, &here)) {
                    beam_set_error(BEAM_ERR_INVALID_ARG,
                                   "this model's count tables are in a form "
                                   "the non-zero cell count cannot read "
                                   "exactly, so no number is returned rather "
                                   "than a wrong one");
                    return 0;
                }
                n += here;
            }
        }
        return n;
    }
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        size_t c;
        c = beam_n_table(m);
        for (size_t i = 0; i < c; ++i) if (m->cnt[d][i] > 0.0) n++;
    }
    return n;
}

double beam_model_log_length_prior(const BeamModel *m, uint32_t length) {
    /*
     * -inf is a real answer here -- it is what a length the model does not
     * generate is worth -- so the three ERROR conditions must be separated
     * from it by the error context rather than by the return value.
     */
    if (!m) return (void)BEAM_ENULL("model"), -INFINITY;
    if (!m->finalized) return (void)BEAM_ENOTBUILT(), -INFINITY;
    if (length >= m->n_len_slots)
        return (void)BEAM_ERANGE("%s: length %u is outside 0..%u",
                                 __func__, (unsigned)length,
                                 (unsigned)m->n_len_slots - 1u), -INFINITY;
    beam_clear_error();
    return m->log_pi[length];
}
double beam_model_log_partition(const BeamModel *m, uint32_t length) {
    /* Same split as `beam_model_log_length_prior` above. */
    if (!m) return (void)BEAM_ENULL("model"), -INFINITY;
    if (!m->finalized) return (void)BEAM_ENOTBUILT(), -INFINITY;
    if (length >= m->n_len_slots)
        return (void)BEAM_ERANGE("%s: length %u is outside 0..%u",
                                 __func__, (unsigned)length,
                                 (unsigned)m->n_len_slots - 1u), -INFINITY;
    beam_clear_error();
    return m->log_z[length];
}
