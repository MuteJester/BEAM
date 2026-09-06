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
         * (beam_n_wide(m) + beam_n_mid(m) + beam_n_len(m));
}
uint64_t beam_model_n_nonzero_cells(const BeamModel *m) {
    if (!m || m->compacted) return 0;
    uint64_t n = 0;
    if (m->cfg.sparse) {
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            for (uint32_t kind = BEAM_TABLE_WIDE;
                 kind <= BEAM_TABLE_LEN; ++kind) {
                const BeamSparseRows *t = beam_sparse_count_table_const(
                    m, d, (BeamTableKind)kind);
                for (uint32_t p = 0; p < t->n_parts; ++p) {
                    const double *v = t->values[p];
                    const size_t nc = (size_t)t->size[p] * m->n_residues;
                    for (size_t i = 0; i < nc; ++i) if (v[i] > 0.0) n++;
                }
            }
        }
        return n;
    }
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        size_t c;
        c = beam_n_wide(m);
        for (size_t i = 0; i < c; ++i) if (m->cnt_wide[d][i] > 0.0) n++;
        c = beam_n_mid(m);
        for (size_t i = 0; i < c; ++i) if (m->cnt_mid[d][i] > 0.0) n++;
        c = beam_n_len(m);
        for (size_t i = 0; i < c; ++i) if (m->cnt_len[d][i] > 0.0) n++;
    }
    return n;
}
uint64_t beam_model_memory_bytes(const BeamModel *m) {
    if (!m) return 0;
    if (m->cfg.sparse) {
        uint64_t n = 0;
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            if (!m->compacted) {
                n += beam_sparse_memory_bytes(&m->sp_cnt_wide[d]);
                n += beam_sparse_memory_bytes(&m->sp_cnt_mid[d]);
                n += beam_sparse_memory_bytes(&m->sp_cnt_len[d]);
            }
            if (m->finalized) {
                n += beam_sparse_memory_bytes(&m->sp_lp_wide[d]);
                n += beam_sparse_memory_bytes(&m->sp_lp_mid[d]);
                n += beam_sparse_memory_bytes(&m->sp_lp_len[d]);
            }
        }
        return n;
    }
    uint64_t per_dir = (uint64_t)(beam_n_wide(m) + beam_n_mid(m)
                                  + beam_n_len(m)) * sizeof(double);
    uint64_t n = 0;
    if (!m->compacted) n += per_dir * BEAM_DIRS;   /* counts */
    if (m->finalized)  n += per_dir * BEAM_DIRS;   /* log tables */
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
