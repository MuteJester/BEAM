/**
 * @file lifecycle.c
 * @brief Allocation, construction and destruction.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"


/* ── Lifecycle ───────────────────────────────────────────────── */

void beam_free_counts(BeamModel *m) {
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        free(m->cnt_wide[d]); m->cnt_wide[d] = NULL;
        free(m->cnt_mid[d]);  m->cnt_mid[d]  = NULL;
        free(m->cnt_len[d]);  m->cnt_len[d]  = NULL;
        beam_sparse_destroy(&m->sp_cnt_wide[d]);
        beam_sparse_destroy(&m->sp_cnt_mid[d]);
        beam_sparse_destroy(&m->sp_cnt_len[d]);
    }
    free(m->cnt_length); m->cnt_length = NULL;
}

void beam_free_tables(BeamModel *m) {
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        free(m->lp_wide[d]); m->lp_wide[d] = NULL;
        free(m->lp_mid[d]);  m->lp_mid[d]  = NULL;
        free(m->lp_len[d]);  m->lp_len[d]  = NULL;
        beam_sparse_destroy(&m->sp_lp_wide[d]);
        beam_sparse_destroy(&m->sp_lp_mid[d]);
        beam_sparse_destroy(&m->sp_lp_len[d]);
    }
    free(m->log_pi); m->log_pi = NULL;
    free(m->log_z);  m->log_z  = NULL;
}

BeamModel *beam_model_new(const BeamModelConfig *cfg) {
    BeamModelConfig c;
    if (cfg) c = *cfg; else beam_model_config_default(&c);
    /* A depth cap at or beyond the longest sequence resolves every depth
     * exactly, so clamping it is a no-op semantically and saves the caller
     * from having to match the two whenever max_length is reduced. */
    if (c.max_length >= 1 && c.depth_cap >= c.max_length)
        c.depth_cap = (uint8_t)(c.max_length - 1u);
    if (!beam_config_valid(&c)) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "invalid BEAM config: order=%u (need 2..%u), "
                      "max_length=%u (need 1..%u), depth_cap=%u (need 1..%u), "
                      "coef sum=%.17g (need 1), weights=%.17g/%.17g (need >= 0)",
                      (unsigned)c.order, (unsigned)BEAM_MAX_ORDER,
                      (unsigned)c.max_length, (unsigned)BEAM_MAX_LENGTH,
                      (unsigned)c.depth_cap,
                      (unsigned)(c.max_length ? c.max_length - 1u : 0u),
                      c.coef_wide + c.coef_mid + c.coef_len,
                      c.weight_forward, c.weight_backward);
        return NULL;
    }

    BeamModel *m = (BeamModel *)calloc(1, sizeof(*m));
    if (!m) {
        beam_set_error(BEAM_ERR_ALLOC, "out of memory allocating the BEAM model");
        return NULL;
    }
    m->cfg          = c;
    m->order        = c.order;
    m->max_len      = c.max_length;
    m->n_len_slots  = (uint32_t)c.max_length + 1u;
    /* The mid expert is order 3, so the state can never be narrower. */
    /* At least 3, and `state_weights_sparse` depends on it: its backward
     * hoist assumes the order-3 context lies entirely inside the state. */
    m->state_order  = m->order > 3u ? m->order : 3u;
    m->n_states     = beam_ipow_u32(BEAM_ALPHABET, m->state_order);
    m->state_shift  = beam_ipow_u32(BEAM_ALPHABET, m->state_order - 1u);
    m->n_ctx_wide   = beam_ipow_u32(BEAM_ALPHABET, m->order);
    m->n_ctx_mid    = beam_ipow_u32(BEAM_ALPHABET, 3u);
    m->n_ctx_len    = beam_ipow_u32(BEAM_ALPHABET, 2u);
    m->n_depth_wide = (uint32_t)c.depth_cap + 1u;

    bool ok = true;
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
        if (m->cfg.sparse) {
            ok = beam_sparse_init(
                     &m->sp_cnt_wide[d],
                     (uint32_t)(beam_n_wide(m) / BEAM_RESIDUES),
                     m->n_depth_wide, true) == BEAM_OK
              && beam_sparse_init(
                     &m->sp_cnt_mid[d],
                     (uint32_t)(beam_n_mid(m) / BEAM_RESIDUES),
                     m->n_depth_wide, true) == BEAM_OK
              && beam_sparse_init(
                     &m->sp_cnt_len[d],
                     (uint32_t)(beam_n_len(m) / BEAM_RESIDUES),
                     m->n_depth_wide, true) == BEAM_OK;
        } else {
            m->cnt_wide[d] = (double *)calloc(beam_n_wide(m), sizeof(double));
            m->cnt_mid[d]  = (double *)calloc(beam_n_mid(m),  sizeof(double));
            m->cnt_len[d]  = (double *)calloc(beam_n_len(m),  sizeof(double));
            ok = m->cnt_wide[d] && m->cnt_mid[d] && m->cnt_len[d];
        }
    }
    m->cnt_length = (double *)calloc(m->n_len_slots, sizeof(double));
    if (!ok || !m->cnt_length) {
        beam_set_error(BEAM_ERR_ALLOC,
                      "out of memory allocating BEAM count tables "
                      "(%llu logical cells at order %u, max_length %u, "
                      "sparse=%s)",
                      (unsigned long long)beam_model_n_cells(m),
                      (unsigned)m->order, (unsigned)m->max_len,
                      m->cfg.sparse ? "true" : "false");
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

void beam_model_destroy(BeamModel *m) {
    if (!m) return;
    beam_live_cache_drop(m);
    beam_free_counts(m);
    beam_free_tables(m);
    free(m);
}

bool beam_model_is_finalized(const BeamModel *m) {
    return m && m->finalized;
}

BeamError beam_model_compact(BeamModel *m) {
    if (!m) return BEAM_FAIL(BEAM_ERR_NULL_ARG, "compact: model must not be NULL");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (!m->finalized)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "compact: the model is not finalized, so releasing "
                        "its counts would leave nothing behind");
    beam_free_counts(m);
    m->compacted = true;
    return BEAM_OK;
}
