/**
 * @file lifecycle.c
 * @brief Allocation, construction and destruction.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/memory.h"
#include "beam/model_build.h"


/* ── Lifecycle ───────────────────────────────────────────────── */

void beam_free_counts(BeamModel *m) {
    /* Counts live in one of two forms and this releases whichever it is. */
    beam_model_release_frozen(m);
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        free(m->cnt[d]); m->cnt[d] = NULL;
        beam_sparse_destroy(&m->sp_cnt[d]);
    }
    free(m->cnt_length); m->cnt_length = NULL;
}

void beam_free_tables(BeamModel *m) {
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        free(m->lp[d]); m->lp[d] = NULL;
        beam_sparse_destroy(&m->sp_lp[d]);
    }
    free(m->log_pi); m->log_pi = NULL;
    free(m->log_z);  m->log_z  = NULL;
}

BeamModel *beam_model_new(const BeamModelConfig *cfg) {
    /* Also pulls memory/allocator.o into a static link, which a constructor
     * alone does not: see beam_memory_init. */
    beam_memory_init();
    BeamModelConfig c;
    if (cfg) c = *cfg; else beam_model_config_default(&c);
    /* Resolved before validation, because the legal range of `order` is read
     * from it and an unknown alphabet is a config that cannot be interpreted
     * rather than one that is merely out of range. */
    const BeamAlphabetInfo *a = beam_alphabet_info(c.alphabet);
    if (!a) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "unknown BEAM alphabet id %u",
                      (unsigned)c.alphabet);
        return NULL;
    }

    char why[BEAM_CONFIG_REASON_MAX];
    const char *reason = beam_config_invalid_reason(&c, why, sizeof why);
    if (reason) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "invalid BEAM config for the %s alphabet: %s",
                      a->name, reason);
        return NULL;
    }

    /*
     * A guard rather than an expectation. Each alphabet's order ceiling is
     * chosen so this cannot fire, and test_alphabet.c pins that for every
     * descriptor. It is here so that raising a ceiling without rechecking
     * fails loudly, instead of wrapping beam_ipow_u32 -- a uint32 multiply
     * loop with no overflow check -- and sizing every table from the wrapped
     * value.
     */
    const uint32_t state_order = c.order;   /* the state holds exactly one context */
    uint64_t n_states64 = 1;
    for (uint32_t i = 0; i < state_order; ++i) n_states64 *= a->radix;
    if (n_states64 > UINT32_MAX) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "the %s alphabet at order %u needs %llu DAG states, "
                      "which does not fit the 32 bits they are counted in",
                      a->name, (unsigned)c.order,
                      (unsigned long long)n_states64);
        return NULL;
    }

    BeamModel *m = (BeamModel *)calloc(1, sizeof(*m));
    if (!m) {
        beam_set_error(BEAM_ERR_ALLOC, "out of memory allocating the BEAM model");
        return NULL;
    }
    m->cfg          = c;
    m->alpha        = a;
    m->n_residues   = a->residues;
    m->n_alphabet   = a->radix;
    m->none_idx     = a->none;
    m->order        = c.order;
    m->max_len      = c.max_length;
    m->n_len_slots  = (uint32_t)c.max_length + 1u;
    /* The state holds exactly the residues one context needs, no more. */
    m->state_order  = state_order;
    m->n_states     = beam_ipow_u32(m->n_alphabet, m->state_order);
    m->state_shift  = beam_ipow_u32(m->n_alphabet, m->state_order - 1u);
    m->n_ctx        = beam_ipow_u32(m->n_alphabet, m->order);

    /*
     * A sparse table addresses its rows with a uint32, and the row count
     * below is cast to one. `max_len * n_ctx` reaches 2.57e9 at the largest
     * configuration the ceilings currently allow -- inside the 4.29e9 a
     * uint32 holds, but by less than a factor of two. That is the same bound
     * the capped wide table had, because the cap was itself clamped to
     * `max_length - 1`, so removing it moved nothing here. The margin is not
     * a thing to rely on silently: a truncating cast would size the row
     * directory from the wrapped value and every lookup past the wrap would
     * alias another row's counts.
     *
     * Checked here rather than in beam_config_valid because it needs the
     * derived dimensions, and checked for every model rather than only sparse
     * ones so that raising a ceiling cannot make the dense path the only one
     * that still works.
     */
    {
        const uint64_t rows = (uint64_t)m->max_len * m->n_ctx;
        /*
         * And refuse a configuration that cannot fit before allocating for
         * it, rather than after.
         *
         * calloc() on Linux succeeds for an absurd size and hands back lazily
         * mapped pages, so a model asking for hundreds of gigabytes does not
         * fail here -- it fails much later, when the tables are first
         * written, by which time the OOM killer is choosing which process on
         * the machine dies. That is not a failure mode a library gets to have:
         * a configuration this build cannot serve has to be refused while
         * refusing is still cheap.
         *
         * The estimate is what construction and finalize allocate together:
         * the count table now and the finalized log table later, both
         * directions. Sparse storage grows its payloads with the data instead,
         * so what a sparse config commits to up front is the row directory.
         */
        const uint64_t rows_total = rows * BEAM_DIRS;
        const uint64_t need = c.sparse
            ? rows_total * 2ull * sizeof(uint32_t)
            : rows_total * m->n_residues * 2ull * sizeof(double);
        const uint64_t budget = beam_model_memory_budget();
        if (need > budget) {
            const char *nu, *bu;
            const double nv = beam_budget_scaled(need, &nu);
            const double bv = beam_budget_scaled(budget, &bu);
            beam_set_error(BEAM_ERR_BUDGET,
                          "this %s configuration needs about %.2f %s for its "
                          "tables (order %u, max_length %u, %s) "
                          "and the memory budget is %.2f %s; raise it with "
                          "beam_model_set_memory_budget / "
                          "BEAM.set_memory_budget, or build a smaller model",
                          a->name, nv, nu,
                          (unsigned)c.order, (unsigned)c.max_length,
                          c.sparse ? "sparse" : "dense",
                          bv, bu);
            free(m);
            return NULL;
        }

        if (rows > UINT32_MAX) {
            beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                          "the %s alphabet at order %u with max_length %u "
                          "needs %llu table rows, which does not fit the 32 "
                          "bits a row index is held in",
                          a->name, (unsigned)c.order, (unsigned)c.max_length,
                          (unsigned long long)rows);
            free(m);
            return NULL;
        }
    }

    bool ok = true;
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
        if (m->cfg.sparse) {
            ok = beam_sparse_init(
                     &m->sp_cnt[d],
                     (uint32_t)(beam_n_table(m) / m->n_residues),
                     m->max_len, m->n_residues, true) == BEAM_OK;
        } else {
            m->cnt[d] = (double *)calloc(beam_n_table(m), sizeof(double));
            ok = m->cnt[d] != NULL;
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
