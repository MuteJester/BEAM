/**
 * @file accounting.c
 * @brief What a model reports about its own size.
 *
 * One of the files under src/native/memory/.
 *
 * `beam_model_memory_bytes` reports the model's TABLES, and is exact to
 * within 0.2 MB of what the build actually allocates. It does not report the
 * per-length live cache, which is not the model -- it is a memoisation that
 * can be dropped and rebuilt -- but which is real memory the process holds
 * and, on a per-repertoire model, is larger than the model itself:
 *
 *              model_memory_bytes   live cache   actually allocated
 *   repertoire         13.8 MB        15.1 MB          29.2 MB
 *   foundation        806.7 MB       382.1 MB       1,190.4 MB
 *
 * Folding the cache into the model's figure would make that number mean two
 * things at once and would change with when it was asked. Reporting it
 * separately, and giving callers a way to release it, is what
 * `beam_model_live_cache_bytes` and `beam_model_drop_caches` are for.
 */
#include "beam/internal.h"
#include "beam/memory.h"

static uint64_t table_bytes(const BeamModel *m);

/* The hierarchical prior's lower levels, held in either storage form. */
static uint64_t lower_level_bytes(const BeamModel *m) {
    uint64_t n = 0;
    if (!m->finalized) return 0;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        for (uint32_t k = 0; k < m->order; ++k)
            n += beam_sparse_memory_bytes(&m->sp_lo[d][k]);
    return n;
}

uint64_t beam_model_memory_bytes(const BeamModel *m) {
    if (!m) return 0;
    return table_bytes(m) + lower_level_bytes(m);
}

/* The top tables and the counts, in whichever storage form the model uses. */
static uint64_t table_bytes(const BeamModel *m) {
    if (m->cfg.sparse) {
        uint64_t n = 0;
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            if (!m->compacted) {
                /* Zero for each of these once the counts are frozen; the
                 * frozen bytes are added below, outside the direction loop. */
                n += beam_sparse_memory_bytes(&m->sp_cnt[d]);
            }
            if (m->finalized) {
                n += beam_sparse_memory_bytes(&m->sp_lp[d]);
            }
        }
        if (!m->compacted) n += beam_model_frozen_bytes(m);
        return n;
    }
    uint64_t per_dir = (uint64_t)beam_n_table(m) * sizeof(double);
    uint64_t n = 0;
    if (!m->compacted) n += per_dir * BEAM_DIRS;   /* counts */
    if (m->finalized)  n += per_dir * BEAM_DIRS;   /* log tables */
    return n;
}

uint64_t beam_model_live_cache_bytes(const BeamModel *m) {
    return beam_live_cache_bytes(m);
}

void beam_model_drop_caches(BeamModel *m) {
    beam_live_cache_drop(m);
}
