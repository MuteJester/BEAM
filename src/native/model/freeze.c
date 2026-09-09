/**
 * @file freeze.c
 * @brief Moving counts between their mutable and frozen forms.
 *
 * Finalize freezes; anything that needs to write counts again thaws first.
 * The two forms hold the same bits -- see frozen_counts.h -- so which one a
 * model is carrying changes nothing it reports, only what it costs.
 */
#include "beam/internal.h"
#include "beam/frozen_counts.h"

/** The three tables of one direction, in the order the kinds enumerate. */
static BeamSparseRows *mutable_table(BeamModel *m, uint32_t dir, uint32_t kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->sp_cnt_wide[dir];
    if (kind == BEAM_TABLE_MID)  return &m->sp_cnt_mid[dir];
    return &m->sp_cnt_len[dir];
}
static BeamFrozenCounts *frozen_table(BeamModel *m, uint32_t dir, uint32_t kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->frz_wide[dir];
    if (kind == BEAM_TABLE_MID)  return &m->frz_mid[dir];
    return &m->frz_len[dir];
}

const BeamFrozenCounts *beam_model_frozen_table(const BeamModel *m,
                                                uint32_t dir, uint32_t kind) {
    return frozen_table((BeamModel *)m, dir, kind);
}

typedef struct { const BeamModel *m; BeamTableKind kind; } PartCtx;
static uint32_t part_of_row(uint32_t row, void *ctx) {
    const PartCtx *c = (const PartCtx *)ctx;
    return beam_count_part_for_row(c->m, c->kind, row);
}

void beam_model_release_frozen(BeamModel *m) {
    if (!m) return;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        for (uint32_t k = BEAM_TABLE_WIDE; k <= BEAM_TABLE_LEN; ++k)
            beam_frozen_destroy(frozen_table(m, d, k));
    m->counts_frozen = false;
}

/**
 * Freeze every count table, releasing the mutable ones.
 *
 * All six are built before any is released, so a failure part-way through
 * leaves the model exactly as it was rather than half-converted. A dense
 * model is left alone: its counts are one flat array with no directory to
 * shed, and this form addresses a sparse table's shape.
 */
BeamError beam_model_freeze_counts(BeamModel *m) {
    if (!m) return BEAM_ENULL("model");
    if (!m->cfg.sparse || m->counts_frozen || m->compacted) return BEAM_OK;

    BeamFrozenCounts built[BEAM_DIRS][3];
    memset(built, 0, sizeof built);
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        for (uint32_t k = BEAM_TABLE_WIDE; k <= BEAM_TABLE_LEN; ++k) {
            const BeamSparseRows *src = mutable_table(m, d, k);
            const BeamError e = beam_frozen_freeze(src, src->n_parts,
                                                   &built[d][k]);
            if (e != BEAM_OK) {
                for (uint32_t a = 0; a < BEAM_DIRS; ++a)
                    for (uint32_t b = 0; b < 3u; ++b)
                        beam_frozen_destroy(&built[a][b]);
                return e;
            }
        }

    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        for (uint32_t k = BEAM_TABLE_WIDE; k <= BEAM_TABLE_LEN; ++k) {
            *frozen_table(m, d, k) = built[d][k];
            beam_sparse_destroy(mutable_table(m, d, k));
        }
    m->counts_frozen = true;
    return BEAM_OK;
}

/* One table, into a caller-owned local; the model is not touched. The
 * contract is in frozen_counts.h, beside the reason it exists. */
BeamError beam_model_thaw_count_table(const BeamModel *m, uint32_t dir,
                                      uint32_t kind, BeamSparseRows *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "destination");
    memset(out, 0, sizeof *out);
    if (!m->counts_frozen) return BEAM_OK;      /* nothing frozen to expand */
    PartCtx ctx = { m, (BeamTableKind)kind };
    return beam_frozen_thaw(frozen_table((BeamModel *)m, dir, kind),
                            beam_count_n_rows(m, (BeamTableKind)kind),
                            m->n_depth_wide,
                            part_of_row, &ctx, out);
}

/**
 * Restore the mutable form, releasing the frozen one.
 *
 * Same all-or-nothing shape as freezing: every table is rebuilt before any
 * frozen one is dropped, so a model that runs out of memory here still has
 * its counts.
 */
BeamError beam_model_thaw_counts(BeamModel *m) {
    if (!m) return BEAM_ENULL("model");
    if (!m->counts_frozen) return BEAM_OK;

    BeamSparseRows built[BEAM_DIRS][3];
    memset(built, 0, sizeof built);
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        for (uint32_t k = BEAM_TABLE_WIDE; k <= BEAM_TABLE_LEN; ++k) {
            PartCtx ctx = { m, (BeamTableKind)k };
            const BeamError e = beam_frozen_thaw(
                frozen_table(m, d, k), beam_count_n_rows(m, (BeamTableKind)k),
                m->n_depth_wide, part_of_row, &ctx, &built[d][k]);
            if (e != BEAM_OK) {
                for (uint32_t a = 0; a < BEAM_DIRS; ++a)
                    for (uint32_t b = 0; b < 3u; ++b)
                        beam_sparse_destroy(&built[a][b]);
                return e;
            }
        }

    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        for (uint32_t k = BEAM_TABLE_WIDE; k <= BEAM_TABLE_LEN; ++k) {
            beam_sparse_destroy(mutable_table(m, d, k));
            *mutable_table(m, d, k) = built[d][k];
            beam_frozen_destroy(frozen_table(m, d, k));
        }
    m->counts_frozen = false;
    return BEAM_OK;
}

uint64_t beam_model_frozen_bytes(const BeamModel *m) {
    if (!m || !m->counts_frozen) return 0u;
    uint64_t n = 0;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        for (uint32_t k = BEAM_TABLE_WIDE; k <= BEAM_TABLE_LEN; ++k)
            n += beam_frozen_memory_bytes(beam_model_frozen_table(m, d, k));
    return n;
}
