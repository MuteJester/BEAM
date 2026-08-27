/**
 * @file table_access.h
 * @brief Dense and sparse row access.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_TABLE_ACCESS_H
#define BEAM_INTERNAL_TABLE_ACCESS_H

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/sparse_rows.h"
#include "beam/indexing.h"

/* ── Dense/sparse row access ────────────────────────────────── */

static inline uint32_t beam_depth_part(const BeamModel *m,
                                           uint32_t depth) {
    return depth < m->cfg.depth_cap ? depth : m->cfg.depth_cap;
}

static inline uint32_t beam_wide_part_for_row(const BeamModel *m,
                                                   uint32_t row) {
    return row / m->n_ctx_wide;
}

static inline uint32_t beam_mid_part_for_row(const BeamModel *m,
                                                  uint32_t row) {
    return beam_depth_part(m, row / m->n_ctx_mid);
}

static inline uint32_t beam_len_part_for_row(const BeamModel *m,
                                                  uint32_t row) {
    const uint32_t depth = (row / m->n_ctx_len) % m->max_len;
    return beam_depth_part(m, depth);
}

static inline const double *beam_cnt_wide_row(const BeamModel *m,
                                                   uint32_t dir,
                                                   uint32_t row) {
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_cnt_wide[dir], row,
                              beam_wide_part_for_row(m, row))
        : m->cnt_wide[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_cnt_mid_row(const BeamModel *m,
                                                  uint32_t dir,
                                                  uint32_t row) {
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_cnt_mid[dir], row,
                              beam_mid_part_for_row(m, row))
        : m->cnt_mid[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_cnt_len_row(const BeamModel *m,
                                                  uint32_t dir,
                                                  uint32_t row) {
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_cnt_len[dir], row,
                              beam_len_part_for_row(m, row))
        : m->cnt_len[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline double *beam_cnt_wide_row_mut(BeamModel *m, uint32_t dir,
                                                 uint32_t row, bool create) {
    return m->cfg.sparse
        ? beam_sparse_row_mut(&m->sp_cnt_wide[dir], row,
                                  beam_wide_part_for_row(m, row), create)
        : m->cnt_wide[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline double *beam_cnt_mid_row_mut(BeamModel *m, uint32_t dir,
                                                uint32_t row, bool create) {
    return m->cfg.sparse
        ? beam_sparse_row_mut(&m->sp_cnt_mid[dir], row,
                                  beam_mid_part_for_row(m, row), create)
        : m->cnt_mid[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline double *beam_cnt_len_row_mut(BeamModel *m, uint32_t dir,
                                                uint32_t row, bool create) {
    return m->cfg.sparse
        ? beam_sparse_row_mut(&m->sp_cnt_len[dir], row,
                                  beam_len_part_for_row(m, row), create)
        : m->cnt_len[dir] + (size_t)row * BEAM_RESIDUES;
}

typedef enum {
    BEAM_TABLE_WIDE = 0,
    BEAM_TABLE_MID  = 1,
    BEAM_TABLE_LEN  = 2
} BeamTableKind;

static inline uint32_t beam_count_n_rows(const BeamModel *m,
                                              BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE)
        return m->n_depth_wide * m->n_ctx_wide;
    if (kind == BEAM_TABLE_MID)
        return m->max_len * m->n_ctx_mid;
    return m->n_len_slots * m->max_len * m->n_ctx_len;
}

static inline uint32_t beam_count_part_for_row(const BeamModel *m,
                                                    BeamTableKind kind,
                                                    uint32_t row) {
    if (kind == BEAM_TABLE_WIDE) return beam_wide_part_for_row(m, row);
    if (kind == BEAM_TABLE_MID) return beam_mid_part_for_row(m, row);
    return beam_len_part_for_row(m, row);
}

static inline BeamSparseRows *beam_sparse_count_table(
        BeamModel *m, uint32_t dir, BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->sp_cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return &m->sp_cnt_mid[dir];
    return &m->sp_cnt_len[dir];
}

static inline const BeamSparseRows *beam_sparse_count_table_const(
        const BeamModel *m, uint32_t dir, BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->sp_cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return &m->sp_cnt_mid[dir];
    return &m->sp_cnt_len[dir];
}

static inline double *beam_dense_count_table(BeamModel *m,
                                                  uint32_t dir,
                                                  BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return m->cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return m->cnt_mid[dir];
    return m->cnt_len[dir];
}

static inline const double *beam_dense_count_table_const(
        const BeamModel *m, uint32_t dir, BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return m->cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return m->cnt_mid[dir];
    return m->cnt_len[dir];
}

static inline const double *beam_count_row(const BeamModel *m,
                                                uint32_t dir,
                                                BeamTableKind kind,
                                                uint32_t row) {
    if (!m->cfg.sparse)
        return beam_dense_count_table_const(m, dir, kind)
             + (size_t)row * BEAM_RESIDUES;
    return beam_sparse_row(beam_sparse_count_table_const(m, dir, kind),
                               row, beam_count_part_for_row(m, kind, row));
}

static inline double *beam_count_row_mut(BeamModel *m, uint32_t dir,
                                              BeamTableKind kind,
                                              uint32_t row, bool create) {
    if (!m->cfg.sparse)
        return beam_dense_count_table(m, dir, kind)
             + (size_t)row * BEAM_RESIDUES;
    return beam_sparse_row_mut(beam_sparse_count_table(m, dir, kind),
                                   row,
                                   beam_count_part_for_row(m, kind, row),
                                   create);
}

static inline bool beam_count_erase(BeamModel *m, uint32_t dir,
                                         BeamTableKind kind, uint32_t row) {
    if (!m->cfg.sparse) return false;
    return beam_sparse_erase(beam_sparse_count_table(m, dir, kind),
                                 row,
                                 beam_count_part_for_row(m, kind, row));
}

static inline const double *beam_lp_wide_row(const BeamModel *m,
                                                  uint32_t dir,
                                                  uint32_t depth,
                                                  uint32_t ctx) {
    const uint32_t d = beam_depth_part(m, depth);
    const uint32_t row = d * m->n_ctx_wide + ctx;
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_lp_wide[dir], row, 0u)
        : m->lp_wide[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_lp_mid_row(const BeamModel *m,
                                                 uint32_t dir,
                                                 uint32_t depth,
                                                 uint32_t ctx) {
    const uint32_t row = depth * m->n_ctx_mid + ctx;
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_lp_mid[dir], row, 0u)
        : m->lp_mid[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_lp_len_row(const BeamModel *m,
                                                 uint32_t dir, uint32_t len,
                                                 uint32_t depth,
                                                 uint32_t ctx) {
    const uint32_t row = (len * m->max_len + depth) * m->n_ctx_len + ctx;
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_lp_len[dir], row, 0u)
        : m->lp_len[dir] + (size_t)row * BEAM_RESIDUES;
}

#endif /* BEAM_INTERNAL_TABLE_ACCESS_H */
