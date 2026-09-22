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

/*
 * Occupied rows and cells per (direction, depth).
 *
 * This is what "how much of the dictionary did we discover" is built from.
 * A sequence of length L reads forward depths 0..L-1 and backward depths
 * 0..L-1, so the rows available to it are exactly those below L, and the
 * rows that COULD exist are `L * n_ctx` per direction. Bucketing occupancy
 * by depth here lets a caller answer that for every length from one sweep.
 *
 * Counted through `beam_lp_row` rather than by walking the payload, for the
 * same reason `n_nonzero_cells` does: a packed table's payload holds
 * dictionary indices, not values, and reading it as doubles gives a number
 * that looks right. The accessor is the only thing that knows which form the
 * table is in.
 *
 * The C-anchored table is packed nearest-first, which relabels the CONTEXT
 * within a row index and leaves the depth part alone. Counting per depth is
 * therefore unaffected by the packing, which is why this does not need the
 * reversal helpers.
 */
BeamError beam_model_depth_occupancy(const BeamModel *m,
                                     uint64_t *rows, uint64_t *cells) {
    if (!m || !rows || !cells)
        return BEAM_ENULL(!m ? "model" : (!rows ? "rows" : "cells"));
    if (!m->finalized) return BEAM_ENOTBUILT();
    const uint32_t W = m->max_len, R = m->n_residues, nc = m->n_ctx;
    for (uint32_t i = 0; i < (uint32_t)BEAM_DIRS * W; ++i) rows[i] = cells[i] = 0;

    /*
     * The COUNTS are what this question is about.
     *
     * It reads "how much of what this length admits did the corpus reach",
     * and that is a fact about observations. This used to read the finalized
     * tables and call a cell reached when its log probability was not -inf,
     * which is a different question with the same answer in the closed world
     * and a useless one in the open world: the Laplace add-one leaves every
     * cell finite, so a model built from two sequences reported that it had
     * discovered 100% of the dictionary, and the same report carried
     * `n_nonzero_cells = 30` beside `cells_observed = 176400`.
     *
     * A compacted model has no counts. For a STRICT one the finalized tables
     * still answer exactly -- an unobserved cell is -inf and stays -inf -- so
     * that case is served from them and the figure survives `compact()`. For
     * an open-world one the smoothing has overwritten the evidence and the
     * counts that held it are gone, so there is no answer to give and this
     * says so rather than returning the saturated one.
     */
    if (m->compacted) {
        if (!m->cfg.strict)
            return BEAM_ECOMPACTED(
                "an open-world model's add-one leaves every finalized cell "
                "finite, so which cells the corpus actually reached is known "
                "only to the counts, and compact() released them");
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            for (uint32_t depth = 0; depth < W; ++depth) {
                uint64_t nr = 0, ncell = 0;
                for (uint32_t c = 0; c < nc; ++c) {
                    const BeamLogRow r = beam_lp_row(m, d, depth, c);
                    if (!r.present) continue;
                    uint32_t live = 0;
                    for (uint32_t x = 0; x < R; ++x)
                        if (beam_log_cell(&r, x) != -INFINITY) live++;
                    if (live) { nr++; ncell += live; }
                }
                rows[d * W + depth] = nr;
                cells[d * W + depth] = ncell;
            }
        }
        return BEAM_OK;
    }

    const uint32_t n_rows = beam_count_n_rows(m);
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        /*
         * Frozen counts already carry the answer: a cell is stored only when
         * it is present, so the residue mask IS the set of reached cells and
         * the row id gives the depth. Thawing to look would cost 95 MB, and
         * `beam_model_n_nonzero_cells` declines it for the same reason.
         */
        if (m->counts_frozen) {
            const BeamFrozenCounts *f = beam_model_frozen_table(m, d);
            for (uint32_t k = 0; k < f->n_rows; ++k) {
                const uint32_t row = beam_frozen_row_id(f, k);
                const uint32_t depth = row / nc;
                if (depth >= W) continue;
                const uint32_t live = (uint32_t)__builtin_popcount(
                    f->mask_width[k] & ((1u << BEAM_FROZEN_MASK_BITS) - 1u));
                if (live) { rows[d * W + depth]++; cells[d * W + depth] += live; }
            }
            continue;
        }
        for (uint32_t row = 0; row < n_rows; ++row) {
            const double *cr = beam_cnt_row(m, d, row);
            if (!cr) continue;                    /* absent sparse row */
            uint32_t live = 0;
            for (uint32_t x = 0; x < R; ++x) if (cr[x] > 0.0) live++;
            if (!live) continue;
            const uint32_t depth = row / nc;
            rows[d * W + depth]++;
            cells[d * W + depth] += live;
        }
    }
    return BEAM_OK;
}

/*
 * When the model was first built, and what observing and finalizing cost.
 *
 * One call for three because they are always wanted together, and because a
 * caller assembling them from three accessors could assemble them from three
 * different models without noticing.
 */
BeamError beam_model_provenance(const BeamModel *m, int64_t *created_unix,
                                int64_t *observe_nanos,
                                int64_t *finalize_nanos) {
    if (!m) return BEAM_ENULL("model");
    if (created_unix)   *created_unix   = m->created_unix;
    if (observe_nanos)  *observe_nanos  = m->observe_nanos;
    if (finalize_nanos) *finalize_nanos = m->finalize_nanos;
    return BEAM_OK;
}
