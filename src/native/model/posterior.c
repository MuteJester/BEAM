/**
 * @file posterior.c
 * @brief Dirichlet-multinomial posterior update.
 */
#include <stdio.h>

#include "beam/internal.h"


/* ── Bayesian posterior ──────────────────────────────────────── */

/**
 * One table's Dirichlet-multinomial update: `dst += kappa * normalise(src)`,
 * row by row.
 *
 * A row is one categorical over the 20 residues, so normalising it is what
 * turns a prior's COUNTS into the prior BELIEF that `kappa` pseudo-counts of
 * weight are then contributed to. A row the prior never saw stays untouched:
 * an empty row carries no belief, and spreading `kappa` uniformly over it
 * would invent one.
 */
static void posterior_table(double *dst, const double *src, size_t n_rows,
                            uint32_t R, double kappa) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t r = 0; r < (int64_t)n_rows; ++r) {
        const double *sr = src + (size_t)r * R;
        double *dr = dst + (size_t)r * R;
        double tot = 0.0;
        for (uint32_t x = 0; x < R; ++x) tot += sr[x];
        if (!(tot > 0.0)) continue;
        const double scale = kappa / tot;
        for (uint32_t x = 0; x < R; ++x) dr[x] += sr[x] * scale;
    }
}

/** Row-addressed posterior used whenever either operand is sparse. */
static bool posterior_rows(BeamModel *dst_m, const BeamModel *src_m,
                           uint32_t dir, BeamTableKind kind,
                           double kappa) {
    const uint32_t nr = beam_count_n_rows(dst_m, kind);
    const uint32_t R = dst_m->n_residues;
    for (uint32_t row = 0; row < nr; ++row) {
        const double *src = beam_count_row(src_m, dir, kind, row);
        if (!src) continue;
        double tot = 0.0;
        for (uint32_t x = 0; x < R; ++x) tot += src[x];
        if (!(tot > 0.0)) continue;
        double *dst = beam_count_row_mut(dst_m, dir, kind, row, true);
        if (!dst) return false;
        const double scale = kappa / tot;
        for (uint32_t x = 0; x < R; ++x)
            dst[x] += src[x] * scale;
    }
    return true;
}

BeamError beam_model_posterior(BeamModel *m, const BeamModel *prior,
                                  double kappa) {
    if (!m || !prior) return BEAM_ENULL(!m ? "model" : "prior");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (m->compacted || prior->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "posterior: the %s model is compacted, so its counts "
                        "are gone", m->compacted ? "target" : "prior");
    if (!(kappa >= 0.0) || !isfinite(kappa))
        return BEAM_ERANGE("posterior: kappa must be finite and >= 0, "
                               "got %.17g", kappa);
    if (m->cfg.alphabet != prior->cfg.alphabet
        || m->order != prior->order || m->max_len != prior->max_len
        || m->n_depth_wide != prior->n_depth_wide)
        return BEAM_FAIL(BEAM_ERR_VARIANT_MISMATCH,
                        "posterior: shape mismatch, target is %s order=%u "
                        "max_length=%u depth_cap=%u and the prior is "
                        "%s order=%u max_length=%u depth_cap=%u",
                        m->alpha->name,
                        (unsigned)m->order, (unsigned)m->max_len,
                        (unsigned)m->n_depth_wide - 1u,
                        prior->alpha->name,
                        (unsigned)prior->order, (unsigned)prior->max_len,
                        (unsigned)prior->n_depth_wide - 1u);
    if (kappa == 0.0) { m->finalized = false; return BEAM_OK; }

    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        if (!m->cfg.sparse && !prior->cfg.sparse) {
            /* Preserve the original dense/open-world path byte for byte. */
            posterior_table(m->cnt_wide[d], prior->cnt_wide[d],
                            beam_n_wide(m) / m->n_residues,
                            m->n_residues, kappa);
            posterior_table(m->cnt_mid[d], prior->cnt_mid[d],
                            beam_n_mid(m) / m->n_residues,
                            m->n_residues, kappa);
            posterior_table(m->cnt_len[d], prior->cnt_len[d],
                            beam_n_len(m) / m->n_residues,
                            m->n_residues, kappa);
        } else if (!posterior_rows(m, prior, d, BEAM_TABLE_WIDE, kappa)
                   || !posterior_rows(m, prior, d, BEAM_TABLE_MID, kappa)
                   || !posterior_rows(m, prior, d, BEAM_TABLE_LEN, kappa)) {
            m->finalized = false;
            return BEAM_EALLOC("a sparse posterior row");
        }
    }
    /* The length histogram is one categorical too, over lengths rather than
     * residues, and it has to move with the rest: leaving it alone would give
     * the posterior the individual's length distribution with the prior's
     * conditionals, which is not a posterior of anything. */
    {
        double tot = 0.0;
        for (uint32_t L = 0; L < prior->n_len_slots; ++L) tot += prior->cnt_length[L];
        if (tot > 0.0) {
            const double scale = kappa / tot;
            for (uint32_t L = 0; L < m->n_len_slots; ++L)
                m->cnt_length[L] += prior->cnt_length[L] * scale;
        }
    }
    /*
     * The sequence counters do NOT move. `total_weight` and `n_observed`
     * report what was observed, and a pseudo-count is not an observation;
     * reporting kappa there would make a leave-one-out corpus size depend on
     * the smoothing strength.
     */
    m->finalized = false;
    return BEAM_OK;
}
