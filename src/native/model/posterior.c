/**
 * @file posterior.c
 * @brief Dirichlet-multinomial posterior update.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"


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
                           uint32_t dir,
                           double kappa) {
    const uint32_t nr = beam_count_n_rows(dst_m);
    const uint32_t R = dst_m->n_residues;
    for (uint32_t row = 0; row < nr; ++row) {
        const double *src = beam_cnt_row(src_m, dir, row);
        if (!src) continue;
        double tot = 0.0;
        for (uint32_t x = 0; x < R; ++x) tot += src[x];
        if (!(tot > 0.0)) continue;
        double *dst = beam_cnt_row_mut(dst_m, dir, row, true);
        if (!dst) return false;
        const double scale = kappa / tot;
        for (uint32_t x = 0; x < R; ++x)
            dst[x] += src[x] * scale;
    }
    return true;
}

BeamError beam_model_posterior(BeamModel *m, const BeamModel *prior,
                                  double kappa) {
    {   /* The prior is read and the model is written; both need count rows. */
        const BeamError ta = beam_model_thaw_counts(m);
        if (ta != BEAM_OK) return ta;
        const BeamError tb = beam_model_thaw_counts((BeamModel *)prior);
        if (tb != BEAM_OK) return tb;
    }

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
        || m->order != prior->order || m->max_len != prior->max_len)
        return BEAM_FAIL(BEAM_ERR_VARIANT_MISMATCH,
                        "posterior: shape mismatch, target is %s order=%u "
                        "max_length=%u and the prior is "
                        "%s order=%u max_length=%u",
                        m->alpha->name,
                        (unsigned)m->order, (unsigned)m->max_len,
                        prior->alpha->name,
                        (unsigned)prior->order, (unsigned)prior->max_len);
    if (kappa == 0.0) { m->finalized = false; return BEAM_OK; }

    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        if (!m->cfg.sparse && !prior->cfg.sparse) {
            /* The dense path, kept byte for byte. */
            posterior_table(m->cnt[d], prior->cnt[d],
                            beam_n_table(m) / m->n_residues,
                            m->n_residues, kappa);
        } else if (!posterior_rows(m, prior, d, kappa)) {
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

BeamError beam_model_set_prior(BeamModel *m, const double *alpha,
                               uint32_t n_alpha, double alpha_length) {
    if (!m) return BEAM_ENULL("model");
    if (m->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                         "set_prior: the model is compacted, so the counts the "
                         "prior would be re-applied to are gone");
    if (alpha && n_alpha != (uint32_t)m->order + 1u)
        return BEAM_ERANGE("set_prior: %u strengths given, and an order-%u "
                           "model has %u levels (0..order)", (unsigned)n_alpha,
                           (unsigned)m->order, (unsigned)m->order + 1u);
    BeamModelConfig c = m->cfg;
    c.alpha_auto = (alpha == NULL);
    if (alpha) for (uint32_t k = 0; k < n_alpha; ++k) c.alpha[k] = alpha[k];
    c.alpha_length_auto = isnan(alpha_length);
    if (!c.alpha_length_auto) c.alpha_length = alpha_length;
    char why[BEAM_CONFIG_REASON_MAX];
    const char *reason = beam_config_invalid_reason(&c, why, sizeof why);
    if (reason)
        return BEAM_ERANGE("set_prior: %s", reason);
    const BeamModelConfig old = m->cfg;
    m->cfg = c;
    const BeamError e = beam_model_finalize(m);
    if (e != BEAM_OK) m->cfg = old;     /* finalize left the old tables */
    return e;
}
