/**
 * @file temper.c
 * @brief Tempering: `P_t(s) proportional to P(s)^t`, in place on a finalized model.
 *
 * The tempered law already drives the surprisal moments, the spectrum and the
 * saddlepoint, where it is a measure a sweep carries rather than a model. This
 * makes it a model, so it can be scored, sampled and re-analysed by everything
 * the ordinary one can.
 *
 * `ln P(s) = ln pi(L) + S(s) - ln Z(L)`, so `t ln P(s)` is `t S(s)` against a
 * length term, and `t S(s)` is what the tables scaled by `t` spell. Three
 * quantities move:
 *
 *   tables   ->  t * tables
 *   Z_t(L)   =   the same forward sweep, on the scaled tables
 *   pi_t(L)  =   pi(L)^t * Z_t(L) / Z(L)^t,  normalised over L
 *
 * and the shift that normalises the last is exactly `ln sum_s P(s)^t`, so the
 * result satisfies `ln P_t(s) = t ln P(s) - ln sum_s P(s)^t` identically
 * rather than to within a tolerance.
 */
#include <math.h>
#include <stdlib.h>

#include "beam/internal.h"
#include "beam/model_build.h"

/*
 * Scale every stored log-probability of one finalized sparse table.
 *
 * `cell_mask` is the discriminator, and it is the one `pack_cells` itself
 * leaves behind: a packed table holds each distinct value once in `dict` and
 * its cells are indices into that, so scaling the dictionary scales every cell
 * reading it, and `values[0]` has been freed. An unpacked table holds
 * `rows * stride` doubles in `values[0]`, where interning may have several
 * dense rows sharing one payload row -- which is why the payload is scaled
 * rather than the rows, so a shared row is scaled once.
 *
 * A non-finite cell never enters the dictionary, `pack_cells` skipping it, and
 * is -inf where it is stored otherwise. `t > 0` leaves -inf at -inf either
 * way, so the support survives with no special case.
 */
static void scale_sparse_log(BeamSparseRows *t, double s) {
    if (!t) return;
    if (t->cell_mask) {
        for (uint32_t i = 0; i < t->n_dict; ++i) t->dict[i] *= s;
        return;
    }
    if (!t->values || !t->values[0] || !t->size || !t->stride) return;
    const uint32_t rows = t->n_pool ? t->n_pool : t->size[0];
    const size_t n = (size_t)rows * t->stride;
    for (size_t i = 0; i < n; ++i) t->values[0][i] *= s;
}

BeamError beam_model_temper(BeamModel *m, double t) {
    if (!m) return BEAM_ENULL("model");
    if (!isfinite(t) || t <= 0.0)
        return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                        "temper: t must be finite and strictly positive, got "
                        "%g. At t = 0 every finite log weight becomes zero and "
                        "every -inf becomes NaN, and a negative t turns an "
                        "unspellable edge into the likeliest one", t);
    if (!m->finalized)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "temper: the model is not finalized, so it has no log "
                        "tables to raise to a power");
    /* The live cache is a projection of the finalized tables, so it describes
     * a model that stops existing below. Dropped BEFORE the change, the order
     * `add` and `compact` both use. */
    beam_live_cache_drop(m);

    /*
     * Z(L) at exponent one. The tempered length prior needs it and the scaling
     * below destroys it, so it is taken first and in full.
     */
    double *z1 = (double *)malloc((size_t)m->n_len_slots * sizeof *z1);
    if (!z1) return BEAM_EALLOC("the untempered partition functions");
    for (uint32_t L = 0; L < m->n_len_slots; ++L) z1[L] = m->log_z[L];

    if (t != 1.0) {
        const size_t n = beam_n_table(m);
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            if (m->lp[d])
                for (size_t i = 0; i < n; ++i) m->lp[d][i] *= t;
            scale_sparse_log(&m->sp_lp[d], t);
            /* An unobserved row IS its nearest observed ancestor, so the
             * ancestors and the uniform row below them are scaled too, or an
             * unobserved row would come back untempered. */
            for (uint32_t k = 0; k < m->order; ++k)
                scale_sparse_log(&m->sp_lo[d][k], t);
        }
        for (uint32_t x = 0; x < m->n_residues; ++x) m->base_row[x] *= t;
    }

    /*
     * Z_t(L), by the same sweep finalize uses. A length whose prior or whose
     * untempered partition was already dead stays dead: tempering reweights
     * the support, it does not widen it.
     */
    for (uint32_t L = 0; L < m->n_len_slots; ++L) {
        if (L >= 1u && isfinite(m->log_pi[L]) && isfinite(z1[L])) {
            double z = -INFINITY;
            const BeamError e = beam_compute_log_z(m, L, &z, NULL);
            if (e != BEAM_OK) { free(z1); return e; }
            m->log_z[L] = z;
        } else {
            m->log_z[L] = -INFINITY;
        }
    }

    /*
     * pi_t(L) proportional to pi(L)^t * Z_t(L) / Z(L)^t.
     *
     * Accumulated as a log-sum-exp against the maximum, because the
     * unnormalised terms are `t ln` of a probability and a large `t` on a long
     * model puts them far below the underflow point while their RATIOS stay
     * perfectly representable.
     */
    double top = -INFINITY;
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        double u = -INFINITY;
        if (isfinite(m->log_pi[L]) && isfinite(z1[L]) && isfinite(m->log_z[L]))
            u = t * m->log_pi[L] + m->log_z[L] - t * z1[L];
        m->log_pi[L] = u;
        if (u > top) top = u;
    }
    if (!isfinite(top)) {
        free(z1);
        return BEAM_FAIL(BEAM_ERR_INVALID_ARG,
                        "temper: no length keeps a finite mass at t = %g, so "
                        "the tempered model would have no support", t);
    }
    double acc = 0.0;
    for (uint32_t L = 1; L < m->n_len_slots; ++L)
        if (isfinite(m->log_pi[L])) acc += exp(m->log_pi[L] - top);
    const double shift = top + log(acc);
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        m->log_pi[L] = (L >= 1u && isfinite(m->log_pi[L]))
                     ? m->log_pi[L] - shift : -INFINITY;
    free(z1);

    /*
     * The counts go, and this is a correctness guard rather than a saving.
     *
     * A tempered model's distribution is no longer a function of its counts,
     * so every count-level operation on it -- add, subtract, posterior, copy,
     * and the re-finalize each of them ends in -- would silently rebuild the
     * UNTEMPERED tables and report a plausible wrong number. Releasing the
     * counts makes all of them refuse instead, through the refusal
     * `compact()` already installs. It is also why tempering does not yet
     * compose by chaining: `tempered(u)` on a tempered model is refused, not
     * approximated. Carrying the exponent in the config is what would lift
     * that, and it belongs with the file format rather than here.
     */
    beam_free_counts(m);
    m->compacted = true;
    return BEAM_OK;
}
