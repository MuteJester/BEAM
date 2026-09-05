/**
 * @file indel.c
 * @brief Insertion and deletion neighbourhoods, and what the model makes of them.
 *
 * Given an observed sequence, the sequences one residue longer that reduce to
 * it by a deletion, and the sequences one residue shorter obtained by
 * deleting from it. Each is scored, and reported two ways: as odds against
 * the source, and as a posterior over the neighbourhood.
 *
 * The two answer different questions and neither substitutes for the other.
 * The odds ask how much more likely `CASSAF` is than `CASSF`, and since
 * `log_pseq` carries the length prior that comparison is well posed across
 * lengths and the prior does not cancel. The posterior asks where and which,
 * GIVEN that one residue was inserted, and there the prior is a common factor
 * and does.
 *
 * NOTHING HERE INFERS AN EDIT. The model has no alignment and no indel
 * process; these are ratios between what it assigns to two strings that
 * happen to differ by one residue, and they describe where a repertoire
 * tolerates insertions rather than what happened to any sequence. PRISM
 * answers the per-sequence question and this does not.
 *
 * Aliases need no deduplication pass. Inserting `S` at gaps 2, 3 and 4 of
 * `CASSF` gives one string, and the leftmost representative is picked by a
 * local test -- `g == 0 || s[g - 1] != a` -- because aliases arise only
 * inside a run of the inserted residue. Deletion mirrors it. Checked against
 * brute-force enumeration in the tests, both for how many survive and for
 * which.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/query.h"
#include "beam/query_internal.h"

void beam_query_neighbourhood_free(BeamNeighbourhood *nb) {
    if (!nb) return;
    free(nb->sequences);
    free(nb->site);
    free(nb->residue);
    free(nb->log_odds);
    free(nb->posterior);
    memset(nb, 0, sizeof(*nb));
}

static BeamError nb_alloc(BeamNeighbourhood *nb, uint32_t cap,
                          uint32_t length) {
    memset(nb, 0, sizeof(*nb));
    nb->length    = length;
    nb->sequences = (char *)malloc((size_t)cap * (length + 1u));
    nb->site      = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    nb->residue   = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    nb->log_odds  = (double *)malloc((size_t)cap * sizeof(double));
    nb->posterior = (double *)malloc((size_t)cap * sizeof(double));
    if (!nb->sequences || !nb->site || !nb->residue || !nb->log_odds
        || !nb->posterior) {
        beam_query_neighbourhood_free(nb);
        return BEAM_EALLOC("the edit neighbourhood");
    }
    return BEAM_OK;
}

/**
 * Turn scores into odds and a posterior.
 *
 * The maximum is subtracted before exponentiating, and that is not
 * decoration. A neighbourhood sitting far below the model's mode -- and in an
 * open-world model a sequence of improbable residues sits thousands of nats
 * below it -- underflows `exp` to zero in every single entry without it, and
 * the posterior comes back as a vector of zeros rather than as the perfectly
 * well-defined distribution it is.
 */
static BeamError nb_finish(BeamNeighbourhood *nb, const double *score,
                           double log_source) {
    nb->log_source = log_source;
    double mx = -INFINITY;
    for (uint32_t k = 0; k < nb->n; ++k) if (score[k] > mx) mx = score[k];

    for (uint32_t k = 0; k < nb->n; ++k)
        nb->log_odds[k] = score[k] - log_source;

    if (mx == -INFINITY) {
        /*
         * Nothing in the neighbourhood is producible. The odds still say so,
         * each being -inf, but there is no distribution over an empty set to
         * report. Handled here rather than by guarding the loop below,
         * because that loop would otherwise form -inf minus -inf.
         */
        for (uint32_t k = 0; k < nb->n; ++k) nb->posterior[k] = 0.0;
        return BEAM_OK;
    }

    /* With `mx` finite, an impossible neighbour gives exp(-inf) = 0 on its
     * own, and the entry that attained the maximum gives exp(0) = 1, so the
     * total is positive without needing to be checked. */
    double total = 0.0;
    for (uint32_t k = 0; k < nb->n; ++k) {
        nb->posterior[k] = exp(score[k] - mx);
        total += nb->posterior[k];
    }
    for (uint32_t k = 0; k < nb->n; ++k) nb->posterior[k] /= total;
    return BEAM_OK;
}

/**
 * The source's shape, checked before anything is asked of the model.
 *
 * Split from the scoring below so that a structural refusal is reported as
 * one. A single residue trips both this and the zero-probability test -- most
 * models spell no sequence of length one -- and "deleting from one residue
 * leaves the empty sequence" is the useful half of that: it is true of every
 * model, whereas the other is a fact about this one.
 */
static BeamError source_shape(const BeamModel *m, const char *sequence,
                              uint32_t *out_len, uint8_t *out_idx) {
    if (!m) return BEAM_ENULL("model");
    if (!sequence) return BEAM_ENULL("sequence");
    if (!m->finalized) return BEAM_ENOTBUILT();

    const size_t n = strlen(sequence);
    if (n == 0u || n > BEAM_MAX_LENGTH_CEILING)
        return BEAM_ERANGE("beam_query: a source sequence of %zu residues is "
                          "outside 1..%u", n,
                          (unsigned)BEAM_MAX_LENGTH_CEILING);

    /*
     * Decoded once, and every byte checked. Two things depend on this.
     *
     * A character outside the alphabet is a MALFORMED source, not a source of
     * probability zero, and the two are different answers: `log_pseq` maps
     * both to -inf, so without this check `AA?A` came back as
     * NO_LIVE_PATHS -- "this cannot happen" -- rather than as bad input.
     *
     * And the alias rule below must compare RESIDUES rather than bytes. The
     * nucleotide alphabet folds case, so `a` and `A` are one residue; comparing
     * raw bytes made every case variant look distinct, and the neighbourhood
     * of `aaaa` came back with the same sequence five times, each holding a
     * fifth of the posterior. Found by review.
     */
    const BeamAlphabetInfo *a = beam_alphabet_info(m->cfg.alphabet);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t r = a->char_to_idx[(unsigned char)sequence[i]];
        if (r >= m->n_residues)
            return BEAM_FAIL(BEAM_ERR_INVALID_SEQUENCE,
                            "beam_query: position %zu of the source is '%c', "
                            "which is not a residue of this model's %s "
                            "alphabet", i, sequence[i], a->name);
        out_idx[i] = r;
    }
    *out_len = (uint32_t)n;
    return BEAM_OK;
}

/** The source's log probability, once its shape is known to be usable. */
static BeamError source_score(const BeamModel *m, const char *sequence,
                              double *out_score) {
    const double s = beam_model_log_pseq(m, sequence);
    if (!isfinite(s))
        return BEAM_ENOPATHS("the source sequence has probability zero under "
                            "this model, so the odds of its neighbours would "
                            "be a ratio to nothing");
    *out_score = s;
    return BEAM_OK;
}

/* ── Insertions ──────────────────────────────────────────────── */

BeamError beam_query_insertions(const BeamModel *m, const char *sequence,
                                BeamNeighbourhood *out) {
    if (!out) return BEAM_ENULL("output");
    /*
     * Emptied before anything can refuse, so that a caller who does not check
     * the return code reads an empty neighbourhood rather than whatever was
     * on its stack, and so that freeing a refused one is safe.
     */
    memset(out, 0, sizeof(*out));
    uint32_t L = 0u;
    double log_source = 0.0;
    uint8_t idx[BEAM_MAX_LENGTH_CEILING];
    BeamError err = source_shape(m, sequence, &L, idx);
    if (err != BEAM_OK) return err;
    err = source_score(m, sequence, &log_source);
    if (err != BEAM_OK) return err;

    const BeamAlphabetInfo *a = beam_alphabet_info(m->cfg.alphabet);
    const uint32_t R = m->n_residues;
    const uint32_t cap = (L + 1u) * R;

    err = nb_alloc(out, cap, L + 1u);
    if (err != BEAM_OK) return err;
    double *score = (double *)malloc((size_t)cap * sizeof(double));
    if (!score) {
        beam_query_neighbourhood_free(out);
        return BEAM_EALLOC("the neighbour scores");
    }

    char *buf = out->sequences;
    for (uint32_t g = 0; g <= L; ++g) {
        for (uint32_t x = 0; x < R; ++x) {
            const char ch = a->idx_to_char[x];
            /*
             * Leftmost representative only. An alias exists exactly when the
             * residue before the gap is the one being inserted, because the
             * strings coincide only inside a run of it. Compared as RESIDUES,
             * so that a case-folded alphabet does not make `a` and `A` look
             * like different neighbours.
             */
            if (g > 0u && idx[g - 1u] == x) continue;

            char *dst = buf + (size_t)out->n * (L + 2u);
            memcpy(dst, sequence, g);
            dst[g] = ch;
            memcpy(dst + g + 1u, sequence + g, L - g);
            dst[L + 1u] = '\0';

            out->site[out->n]    = g;
            out->residue[out->n] = x;
            score[out->n]        = beam_model_log_pseq(m, dst);
            out->n++;
        }
    }
    err = nb_finish(out, score, log_source);
    free(score);
    return err;
}

/* ── Deletions ───────────────────────────────────────────────── */

BeamError beam_query_deletions(const BeamModel *m, const char *sequence,
                               BeamNeighbourhood *out) {
    if (!out) return BEAM_ENULL("output");
    /*
     * Emptied before anything can refuse, so that a caller who does not check
     * the return code reads an empty neighbourhood rather than whatever was
     * on its stack, and so that freeing a refused one is safe.
     */
    memset(out, 0, sizeof(*out));
    uint32_t L = 0u;
    double log_source = 0.0;
    uint8_t idx[BEAM_MAX_LENGTH_CEILING];
    BeamError err = source_shape(m, sequence, &L, idx);
    if (err != BEAM_OK) return err;
    /* Before scoring: this is true of every model, and the zero-probability
     * refusal below would otherwise mask it for most of them. */
    if (L < 2u)
        return BEAM_ERANGE("beam_query_deletions: deleting from a single "
                          "residue leaves the empty sequence, which this "
                          "model does not score");
    err = source_score(m, sequence, &log_source);
    if (err != BEAM_OK) return err;

    err = nb_alloc(out, L, L - 1u);
    if (err != BEAM_OK) return err;
    double *score = (double *)malloc((size_t)L * sizeof(double));
    if (!score) {
        beam_query_neighbourhood_free(out);
        return BEAM_EALLOC("the neighbour scores");
    }

    char *buf = out->sequences;
    for (uint32_t p = 0; p < L; ++p) {
        /* Leftmost of a run again: deleting either of two equal neighbours
         * leaves the same string. Residues, not bytes; see above. */
        if (p > 0u && idx[p - 1u] == idx[p]) continue;

        char *dst = buf + (size_t)out->n * L;
        memcpy(dst, sequence, p);
        memcpy(dst + p, sequence + p + 1u, L - p - 1u);
        dst[L - 1u] = '\0';

        out->site[out->n]    = p;
        out->residue[out->n] = UINT32_MAX;    /* a deletion adds nothing */
        score[out->n]        = beam_model_log_pseq(m, dst);
        out->n++;
    }
    err = nb_finish(out, score, log_source);
    free(score);
    return err;
}
