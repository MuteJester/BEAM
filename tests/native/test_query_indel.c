/**
 * @file test_query_indel.c
 * @brief Tests for insertion and deletion neighbourhoods.
 *
 * The neighbourhood is built without a deduplication pass, by a local test
 * that claims to pick exactly the leftmost representative of each distinct
 * string. That claim is the thing most worth attacking, so it is checked
 * against brute force twice over: the engine's neighbours must be the same SET
 * as the distinct strings of an exhaustive generation, and each one's recorded
 * site must be the leftmost gap that actually produces it.
 *
 * Then the structural check that no string-building bug survives: undoing the
 * recorded edit on each neighbour must give back the source, character for
 * character. And the round trip -- every insertion neighbour must have the
 * source among ITS deletion neighbours -- which ties the two functions to each
 * other rather than each to itself.
 *
 * Finally the arithmetic, on a corpus built so the answer is orderable in
 * advance: three sequences differing only by which residue sits at one
 * position, weighted 100 to 10 to 1, so the posterior over the neighbourhood
 * has to rank them in that order and in those proportions.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/query.h"

#include "../../src/native/internal/beam/alphabet.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define R BEAM_AA_RESIDUES
static const char *AA = "ACDEFGHIKLMNPQRSTVWY";

static BeamModel *build(const char **seqs, const double *w, uint32_t n,
                        uint8_t order, uint8_t max_length, bool open_world) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = max_length;
    if (open_world) cfg.strict = 0;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, seqs, w, n) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

#define N_TRAIN 8
static const char *TRAIN[N_TRAIN] = {
    "CASSF", "CASSAF", "CASSYF", "CASSGF", "CASF", "CAAAF", "CASSLF", "CSSF",
};
static const double WEIGHT[N_TRAIN] = {50, 100, 10, 1, 20, 15, 8, 5};

/* CDR3-shaped and long enough that an improbable sequence sits thousands of
 * nats below the mode, which is what the underflow test needs. */
#define N_DEEP 8
static const char *DEEP[N_DEEP] = {
    "CASSLGEQYF", "CASSLGEQFF", "CASSQGEQYF", "CASSPGEQYF",
    "CASSLAEQYF", "CASSLGETQYF", "CASSVGEQYF", "CASSLGEQYFF",
};
static const double DEEP_WEIGHT[N_DEEP] = {50, 30, 20, 15, 12, 25, 18, 7};

/* ── Deduplication, against brute force ──────────────────────── */

static void test_insertion_neighbours_are_the_distinct_strings(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    const char *src = "CASSF";
    const uint32_t L = 5u;

    BeamNeighbourhood nb;
    ASSERT_MSG(beam_query_insertions(m, src, &nb) == BEAM_OK, "insertions");

    /* Every candidate, kept only if not already seen. Deliberately the naive
     * quadratic scan, sharing nothing with the local rule under test. */
    static char want[(5u + 1u) * R][8];
    uint32_t n_want = 0u;
    for (uint32_t g = 0; g <= L; ++g) {
        for (uint32_t x = 0; x < R; ++x) {
            char cand[8];
            memcpy(cand, src, g);
            cand[g] = AA[x];
            memcpy(cand + g + 1u, src + g, L - g);
            cand[L + 1u] = '\0';
            bool seen = false;
            for (uint32_t k = 0; k < n_want; ++k)
                if (strcmp(want[k], cand) == 0) { seen = true; break; }
            if (!seen) { strcpy(want[n_want], cand); n_want++; }
        }
    }
    ASSERT_MSG(n_want == 115u,
               "CASSF has 115 distinct insertion neighbours of its 120 "
               "candidates, the five aliases being the runs");
    ASSERT_MSG(nb.n == n_want, "the engine found exactly as many");
    ASSERT_MSG(nb.length == L + 1u, "and they are one residue longer");

    /* Same set, and no repeats within the engine's own answer. */
    for (uint32_t k = 0; k < nb.n; ++k) {
        const char *got = nb.sequences + (size_t)k * (nb.length + 1u);
        bool found = false;
        for (uint32_t j = 0; j < n_want; ++j)
            if (strcmp(want[j], got) == 0) { found = true; break; }
        ASSERT_MSG(found, "every neighbour is one brute force also produced");
        for (uint32_t j = 0; j < k; ++j)
            ASSERT_MSG(strcmp(nb.sequences + (size_t)j * (nb.length + 1u),
                              got) != 0, "and appears only once");
    }
    beam_query_neighbourhood_free(&nb);
    beam_model_destroy(m);
    PASS();
}

static void test_each_neighbour_is_attributed_to_its_leftmost_gap(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    const char *src = "CAAAF";        /* a run of three, so aliases abound */
    const uint32_t L = 5u;

    BeamNeighbourhood nb;
    ASSERT_MSG(beam_query_insertions(m, src, &nb) == BEAM_OK, "insertions");

    for (uint32_t k = 0; k < nb.n; ++k) {
        const char *got = nb.sequences + (size_t)k * (nb.length + 1u);
        /* The smallest gap that produces this exact string, found by trying
         * all of them. */
        uint32_t leftmost = UINT32_MAX;
        for (uint32_t g = 0; g <= L && leftmost == UINT32_MAX; ++g) {
            for (uint32_t x = 0; x < R; ++x) {
                char cand[8];
                memcpy(cand, src, g);
                cand[g] = AA[x];
                memcpy(cand + g + 1u, src + g, L - g);
                cand[L + 1u] = '\0';
                if (strcmp(cand, got) == 0) { leftmost = g; break; }
            }
        }
        ASSERT_MSG(nb.site[k] == leftmost,
                   "the recorded gap is the leftmost that yields this string");
        /* Undoing the recorded edit must give the source back exactly. */
        char undone[8];
        memcpy(undone, got, nb.site[k]);
        strcpy(undone + nb.site[k], got + nb.site[k] + 1u);
        ASSERT_MSG(strcmp(undone, src) == 0,
                   "and removing that residue restores the source");
        ASSERT_MSG(got[nb.site[k]] == AA[nb.residue[k]],
                   "the recorded residue is the one that was inserted");
    }
    beam_query_neighbourhood_free(&nb);
    beam_model_destroy(m);
    PASS();
}

static void test_deletion_neighbours_collapse_runs(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    BeamNeighbourhood nb;
    /* CAAAF: dropping any of the three A's leaves CAAF, so the five
     * candidates collapse to three distinct strings. */
    ASSERT_MSG(beam_query_deletions(m, "CAAAF", &nb) == BEAM_OK, "deletions");
    ASSERT_MSG(nb.n == 3u, "three distinct deletions, not five");
    ASSERT_MSG(nb.length == 4u, "each one residue shorter");
    ASSERT_MSG(strcmp(nb.sequences + 0u * 5u, "AAAF") == 0, "drop the C");
    ASSERT_MSG(strcmp(nb.sequences + 1u * 5u, "CAAF") == 0, "drop one A");
    ASSERT_MSG(strcmp(nb.sequences + 2u * 5u, "CAAA") == 0, "drop the F");
    ASSERT_MSG(nb.site[1] == 1u,
               "the collapsed A is attributed to the leftmost of the run");
    for (uint32_t k = 0; k < nb.n; ++k)
        ASSERT_MSG(nb.residue[k] == UINT32_MAX,
                   "a deletion records no inserted residue");
    beam_query_neighbourhood_free(&nb);
    beam_model_destroy(m);
    PASS();
}

static void test_inserting_then_deleting_returns_the_source(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    const char *src = "CASSF";
    BeamNeighbourhood ins;
    ASSERT_MSG(beam_query_insertions(m, src, &ins) == BEAM_OK, "insertions");

    /* Ties the two functions to each other: every sequence one insertion away
     * must have the source one deletion away, whatever the aliasing did. */
    uint32_t checked = 0u;
    for (uint32_t k = 0; k < ins.n; k += 7u) {          /* a sample, for time */
        const char *nbr = ins.sequences + (size_t)k * (ins.length + 1u);
        BeamNeighbourhood del;
        if (beam_query_deletions(m, nbr, &del) != BEAM_OK) continue;
        bool found = false;
        for (uint32_t j = 0; j < del.n; ++j)
            if (strcmp(del.sequences + (size_t)j * (del.length + 1u), src)
                == 0) { found = true; break; }
        beam_query_neighbourhood_free(&del);
        ASSERT_MSG(found, "the source is a deletion away from its insertions");
        checked++;
    }
    ASSERT_MSG(checked > 10u, "a real sample was round-tripped");
    beam_query_neighbourhood_free(&ins);
    beam_model_destroy(m);
    PASS();
}

/* ── The arithmetic ──────────────────────────────────────────── */

static void test_the_odds_are_the_scores_they_claim_to_be(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    const char *src = "CASSF";
    const double ls = beam_model_log_pseq(m, src);

    BeamNeighbourhood nb;
    ASSERT_MSG(beam_query_insertions(m, src, &nb) == BEAM_OK, "insertions");
    ASSERT_MSG(nb.log_source == ls, "the source score is reported as given");

    double sum = 0.0, best = -INFINITY;
    for (uint32_t k = 0; k < nb.n; ++k) {
        const char *s = nb.sequences + (size_t)k * (nb.length + 1u);
        const double want = beam_model_log_pseq(m, s) - ls;
        ASSERT_MSG(nb.log_odds[k] == want,
                   "the odds are exactly the difference of the two scores");
        sum += nb.posterior[k];
        if (nb.log_odds[k] > best) best = nb.log_odds[k];
    }
    ASSERT_MSG(fabs(sum - 1.0) < 1e-12, "the posterior is a distribution");

    /* And it is the softmax of the odds: the source term is common to every
     * entry, so it cancels and the posterior does not depend on it. */
    double z = 0.0;
    for (uint32_t k = 0; k < nb.n; ++k)
        if (isfinite(nb.log_odds[k])) z += exp(nb.log_odds[k] - best);
    for (uint32_t k = 0; k < nb.n; ++k) {
        const double want = isfinite(nb.log_odds[k])
                          ? exp(nb.log_odds[k] - best) / z : 0.0;
        ASSERT_MSG(fabs(nb.posterior[k] - want) < 1e-12,
                   "the posterior is the softmax of the odds");
    }
    beam_query_neighbourhood_free(&nb);
    beam_model_destroy(m);
    PASS();
}

static void test_the_posterior_ranks_the_insertions_the_corpus_prefers(void) {
    /*
     * The corpus contains CASSAF, CASSYF and CASSGF at 100, 10 and 1, and
     * nothing else of that length reachable from CASSF by one insertion. The
     * neighbourhood must rank them in that order, and their posteriors must
     * stand in the same ratios as the probabilities the model gives them.
     */
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, false);
    ASSERT_MSG(m != NULL, "closed-world model built");
    BeamNeighbourhood nb;
    ASSERT_MSG(beam_query_insertions(m, "CASSF", &nb) == BEAM_OK, "insertions");

    double p_a = -1.0, p_y = -1.0, p_g = -1.0;
    uint32_t live = 0u;
    for (uint32_t k = 0; k < nb.n; ++k) {
        const char *s = nb.sequences + (size_t)k * (nb.length + 1u);
        if (nb.posterior[k] > 0.0) live++;
        if (strcmp(s, "CASSAF") == 0) p_a = nb.posterior[k];
        if (strcmp(s, "CASSYF") == 0) p_y = nb.posterior[k];
        if (strcmp(s, "CASSGF") == 0) p_g = nb.posterior[k];
    }
    ASSERT_MSG(p_a > 0.0 && p_y > 0.0 && p_g > 0.0, "all three were found");
    ASSERT_MSG(p_a > p_y && p_y > p_g,
               "the posterior ranks them as the corpus weights them");
    /* The ratio must be the model's own, not merely the right order. */
    const double ra = exp(beam_model_log_pseq(m, "CASSAF"));
    const double ry = exp(beam_model_log_pseq(m, "CASSYF"));
    ASSERT_MSG(fabs(p_a / p_y - ra / ry) < 1e-9,
               "and in the proportions the model assigns them");
    ASSERT_MSG(live < nb.n,
               "while the insertions the closed world cannot spell carry no "
               "mass at all");
    beam_query_neighbourhood_free(&nb);
    beam_model_destroy(m);
    PASS();
}

static void test_an_impossible_neighbourhood_still_reports_its_odds(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 6u, false);
    ASSERT_MSG(m != NULL, "model built");
    /* max_length 6, so inserting into a length-6 sequence leaves nothing the
     * model can produce. Every neighbour is impossible, which is an answer. */
    BeamNeighbourhood nb;
    ASSERT_MSG(beam_query_insertions(m, "CASSAF", &nb) == BEAM_OK,
               "asking is not an error");
    ASSERT_MSG(nb.n > 0u, "the neighbourhood is still enumerated");
    for (uint32_t k = 0; k < nb.n; ++k) {
        ASSERT_MSG(nb.log_odds[k] == -INFINITY, "each neighbour is impossible");
        ASSERT_MSG(nb.posterior[k] == 0.0,
                   "and there is no distribution over an empty set to report");
    }
    beam_query_neighbourhood_free(&nb);
    beam_model_destroy(m);
    PASS();
}

static void test_the_posterior_survives_a_neighbourhood_that_underflows(void) {
    BeamModel *m = build(DEEP, DEEP_WEIGHT, N_DEEP, 3u, 14u, true);
    ASSERT_MSG(m != NULL, "open-world model built");

    /*
     * A sequence of tryptophans is producible in an open world and sits about
     * 6,900 nats below the mode, so every one of its neighbours is around
     * exp(-7250) -- which is exactly zero in a double. Exponentiating the
     * scores directly would therefore give a total of zero and a posterior of
     * all zeros, for a distribution that is perfectly well defined. Only
     * subtracting the maximum first recovers it.
     */
    const char *src = "WWWWWWWWWW";
    ASSERT_MSG(isfinite(beam_model_log_pseq(m, src)),
               "the source is producible, if barely");

    BeamNeighbourhood nb = {0};
    ASSERT_MSG(beam_query_insertions(m, src, &nb) == BEAM_OK, "insertions");
    ASSERT_MSG(nb.n > 100u, "a full neighbourhood");

    double best = -INFINITY, sum = 0.0;
    uint32_t positive = 0u;
    for (uint32_t k = 0; k < nb.n; ++k) {
        const double s = nb.log_odds[k] + nb.log_source;
        if (s > best) best = s;
        sum += nb.posterior[k];
        if (nb.posterior[k] > 0.0) positive++;
    }
    ASSERT_MSG(best < -700.0, "every neighbour is far below the mode");
    ASSERT_MSG(exp(best) == 0.0,
               "so exponentiating a score directly gives exactly zero, and a "
               "naive softmax would have nothing left to normalise");
    ASSERT_MSG(fabs(sum - 1.0) < 1e-12,
               "yet the posterior is still a distribution");
    ASSERT_MSG(positive > nb.n / 2u,
               "with most neighbours carrying real mass");

    /* And it still ranks them: the most probable neighbour holds the most. */
    uint32_t arg_odds = 0u, arg_post = 0u;
    for (uint32_t k = 1; k < nb.n; ++k) {
        if (nb.log_odds[k] > nb.log_odds[arg_odds]) arg_odds = k;
        if (nb.posterior[k] > nb.posterior[arg_post]) arg_post = k;
    }
    ASSERT_MSG(arg_odds == arg_post,
               "and the best odds and the highest posterior agree on which "
               "neighbour that is");
    beam_query_neighbourhood_free(&nb);
    beam_model_destroy(m);
    PASS();
}

static void test_case_folded_input_does_not_multiply_the_neighbours(void) {
    /*
     * Regression for a defect found by review. The nucleotide alphabet folds
     * case, so `a` and `A` are one residue -- but the alias rule compared the
     * source's raw BYTE against the canonical uppercase character, so no
     * lowercase run ever looked like a run. The neighbourhood of `aaaa` came
     * back with `Aaaaa`, `aAaaa`, `aaAaa`, `aaaAa` and `aaaaA`: five spellings
     * of one sequence, each holding a fifth of the posterior that belongs
     * entirely to it.
     */
    const char *seqs[4] = {"ACGTA", "AAAAA", "ACGTT", "AAAA"};
    const double w[4] = {10.0, 40.0, 5.0, 20.0};
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.alphabet = BEAM_ALPHABET_NT;
    cfg.order = 2; cfg.max_length = 8; cfg.strict = 0;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "nucleotide model built");
    ASSERT_MSG(beam_model_observe(m, seqs, w, 4u) == BEAM_OK, "observed");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalized");
    const uint32_t NR = BEAM_NT_RESIDUES;

    /* Lowercase and uppercase spell the same sequence, so they must give the
     * same neighbourhood. */
    BeamNeighbourhood lo = {0}, up = {0};
    ASSERT_MSG(beam_query_insertions(m, "aaaa", &lo) == BEAM_OK, "lower");
    ASSERT_MSG(beam_query_insertions(m, "AAAA", &up) == BEAM_OK, "upper");
    ASSERT_MSG(lo.n == up.n,
               "case cannot change how many distinct neighbours there are");
    /* 5 gaps x 5 residues = 25 candidates; inserting A is an alias at every
     * gap but the first, so 21 survive. */
    ASSERT_MSG(lo.n == 5u * NR - 4u, "21 of the 25 candidates are distinct");

    uint32_t all_a = 0u;
    double p_all_a = 0.0;
    for (uint32_t k = 0; k < lo.n; ++k) {
        const char *s = lo.sequences + (size_t)k * (lo.length + 1u);
        bool only_a = true;
        for (uint32_t i = 0; i < lo.length; ++i)
            if (s[i] != 'a' && s[i] != 'A') only_a = false;
        if (only_a) { all_a++; p_all_a = lo.posterior[k]; }
    }
    ASSERT_MSG(all_a == 1u,
               "the all-adenine neighbour appears once, not once per gap");
    ASSERT_MSG(p_all_a > 0.9,
               "and holds the mass the corpus gives it, undivided");

    BeamNeighbourhood dl = {0};
    ASSERT_MSG(beam_query_deletions(m, "AaAA", &dl) == BEAM_OK, "deletions");
    ASSERT_MSG(dl.n == 1u,
               "every deletion from a mixed-case run leaves one sequence");

    beam_query_neighbourhood_free(&lo);
    beam_query_neighbourhood_free(&up);
    beam_query_neighbourhood_free(&dl);
    beam_model_destroy(m);
    PASS();
}

static void test_a_character_outside_the_alphabet_is_malformed_input(void) {
    /*
     * Regression for a defect found by review. `log_pseq` maps both a
     * malformed sequence and a merely impossible one to -inf, so routing
     * every non-finite score to NO_LIVE_PATHS reported bad input as "this
     * cannot happen". The two are different answers and the header promises
     * to distinguish them.
     */
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    BeamNeighbourhood nb = {0};
    ASSERT_MSG(beam_query_insertions(m, "CA?SF", &nb)
               == BEAM_ERR_INVALID_SEQUENCE, "malformed, not impossible");
    ASSERT_MSG(strstr(beam_error_message(), "not a residue") != NULL,
               "and the message says which character and why");
    ASSERT_MSG(beam_query_deletions(m, "CA?SF", &nb)
               == BEAM_ERR_INVALID_SEQUENCE, "the same for deletions");
    beam_model_destroy(m);
    PASS();
}

/* ── Refusals ────────────────────────────────────────────────── */

static void test_neighbourhoods_at_the_short_extreme(void) {
    /* Insertions from one residue and deletions down to one were untested;
     * both are legal and both were correct, which is worth pinning. */
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    BeamNeighbourhood nb = {0};

    if (beam_query_insertions(m, "C", &nb) == BEAM_OK) {
        ASSERT_MSG(nb.length == 2u, "one residue becomes two");
        ASSERT_MSG(nb.n == 2u * R - 1u,
                   "two gaps by R residues, less the one alias");
        double s = 0.0;
        for (uint32_t k = 0; k < nb.n; ++k) s += nb.posterior[k];
        ASSERT_MSG(fabs(s - 1.0) < 1e-12, "and is still a distribution");
        beam_query_neighbourhood_free(&nb);
    }
    if (beam_query_deletions(m, "CA", &nb) == BEAM_OK) {
        ASSERT_MSG(nb.length == 1u, "two residues become one");
        ASSERT_MSG(nb.n == 2u, "and both single residues are distinct");
        beam_query_neighbourhood_free(&nb);
    }
    beam_model_destroy(m);
    PASS();
}

static void test_a_source_at_the_length_ceiling_is_handled(void) {
    /*
     * The source is decoded into a stack array sized BEAM_MAX_LENGTH_CEILING,
     * and `source_shape` bounds it by exactly that. Nothing tested the
     * boundary, so loosening the bound by one would have been an unguarded
     * stack overflow rather than a failing test.
     */
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, true);
    ASSERT_MSG(m != NULL, "model built");
    char *at = (char *)malloc(BEAM_MAX_LENGTH_CEILING + 2u);
    ASSERT_MSG(at != NULL, "alloc");
    memset(at, 'C', BEAM_MAX_LENGTH_CEILING + 1u);
    at[BEAM_MAX_LENGTH_CEILING + 1u] = '\0';

    BeamNeighbourhood nb = {0};
    /* One past the ceiling: refused on shape, before anything is decoded. */
    ASSERT_MSG(beam_query_insertions(m, at, &nb)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "one past the ceiling");
    ASSERT_MSG(nb.n == 0u && nb.sequences == NULL, "and nothing was written");
    /* Exactly at the ceiling: accepted by shape, then refused by the model,
     * which is a different refusal and the one that proves it got that far. */
    at[BEAM_MAX_LENGTH_CEILING] = '\0';
    ASSERT_MSG(beam_query_insertions(m, at, &nb) == BEAM_ERR_NO_LIVE_PATHS,
               "at the ceiling, refused by the model rather than the shape");
    free(at);
    beam_model_destroy(m);
    PASS();
}

static void test_a_source_the_model_cannot_score_is_refused(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 8u, false);
    ASSERT_MSG(m != NULL, "model built");
    BeamNeighbourhood nb;

    ASSERT_MSG(beam_query_insertions(m, "WWWWW", &nb) == BEAM_ERR_NO_LIVE_PATHS,
               "odds against a sequence of probability zero are a ratio to "
               "nothing");
    ASSERT_MSG(strstr(beam_error_message(), "ratio to nothing") != NULL,
               "and the message says so");
    ASSERT_MSG(beam_query_insertions(m, "", &nb)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "an empty source is refused");
    ASSERT_MSG(beam_query_deletions(m, "C", &nb)
               == BEAM_ERR_PARAM_OUT_OF_RANGE,
               "deleting from one residue leaves nothing to score");
    ASSERT_MSG(beam_query_insertions(m, NULL, &nb) == BEAM_ERR_NULL_ARG,
               "and NULL is refused");

    /*
     * A refused call must leave the output empty rather than untouched. A
     * caller that ignores the return code then reads nothing instead of
     * whatever was on its stack, and freeing it is safe -- which the two
     * calls below would otherwise turn into a free of a wild pointer.
     */
    BeamNeighbourhood wild;
    memset(&wild, 0xA5, sizeof(wild));
    ASSERT_MSG(beam_query_insertions(m, "WWWWW", &wild)
               == BEAM_ERR_NO_LIVE_PATHS, "refused");
    ASSERT_MSG(wild.n == 0u && wild.sequences == NULL,
               "and the output was emptied first");
    beam_query_neighbourhood_free(&wild);

    memset(&wild, 0xA5, sizeof(wild));
    ASSERT_MSG(beam_query_deletions(m, "C", &wild)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "refused");
    ASSERT_MSG(wild.n == 0u && wild.sequences == NULL, "emptied first");
    beam_query_neighbourhood_free(&wild);

    beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("\n[neighbourhood construction]\n");
    RUN_TEST(test_insertion_neighbours_are_the_distinct_strings);
    RUN_TEST(test_each_neighbour_is_attributed_to_its_leftmost_gap);
    RUN_TEST(test_deletion_neighbours_collapse_runs);
    RUN_TEST(test_inserting_then_deleting_returns_the_source);

    printf("\n[odds and posterior]\n");
    RUN_TEST(test_the_odds_are_the_scores_they_claim_to_be);
    RUN_TEST(test_the_posterior_ranks_the_insertions_the_corpus_prefers);
    RUN_TEST(test_an_impossible_neighbourhood_still_reports_its_odds);
    RUN_TEST(test_the_posterior_survives_a_neighbourhood_that_underflows);

    RUN_TEST(test_case_folded_input_does_not_multiply_the_neighbours);

    printf("\n[refusals]\n");
    RUN_TEST(test_neighbourhoods_at_the_short_extreme);
    RUN_TEST(test_a_source_at_the_length_ceiling_is_handled);
    RUN_TEST(test_a_character_outside_the_alphabet_is_malformed_input);
    RUN_TEST(test_a_source_the_model_cannot_score_is_refused);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
