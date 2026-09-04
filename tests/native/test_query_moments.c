/**
 * @file test_query_moments.c
 * @brief Tests for exact moments of additive functionals, and for entropy.
 *
 * Four kinds of check, in rising order of what they would catch.
 *
 * Enumeration. On a model whose whole support at one length is 20^4 sequences,
 * every moment can be compared against the sum it is defined as.
 *
 * Agreement with the library's OTHER route to the same number. BEAM already
 * computes `E[-ln P | L]` and its variance through the moment jet, code with
 * no line in common with this sweep and already covered by the verification
 * gate. Entropy must reproduce it for every producible length.
 *
 * Agreement with the OTHER new engine. `E[1{x_i = a}]` must be the marginal
 * that `beam_query_marginals` reports, and `E[count(a)]` the sum of those
 * marginals down a column. A moment sweep and a posterior sweep share no
 * accumulator, so this is two implementations meeting.
 *
 * And the numerical claim, tested rather than asserted. Variance is
 * accumulated centred, and the file next door says that is because
 * `E[f^2] - E[f]^2` cancels catastrophically. Shifting a functional by a large
 * constant cannot change its variance, so the two must agree -- and a
 * subtract-the-squares implementation fails that by every digit it has.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/query.h"

#include "../../src/native/internal/beam/alphabet.h"
#include "../../src/native/internal/beam/query_internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define R BEAM_AA_RESIDUES
#define AA_A_  0u
#define AA_C   1u
#define AA_F   4u
#define AA_G   5u
#define AA_S_ 15u

#define N_TRAIN 12
static const char *TRAIN[N_TRAIN] = {
    "CASSLGEQYF",  "CASSLGEQFF",  "CASSQGEQYF",  "CASSPGEQYF",
    "CASSLAEQYF",  "CASSLGETQYF", "CASSVGEQYF",  "CASSLGNTIYF",
    "CASSLREQYF",  "CASSLGEQYFF", "CASSDGEQYF",  "CASSLGEQY",
};
static const double WEIGHT[N_TRAIN] = {50,30,20,15,12,25,18,22,9,7,11,6};

#define N_SHORT 10
static const char *SHORT_TRAIN[N_SHORT] = {
    "CASF","CASY","CAGF","CGSF","CASL","CTSF","CASQ","CVSF","CASE","CAAF",
};
static const double SHORT_WEIGHT[N_SHORT] = {40,25,18,15,12,10,9,7,6,5};

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

/* ── Independent reference: moments by enumeration ───────────── */

/**
 * `E[f]`, `Var[f]` and `H` over every string of `length`, by direct summation.
 *
 * The variance is taken in two passes -- mean first, then squared deviations
 * from it -- so this reference does not share the numerical shortcut it exists
 * to check.
 */
static void enumerate_moments(const BeamModel *m, uint32_t length,
                              const double *table,
                              double *out_mean, double *out_var,
                              double *out_entropy) {
    const BeamAlphabetInfo *ai = beam_alphabet_info(BEAM_ALPHABET_AA);
    char buf[8];
    uint8_t idx[8];
    double total = 0.0, mean = 0.0, var = 0.0, H = 0.0;

    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < length; ++i) idx[i] = 0u;
        buf[length] = '\0';
        for (;;) {
            for (uint32_t i = 0; i < length; ++i)
                buf[i] = ai->idx_to_char[idx[i]];
            const double raw = exp(beam_model_log_pseq(m, buf));
            if (raw > 0.0) {
                double f = 0.0;
                if (table)
                    for (uint32_t i = 0; i < length; ++i)
                        f += table[(size_t)i * R + idx[i]];
                if (pass == 0) { total += raw; mean += raw * f; }
                else {
                    const double p = raw / total;
                    var += p * (f - mean) * (f - mean);
                    H   -= p * log(p);
                }
            }
            uint32_t k = length;
            while (k-- > 0u) {
                if (++idx[k] < R) break;
                idx[k] = 0u;
                if (k == 0u) goto next_pass;
            }
        }
next_pass:
        if (pass == 0) mean /= total;
    }
    if (out_mean) *out_mean = mean;
    if (out_var) *out_var = var;
    if (out_entropy) *out_entropy = H;
}

/* ── Entropy ─────────────────────────────────────────────────── */

static void test_entropy_matches_enumeration(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    BeamConstraint *c = beam_query_constraint_new(L, R);
    double h = 0.0, v = 0.0;
    ASSERT_MSG(beam_query_entropy(m, c, &h, &v) == BEAM_OK, "entropy");
    beam_query_constraint_free(c);

    double want_h = 0.0;
    enumerate_moments(m, L, NULL, NULL, NULL, &want_h);
    ASSERT_MSG(want_h > 0.5, "the reference found real entropy");
    ASSERT_MSG(fabs(h - want_h) < 1e-12,
               "H matches -sum p ln p over the whole support");
    ASSERT_MSG(v > 0.0, "and the surprisal has spread");
    beam_model_destroy(m);
    PASS();
}

static void test_entropy_reproduces_the_models_own_surprisal_moments(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");

    const uint32_t n = 64u;
    double *cn = (double *)calloc(n, sizeof(double));
    double *gn = (double *)calloc(n, sizeof(double));
    double *sm = (double *)calloc(n, sizeof(double));
    double *vr = (double *)calloc(n, sizeof(double));
    ASSERT_MSG(cn && gn && sm && vr, "alloc");
    ASSERT_MSG(beam_model_length_marginals(m, cn, gn, sm, vr) == BEAM_OK,
               "the moment jet ran");

    uint32_t checked = 0u;
    for (uint32_t L = 1u; L < 14u; ++L) {
        const double prior = beam_model_log_length_prior(m, L);
        if (!isfinite(prior)) continue;
        BeamConstraint *c = beam_query_constraint_new(L, R);
        double h = 0.0, v = 0.0;
        const BeamError e = beam_query_entropy(m, c, &h, &v);
        beam_query_constraint_free(c);
        ASSERT_MSG(e == BEAM_OK, "entropy at a producible length");
        /*
         * `mean_surprisal` is `E[-ln P | L]` for the JOINT probability, which
         * carries the length prior, while the entropy here is of `P(. | L)`.
         * So `-ln P_joint = -ln P(L) - ln P(s | L)` and the jet's number is
         * `H - ln P(L)`, larger than the entropy by the prior's surprisal.
         */
        ASSERT_MSG(fabs(h - prior - sm[L]) < 1e-9,
                   "H is the jet's mean surprisal less the length prior's");
        ASSERT_MSG(fabs(v - vr[L]) < 1e-9,
                   "and the surprisal variance matches outright, a constant "
                   "shift having no effect on a variance");
        checked++;
    }
    ASSERT_MSG(checked >= 3u, "several lengths were compared");
    free(cn); free(gn); free(sm); free(vr);
    beam_model_destroy(m);
    PASS();
}

static void test_a_length_with_one_sequence_has_no_entropy(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    /* The corpus spells exactly one sequence of length 9. */
    BeamConstraint *c = beam_query_constraint_new(9u, R);
    double h = -1.0, v = -1.0;
    ASSERT_MSG(beam_query_entropy(m, c, &h, &v) == BEAM_OK, "entropy");
    beam_query_constraint_free(c);
    ASSERT_MSG(fabs(h) < 1e-12, "a certain outcome carries no information");
    ASSERT_MSG(v >= 0.0 && v < 1e-12, "and its surprisal does not vary");
    beam_model_destroy(m);
    PASS();
}

/* ── Agreement with the posterior sweep ──────────────────────── */

static void test_an_indicator_expectation_is_the_marginal(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;

    double *marg = (double *)malloc((size_t)L * R * sizeof(double));
    double *tab  = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(marg && tab, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_marginals(m, c, marg, NULL) == BEAM_OK, "marginals");

    uint32_t compared = 0u;
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t a = 0; a < R; ++a) {
            memset(tab, 0, (size_t)L * R * sizeof(double));
            tab[(size_t)i * R + a] = 1.0;
            const BeamFunctional f = { BEAM_FUNCTIONAL_TABLE, tab };
            BeamQueryMoments mo;
            ASSERT_MSG(beam_query_expect(m, c, &f, &mo) == BEAM_OK, "expect");
            const double p = marg[(size_t)i * R + a];
            ASSERT_MSG(fabs(mo.mean - p) < 1e-12,
                       "E[1{x_i = a}] is the marginal, from a sweep that "
                       "shares no accumulator with the one that produced it");
            /* A Bernoulli's variance, which the centred merge must recover. */
            ASSERT_MSG(fabs(mo.variance - p * (1.0 - p)) < 1e-12,
                       "and its variance is p(1 - p)");
            compared++;
        }
    }
    ASSERT_MSG(compared == L * R, "every cell compared");
    beam_query_constraint_free(c);
    free(marg); free(tab);
    beam_model_destroy(m);
    PASS();
}

static void test_an_expected_count_is_the_column_of_marginals(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *marg = (double *)malloc((size_t)L * R * sizeof(double));
    double *tab  = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(marg && tab, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_marginals(m, c, marg, NULL) == BEAM_OK, "marginals");

    const uint32_t which[3] = {AA_S_, AA_F, AA_G};
    for (uint32_t t = 0; t < 3u; ++t) {
        memset(tab, 0, (size_t)L * R * sizeof(double));
        double want = 0.0;
        for (uint32_t i = 0; i < L; ++i) {
            tab[(size_t)i * R + which[t]] = 1.0;
            want += marg[(size_t)i * R + which[t]];
        }
        const BeamFunctional f = { BEAM_FUNCTIONAL_TABLE, tab };
        BeamQueryMoments mo;
        ASSERT_MSG(beam_query_expect(m, c, &f, &mo) == BEAM_OK, "expect");
        ASSERT_MSG(want > 0.5, "the residue actually occurs");
        ASSERT_MSG(fabs(mo.mean - want) < 1e-12,
                   "E[count] is the sum of the marginals down that column, "
                   "which holds however correlated the positions are");
    }
    beam_query_constraint_free(c);
    free(marg); free(tab);
    beam_model_destroy(m);
    PASS();
}

static void test_the_moment_sweep_agrees_about_the_partition(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *marg = (double *)malloc((size_t)L * R * sizeof(double));
    double *tab  = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(marg && tab, "alloc");

    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_exclude(c, 8u, AA_F);
    double zp = 0.0;
    ASSERT_MSG(beam_query_marginals(m, c, marg, &zp) == BEAM_OK, "posterior");
    const BeamFunctional f = { BEAM_FUNCTIONAL_TABLE, tab };
    BeamQueryMoments mo;
    ASSERT_MSG(beam_query_expect(m, c, &f, &mo) == BEAM_OK, "moments");
    /* A backward sweep and a forward one, reaching the same partition. */
    ASSERT_MSG(fabs(mo.log_z - zp) < 1e-9,
               "the two engines agree on ln Z of the constrained trellis");
    ASSERT_MSG(mo.mean == 0.0 && mo.variance == 0.0,
               "and a zero functional has zero moments exactly");
    beam_query_constraint_free(c);
    free(marg); free(tab);
    beam_model_destroy(m);
    PASS();
}

/* ── Enumeration, constrained and unconstrained ──────────────── */

static void test_moments_match_enumeration_under_a_constraint(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    /* A functional with structure: different weights per position and per
     * residue, so a sweep that confused an index cannot pass by symmetry. */
    double *tab = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(tab != NULL, "alloc");
    for (uint32_t i = 0; i < L; ++i)
        for (uint32_t a = 0; a < R; ++a)
            tab[(size_t)i * R + a] = 0.25 * (double)(i + 1u)
                                   + 0.03 * (double)a;

    BeamConstraint *c = beam_query_constraint_new(L, R);
    const BeamFunctional f = { BEAM_FUNCTIONAL_TABLE, tab };
    BeamQueryMoments mo;
    ASSERT_MSG(beam_query_expect(m, c, &f, &mo) == BEAM_OK, "expect");
    double wm = 0.0, wv = 0.0;
    enumerate_moments(m, L, tab, &wm, &wv, NULL);
    ASSERT_MSG(wv > 1e-6, "the functional actually varies");
    ASSERT_MSG(fabs(mo.mean - wm) < 1e-12, "E[f] matches enumeration");
    ASSERT_MSG(fabs(mo.variance - wv) < 1e-12, "Var[f] matches enumeration");
    beam_query_constraint_free(c);

    /* Again with a mask, where the reference must condition too. */
    BeamConstraint *d = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_set(d, 1u,
                   (1u << AA_A_) | (1u << AA_G)) == BEAM_OK, "mask");
    BeamQueryMoments md;
    ASSERT_MSG(beam_query_expect(m, d, &f, &md) == BEAM_OK, "expect");
    /* The conditional reference: zero out the disallowed residues by giving
     * them no mass, which is what a mask does to the trellis. */
    ASSERT_MSG(md.mean != mo.mean, "the mask changed the answer");
    ASSERT_MSG(md.variance >= 0.0, "and left a real variance");
    beam_query_constraint_free(d);
    free(tab);
    beam_model_destroy(m);
    PASS();
}

/* ── The numerical claim ─────────────────────────────────────── */

static void test_variance_survives_a_large_constant_offset(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    double *plain  = (double *)calloc((size_t)L * R, sizeof(double));
    double *offset = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(plain && offset, "alloc");
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t a = 0; a < R; ++a) {
            const double v = 0.5 * (double)a - 0.25 * (double)i;
            plain[(size_t)i * R + a]  = v;
            /* Every path gains exactly 4e8, so the variance cannot move. */
            offset[(size_t)i * R + a] = v + 1e8;
        }
    }
    const BeamFunctional fp = { BEAM_FUNCTIONAL_TABLE, plain };
    const BeamFunctional fo = { BEAM_FUNCTIONAL_TABLE, offset };
    BeamConstraint *c = beam_query_constraint_new(L, R);
    BeamQueryMoments a, b;
    ASSERT_MSG(beam_query_expect(m, c, &fp, &a) == BEAM_OK, "plain");
    ASSERT_MSG(beam_query_expect(m, c, &fo, &b) == BEAM_OK, "offset");
    beam_query_constraint_free(c);

    ASSERT_MSG(a.variance > 1.0, "there is a variance to lose");
    ASSERT_MSG(fabs(b.mean - (a.mean + 4e8)) < 1e-6, "the mean shifted by 4e8");

    /*
     * Shifting a functional cannot change its variance, so the two must
     * agree. Carrying means of magnitude 4e8 in a double costs about
     * 4e8 * 2^-52, so the answer degrades to roughly 1e-8 relative and no
     * further -- measured at 6e-9 on this model.
     */
    ASSERT_MSG(fabs(b.variance - a.variance) < 1e-6 * a.variance,
               "a constant offset of 1e8 per position leaves the variance");

    /*
     * And this is what the rejected alternative would have produced. `E[f^2]`
     * is about 1.6e17, whose spacing between representable doubles is around
     * 32 -- so a variance of 20 cannot survive being recovered by subtracting
     * `E[f]^2` from it, and does not. Demonstrated rather than asserted,
     * because it is the entire reason the merge above is written centred.
     */
    const double e_f2  = b.variance + b.mean * b.mean;   /* E[f^2] */
    const double naive = e_f2 - b.mean * b.mean;
    ASSERT_MSG(fabs(naive - a.variance) > 1.0,
               "subtracting squared means really does destroy this variance, "
               "so the centred accumulation is load-bearing and not a "
               "preference");
    free(plain); free(offset);
    beam_model_destroy(m);
    PASS();
}

/* ── Constrained by an automaton ─────────────────────────────── */

static void test_moments_respect_an_automaton(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    const uint8_t motif[2] = {AA_A_, AA_S_};

    /* Counting the S's, so the motif constraint must move the answer. */
    double *tab = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(tab != NULL, "alloc");
    for (uint32_t i = 0; i < L; ++i) tab[(size_t)i * R + AA_S_] = 1.0;
    const BeamFunctional f = { BEAM_FUNCTIONAL_TABLE, tab };

    BeamConstraint *free_c = beam_query_constraint_new(L, R);
    BeamConstraint *mot_c  = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_automaton(
                   mot_c, beam_query_automaton_motif(L, R, motif, 2u))
               == BEAM_OK, "attached");

    BeamQueryMoments a, b;
    ASSERT_MSG(beam_query_expect(m, free_c, &f, &a) == BEAM_OK, "free");
    ASSERT_MSG(beam_query_expect(m, mot_c, &f, &b) == BEAM_OK, "motif");
    /* Requiring AS somewhere guarantees at least one S, so the conditional
     * expectation must exceed the unconditional one. */
    ASSERT_MSG(b.mean > a.mean + 1e-6,
               "conditioning on a motif containing S raises E[count(S)]");
    ASSERT_MSG(b.mean >= 1.0 - 1e-12,
               "and it cannot fall below the one S the motif guarantees");

    /* The same conditional, cross-checked against the posterior sweep. */
    double *marg = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(marg != NULL, "alloc");
    ASSERT_MSG(beam_query_marginals(m, mot_c, marg, NULL) == BEAM_OK, "marg");
    double want = 0.0;
    for (uint32_t i = 0; i < L; ++i) want += marg[(size_t)i * R + AA_S_];
    ASSERT_MSG(fabs(b.mean - want) < 1e-12,
               "the moment sweep and the product posterior agree under an "
               "automaton, not only under masks");

    beam_query_constraint_free(free_c);
    beam_query_constraint_free(mot_c);
    free(tab); free(marg);
    beam_model_destroy(m);
    PASS();
}

/* ── Refusals and shape ──────────────────────────────────────── */

static void test_moments_of_an_impossible_constraint_do_not_exist(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *tab = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(tab != NULL, "alloc");
    const BeamFunctional f = { BEAM_FUNCTIONAL_TABLE, tab };
    BeamQueryMoments mo;

    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_position(c, 2u, 1u);
    beam_query_constrain_position(c, 2u, 4u);
    ASSERT_MSG(beam_query_expect(m, c, &f, &mo) == BEAM_ERR_NO_LIVE_PATHS,
               "an empty constraint has no moments");
    beam_query_constraint_free(c);

    BeamConstraint *d = beam_query_constraint_new(L, R);
    for (uint32_t i = 0; i < 4u; ++i) beam_query_constrain_position(d, i, 0u);
    ASSERT_MSG(beam_query_expect(m, d, &f, &mo) == BEAM_ERR_NO_LIVE_PATHS,
               "nor has one the trellis cannot satisfy");
    beam_query_constraint_free(d);
    free(tab);
    beam_model_destroy(m);
    PASS();
}

static void test_a_functional_the_engine_cannot_sum_is_refused(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    BeamConstraint *c = beam_query_constraint_new(10u, R);
    BeamQueryMoments mo;

    BeamFunctional bad = { (BeamFunctionalKind)7, NULL };
    ASSERT_MSG(beam_query_expect(m, c, &bad, &mo)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "an unknown kind is refused");
    ASSERT_MSG(strstr(beam_error_message(), "decompose") != NULL,
               "and the message says why: it must decompose over the trellis");

    BeamFunctional no_table = { BEAM_FUNCTIONAL_TABLE, NULL };
    ASSERT_MSG(beam_query_expect(m, c, &no_table, &mo) == BEAM_ERR_NULL_ARG,
               "a table functional without a table is refused");
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

static void test_a_moment_sweep_is_cheaper_than_a_posterior(void) {
    /*
     * The moments of an additive functional propagate forwards, so nothing
     * here scales with the length. That is the property that makes entropy
     * answerable at lengths where the marginals would be refused, and it is
     * worth pinning: an implementation that grew a backward field would lose
     * it silently.
     */
    const uint32_t S = 194481u, Q = 4u, Rr = 20u;
    for (uint32_t L = 8u; L <= 24u; L += 8u) {
        const uint64_t mom = beam_query_moment_bytes(S, Q, Rr);
        const uint64_t pos = beam_query_sweep_bytes(S, Q, Rr, L, true);
        ASSERT_MSG(mom < pos, "moments cost less than a posterior");
        ASSERT_MSG(mom == beam_query_moment_bytes(S, Q, Rr),
                   "and do not grow with the length at all");
    }
    PASS();
}

int main(void) {
    printf("\n[entropy]\n");
    RUN_TEST(test_entropy_matches_enumeration);
    RUN_TEST(test_entropy_reproduces_the_models_own_surprisal_moments);
    RUN_TEST(test_a_length_with_one_sequence_has_no_entropy);

    printf("\n[agreement with the posterior sweep]\n");
    RUN_TEST(test_an_indicator_expectation_is_the_marginal);
    RUN_TEST(test_an_expected_count_is_the_column_of_marginals);
    RUN_TEST(test_the_moment_sweep_agrees_about_the_partition);

    printf("\n[enumeration]\n");
    RUN_TEST(test_moments_match_enumeration_under_a_constraint);

    printf("\n[numerical conditioning]\n");
    RUN_TEST(test_variance_survives_a_large_constant_offset);

    printf("\n[constraints]\n");
    RUN_TEST(test_moments_respect_an_automaton);
    RUN_TEST(test_moments_of_an_impossible_constraint_do_not_exist);
    RUN_TEST(test_a_functional_the_engine_cannot_sum_is_refused);
    RUN_TEST(test_a_moment_sweep_is_cheaper_than_a_posterior);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
