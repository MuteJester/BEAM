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
#define AA_W 18u

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

/**
 * `E[f]` and `Var[f]` over the sequences whose residue at `pos` lies in
 * `keep`, by direct summation. The conditional reference the test above needs.
 */
static void enumerate_moments_where(const BeamModel *m, uint32_t length,
                                    const double *table, uint32_t pos,
                                    BeamResidueMask keep,
                                    double *out_mean, double *out_var) {
    const BeamAlphabetInfo *ai = beam_alphabet_info(BEAM_ALPHABET_AA);
    char buf[8];
    uint8_t idx[8];
    double total = 0.0, mean = 0.0, var = 0.0;

    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < length; ++i) idx[i] = 0u;
        buf[length] = '\0';
        for (;;) {
            if ((keep >> idx[pos]) & 1u) {
                for (uint32_t i = 0; i < length; ++i)
                    buf[i] = ai->idx_to_char[idx[i]];
                const double raw = exp(beam_model_log_pseq(m, buf));
                if (raw > 0.0) {
                    double f = 0.0;
                    for (uint32_t i = 0; i < length; ++i)
                        f += table[(size_t)i * R + idx[i]];
                    if (pass == 0) { total += raw; mean += raw * f; }
                    else var += (raw / total) * (f - mean) * (f - mean);
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
    *out_mean = mean;
    *out_var = var;
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

    /*
     * Again with a mask, and this time the reference conditions too. The
     * earlier version of this only checked that the mean had MOVED and the
     * variance was non-negative, which any number satisfies; review was right
     * to call it out.
     */
    const BeamResidueMask keep = (1u << AA_A_) | (1u << AA_G);
    BeamConstraint *d = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_set(d, 1u, keep) == BEAM_OK, "mask");
    BeamQueryMoments md;
    ASSERT_MSG(beam_query_expect(m, d, &f, &md) == BEAM_OK, "expect");
    beam_query_constraint_free(d);

    double cm = 0.0, cv = 0.0;
    enumerate_moments_where(m, L, tab, 1u, keep, &cm, &cv);
    ASSERT_MSG(fabs(md.mean - cm) < 1e-12,
               "E[f | mask] matches the enumerated conditional");
    ASSERT_MSG(fabs(md.variance - cv) < 1e-12,
               "and so does Var[f | mask]");
    ASSERT_MSG(fabs(md.mean - mo.mean) > 1e-9, "the mask really did change it");
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

static void test_variance_survives_an_offset_that_squares_to_infinity(void) {
    /*
     * Regression for a defect found by review. On the FIRST contribution into
     * an accumulator the running mean is still zero, so the deviation is the
     * whole functional value rather than a difference. At 1e155 that squares
     * to infinity, and `infinity * 0` -- the zero being the accumulator's
     * empty weight -- is NaN, not zero. The NaN travelled to the end and was
     * reported as a variance of ZERO, because `NaN > 0` is false: a silent
     * wrong answer, with BEAM_OK.
     *
     * The first contribution is now taken as its own case and no deviation is
     * formed. All the path values and the true variance here are ordinary
     * finite doubles; only the intermediate was not.
     */
    const char *seqs[2] = {"CASF", "CAGF"};
    const double w[2] = {1.0, 1.0};
    BeamModel *m = build(seqs, w, 2u, 2u, 6u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    double *tab = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(tab != NULL, "alloc");
    /* Every path totals about 4e155; the two differ by 1e150, so the true
     * variance is about 2.5e299 -- large, and perfectly representable. */
    for (uint32_t i = 0; i < L; ++i)
        for (uint32_t a = 0; a < R; ++a)
            tab[(size_t)i * R + a] = 1e155;
    tab[(size_t)2 * R + AA_S_] = 1e155 + 1e150;

    const BeamFunctional f = { BEAM_FUNCTIONAL_TABLE, tab };
    BeamConstraint *c = beam_query_constraint_new(L, R);
    BeamQueryMoments mo;
    ASSERT_MSG(beam_query_expect(m, c, &f, &mo) == BEAM_OK, "expect");
    beam_query_constraint_free(c);

    /* Two equally likely paths differing by 1e150: variance is (d/2)^2. */
    const double want = 0.25 * 1e150 * 1e150;
    ASSERT_MSG(mo.variance > 0.0,
               "a finite variance is not reported as zero");
    ASSERT_MSG(fabs(mo.variance - want) < 1e-6 * want,
               "and it is the variance the two paths actually have");
    ASSERT_MSG(isfinite(mo.mean) && mo.mean > 1e155, "the mean is finite too");
    free(tab);
    beam_model_destroy(m);
    PASS();
}

static void test_many_rare_paths_still_mix_into_an_entropy(void) {
    /*
     * The test below pins every position, so each state has exactly one
     * arrival and the merge never has to COMBINE two contributions. This one
     * leaves a position free, giving twenty paths all around 6,900 nats below
     * the mode, so the mixture is exercised where its weights would underflow.
     *
     * Taking those weights as `exp(a) / (exp(a) + exp(b))` -- the obvious
     * form -- gives 0/0 at these magnitudes. The merge subtracts the larger
     * log first, so its two weights are exact complements whatever the inputs,
     * and that is the whole reason this file carries logarithms.
     */
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, true);
    ASSERT_MSG(m != NULL, "open-world model built");
    const uint32_t L = 10u, FREE = 4u;

    BeamConstraint *c = beam_query_constraint_new(L, R);
    for (uint32_t i = 0; i < L; ++i)
        if (i != FREE)
            ASSERT_MSG(beam_query_constrain_position(c, i, AA_W) == BEAM_OK,
                       "pin");

    double h = -1.0, v = -1.0;
    ASSERT_MSG(beam_query_entropy(m, c, &h, &v) == BEAM_OK, "entropy");
    ASSERT_MSG(h > 0.1,
               "twenty admitted sequences carry real entropy, so the merge "
               "genuinely combined contributions rather than taking one");
    /* Not that the surprisal must VARY: if the free position is uniform the
     * admitted sequences are equally likely and its spread is exactly zero,
     * which is the right answer and not a collapsed one. */
    ASSERT_MSG(v >= 0.0 && isfinite(v), "with a finite surprisal spread");

    /*
     * With every other position determined, the sequence entropy IS the
     * entropy of the free position's marginal -- computed by the posterior
     * sweep, which shares no accumulator with the moment sweep.
     */
    double *marg = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(marg != NULL, "alloc");
    ASSERT_MSG(beam_query_marginals(m, c, marg, NULL) == BEAM_OK, "marginals");
    double want = 0.0;
    for (uint32_t a = 0; a < R; ++a) {
        const double q = marg[(size_t)FREE * R + a];
        if (q > 0.0) want -= q * log(q);
    }
    ASSERT_MSG(fabs(h - want) < 1e-9,
               "and it matches the entropy of that position's marginal");
    free(marg);
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

static void test_a_rare_but_certain_event_has_moments(void) {
    /*
     * Regression for a defect found by review. The sink rescale used to be
     * taken over every trellis state, whether or not the sweep had reached it
     * or the automaton would accept it. A conditioned event more than about
     * 745 nats below an unreachable maximum then underflowed to no mass at
     * all, and a perfectly producible question came back NO_LIVE_PATHS --
     * while `beam_query_log_probability`, which is log-space throughout,
     * answered it without difficulty.
     *
     * Pinning every position leaves exactly one sequence, so its conditional
     * entropy and variance are exactly zero and its log Z is its own score.
     */
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, true);
    ASSERT_MSG(m != NULL, "open-world model built");
    const uint32_t L = 10u;
    const char *rare = "WWWWWWWWWW";
    const double score = beam_model_log_pseq(m, rare);
    ASSERT_MSG(isfinite(score) && score < -700.0,
               "producible, and thousands of nats below the mode");

    BeamConstraint *c = beam_query_constraint_new(L, R);
    for (uint32_t i = 0; i < L; ++i)
        ASSERT_MSG(beam_query_constrain_position(c, i, AA_W) == BEAM_OK, "pin");

    double lp = 0.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &lp) == BEAM_OK,
               "the log-space route always answered this");
    double h = -1.0, v = -1.0;
    ASSERT_MSG(beam_query_entropy(m, c, &h, &v) == BEAM_OK,
               "and so must the moment sweep");
    ASSERT_MSG(fabs(h) < 1e-9, "one sequence carries no entropy");
    ASSERT_MSG(v >= 0.0 && v < 1e-9, "and its surprisal does not vary");

    const BeamFunctional f = { BEAM_FUNCTIONAL_LOG_WEIGHT, NULL };
    BeamQueryMoments mo;
    ASSERT_MSG(beam_query_expect(m, c, &f, &mo) == BEAM_OK, "expect");
    ASSERT_MSG(fabs(mo.log_z - (lp + beam_model_log_partition(m, L))) < 1e-6,
               "and the two routes agree on the constrained partition");
    beam_query_constraint_free(c);
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

    /*
     * A non-finite entry is refused, naming the cell. `np.log(p)` is the
     * obvious way to build a functional and gives -inf wherever p is zero; the
     * sweep then formed a difference of two infinities and the NaN reached the
     * end, where `NaN > 0` being false reported it as a VARIANCE OF ZERO.
     * Found by review, and the same shape as a defect found on the overflow
     * path before it.
     */
    double *bad_tab = (double *)calloc((size_t)10u * R, sizeof(double));
    ASSERT_MSG(bad_tab != NULL, "alloc");
    for (uint32_t probe = 0; probe < 3u; ++probe) {
        bad_tab[(size_t)2 * R + 4] = (probe == 0) ? -INFINITY
                                   : (probe == 1) ? INFINITY : NAN;
        BeamFunctional bad_val = { BEAM_FUNCTIONAL_TABLE, bad_tab };
        ASSERT_MSG(beam_query_expect(m, c, &bad_val, &mo)
                   == BEAM_ERR_PARAM_OUT_OF_RANGE,
                   "a non-finite functional value is refused");
        ASSERT_MSG(strstr(beam_error_message(), "position 2 residue 4") != NULL,
                   "and the message names the cell");
    }
    free(bad_tab);
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
    const uint64_t S = 194481u, Q = 4u, Rr = 20u;
    /*
     * Against arithmetic done here rather than another call to the same
     * helper: three quantities per product state, double-buffered, one layer
     * of weights, and the sink. A helper that returned a constant zero would
     * pass a comparison with itself and silently disable the budget guard.
     */
    const uint64_t expect = 6ull * S * Q * sizeof(double)
                          + S * Rr * sizeof(double)
                          + S * sizeof(double);
    ASSERT_MSG(beam_query_moment_bytes(S, Q, Rr) == expect,
               "the moment sweep is costed as what it allocates");
    for (uint32_t L = 8u; L <= 24u; L += 8u)
        ASSERT_MSG(beam_query_moment_bytes(S, Q, Rr)
                   < beam_query_sweep_bytes(S, Q, Rr, L, true),
                   "and costs less than a posterior at every length");
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
    RUN_TEST(test_variance_survives_an_offset_that_squares_to_infinity);
    RUN_TEST(test_a_rare_but_certain_event_has_moments);
    RUN_TEST(test_many_rare_paths_still_mix_into_an_entropy);

    printf("\n[constraints]\n");
    RUN_TEST(test_moments_respect_an_automaton);
    RUN_TEST(test_moments_of_an_impossible_constraint_do_not_exist);
    RUN_TEST(test_a_functional_the_engine_cannot_sum_is_refused);
    RUN_TEST(test_a_moment_sweep_is_cheaper_than_a_posterior);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
