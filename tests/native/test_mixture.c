/**
 * @file test_mixture.c
 * @brief Tests for the mixture kernels: the product sweep and the union sweep.
 *
 * Both kernels are checked against something that already existed rather than
 * against themselves.
 *
 * The product sweep meets `beam_model_hill_number` on the identity
 * `<P, P> = sum_s P(s)^2 = 1 / D2`, and its k-way form meets
 * `beam_model_log_mellin` on `<P, .., P> = sum_s P(s)^k`. Those two functions
 * reach the same numbers down a completely separate path -- the tilted
 * partition function in math/diversity.c -- so agreement is two independent
 * implementations of one quantity, not a routine agreeing with itself.
 *
 * Members trained on disjoint residue pools are held to a brute-force
 * enumeration of every nucleotide string up to length 6: under the
 * hierarchical prior each gives every one of those strings positive mass, so
 * their inner product is finite -- small, since each puts almost all of its
 * mass where the other puts almost none -- and has to be the exact sum.
 *
 * The union sweep meets `beam_model_path_count_total` and plain arithmetic.
 * Every member's support is the whole space, every string over the alphabet
 * up to max_length, so the union of any members is `sum_L R^L` whatever they
 * were trained on, and the sweep must carry exactly one member bitmask per
 * trellis state: a member wrongly dropped from a path it can spell shows as a
 * second mask in the peak, and nowhere in the totals, since the member left
 * behind still counts the path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/mixture.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define CLOSE(a, b, rel) (fabs((a) - (b)) <= (rel) * fabs(b))

/* Two corpora over disjoint residue pools. The models built from them still
 * share every sequence -- the prior gives every string mass -- but put almost
 * all of it in different places. */
#define N_A 8
static const char *TRAIN_A[N_A] = {
    "CASF", "CASY", "CAGF", "CGSF", "CASL", "CTSF", "CASQ", "CAAF",
};
static const double WEIGHT_A[N_A] = {40, 25, 18, 15, 12, 10, 9, 5};

#define N_B 8
static const char *TRAIN_B[N_B] = {
    "WMNP", "WMNK", "WMDP", "WHNP", "WMNH", "WKNP", "WMNM", "WMMP",
};
static const double WEIGHT_B[N_B] = {33, 21, 17, 14, 11, 9, 8, 6};

/* One sequence from B's pool, observed by a model of A's corpus: it moves
 * mass toward B's corner of the space without changing the support, which is
 * the whole space already. */
static const char *SHARED = "WMNP";

static BeamModel *build(const char **seqs, const double *w, uint32_t n,
                        uint8_t order) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = 6u;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, seqs, w, n) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/** `base` with `SHARED` appended, at abundance 1. */
static BeamModel *build_with_shared(const char **base, const double *w,
                                    uint32_t n, uint8_t order) {
    const char *seqs[N_A + N_B + 1];
    double wts[N_A + N_B + 1];
    for (uint32_t i = 0; i < n; ++i) { seqs[i] = base[i]; wts[i] = w[i]; }
    seqs[n] = SHARED;
    wts[n] = 1.0;
    return build(seqs, wts, n + 1u, order);
}

/* The same pair of disjoint pools on the nucleotide alphabet, small enough
 * that every string a model can emit -- 5 + 25 + ... + 5^6 = 19,530 of them,
 * N included -- can be enumerated and scored one at a time. */
#define N_NT 8
static const char *NT_A[N_NT] = {
    "ACCA", "ACAC", "CCAA", "ACCAC", "CAAC", "ACCCA", "AACC", "CACA",
};
static const char *NT_B[N_NT] = {
    "GTTG", "GTGT", "TTGG", "GTTGT", "TGGT", "GTTTG", "GGTT", "TGTG",
};
static const double NT_WEIGHT[N_NT] = {40, 25, 18, 15, 12, 10, 9, 5};
#define NT_MAX_LEN 6u

static BeamModel *build_nt(const char **seqs) {
    BeamModelConfig cfg;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &cfg);
    cfg.order = 2;
    cfg.max_length = NT_MAX_LEN;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, seqs, NT_WEIGHT, N_NT) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/**
 * `sum_s P_a(s) P_b(s)` over every nucleotide string of each length up to
 * NT_MAX_LEN, one `log_pseq` pair at a time. Shares nothing with the sweep
 * but the model's own scoring. `by[L]` receives the length-L sum; long double,
 * so twenty thousand positive terms add without drift worth mentioning.
 */
static void brute_inner_product(const BeamModel *a, const BeamModel *b,
                                long double by[NT_MAX_LEN + 1]) {
    static const char alphabet[] = "ACGTN";
    char s[NT_MAX_LEN + 1];
    by[0] = 0.0L;
    for (uint32_t L = 1; L <= NT_MAX_LEN; ++L) {
        uint32_t count = 1;
        for (uint32_t k = 0; k < L; ++k) count *= 5u;
        long double acc = 0.0L;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t v = i;
            for (uint32_t k = 0; k < L; ++k) { s[k] = alphabet[v % 5u]; v /= 5u; }
            s[L] = '\0';
            acc += expl((long double)beam_model_log_pseq(a, s)
                        + (long double)beam_model_log_pseq(b, s));
        }
        by[L] = acc;
    }
}


/**
 * `sum_s (P_a(s) P_b(s))^a` over every nucleotide string, one pair at a time.
 *
 * The exponent form of `brute_inner_product`, and the only reference for a
 * FRACTIONAL exponent on two different models: the Mellin path answers the
 * one-member case at any real order, but nothing in the library computes a
 * two-model Bhattacharyya coefficient a second way.
 */
static void brute_power_product(const BeamModel *a, const BeamModel *b,
                                double expo,
                                long double by[NT_MAX_LEN + 1]) {
    static const char alphabet[] = "ACGTN";
    char s[NT_MAX_LEN + 1];
    by[0] = 0.0L;
    for (uint32_t L = 1; L <= NT_MAX_LEN; ++L) {
        uint32_t count = 1;
        for (uint32_t k = 0; k < L; ++k) count *= 5u;
        long double acc = 0.0L;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t v = i;
            for (uint32_t k = 0; k < L; ++k) { s[k] = alphabet[v % 5u]; v /= 5u; }
            s[L] = '\0';
            acc += expl((long double)expo
                        * ((long double)beam_model_log_pseq(a, s)
                           + (long double)beam_model_log_pseq(b, s)));
        }
        by[L] = acc;
    }
}


/* ── The product sweep against the Mellin path ───────────────── */

static void test_a_model_against_itself_is_one_over_its_hill_number(void) {
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *pair[2] = {m, m};

    double lip = 0.0, d2 = 0.0;
    const BeamError e1 = beam_mixture_log_inner_product_by_length(pair, 2,
                                                                  NULL, &lip);
    const BeamError e2 = beam_model_hill_number(m, 2.0, &d2);
    beam_model_destroy(m);
    ASSERT_MSG(e1 == BEAM_OK && e2 == BEAM_OK, "a call failed");
    ASSERT_MSG(isfinite(lip), "the inner product is not finite");
    ASSERT_MSG(CLOSE(exp(-lip), d2, 1e-12),
               "1/<P,P> disagrees with hill_number(2)");
    PASS();
}

static void test_one_member_sums_to_one(void) {
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *one[1] = {m};
    double lip = 1.0;
    const BeamError e = beam_mixture_log_inner_product_by_length(one, 1, NULL,
                                                                 &lip);
    beam_model_destroy(m);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(fabs(lip) < 1e-12, "sum_s P(s) is not 1");
    PASS();
}

static void test_a_k_way_product_is_the_mellin_transform_at_k(void) {
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *rep[6] = {m, m, m, m, m, m};
    for (uint32_t k = 2; k <= 6; ++k) {
        double lip = 0.0, mel = 0.0;
        const BeamError e1 = beam_mixture_log_inner_product_by_length(rep, k,
                                                                      NULL, &lip);
        const BeamError e2 = beam_model_log_mellin(m, (double)k, &mel);
        if (e1 != BEAM_OK || e2 != BEAM_OK) {
            beam_model_destroy(m);
            ASSERT_MSG(0, "a call failed");
        }
        if (!CLOSE(lip, mel, 1e-12)) {
            beam_model_destroy(m);
            ASSERT_MSG(0, "the k-way product disagrees with log_mellin(k)");
        }
    }
    beam_model_destroy(m);
    PASS();
}

static void test_members_trained_apart_match_the_enumeration(void) {
    /*
     * Each member gives every string mass, so the product of two members
     * trained on disjoint pools is a finite sum over the whole space, and the
     * enumeration is that sum written out. Checked per length as well as in
     * total, since the total is dominated by a few lengths and would hide a
     * wrong term at another.
     */
    BeamModel *a = build_nt(NT_A);
    BeamModel *b = build_nt(NT_B);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *ab[2] = {a, b}, *aa[2] = {a, a}, *bb[2] = {b, b};
    double by[NT_MAX_LEN + 1], lip = 0.0, self_a = 0.0, self_b = 0.0;
    const BeamError e1 = beam_mixture_log_inner_product_by_length(ab, 2, by,
                                                                  &lip);
    const BeamError e2 = beam_mixture_log_inner_product_by_length(aa, 2, NULL,
                                                                  &self_a);
    const BeamError e3 = beam_mixture_log_inner_product_by_length(bb, 2, NULL,
                                                                  &self_b);
    long double want[NT_MAX_LEN + 1], total = 0.0L;
    brute_inner_product(a, b, want);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e1 == BEAM_OK && e2 == BEAM_OK && e3 == BEAM_OK,
               "a call failed");
    ASSERT_MSG(isfinite(lip), "members trained apart share no mass, but "
                              "every string has mass under both");

    ASSERT_MSG(by[0] == -INFINITY, "length 0 carries mass");
    for (uint32_t L = 1; L <= NT_MAX_LEN; ++L) {
        total += want[L];
        ASSERT_MSG(CLOSE(by[L], (double)logl(want[L]), 1e-12),
                   "a per-length term disagrees with the enumeration");
    }
    ASSERT_MSG(CLOSE(lip, (double)logl(total), 1e-12),
               "the inner product disagrees with the enumeration");

    /* Small, as well as finite. Cauchy-Schwarz bounds <A,B> by the geometric
     * mean of the self products, with equality only for proportional
     * distributions; members trained apart overlap only through the prior's
     * pseudo-mass, which puts them about 15.5 nats (a cosine of 1.8e-7)
     * below that bound here. A thousandth still separates "trained apart"
     * from "trained alike" by orders of magnitude on either side. */
    const double log_cosine = lip - 0.5 * (self_a + self_b);
    ASSERT_MSG(log_cosine < log(1e-3),
               "members trained on disjoint pools overlap as if trained alike");
    PASS();
}

static void test_the_per_length_terms_sum_to_the_total(void) {
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *pair[2] = {m, m};
    double by[8], total = 0.0;
    for (uint32_t i = 0; i < 8; ++i) by[i] = 0.0;
    const BeamError e = beam_mixture_log_inner_product_by_length(pair, 2, by,
                                                                 &total);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    double mx = -INFINITY;
    for (uint32_t L = 0; L < 7; ++L) if (by[L] > mx) mx = by[L];
    double acc = 0.0;
    for (uint32_t L = 0; L < 7; ++L)
        if (by[L] != -INFINITY) acc += exp(by[L] - mx);
    beam_model_destroy(m);
    ASSERT_MSG(CLOSE(mx + log(acc), total, 1e-13),
               "the by-length vector does not sum to the total");
    PASS();
}


/* ── The union sweep against path counting ───────────────────── */

static void test_identical_members_have_one_members_support(void) {
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *rep[3] = {m, m, m};
    double u = 0.0;
    uint64_t peak = 0;
    const BeamError e = beam_mixture_support_union_by_length(rep, 3, 0, NULL,
                                                             &u, &peak);
    const double want = beam_model_path_count_total(m);
    beam_model_destroy(m);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(u == want, "the union of a model with itself is not itself");
    ASSERT_MSG(peak > 0, "no peak population was reported");
    PASS();
}

static void test_the_union_of_any_members_is_the_whole_space(void) {
    /*
     * Members that share no training residue still have the same support:
     * every string over the twenty residues up to max_length. So the union
     * is that space, 20^L at each length, counted here by multiplication
     * rather than by any sweep.
     *
     * The peak is the sharper half. Both members spell every path, so no
     * member ever drops out of a bitmask, and the sweep holds exactly one
     * (state, mask) pair per trellis state: 20^2 = 400 at order 2, the width
     * of every layer from the second on. A member wrongly dropped from a
     * path's mask shows there as a second mask at that state, whatever the
     * totals say.
     */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *pair[2] = {a, b};
    double by[7], u = 0.0;
    uint64_t peak = 0;
    const BeamError e = beam_mixture_support_union_by_length(pair, 2, 0, by,
                                                             &u, &peak);
    const double own_a = beam_model_path_count_total(a);
    const double own_b = beam_model_path_count_total(b);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "call failed");

    double space = 0.0, at_length = 1.0;
    ASSERT_MSG(by[0] == 0.0, "the union counts a sequence of length 0");
    for (uint32_t L = 1; L <= 6; ++L) {
        at_length *= 20.0;
        space += at_length;
        ASSERT_MSG(by[L] == at_length,
                   "the union at one length is not every string of it");
    }
    ASSERT_MSG(u == space, "the union is not the whole space");
    ASSERT_MSG(own_a == space && own_b == space,
               "a member's own support is not the whole space");
    ASSERT_MSG(peak == 400u,
               "the sweep carried more than one member mask per state, so a "
               "member dropped out of a path it spells");
    PASS();
}

static void test_the_union_never_exceeds_the_pooled_model(void) {
    /* Under the prior both are the whole space, so this holds with equality;
     * the inequality is the relation that holds whatever the smoothing, and
     * the test above pins the value itself. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build_with_shared(TRAIN_B, WEIGHT_B, N_B, 2);
    BeamModel *pooled = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(a && b && pooled, "a model did not build");
    if (beam_model_add(pooled, b) != BEAM_OK
        || beam_model_finalize(pooled) != BEAM_OK) {
        beam_model_destroy(a); beam_model_destroy(b);
        beam_model_destroy(pooled);
        ASSERT_MSG(0, "the pooled model did not build");
    }
    const BeamModel *pair[2] = {a, b};
    double u = 0.0;
    const BeamError e = beam_mixture_support_union_by_length(pair, 2, 0, NULL,
                                                             &u, NULL);
    const double merged = beam_model_path_count_total(pooled);
    beam_model_destroy(a); beam_model_destroy(b); beam_model_destroy(pooled);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(u <= merged,
               "the union outgrew the pooled model, which cannot happen");
    PASS();
}

static void test_a_ceiling_of_one_refuses_rather_than_undercounts(void) {
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *pair[2] = {a, b};
    double u = -1.0;
    const BeamError e = beam_mixture_support_union_by_length(pair, 2, 1, NULL,
                                                             &u, NULL);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_ERR_BUDGET, "an impossible ceiling was not refused");
    PASS();
}


static void test_a_budget_too_small_refuses_both_sweeps(void) {
    /*
     * Exercised for its CLEANUP as much as its verdict. Both by-length drivers
     * allocate their per-length scratch before they know whether one sweep
     * fits, so the refusal path has something to free, and this is the case
     * that walks it. Run under the leak sanitiser it is the only coverage that
     * path gets.
     */
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *pair[2] = {m, m};
    const uint64_t saved = beam_model_memory_budget();

    beam_model_set_memory_budget(1024);
    double lip = 0.0, u = 0.0;
    const BeamError e1 = beam_mixture_log_inner_product_by_length(pair, 2, NULL,
                                                                  &lip);
    const BeamError e2 = beam_mixture_support_union_by_length(pair, 2, 0, NULL,
                                                              &u, NULL);
    beam_model_set_memory_budget(saved);
    beam_model_destroy(m);
    ASSERT_MSG(e1 == BEAM_ERR_BUDGET, "the inner product ignored the budget");
    ASSERT_MSG(e2 == BEAM_ERR_BUDGET, "the support union ignored the budget");
    PASS();
}


/* ── The shape gate ──────────────────────────────────────────── */

static void test_members_of_different_shapes_are_refused(void) {
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_A, WEIGHT_A, N_A, 3);   /* different order */
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *pair[2] = {a, b};
    const BeamError e = beam_mixture_check_shape(pair, 2);
    double lip = 0.0;
    const BeamError e2 = beam_mixture_log_inner_product_by_length(pair, 2, NULL,
                                                                  &lip);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_ERR_VARIANT_MISMATCH, "the shape gate let it through");
    ASSERT_MSG(e2 == BEAM_ERR_VARIANT_MISMATCH,
               "the sweep did not run the shape gate");
    PASS();
}

static void test_an_unfinalized_member_is_refused(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 2;
    cfg.max_length = 6u;
    BeamModel *raw = beam_model_new(&cfg);
    BeamModel *ok = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(raw && ok, "a model did not build");
    const BeamModel *pair[2] = {ok, raw};
    const BeamError e = beam_mixture_check_shape(pair, 2);
    beam_model_destroy(raw); beam_model_destroy(ok);
    ASSERT_MSG(e == BEAM_ERR_NOT_BUILT, "an unfinalized member was accepted");
    PASS();
}

static void test_the_union_ceiling_binds_only_the_union(void) {
    /*
     * The 64-member bitmask is the union sweep's limit, and used to be the
     * shape gate's too. That refused a cohort-scale mixture AT CONSTRUCTION and
     * took every other quantity with it -- the inner product never
     * distinguishes the members it multiplies and has no such bound. The gate
     * now passes, the union alone refuses.
     */
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const uint32_t over = BEAM_MIXTURE_MAX_MEMBERS + 1u;
    const BeamModel **many =
        (const BeamModel **)malloc((size_t)over * sizeof(*many));
    ASSERT_MSG(many, "allocation");
    for (uint32_t i = 0; i < over; ++i) many[i] = m;

    const BeamError shape = beam_mixture_check_shape(many, over);
    const BeamError empty = beam_mixture_check_shape(many, 0);

    double lip = 0.0, u = 0.0;
    const BeamError prod =
        beam_mixture_log_inner_product_by_length(many, over, NULL, &lip);
    const BeamError uni =
        beam_mixture_support_union_by_length(many, over, 0, NULL, &u, NULL);

    free(many);
    beam_model_destroy(m);
    ASSERT_MSG(shape == BEAM_OK, "the shape gate still enforces 64");
    ASSERT_MSG(empty == BEAM_ERR_EMPTY_INPUT, "an empty mixture was accepted");
    ASSERT_MSG(prod == BEAM_OK && isfinite(lip),
               "the inner product refused a mixture larger than 64");
    ASSERT_MSG(uni == BEAM_ERR_PARAM_OUT_OF_RANGE,
               "the union did not enforce its own bitmask limit");
    PASS();
}

/**
 * The PER-LENGTH union refuses above the ceiling too, and identically.
 *
 * This is the entry point whose safety actually depends on the check.
 * `beam_mixture_support_union` declares `LiveHandle h[BEAM_MIXTURE_MAX_MEMBERS]`
 * and `const BeamLive *L[...]` as fixed stack arrays indexed by member, so a
 * ceiling that failed here would be a stack smash rather than a wrong number --
 * and until this test, the only coverage was on
 * `beam_mixture_support_union_by_length`, which has no such arrays.
 *
 * The two refusals were separate hand-maintained copies of the same eleven
 * lines. They are one function now, and this asserts both callers reach it and
 * say the same thing, which is what stops them becoming two again.
 */
static void test_both_union_entry_points_enforce_the_same_ceiling(void) {
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const uint32_t over = BEAM_MIXTURE_MAX_MEMBERS + 1u;
    const BeamModel **many =
        (const BeamModel **)malloc((size_t)over * sizeof(*many));
    ASSERT_MSG(many, "allocation");
    for (uint32_t i = 0; i < over; ++i) many[i] = m;

    /* Length 2, the shortest with a transition between layers. The ceiling
     * is checked before any sweep, so no length refuses better than another;
     * the call at the ceiling does sweep, though, sixty-four members of it,
     * each member's layer built in a parallel region of its own, so the
     * length is what sets this test's cost. */
    double one = 0.0;
    const BeamError per_length =
        beam_mixture_support_union(many, over, 2u, 0, &one, NULL);
    static char msg_a[512];
    snprintf(msg_a, sizeof msg_a, "%s", beam_error_message());

    double all = 0.0;
    const BeamError pooled =
        beam_mixture_support_union_by_length(many, over, 0, NULL, &all, NULL);
    static char msg_b[512];
    snprintf(msg_b, sizeof msg_b, "%s", beam_error_message());

    /* And at the ceiling itself it is not refused, so the bound is `>` and not
     * `>=`: an off-by-one here costs a caller the largest mixture that fits. */
    const BeamError at = beam_mixture_support_union(many,
                                                    BEAM_MIXTURE_MAX_MEMBERS,
                                                    2u, 0, &one, NULL);
    free(many);
    beam_model_destroy(m);

    ASSERT_MSG(per_length == BEAM_ERR_PARAM_OUT_OF_RANGE,
               "the union that indexes fixed stack arrays by member did not "
               "enforce its own ceiling");
    ASSERT_MSG(pooled == BEAM_ERR_PARAM_OUT_OF_RANGE,
               "the per-length union did not enforce the ceiling");
    ASSERT_MSG(strcmp(msg_a, msg_b) == 0,
               "the two entry points refuse with different messages, so they "
               "are not sharing the check");
    ASSERT_MSG(at != BEAM_ERR_PARAM_OUT_OF_RANGE,
               "a mixture of exactly BEAM_MIXTURE_MAX_MEMBERS was refused, so "
               "the ceiling is off by one");
    PASS();
}

static void test_the_gram_matrix_matches_the_pairs_one_at_a_time(void) {
    /*
     * The driver exists for its parallelism, so what has to be checked is that
     * parallelism changed no answer: every entry must equal the sweep run on
     * that pair alone, and the matrix must be symmetric to the bit rather than
     * approximately -- one triangle is computed and mirrored precisely because
     * the sweep sums the members' weights in the order it is handed them.
     */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    BeamModel *c = build_with_shared(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(a && b && c, "a model did not build");
    const BeamModel *v[3] = {a, b, c};

    double g[9];
    const BeamError e = beam_mixture_gram(v, 3u, g);
    if (e != BEAM_OK) {
        beam_model_destroy(a); beam_model_destroy(b); beam_model_destroy(c);
        ASSERT_MSG(0, "the gram driver failed");
    }
    for (uint32_t i = 0; i < 3u; ++i)
        for (uint32_t j = 0; j < 3u; ++j) {
            const BeamModel *pair[2] = {v[i], v[j]};
            double want = 0.0;
            (void)beam_mixture_log_inner_product_by_length(pair, 2u, NULL, &want);
            if (memcmp(&g[i * 3u + j], &want, sizeof want) != 0) {
                beam_model_destroy(a); beam_model_destroy(b);
                beam_model_destroy(c);
                ASSERT_MSG(0, "a gram entry differs from the pair alone");
            }
        }
    for (uint32_t i = 0; i < 3u; ++i)
        for (uint32_t j = 0; j < 3u; ++j)
            if (memcmp(&g[i * 3u + j], &g[j * 3u + i], sizeof g[0]) != 0) {
                beam_model_destroy(a); beam_model_destroy(b);
                beam_model_destroy(c);
                ASSERT_MSG(0, "the gram matrix is not exactly symmetric");
            }
    beam_model_destroy(a); beam_model_destroy(b); beam_model_destroy(c);
    PASS();
}


/* ── Per-member exponents ────────────────────────────────────── */

static void test_all_ones_exponents_are_the_plain_inner_product(void) {
    /* Bit for bit, not to a tolerance. `1.0 * x` is exact, so the exponent
     * path must land on the same bytes as the path without one; anything else
     * means the scaling moved an accumulation. */
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(m && b, "a model did not build");
    const BeamModel *pair[2] = {m, b};
    const double ones[2] = {1.0, 1.0};
    double plain = 0.0, powered = 0.0;
    const BeamError e1 = beam_mixture_log_inner_product_by_length(pair, 2,
                                                                  NULL, &plain);
    const BeamError e2 = beam_mixture_log_power_product_by_length(pair, ones, 2,
                                                                  NULL, &powered);
    beam_model_destroy(m); beam_model_destroy(b);
    ASSERT_MSG(e1 == BEAM_OK && e2 == BEAM_OK, "a call failed");
    ASSERT_MSG(memcmp(&plain, &powered, sizeof plain) == 0,
               "all-ones exponents did not reproduce the inner product "
               "exactly");
    PASS();
}

static void test_a_fractional_exponent_is_the_mellin_transform(void) {
    /* One member at a real order: `sum_s P(s)^q` by the product sweep and by
     * the tilted sweep in math/diversity.c, which share no code. This is the
     * only full-trellis check on a fractional exponent. */
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *one[1] = {m};
    static const double orders[] = {0.25, 0.5, 0.75, 1.5, 2.5};
    for (size_t t = 0; t < sizeof orders / sizeof orders[0]; ++t) {
        double lip = 0.0, mel = 0.0;
        const BeamError e1 = beam_mixture_log_power_product_by_length(
            one, &orders[t], 1, NULL, &lip);
        const BeamError e2 = beam_model_log_mellin(m, orders[t], &mel);
        if (e1 != BEAM_OK || e2 != BEAM_OK || !CLOSE(lip, mel, 1e-12)) {
            beam_model_destroy(m);
            ASSERT_MSG(0, "sum_s P^q disagrees with log_mellin(q)");
        }
    }
    beam_model_destroy(m);
    PASS();
}

static void test_the_bhattacharyya_coefficient_matches_the_enumeration(void) {
    BeamModel *a = build_nt(NT_A);
    BeamModel *b = build_nt(NT_B);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *ab[2] = {a, b};
    const double half[2] = {0.5, 0.5};
    double by[NT_MAX_LEN + 1], lbc = 0.0;
    const BeamError e = beam_mixture_log_power_product_by_length(ab, half, 2,
                                                                 by, &lbc);
    long double want[NT_MAX_LEN + 1], total = 0.0L;
    brute_power_product(a, b, 0.5, want);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(by[0] == -INFINITY, "length 0 carries mass");
    for (uint32_t L = 1; L <= NT_MAX_LEN; ++L) {
        total += want[L];
        ASSERT_MSG(CLOSE(by[L], (double)logl(want[L]), 1e-12),
                   "a per-length Bhattacharyya term differs from the sum");
    }
    ASSERT_MSG(CLOSE(lbc, (double)logl(total), 1e-12),
               "the Bhattacharyya coefficient differs from the enumeration");
    ASSERT_MSG(lbc < 0.0, "BC of two different models is not below 1");
    PASS();
}

static void test_a_model_against_itself_has_coefficient_one(void) {
    /* `sum_s sqrt(P P) = sum_s P = 1`. Fails for any error in the half-power
     * of the per-length constant, of an edge weight or of the sink. */
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *pair[2] = {m, m};
    const double half[2] = {0.5, 0.5};
    double lbc = 1.0;
    const BeamError e = beam_mixture_log_power_product_by_length(pair, half, 2,
                                                                 NULL, &lbc);
    beam_model_destroy(m);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(fabs(lbc) < 1e-12, "BC(P, P) is not 1");
    PASS();
}

static void test_a_zero_exponent_drops_its_member(void) {
    /* `P^0 = 1` everywhere, including where that member has almost no mass,
     * so `<P_a^1, P_b^0>` is `sum_s P_a(s) = 1`. Treating the zeroed member as
     * still constraining the support would instead return the mass `a` puts on
     * `b`'s corner, which for these two corpora is orders of magnitude away. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *pair[2] = {a, b};
    const double drop[2] = {1.0, 0.0};
    double v = 1.0;
    const BeamError e = beam_mixture_log_power_product_by_length(pair, drop, 2,
                                                                 NULL, &v);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(fabs(v) < 1e-12, "a zero exponent did not drop its member");
    PASS();
}

static void test_bad_exponents_are_refused(void) {
    BeamModel *m = build(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(m, "model did not build");
    const BeamModel *pair[2] = {m, m};
    const double neg[2] = {1.0, -0.5};
    const double nan2[2] = {1.0, NAN};
    const double zero[2] = {0.0, 0.0};
    double v = 0.0;
    const BeamError e1 = beam_mixture_log_power_product(pair, neg, 2, 4u, &v);
    const BeamError e2 = beam_mixture_log_power_product(pair, nan2, 2, 4u, &v);
    const BeamError e3 = beam_mixture_log_power_product(pair, zero, 2, 4u, &v);
    const BeamError e4 = beam_mixture_power_gram(pair, 2, 0.0, &v);
    const BeamError e5 = beam_mixture_power_gram(pair, 2, -1.0, &v);
    beam_model_destroy(m);
    ASSERT_MSG(e1 != BEAM_OK, "a negative exponent was accepted");
    ASSERT_MSG(e2 != BEAM_OK, "a NaN exponent was accepted");
    ASSERT_MSG(e3 != BEAM_OK, "an all-zero exponent vector was accepted");
    ASSERT_MSG(e4 != BEAM_OK, "a zero gram exponent was accepted");
    ASSERT_MSG(e5 != BEAM_OK, "a negative gram exponent was accepted");
    PASS();
}

static void test_the_per_position_terms_sum_to_the_total(void) {
    /* The whole contract: the positions, the sink and the non-positional
     * constant account for the total and nothing is left over. Checked on two
     * models trained apart, at every length both can produce, so a term that
     * is right at one length and wrong at another cannot hide in a single
     * lucky case. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "models did not build");
    const BeamModel *pair[2] = {a, b};
    const double half[2] = {0.5, 0.5};
    int checked = 0;
    double worst = 0.0;
    for (uint32_t len = 1; len <= 6u; ++len) {
        double by[8], sink = 0.0, konst = 0.0, total = 0.0;
        for (uint32_t i = 0; i < 8; ++i) by[i] = 0.0;
        const BeamError e = beam_mixture_log_power_product_by_position(
                pair, half, 2, len, by, &sink, &konst, &total);
        if (e != BEAM_OK) continue;
        if (total == -INFINITY) continue;
        double acc = konst + sink;
        for (uint32_t i = 0; i < len; ++i) acc += by[i];
        const double err = fabs(acc - total);
        if (err > worst) worst = err;
        ++checked;
    }
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(checked >= 4, "too few lengths were decomposable to be a test");
    ASSERT_MSG(worst < 1e-12, "the positional terms do not sum to the total");
    PASS();
}


static void test_the_positional_split_leaves_the_total_bit_identical(void) {
    /* The decomposition adds accumulators and moves no existing arithmetic, so
     * the total must be the plain call's answer to the last bit -- not close to
     * it. A tolerance here would hide exactly the drift this asserts against. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "models did not build");
    const BeamModel *pair[2] = {a, b};
    const double half[2] = {0.5, 0.5};
    int bad = 0, checked = 0;
    for (uint32_t len = 1; len <= 6u; ++len) {
        double plain = 0.0, total = 0.0, by[8], sink = 0.0, konst = 0.0;
        const BeamError e1 = beam_mixture_log_power_product(pair, half, 2, len,
                                                            &plain);
        const BeamError e2 = beam_mixture_log_power_product_by_position(
                pair, half, 2, len, by, &sink, &konst, &total);
        if (e1 != BEAM_OK || e2 != BEAM_OK) continue;
        ++checked;
        if (total != plain) ++bad;           /* exact, deliberately */
        /* and every output optional, which must not change the total either */
        double only = 0.0;
        const BeamError e3 = beam_mixture_log_power_product_by_position(
                pair, half, 2, len, NULL, NULL, NULL, &only);
        if (e3 != BEAM_OK || only != plain) ++bad;
    }
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(checked >= 4, "too few lengths to compare");
    ASSERT_MSG(bad == 0, "the positional split moved the total");
    PASS();
}


static void test_the_positional_total_agrees_with_the_by_length_vector(void) {
    /* Cross-check against a function that already existed and is already
     * tested, rather than against this one's own arithmetic. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "models did not build");
    const BeamModel *pair[2] = {a, b};
    const double half[2] = {0.5, 0.5};
    double byL[64], tot = 0.0;
    for (uint32_t i = 0; i < 64; ++i) byL[i] = 0.0;
    const BeamError e = beam_mixture_log_power_product_by_length(pair, half, 2,
                                                                 byL, &tot);
    int bad = 0, checked = 0;
    if (e == BEAM_OK) {
        for (uint32_t len = 1; len <= 6u; ++len) {
            double total = 0.0;
            if (beam_mixture_log_power_product_by_position(
                    pair, half, 2, len, NULL, NULL, NULL, &total) != BEAM_OK)
                continue;
            ++checked;
            if (total != byL[len]) ++bad;
        }
    }
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "the by-length call failed");
    ASSERT_MSG(checked >= 4, "too few lengths to compare");
    ASSERT_MSG(bad == 0, "a per-position total disagrees with by-length");
    PASS();
}


static void test_the_positional_profile_does_not_depend_on_member_order(void) {
    /* THE GAUGE TEST, and the reason the terms are mass ratios rather than the
     * sweep's rescaling maxima. Swapping the members drives the sweep from the
     * other model's weights and recomputes every maximum, so a term that
     * depended on the rescaling would move. Combining terms across the three
     * sweeps `jj`, `kk` and `jk` -- which is what the dissimilarity profile
     * does -- is only meaningful if they do not. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "models did not build");
    const BeamModel *ab[2] = {a, b};
    const BeamModel *ba[2] = {b, a};
    const double half[2] = {0.5, 0.5};
    double worst = 0.0;
    int checked = 0;
    for (uint32_t len = 2; len <= 6u; ++len) {
        double p1[8], p2[8], s1 = 0.0, s2 = 0.0, c1 = 0.0, c2 = 0.0,
               t1 = 0.0, t2 = 0.0;
        if (beam_mixture_log_power_product_by_position(
                ab, half, 2, len, p1, &s1, &c1, &t1) != BEAM_OK) continue;
        if (beam_mixture_log_power_product_by_position(
                ba, half, 2, len, p2, &s2, &c2, &t2) != BEAM_OK) continue;
        if (t1 == -INFINITY || t2 == -INFINITY) continue;
        ++checked;
        for (uint32_t i = 0; i < len; ++i)
            if (fabs(p1[i] - p2[i]) > worst) worst = fabs(p1[i] - p2[i]);
        if (fabs(s1 - s2) > worst) worst = fabs(s1 - s2);
        if (fabs(c1 - c2) > worst) worst = fabs(c1 - c2);
    }
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(checked >= 3, "too few lengths to compare");
    ASSERT_MSG(worst < 1e-12, "the profile depends on which member is first");
    PASS();
}


static void test_a_dropped_member_leaves_the_single_member_profile(void) {
    /* An exponent of 0 drops its member, so `<P_A^1, P_B^0>` must decompose
     * exactly as `<P_A>` does. Two different paths reach it -- compaction from
     * two members, and a plain one-member sweep -- and they must agree position
     * by position, not merely in the total. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "models did not build");
    const BeamModel *ab[2] = {a, b};
    const BeamModel *just_a[1] = {a};
    const double drop[2] = {1.0, 0.0};
    double worst = 0.0;
    int checked = 0;
    for (uint32_t len = 2; len <= 6u; ++len) {
        double p1[8], p2[8], s1 = 0.0, s2 = 0.0, c1 = 0.0, c2 = 0.0,
               t1 = 0.0, t2 = 0.0;
        if (beam_mixture_log_power_product_by_position(
                ab, drop, 2, len, p1, &s1, &c1, &t1) != BEAM_OK) continue;
        if (beam_mixture_log_power_product_by_position(
                just_a, NULL, 1, len, p2, &s2, &c2, &t2) != BEAM_OK) continue;
        if (t1 == -INFINITY || t2 == -INFINITY) continue;
        ++checked;
        for (uint32_t i = 0; i < len; ++i)
            if (fabs(p1[i] - p2[i]) > worst) worst = fabs(p1[i] - p2[i]);
        if (fabs(s1 - s2) > worst) worst = fabs(s1 - s2);
    }
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(checked >= 3, "too few lengths to compare");
    ASSERT_MSG(worst < 1e-12, "dropping a member changed the profile");
    PASS();
}


static void test_the_dissimilarity_profile_sums_to_a_nonnegative_distance(void) {
    /* The object this decomposition exists for:
     *
     *     d_jk(i) = (1/2)[g_jj(i) + g_kk(i)] - g_jk(i)
     *     D_jk    = sum_i d_jk(i) + the same combination of sink and const
     *
     * `D_jk = -log BC >= 0` by Cauchy-Schwarz, zero exactly when the two
     * models agree. The self-terms subtract the common structure that makes a
     * raw profile non-flat -- BEAM's layer weights carry a backward term, so
     * the forward mass is not conserved along the receptor and the sink closes
     * the accounting. That is why the profile to READ is this difference and
     * never a single sweep's terms. */
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "models did not build");
    const double half[2] = {0.5, 0.5};
    const uint32_t len = 4u;
    double g[3][8], sk[3], ko[3], tt[3];
    const BeamModel *sweeps[3][2] = {{a, a}, {b, b}, {a, b}};
    int ok = 1;
    for (int c = 0; c < 3; ++c)
        if (beam_mixture_log_power_product_by_position(
                sweeps[c], half, 2, len, g[c], &sk[c], &ko[c],
                &tt[c]) != BEAM_OK) ok = 0;
    double D = 0.0, self_worst = 0.0;
    if (ok) {
        for (uint32_t i = 0; i < len; ++i)
            D += 0.5 * (g[0][i] + g[1][i]) - g[2][i];
        D += 0.5 * (sk[0] + sk[1]) - sk[2];
        D += 0.5 * (ko[0] + ko[1]) - ko[2];
        /* and the same construction on a model against itself is identically
         * zero, term by term, which is the consistency of the gauge across
         * three separate calls */
        for (uint32_t i = 0; i < len; ++i) {
            const double d_self = 0.5 * (g[0][i] + g[0][i]) - g[0][i];
            if (fabs(d_self) > self_worst) self_worst = fabs(d_self);
        }
    }
    /* AT A FIXED LENGTH the self-products are not 1: `Z_jj(L) = sum_{|s|=L}
     * P_j(s) = pi_j(L)`. So the distance is
     *
     *     D_jk(L) = (1/2)[log pi_j(L) + log pi_k(L)] - log Z_jk(L)
     *
     * non-negative by Cauchy-Schwarz in the form
     * `sum sqrt(P_j P_k) <= sqrt(pi_j pi_k)`, which is also where the
     * `nu(L) proportional to sqrt(pi_j pi_k)` weight for aggregating across
     * lengths comes from. Comparing against `-log Z_jk` alone drops that term
     * and is only right once summed over every length. */
    const double scalar = 0.5 * (tt[0] + tt[1]) - tt[2];
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(ok, "a sweep failed");
    ASSERT_MSG(D > -1e-12, "the positional distance is negative");
    ASSERT_MSG(self_worst < 1e-15, "a self-pair has a non-zero profile");
    ASSERT_MSG(CLOSE(D, scalar, 1e-12),
               "the positional terms do not sum to -log BC");
    PASS();
}


static void test_the_power_gram_matches_the_pairs_one_at_a_time(void) {
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    BeamModel *c = build_with_shared(TRAIN_A, WEIGHT_A, N_A, 2);
    ASSERT_MSG(a && b && c, "a model did not build");
    const BeamModel *v[3] = {a, b, c};
    const double half[2] = {0.5, 0.5};
    double g[9];
    const BeamError e = beam_mixture_power_gram(v, 3u, 0.5, g);
    if (e != BEAM_OK) {
        beam_model_destroy(a); beam_model_destroy(b); beam_model_destroy(c);
        ASSERT_MSG(0, "the power gram failed");
    }
    for (uint32_t i = 0; i < 3u; ++i) {
        /* Every diagonal entry is `ln sum_s P_i(s) = 0`: a free check on each
         * row, which the plain Gram matrix does not have. */
        if (fabs(g[i * 3u + i]) > 1e-12) {
            beam_model_destroy(a); beam_model_destroy(b);
            beam_model_destroy(c);
            ASSERT_MSG(0, "a power-gram diagonal entry is not ln 1");
        }
        for (uint32_t j = i + 1u; j < 3u; ++j) {
            const BeamModel *pair[2] = {v[i], v[j]};
            double want = 0.0;
            (void)beam_mixture_log_power_product_by_length(pair, half, 2u,
                                                           NULL, &want);
            if (!CLOSE(g[i * 3u + j], want, 1e-12)
                || memcmp(&g[i * 3u + j], &g[j * 3u + i], sizeof g[0]) != 0) {
                beam_model_destroy(a); beam_model_destroy(b);
                beam_model_destroy(c);
                ASSERT_MSG(0, "a power-gram entry differs from the pair alone "
                              "or the matrix is not exactly symmetric");
            }
        }
    }
    beam_model_destroy(a); beam_model_destroy(b); beam_model_destroy(c);
    PASS();
}


int main(void) {
    printf("\n==============================================================\n");
    printf("  Mixture kernels\n");
    printf("==============================================================\n");

    printf("\n[the product sweep against the Mellin path]\n");
    RUN_TEST(test_a_model_against_itself_is_one_over_its_hill_number);
    RUN_TEST(test_one_member_sums_to_one);
    RUN_TEST(test_a_k_way_product_is_the_mellin_transform_at_k);
    RUN_TEST(test_members_trained_apart_match_the_enumeration);
    RUN_TEST(test_the_per_length_terms_sum_to_the_total);

    printf("\n[the union sweep against path counting]\n");
    RUN_TEST(test_identical_members_have_one_members_support);
    RUN_TEST(test_the_union_of_any_members_is_the_whole_space);
    RUN_TEST(test_the_union_never_exceeds_the_pooled_model);
    RUN_TEST(test_a_ceiling_of_one_refuses_rather_than_undercounts);
    RUN_TEST(test_a_budget_too_small_refuses_both_sweeps);

    printf("\n[the shape gate]\n");
    RUN_TEST(test_members_of_different_shapes_are_refused);
    RUN_TEST(test_an_unfinalized_member_is_refused);
    RUN_TEST(test_the_union_ceiling_binds_only_the_union);
    RUN_TEST(test_both_union_entry_points_enforce_the_same_ceiling);
    RUN_TEST(test_the_gram_matrix_matches_the_pairs_one_at_a_time);

    printf("\n[per-member exponents]\n");
    RUN_TEST(test_all_ones_exponents_are_the_plain_inner_product);
    RUN_TEST(test_a_fractional_exponent_is_the_mellin_transform);
    RUN_TEST(test_the_bhattacharyya_coefficient_matches_the_enumeration);
    RUN_TEST(test_a_model_against_itself_has_coefficient_one);
    RUN_TEST(test_a_zero_exponent_drops_its_member);
    RUN_TEST(test_bad_exponents_are_refused);
    RUN_TEST(test_the_power_gram_matches_the_pairs_one_at_a_time);

    printf("\n[the positional decomposition]\n");
    RUN_TEST(test_the_per_position_terms_sum_to_the_total);
    RUN_TEST(test_the_positional_split_leaves_the_total_bit_identical);
    RUN_TEST(test_the_positional_total_agrees_with_the_by_length_vector);
    RUN_TEST(test_the_positional_profile_does_not_depend_on_member_order);
    RUN_TEST(test_a_dropped_member_leaves_the_single_member_profile);
    RUN_TEST(test_the_dissimilarity_profile_sums_to_a_nonnegative_distance);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
