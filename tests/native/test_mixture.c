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

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
