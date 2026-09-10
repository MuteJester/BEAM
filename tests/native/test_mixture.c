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
 * The union sweep meets `beam_model_path_count_total` on cases where the
 * answer is arithmetic: identical members give one member's support, members
 * that share nothing give the sum, and a pair sharing exactly one sequence
 * gives the sum less one. The last is the one that matters -- an off-by-one in
 * how a shared sequence is attributed is invisible in a large total.
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

/* Two corpora over disjoint residue pools, so the models built from them
 * cannot share a single sequence. */
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

/* `TRAIN_A` plus one sequence from B's pool, and vice versa, so the two
 * supports meet in exactly that one sequence. */
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

/** `TRAIN_A` with `SHARED` appended, at abundance 1. */
static BeamModel *build_with_shared(const char **base, const double *w,
                                    uint32_t n, uint8_t order) {
    const char *seqs[N_A + N_B + 1];
    double wts[N_A + N_B + 1];
    for (uint32_t i = 0; i < n; ++i) { seqs[i] = base[i]; wts[i] = w[i]; }
    seqs[n] = SHARED;
    wts[n] = 1.0;
    return build(seqs, wts, n + 1u, order);
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

static void test_disjoint_members_have_no_inner_product(void) {
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *pair[2] = {a, b};
    double lip = 0.0;
    const BeamError e = beam_mixture_log_inner_product_by_length(pair, 2, NULL,
                                                                 &lip);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(lip == -INFINITY, "disjoint supports must give -inf, not 0");
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

static void test_disjoint_members_have_the_summed_support(void) {
    BeamModel *a = build(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *pair[2] = {a, b};
    double u = 0.0;
    const BeamError e = beam_mixture_support_union_by_length(pair, 2, 0, NULL,
                                                             &u, NULL);
    const double want = beam_model_path_count_total(a)
                      + beam_model_path_count_total(b);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(u == want, "disjoint supports must add");
    PASS();
}

static void test_one_shared_sequence_is_counted_exactly_once(void) {
    /*
     * The case an off-by-one hides in. Each model is given the other's single
     * sequence, so the supports meet in exactly one place and the union must be
     * the sum less one -- not the sum, which is what dropping the shared mask
     * gives, and not one less again, which is what double-counting gives.
     */
    BeamModel *a = build_with_shared(TRAIN_A, WEIGHT_A, N_A, 2);
    BeamModel *b = build(TRAIN_B, WEIGHT_B, N_B, 2);
    ASSERT_MSG(a && b, "a model did not build");
    ASSERT_MSG(beam_model_log_pseq(a, SHARED) != -INFINITY,
               "the shared sequence is not in A");
    ASSERT_MSG(beam_model_log_pseq(b, SHARED) != -INFINITY,
               "the shared sequence is not in B");
    const BeamModel *pair[2] = {a, b};
    double u = 0.0;
    const BeamError e = beam_mixture_support_union_by_length(pair, 2, 0, NULL,
                                                             &u, NULL);
    const double sum = beam_model_path_count_total(a)
                     + beam_model_path_count_total(b);
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "call failed");
    ASSERT_MSG(u == sum - 1.0, "a shared sequence was not counted exactly once");
    PASS();
}

static void test_the_union_never_exceeds_the_pooled_model(void) {
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

    double one = 0.0;
    const BeamError per_length =
        beam_mixture_support_union(many, over, 5u, 0, &one, NULL);
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
                                                    5u, 0, &one, NULL);
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
    RUN_TEST(test_disjoint_members_have_no_inner_product);
    RUN_TEST(test_the_per_length_terms_sum_to_the_total);

    printf("\n[the union sweep against path counting]\n");
    RUN_TEST(test_identical_members_have_one_members_support);
    RUN_TEST(test_disjoint_members_have_the_summed_support);
    RUN_TEST(test_one_shared_sequence_is_counted_exactly_once);
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
