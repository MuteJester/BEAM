/**
 * @file test_expected_counts_budget.c
 * @brief `beam_model_expected_counts` holds its scratch inside the budget.
 *
 * This analytic used to hold, unbudgeted and unreported, more memory than the
 * model it was asked about. Six hash maps accumulate one row per `(depth,
 * context)` the trellis reaches, and on the published foundation model they
 * measured 896 MB against an 806 MB model, growing to a 1,566 MB peak while
 * the answer was copied out of them. Nothing checked any of it.
 *
 * The reason it was never bounded up front is worth stating, because it is
 * what shapes the fix. Every other program in `positional.c` multiplies out
 * its shape before it starts. The maps cannot: they are keyed on the DP's
 * `(depth, ctx)` space, whose only shape-derived bound is the dense table --
 * 148 million cells at order 4, which is the address space the map exists in
 * order NOT to allocate. A guard built on that bound refuses every call. So
 * the check happens at each doubling instead, where the figure being asked
 * for is exactly known.
 *
 * Three properties are pinned here.
 *
 * The first is that the accumulators refuse BEFORE allocating, and say so as
 * a budget refusal rather than as a bare allocation failure -- `rowmap_row`
 * returns a `double *` and has only NULL to fail with, so which failure it was
 * has to be carried out of band or the message naming the figure is lost.
 *
 * The second is that the sweep and the accumulators SHARE ONE ACCOUNT. They
 * are live at the same moment, and each used to check the whole budget
 * independently, so a call could hold the backward field and six maps that
 * separately passed at 90% of the figure the user set. Two checks against the
 * same limit are not a limit. The test for this is the sharp one: it picks a
 * budget the backward field fits inside on its own, and requires a refusal.
 *
 * The third is that a block can be released on its own, which is what lets a
 * caller that copies the answer out avoid holding two copies of all six.
 *
 * Two models are used, and the second is not decoration. The small one never
 * fills a map past the capacity `rowmap_init` gives it, so on that model the
 * doubling paths -- which are the ones the foundation model spends 896 MB in
 * -- never run at all. The wide one drives sixteen thousand rows through them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

/** The wide model, built once for the binary. See `build_wide`. */
static BeamModel *g_wide = NULL;

#include "test_utils.h"

/*
 * Ten sequences of length 5 and ten of length 8, so the model has more than
 * one live length and the pooled sweep runs more than once. Order 2 puts
 * 9,261 states in the trellis, which makes the backward field the largest
 * single allocation and the interesting one to share a budget with.
 */
#define N_TRAIN 20
static const char *TRAIN[N_TRAIN] = {
    "CASSF", "CASSY", "CAGSF", "CGSSF", "CASSL",
    "CTSSF", "CASSQ", "CASRW", "CAWSF", "CAYSL",
    "CASSFGHK", "CASSYWMQ", "CAGSFKPD", "CGSSFPDE", "CASSLNRT",
    "CTSSFRWY", "CASSQEVI", "CASRWMTL", "CAWSFGNE", "CAYSLKQD",
};
static const double WEIGHT[N_TRAIN] = {
    40, 25, 18, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4,
};

static BeamModel *build(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 2u;
    cfg.max_length = 12u;
    cfg.sparse = true;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/**
 * The backward field for the LONGEST live length, from `fb_alloc`'s own
 * expression.
 *
 * Duplicated here deliberately and only here: the test's whole point is to
 * pick a budget on one particular side of this figure, so it has to know it.
 * The longest length is the largest, so a budget at or above this admits the
 * sweep at every length the pooled call will run.
 */
static uint64_t backward_field_bytes(const BeamModel *m) {
    uint32_t longest = 0;
    for (uint32_t L = 1; L <= m->max_len; ++L)
        if (beam_length_live(m, L)) longest = L;
    return ((uint64_t)longest + 1u) * m->n_states * sizeof(double)
         + (uint64_t)m->n_states * (m->n_residues + 3u) * sizeof(double);
}

/** The six maps at the capacity `rowmap_init` gives them, before any growth. */
static uint64_t accumulators_at_rest(const BeamModel *m) {
    const uint64_t dir  = (uint64_t)1024u * 2u * sizeof(uint32_t);
    const uint64_t rows = (uint64_t)512u
                        * ((uint64_t)m->n_residues * sizeof(double)
                           + sizeof(uint32_t));
    return 6u * (dir + rows);
}

/** Every block sums to the same expected length, to floating-point slack. */
static bool blocks_sum_to_expectation(const BeamExpectedCounts *ec) {
    for (uint32_t d = 0; d < 2u; ++d)
        for (uint32_t e = 0; e < 3u; ++e) {
            const BeamCountBlock *b = &ec->block[d][e];
            double s = 0.0;
            for (uint64_t i = 0;
                 i < (uint64_t)b->n_rows * ec->n_residues; ++i)
                s += b->counts[i];
            if (fabs(s - ec->expected_length) > 1e-6 * ec->expected_length)
                return false;
        }
    return true;
}

/**
 * Under the budget the library ships with, the answer arrives and is whole.
 *
 * The block-sum identity is the cheapest check that emitting and releasing one
 * map at a time -- rather than all six then all six -- still copies every row
 * out of every map.
 */
static void test_the_shipped_budget_answers_and_every_block_is_whole(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    beam_model_set_memory_budget(0);            /* 0 restores the default */

    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    ASSERT_MSG(err == BEAM_OK, beam_error_message());
    ASSERT_MSG(ec.expected_length > 0.0, "expected length is not positive");
    ASSERT_MSG(blocks_sum_to_expectation(&ec),
               "a block does not sum to the expected length");
    beam_model_expected_counts_free(&ec);
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

/**
 * Below what the maps need at rest, the refusal is a budget refusal and names
 * the accumulators.
 */
static void test_the_accumulators_refuse_and_say_it_was_the_budget(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    const uint64_t rest = accumulators_at_rest(m);

    beam_model_set_memory_budget(rest - 1u);
    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    beam_model_set_memory_budget(0);

    ASSERT_MSG(err == BEAM_ERR_BUDGET, "a refusal below the maps' own size "
                                       "was not reported as a budget refusal");
    ASSERT_MSG(strstr(beam_error_message(), "expected-count working set"),
               "the refusal does not name what refused");
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

/**
 * THE SHARP ONE. The budget is set above what the backward field needs on its
 * own, and below what the field plus the maps need together. The sweep's own
 * check passes at every length -- it is asserted here that it does -- so a
 * refusal can only come from the two sharing an account.
 *
 * Before they shared one, this call succeeded while holding the field and six
 * maps that had each passed a check against the whole figure.
 */
static void test_the_sweep_and_the_accumulators_share_one_budget(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    const uint64_t field = backward_field_bytes(m);
    const uint64_t rest  = accumulators_at_rest(m);
    ASSERT_MSG(rest > 65536u, "the margin below is not inside the maps' size");
    const uint64_t budget = field + 65536u;     /* fits the field, not both */

    ASSERT_MSG(field <= budget,
               "the budget does not admit the backward field on its own, so "
               "this test would pass without the shared account");

    beam_model_set_memory_budget(budget);
    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    beam_model_set_memory_budget(0);

    ASSERT_MSG(err == BEAM_ERR_BUDGET,
               "the sweep and the accumulators each fit the budget alone and "
               "the call was allowed to hold both");
    ASSERT_MSG(strstr(beam_error_message(), "already hold"),
               "the refusal does not say the accumulators were holding part "
               "of the budget");
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

/**
 * A block can be released while the others are still in use.
 *
 * This is what lets the Python binding copy the six out one at a time instead
 * of holding a second copy of the whole answer. The whole-struct free has to
 * stay correct over the block that is already gone.
 */
static void test_one_block_can_be_released_before_the_rest(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    beam_model_set_memory_budget(0);

    BeamExpectedCounts ec;
    ASSERT_MSG(beam_model_expected_counts(m, 0u, &ec) == BEAM_OK,
               beam_error_message());
    ASSERT_MSG(ec.block[0][0].n_rows > 0 && ec.block[0][1].n_rows > 0,
               "the model produced an empty block, so this proves nothing");

    const uint32_t kept_rows = ec.block[0][1].n_rows;
    double kept_sum = 0.0;
    for (uint64_t i = 0; i < (uint64_t)kept_rows * ec.n_residues; ++i)
        kept_sum += ec.block[0][1].counts[i];

    beam_count_block_free(&ec.block[0][0]);
    ASSERT_MSG(ec.block[0][0].n_rows == 0 && !ec.block[0][0].rows
               && !ec.block[0][0].counts,
               "the released block was not zeroed");

    double still = 0.0;
    for (uint64_t i = 0; i < (uint64_t)kept_rows * ec.n_residues; ++i)
        still += ec.block[0][1].counts[i];
    ASSERT_MSG(ec.block[0][1].n_rows == kept_rows && still == kept_sum,
               "releasing one block disturbed another");

    beam_model_expected_counts_free(&ec);       /* must not double-free */
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

/*
 * A model whose accumulators GROW.
 *
 * The corpus above produces 374 rows across all six maps, which fits the
 * capacity they start at, so it exercises `rowmap_init` and nothing else. The
 * doubling paths need more rows than that, and rows come from distinct
 * `(length, depth, context)` triples: 240 sequences spread over twenty-one
 * lengths reach tens of thousands of them. Pseudo-random residues from a fixed
 * seed, so the figure is the same on every machine and every run.
 *
 * NUCLEOTIDES, and that is a runtime decision rather than a coverage one. What
 * these tests need is many distinct rows, which comes from the spread of
 * lengths; what costs the time is `n_states`, and radix 6 puts 216 of them in
 * the trellis where radix 21 puts 9,261. The amino-acid version of this model
 * ran the pair of test binaries in ninety-five seconds for the same assertions.
 */
#define N_WIDE 240
#define WIDE_MAX_LEN 24u
static char wide_buf[N_WIDE][WIDE_MAX_LEN + 1u];
static const char *wide_seq[N_WIDE];
static double wide_wt[N_WIDE];

static BeamModel *build_wide_once(void) {
    static const char *NT = "ACGT";
    uint32_t s = 12345u;
    for (uint32_t i = 0; i < N_WIDE; ++i) {
        const uint32_t len = 4u + (i % 21u);
        for (uint32_t j = 0; j < len; ++j) {
            s = s * 1103515245u + 12345u;
            wide_buf[i][j] = NT[(s >> 16) % 4u];
        }
        wide_buf[i][len] = '\0';
        wide_seq[i] = wide_buf[i];
        wide_wt[i] = 1.0 + (double)(i % 7u);
    }
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.alphabet = BEAM_ALPHABET_NT;
    cfg.order = 2u;
    cfg.max_length = WIDE_MAX_LEN;
    cfg.strict = false;
    /* No `sparse`: the nt alphabet refuses it, and these tests are about the
     * accumulators, which do not depend on how the model stores its tables. */
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, wide_seq, wide_wt, N_WIDE) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}


/*
 * Built once for the binary. `beam_model_finalize` on this shape costs more
 * than every expected-count call in the file put together, and each test wants
 * the same model, so rebuilding it per test was most of the runtime.
 */
static BeamModel *build_wide(void) {
    if (!g_wide) g_wide = build_wide_once();
    return g_wide;
}

/**
 * The DOUBLING paths refuse too, and they are the only thing that can.
 *
 * The budget here admits the backward field and the six maps at their starting
 * capacity together -- both asserted -- so `fb_alloc` passes and every
 * `rowmap_init` passes. What does not fit is the growth: this model puts about
 * sixteen thousand rows through `rowmap_row` and `rowmap_rehash`, and those
 * are where the check has to be, because there is no expression that could
 * have predicted sixteen thousand from the model's shape.
 */
static void test_the_doubling_paths_refuse_too(void) {
    BeamModel *m = build_wide();
    ASSERT_MSG(m, "wide model did not build");
    const uint64_t field = backward_field_bytes(m);
    const uint64_t rest  = accumulators_at_rest(m);
    const uint64_t budget = field + rest + 262144u;

    ASSERT_MSG(field <= budget && field + rest <= budget,
               "the budget does not admit the field and the maps at rest, so "
               "an earlier check could have produced this refusal");

    beam_model_set_memory_budget(budget);
    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    beam_model_set_memory_budget(0);

    ASSERT_MSG(err == BEAM_ERR_BUDGET,
               "the maps grew past the budget without anything refusing");
    ASSERT_MSG(strstr(beam_error_message(), "expected-count working set"),
               "the refusal does not name what refused");

    /* And with room, the same model answers -- so the budget above refused
     * growth rather than something the model could never have done. */
    beam_model_set_memory_budget(0);
    ASSERT_MSG(beam_model_expected_counts(m, 0u, &ec) == BEAM_OK,
               beam_error_message());
    uint64_t rows = 0;
    for (uint32_t d = 0; d < 2u; ++d)
        for (uint32_t e = 0; e < 3u; ++e) rows += ec.block[d][e].n_rows;
    ASSERT_MSG(rows > 6u * 512u,
               "the wide model did not fill a single map past its starting "
               "capacity, so no doubling path ran");
    ASSERT_MSG(blocks_sum_to_expectation(&ec),
               "a block of the wide model does not sum to the expected length");
    beam_model_expected_counts_free(&ec);
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("expected-count budget\n");

    RUN_TEST(test_the_shipped_budget_answers_and_every_block_is_whole);
    RUN_TEST(test_the_accumulators_refuse_and_say_it_was_the_budget);
    RUN_TEST(test_the_sweep_and_the_accumulators_share_one_budget);
    RUN_TEST(test_the_doubling_paths_refuse_too);
    RUN_TEST(test_one_block_can_be_released_before_the_rest);
    if (g_wide) beam_model_destroy(g_wide);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
