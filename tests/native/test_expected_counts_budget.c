/**
 * @file test_expected_counts_budget.c
 * @brief `beam_model_expected_counts` holds its scratch inside the budget.
 *
 * This analytic used to hold, unbudgeted and unreported, more memory than the
 * model it was asked about. Six hash maps accumulated one row per `(depth,
 * context)` the trellis reached, and on the published foundation model they
 * measured 896 MB against an 806 MB model, growing to a 1,566 MB peak while
 * the answer was copied out of them. Nothing checked any of it. There are two
 * maps now, one per direction, and the same reasoning holds for them.
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
 * That holds for the maps' first allocation and for every doubling after it.
 *
 * The second is that the sweep and the accumulators SHARE ONE ACCOUNT. They
 * are live at the same moment, and each used to check the whole budget
 * independently, so a call could hold the backward field and the maps that
 * separately passed at 90% of the figure the user set. Two checks against the
 * same limit are not a limit. The test for this is the sharp one: it picks a
 * budget the backward field fits inside on its own, and requires a refusal.
 *
 * The third is that a block can be released on its own, which is what lets a
 * caller that copies the answer out avoid holding two copies of both.
 *
 * Every budget below is derived from the sizes the library charges -- the
 * field from `fb_alloc`'s expression, the maps from `rowmap_init`'s -- and
 * not from a measured margin, so each sits on the side of a figure the test
 * can name.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

/*
 * Ten sequences of length 5 and ten of length 8, amino acids at order 2: 441
 * states in the trellis.
 *
 * ONE MODEL, where there used to be two. Under the hierarchical prior every
 * string up to `max_length` has positive probability, so the sweep reaches
 * every `(depth, context)` the trellis can form, whatever the corpus spelled.
 * How many rows a map receives is therefore a property of the model's SHAPE
 * (see `reachable_rows`): 4,021 per map here, against the 512 `rowmap_init`
 * starts with, so the doubling paths -- where the foundation model spent its
 * 896 MB -- run on this model three times per map. A second, wide model
 * used to exist only to make them run; this one now does that on its own.
 *
 * Sparse storage, the default, and the form a per-repertoire model is built
 * in. The accumulators do not depend on how the tables are stored.
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
 * The backward field for one length, from `fb_alloc`'s own expression.
 *
 * Duplicated here deliberately and only here: the tests' whole point is to
 * pick a budget on one particular side of this figure, so they have to know
 * it.
 */
static uint64_t field_bytes(const BeamModel *m, uint32_t len) {
    return ((uint64_t)len + 1u) * m->n_states * sizeof(double)
         + (uint64_t)m->n_states * (m->n_residues + 3u) * sizeof(double);
}

/** The shortest and the longest live length: the first and the largest
 *  field the pooled call allocates. */
static uint32_t shortest_live(const BeamModel *m) {
    for (uint32_t L = 1; L <= m->max_len; ++L)
        if (beam_length_live(m, L)) return L;
    return 0;
}
static uint32_t longest_live(const BeamModel *m) {
    uint32_t longest = 0;
    for (uint32_t L = 1; L <= m->max_len; ++L)
        if (beam_length_live(m, L)) longest = L;
    return longest;
}

/** One accumulator row's bytes, `vals` and `ids` together, as charged. */
static uint64_t row_bytes(const BeamModel *m) {
    return (uint64_t)m->n_residues * sizeof(double) + sizeof(uint32_t);
}

/** The two maps at the capacity `rowmap_init` gives them, before any growth:
 *  a 1,024-slot key directory of two uint32 arrays and room for 512 rows. */
static uint64_t accumulators_at_rest(const BeamModel *m) {
    const uint64_t dir  = (uint64_t)1024u * 2u * sizeof(uint32_t);
    const uint64_t rows = (uint64_t)512u * row_bytes(m);
    return 2u * (dir + rows);
}

/**
 * Rows each map ends with when every path has mass: one per `(depth,
 * context)` the trellis can form. A context holds the `order` residues nearest
 * the scored one, padded with the sentinel past either end, so at depth `d`
 * there are `R^min(d, order)` of them. The same count in both directions.
 */
static uint64_t reachable_rows(const BeamModel *m) {
    uint64_t rows = 0;
    for (uint32_t d = 0; d < m->max_len; ++d) {
        uint64_t n = 1;
        for (uint32_t k = 0; k < d && k < m->order; ++k) n *= m->n_residues;
        rows += n;
    }
    return rows;
}

/** Every block sums to the same expected length, to floating-point slack. */
static bool blocks_sum_to_expectation(const BeamExpectedCounts *ec) {
    for (uint32_t d = 0; d < 2u; ++d) {
        const BeamCountBlock *b = &ec->block[d];
        double s = 0.0;
        for (uint64_t i = 0; i < (uint64_t)b->n_rows * ec->n_residues; ++i)
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
 * map at a time -- rather than both then both -- still copies every row out
 * of every map.
 */
static void test_the_shipped_budget_answers_and_every_block_is_whole(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    beam_model_set_memory_budget(0);            /* 0 restores the default */

    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    if (err != BEAM_OK) { beam_model_destroy(m);
                          ASSERT_MSG(false, beam_error_message()); }
    const bool positive = ec.expected_length > 0.0;
    const bool whole = blocks_sum_to_expectation(&ec);
    beam_model_expected_counts_free(&ec);
    beam_model_destroy(m);
    ASSERT_MSG(positive, "expected length is not positive");
    ASSERT_MSG(whole, "a block does not sum to the expected length");
    PASS();
}

/**
 * Below what the maps need at rest, the refusal is a budget refusal and names
 * the working set.
 */
static void test_the_accumulators_refuse_and_say_it_was_the_budget(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    const uint64_t rest = accumulators_at_rest(m);

    beam_model_set_memory_budget(rest - 1u);
    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    char why[512];
    snprintf(why, sizeof why, "%s", beam_error_message());
    beam_model_set_memory_budget(0);
    if (err == BEAM_OK) beam_model_expected_counts_free(&ec);
    beam_model_destroy(m);

    ASSERT_MSG(err == BEAM_ERR_BUDGET, "a refusal below the maps' own size "
                                       "was not reported as a budget refusal");
    ASSERT_MSG(strstr(why, "expected-count working set"),
               "the refusal does not name what refused");
    PASS();
}

/**
 * THE SHARP ONE. The budget admits the backward field of every length on its
 * own, and admits the maps at rest on their own, but falls ONE BYTE short of
 * the maps at rest plus the first field the pooled call allocates. Both of
 * those "on its own" are asserted, so a refusal can only come from the two
 * sharing an account -- and the message has to say the accumulators were
 * already holding part of it.
 *
 * Before they shared one, this call succeeded while holding the field and the
 * maps that had each passed a check against the whole figure.
 */
static void test_the_sweep_and_the_accumulators_share_one_budget(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    const uint32_t first = shortest_live(m), last = longest_live(m);
    const uint64_t rest   = accumulators_at_rest(m);
    const uint64_t budget = rest + field_bytes(m, first) - 1u;

    ASSERT_MSG(first > 0u, "the model has no live length");
    ASSERT_MSG(field_bytes(m, last) <= budget,
               "the budget does not admit the backward field on its own, so "
               "this test would pass without the shared account");
    ASSERT_MSG(rest <= budget,
               "the budget does not admit the maps at rest, so the refusal "
               "would come from them alone");

    beam_model_set_memory_budget(budget);
    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    char why[512];
    snprintf(why, sizeof why, "%s", beam_error_message());
    beam_model_set_memory_budget(0);
    if (err == BEAM_OK) beam_model_expected_counts_free(&ec);
    beam_model_destroy(m);

    ASSERT_MSG(err == BEAM_ERR_BUDGET,
               "the sweep and the accumulators each fit the budget alone and "
               "the call was allowed to hold both");
    ASSERT_MSG(strstr(why, "already hold"),
               "the refusal does not say the accumulators were holding part "
               "of the budget");
    PASS();
}

/**
 * The DOUBLING paths refuse too, and they are the only thing that can.
 *
 * The budget is exactly the maps at rest plus the largest backward field, so
 * every `rowmap_init` and every `fb_alloc` fits and nothing else does. The
 * first growth is a map's rows doubling from 512 to 1,024, which charges the
 * new 1,024 rows on top of everything held; that exceeds the budget whenever
 * it is larger than the whole field, which is asserted. Those paths are
 * where the check has to be, because nothing in the model's shape bounds
 * them short of the dense table.
 */
static void test_the_doubling_paths_refuse_too(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    const uint64_t field  = field_bytes(m, longest_live(m));
    const uint64_t rest   = accumulators_at_rest(m);
    const uint64_t grow   = 1024u * row_bytes(m);
    const uint64_t budget = rest + field;

    ASSERT_MSG(grow > field,
               "the first doubling fits in the room a shorter field leaves, "
               "so the budget below might not refuse it");

    beam_model_set_memory_budget(budget);
    BeamExpectedCounts ec;
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    char why[512];
    snprintf(why, sizeof why, "%s", beam_error_message());
    beam_model_set_memory_budget(0);

    if (err != BEAM_ERR_BUDGET) {
        if (err == BEAM_OK) beam_model_expected_counts_free(&ec);
        beam_model_destroy(m);
        ASSERT_MSG(false, "the maps grew past the budget without anything "
                          "refusing");
    }
    if (!strstr(why, "expected-count working set")) {
        beam_model_destroy(m);
        ASSERT_MSG(false, "the refusal does not name what refused");
    }

    /* And with room, the same model answers -- so the budget above refused
     * growth rather than something the model could never have done -- and
     * both maps really did grow past where they started. */
    if (beam_model_expected_counts(m, 0u, &ec) != BEAM_OK) {
        beam_model_destroy(m);
        ASSERT_MSG(false, beam_error_message());
    }
    const uint64_t want = reachable_rows(m);
    const bool grew  = ec.block[0].n_rows > 512u && ec.block[1].n_rows > 512u;
    const bool every = ec.block[0].n_rows == want && ec.block[1].n_rows == want;
    const bool whole = blocks_sum_to_expectation(&ec);
    beam_model_expected_counts_free(&ec);
    beam_model_destroy(m);
    ASSERT_MSG(grew, "a map never grew past its starting capacity, so no "
                     "doubling path ran");
    ASSERT_MSG(every, "a map did not receive every (depth, context) the "
                      "trellis can form, though every path has mass");
    ASSERT_MSG(whole, "a block does not sum to the expected length");
    PASS();
}

/**
 * A block can be released while the other is still in use.
 *
 * This is what lets the Python binding copy the blocks out one at a time
 * instead of holding a second copy of the whole answer. The whole-struct free
 * has to stay correct over the block that is already gone.
 */
static void test_one_block_can_be_released_before_the_rest(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
    beam_model_set_memory_budget(0);

    BeamExpectedCounts ec;
    if (beam_model_expected_counts(m, 0u, &ec) != BEAM_OK) {
        beam_model_destroy(m);
        ASSERT_MSG(false, beam_error_message());
    }
    beam_model_destroy(m);
    ASSERT_MSG(ec.block[0].n_rows > 0 && ec.block[1].n_rows > 0,
               "the model produced an empty block, so this proves nothing");

    /* Release the FORWARD block and require the BACKWARD one untouched. */
    const uint32_t kept_rows = ec.block[1].n_rows;
    double kept_sum = 0.0;
    for (uint64_t i = 0; i < (uint64_t)kept_rows * ec.n_residues; ++i)
        kept_sum += ec.block[1].counts[i];

    beam_count_block_free(&ec.block[0]);
    ASSERT_MSG(ec.block[0].n_rows == 0 && !ec.block[0].rows
               && !ec.block[0].counts,
               "the released block was not zeroed");

    double still = 0.0;
    for (uint64_t i = 0; i < (uint64_t)kept_rows * ec.n_residues; ++i)
        still += ec.block[1].counts[i];
    ASSERT_MSG(ec.block[1].n_rows == kept_rows && still == kept_sum,
               "releasing one block disturbed another");

    beam_model_expected_counts_free(&ec);       /* must not double-free */
    PASS();
}

int main(void) {
    printf("expected-count budget\n");

    RUN_TEST(test_the_shipped_budget_answers_and_every_block_is_whole);
    RUN_TEST(test_the_accumulators_refuse_and_say_it_was_the_budget);
    RUN_TEST(test_the_sweep_and_the_accumulators_share_one_budget);
    RUN_TEST(test_the_doubling_paths_refuse_too);
    RUN_TEST(test_one_block_can_be_released_before_the_rest);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
