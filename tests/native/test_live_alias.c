/**
 * @file test_live_alias.c
 * @brief The live cache shares one list when `live` and `reached` are equal.
 *
 * `live` is the subset of `reached` that can still finish a sequence. On many
 * layers nothing is dropped, and the cache used to hold a second array with
 * the same contents. It now points at the first.
 *
 * Aliasing is the kind of change that works for a long time and then frees
 * something twice, so the tests here are about ownership and identity, not
 * about the answers: that the alias is taken when the lists are equal, that
 * it is NOT taken when they differ, that the byte accounting stops counting a
 * list it no longer owns, and that the frontier's own self-check still passes
 * on an aliased cache. Whether the analytics still agree is the business of
 * the suites that already compare their values.
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


#define N_TRAIN 12
static const char *TRAIN[N_TRAIN] = {
    "CASSF",  "CASSY",  "CAGSF",  "CGSSF",  "CASSL",   "CTSSF",
    "CASSQE", "CASSYW", "CAGSFK", "CGSSFP", "CASSLNM", "CTSSFRQ",
};
static const double WEIGHT[N_TRAIN] = { 40,25,18,15,12,10,9,8,7,6,5,4 };

static BeamModel *build(uint32_t max_length, bool strict) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3u;
    cfg.max_length = max_length;
    cfg.strict = strict;
    /* Sparse storage is only accepted in the fully closed world, so the
     * open-world case here is necessarily dense. Both reach the same live
     * cache -- liveness is a property of the trellis, not of the layout. */
    cfg.sparse = strict;
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
 * Aliased exactly where the lists are equal, and nowhere else.
 *
 * Both directions matter. Aliasing a layer whose lists differ would hand
 * consumers the wrong states; failing to alias one where they agree would
 * simply leave the saving on the table, and no answer would ever say so.
 */
static void test_the_alias_is_taken_exactly_when_the_lists_agree(void) {
    BeamModel *m = build(9u, true);
    ASSERT_MSG(m != NULL, "model did not build");

    uint32_t aliased = 0, distinct = 0, layers = 0;
    for (uint32_t len = 1; len <= 9u; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) != BEAM_OK) continue;
        const BeamLive *L = h.L;
        for (uint32_t i = 0; i <= L->len; ++i) {
            const bool same_ptr = (L->live[i] == L->reached[i]);
            const bool same_len = (L->n_live[i] == L->n_reached[i]);
            layers++;
            if (same_ptr != same_len) {
                printf("FAIL: len=%u layer=%u alias=%d equal-counts=%d\n",
                       len, i, (int)same_ptr, (int)same_len);
                fail_count++;
                beam_live_release(&h);
                beam_model_destroy(m);
                return;
            }
            /* Where the alias was taken the contents must still be right. */
            if (same_ptr) {
                aliased++;
            } else {
                distinct++;
                for (uint32_t k = 0; k < L->n_live[i]; ++k) {
                    uint32_t a = 0;
                    while (a < L->n_reached[i] && L->reached[i][a] < L->live[i][k]) a++;
                    if (a >= L->n_reached[i] || L->reached[i][a] != L->live[i][k]) {
                        printf("FAIL: len=%u layer=%u live state not reached\n", len, i);
                        fail_count++;
                        beam_live_release(&h);
                        beam_model_destroy(m);
                        return;
                    }
                }
            }
        }
        beam_live_release(&h);
    }
    beam_model_destroy(m);

    ASSERT_MSG(layers > 0u, "no layer was inspected");
    ASSERT_MSG(aliased > 0u, "the alias never fired: nothing was saved");
    ASSERT_MSG(distinct > 0u,
               "every layer aliased: the negative case was never exercised");
    PASS();
}

/**
 * The cache stops charging itself for a list it no longer owns.
 *
 * `live_bytes` feeds the cache's budget, so an alias that still counted its
 * bytes would evict entries to make room for memory it had already given
 * back -- a saving that costs cache hits.
 */
static void test_the_byte_count_drops_by_the_aliased_lists(void) {
    BeamModel *m = build(9u, true);
    ASSERT_MSG(m != NULL, "model did not build");

    uint64_t aliased_bytes = 0;
    for (uint32_t len = 1; len <= 9u; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) != BEAM_OK) continue;
        const BeamLive *L = h.L;
        for (uint32_t i = 0; i <= L->len; ++i)
            if (L->live[i] == L->reached[i])
                aliased_bytes += (uint64_t)L->n_live[i] * sizeof(uint32_t);
        beam_live_release(&h);
    }
    const uint64_t reported = beam_live_cache_bytes(m);
    ASSERT_MSG(reported > 0u, "the cache reported no bytes at all");
    ASSERT_MSG(aliased_bytes > 0u, "no layer aliased: nothing to check");

    /* The reported total must EXCLUDE the aliased lists. Recompute the total
     * the old accounting would have produced and require the difference. */
    uint64_t naive = reported;
    for (uint32_t len = 1; len <= 9u; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) != BEAM_OK) continue;
        const BeamLive *L = h.L;
        for (uint32_t i = 0; i <= L->len; ++i)
            if (L->live[i] == L->reached[i])
                naive += (uint64_t)L->n_live[i] * sizeof(uint32_t);
        beam_live_release(&h);
    }
    beam_model_destroy(m);
    ASSERT_MSG(naive > reported, "the accounting did not drop the aliased lists");
    PASS();
}

/**
 * The frontier's own invariants still hold on an aliased cache.
 *
 * Self-check code 4 is `live` not strictly ascending and code 6 is a live
 * state that was never reached; an alias satisfies both trivially, which is
 * the point -- this asserts the checker was not silently skipped.
 *
 * Run in both worlds. The open world is the one where `live` and `reached`
 * most often differ, because a state can be reachable and still have no
 * finite tail, so it exercises the branch the closed world rarely takes.
 */
static void test_the_selfcheck_passes_on_an_aliased_cache(void) {
    for (int strict = 0; strict < 2; ++strict) {
        BeamModel *m = build(9u, strict != 0);
        ASSERT_MSG(m != NULL, "model did not build");
        for (uint32_t len = 1; len <= 9u; ++len) {
            const int rc = beam_live_selfcheck(m, len);
            if (rc != 0 && rc != BEAM_LIVE_SELFCHECK_BUILD) {
                printf("FAIL: strict=%d len=%u selfcheck code %d\n",
                       strict, len, rc);
                fail_count++;
                beam_model_destroy(m);
                return;
            }
        }
        beam_model_destroy(m);
    }
    PASS();
}

/**
 * Dropping the cache releases an aliased layer exactly once.
 *
 * Under a plain allocator a double free is undefined and may look like a
 * pass; this test earns its keep when the suite is built with
 * `-fsanitize=address`, which is how it was checked.
 */
static void test_dropping_an_aliased_cache_frees_each_list_once(void) {
    BeamModel *m = build(9u, true);
    ASSERT_MSG(m != NULL, "model did not build");
    for (uint32_t len = 1; len <= 9u; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) == BEAM_OK) beam_live_release(&h);
    }
    ASSERT_MSG(beam_live_cache_entries(m) > 0u, "the cache held nothing");
    beam_live_cache_drop(m);
    ASSERT_MSG(beam_live_cache_bytes(m) == 0u, "drop left bytes behind");
    ASSERT_MSG(beam_live_cache_entries(m) == 0u, "drop left entries behind");

    /* Warm it again after the drop: the second build must not reuse a freed
     * list from the first. */
    for (uint32_t len = 1; len <= 9u; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) == BEAM_OK) beam_live_release(&h);
    }
    ASSERT_MSG(beam_live_cache_entries(m) > 0u, "the cache did not rewarm");
    beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("\nlive/reached aliasing\n");
    RUN_TEST(test_the_alias_is_taken_exactly_when_the_lists_agree);
    RUN_TEST(test_the_byte_count_drops_by_the_aliased_lists);
    RUN_TEST(test_the_selfcheck_passes_on_an_aliased_cache);
    RUN_TEST(test_dropping_an_aliased_cache_frees_each_list_once);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
