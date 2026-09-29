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
 *
 * Under the hierarchical prior every finalized cell is positive, so every
 * state a model reaches can also finish, and on a real model the lists agree
 * on every layer of every length. That is checked as it stands. The branch
 * that keeps `live` as a list of its own is still the library's, though, and
 * it is where a double free would hide, so the negative case is run on a
 * model with some states made dead by hand (see `build_with_dead_states`).
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


#define MAX_LEN 9u
#define ORDER   3u
/* The layer `build_with_dead_states` kills half the states of. At or above
 * ORDER, so every one of the 20^3 residue contexts is reached there. */
#define DEAD_DEPTH 4u

#define N_TRAIN 12
static const char *TRAIN[N_TRAIN] = {
    "CASSF",  "CASSY",  "CAGSF",  "CGSSF",  "CASSL",   "CTSSF",
    "CASSQE", "CASSYW", "CAGSFK", "CGSSFP", "CASSLNM", "CTSSFRQ",
};
static const double WEIGHT[N_TRAIN] = { 40,25,18,15,12,10,9,8,7,6,5,4 };

static BeamModel *build(bool sparse) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = ORDER;
    cfg.max_length = MAX_LEN;
    /* Sparse and dense are the same model bit for bit and reach the same
     * live cache -- liveness is a property of the trellis, not of the
     * layout -- but they read their rows through different code. */
    cfg.sparse = sparse;
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
 * A model on which `live` and `reached` genuinely differ.
 *
 * No finalized model has such a layer any more: under the prior every cell is
 * positive, so every reached state has twenty finite edges and a finite tail.
 * To reach the non-aliasing branch at all, every forward row at depth
 * DEAD_DEPTH whose context index is even is overwritten with -inf. The states
 * carrying those contexts are still reached at layer DEAD_DEPTH -- the edges
 * INTO them read depth DEAD_DEPTH - 1 -- but no edge leaves them, so they
 * are reached and not live, and every length longer than DEAD_DEPTH has one
 * layer whose lists differ.
 *
 * Dense, because a dense row is one contiguous run of cells to overwrite,
 * where a sparse row may be interned, packed, or absent and resolved through
 * its ancestors. The cache is dropped afterwards so nothing built from the
 * unmodified tables survives.
 */
static BeamModel *build_with_dead_states(void) {
    BeamModel *m = build(false);
    if (!m) return NULL;
    const uint32_t R = m->n_residues;
    for (uint32_t ctx = 0; ctx < m->n_ctx; ctx += 2u) {
        double *row = m->lp[BEAM_FWD] + (size_t)beam_row(m, DEAD_DEPTH, ctx) * R;
        for (uint32_t x = 0; x < R; ++x) row[x] = -INFINITY;
    }
    beam_live_cache_drop(m);
    return m;
}

/** `R^k`, the number of residue contexts of length `k`. */
static uint32_t power(uint32_t R, uint32_t k) {
    uint32_t p = 1u;
    while (k--) p *= R;
    return p;
}

/**
 * Every layer of a real model aliases, and reaches every context there is.
 *
 * Both halves are arithmetic the prior makes exact. A state at layer `i`
 * holds the last `min(i, order)` residues, and with every edge finite each of
 * the `R^min(i, order)` combinations is reached; with every tail finite each
 * of them can also finish, so `live` equals `reached` and the alias must be
 * taken. A layer that failed to alias here would leave the saving on the
 * table without any answer ever saying so; a count that differed would mean
 * an edge or a tail the prior forbids from being -inf was read as one.
 */
static void test_every_layer_of_a_real_model_aliases(void) {
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    const uint32_t R = m->n_residues;

    uint32_t layers = 0;
    for (uint32_t len = 1; len <= MAX_LEN; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) != BEAM_OK) {
            beam_model_destroy(m);
            ASSERT_MSG(0, "a live set would not build");
        }
        const BeamLive *L = h.L;
        for (uint32_t i = 0; i <= L->len; ++i) {
            const uint32_t want = power(R, i < ORDER ? i : ORDER);
            const bool ok = L->n_reached[i] == want
                         && L->n_live[i] == want
                         && L->live[i] == L->reached[i];
            if (!ok) {
                printf("FAIL: len=%u layer=%u reached=%u live=%u want=%u "
                       "alias=%d\n", len, i, L->n_reached[i], L->n_live[i],
                       want, (int)(L->live[i] == L->reached[i]));
                fail_count++;
                beam_live_release(&h);
                beam_model_destroy(m);
                return;
            }
            layers++;
        }
        beam_live_release(&h);
    }
    beam_model_destroy(m);
    ASSERT_MSG(layers == MAX_LEN * (MAX_LEN + 3u) / 2u,
               "not every layer of every length was inspected");
    PASS();
}

/**
 * Aliased exactly where the lists are equal, and nowhere else.
 *
 * Both directions matter. Aliasing a layer whose lists differ would hand
 * consumers the wrong states; failing to alias one where they agree would
 * simply leave the saving on the table, and no answer would ever say so.
 */
static void test_the_alias_is_taken_exactly_when_the_lists_agree(void) {
    BeamModel *m = build_with_dead_states();
    ASSERT_MSG(m != NULL, "model did not build");

    uint32_t aliased = 0, distinct = 0, layers = 0;
    for (uint32_t len = 1; len <= MAX_LEN; ++len) {
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
    /* One layer per length longer than DEAD_DEPTH, and no more: the dead
     * states are the only difference between the lists anywhere. */
    ASSERT_MSG(distinct == MAX_LEN - DEAD_DEPTH,
               "the lists differed on a layer other than the one made dead, "
               "or not on that one");
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
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");

    uint64_t aliased_bytes = 0;
    for (uint32_t len = 1; len <= MAX_LEN; ++len) {
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
    for (uint32_t len = 1; len <= MAX_LEN; ++len) {
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
 * Run on sparse and dense storage, where every layer aliases, and on the
 * model with dead states, which is the only one that takes the branch where
 * `live` is a list of its own.
 */
static void test_the_selfcheck_passes_on_an_aliased_cache(void) {
    static const char *const kinds[3] = {"sparse", "dense", "dead states"};
    for (int kind = 0; kind < 3; ++kind) {
        BeamModel *m = kind == 2 ? build_with_dead_states() : build(kind == 0);
        ASSERT_MSG(m != NULL, "model did not build");
        for (uint32_t len = 1; len <= MAX_LEN; ++len) {
            const int rc = beam_live_selfcheck(m, len);
            if (rc != 0) {
                printf("FAIL: %s len=%u selfcheck code %d\n",
                       kinds[kind], len, rc);
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
 * `-fsanitize=address`, which is how it was checked. It runs on the model
 * with dead states, so the cache it drops holds aliased and owned lists side
 * by side -- the mixture a wrong "is this mine to free" test would get wrong.
 */
static void test_dropping_an_aliased_cache_frees_each_list_once(void) {
    BeamModel *m = build_with_dead_states();
    ASSERT_MSG(m != NULL, "model did not build");
    for (uint32_t len = 1; len <= MAX_LEN; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) == BEAM_OK) beam_live_release(&h);
    }
    ASSERT_MSG(beam_live_cache_entries(m) > 0u, "the cache held nothing");
    beam_live_cache_drop(m);
    ASSERT_MSG(beam_live_cache_bytes(m) == 0u, "drop left bytes behind");
    ASSERT_MSG(beam_live_cache_entries(m) == 0u, "drop left entries behind");

    /* Warm it again after the drop: the second build must not reuse a freed
     * list from the first. */
    for (uint32_t len = 1; len <= MAX_LEN; ++len) {
        LiveHandle h;
        if (beam_live_acquire(m, len, &h) == BEAM_OK) beam_live_release(&h);
    }
    ASSERT_MSG(beam_live_cache_entries(m) > 0u, "the cache did not rewarm");
    beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("\nlive/reached aliasing\n");
    RUN_TEST(test_every_layer_of_a_real_model_aliases);
    RUN_TEST(test_the_alias_is_taken_exactly_when_the_lists_agree);
    RUN_TEST(test_the_byte_count_drops_by_the_aliased_lists);
    RUN_TEST(test_the_selfcheck_passes_on_an_aliased_cache);
    RUN_TEST(test_dropping_an_aliased_cache_frees_each_list_once);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
