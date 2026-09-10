/**
 * @file test_expected_counts_accounting.c
 * @brief What `expected_counts` reports holding is what it actually held.
 *
 * `beam_model_expected_counts` charges every allocation against one account
 * and reports the peak as `working_set_bytes`, and every other test of that
 * machinery can only ask the account about itself. If an allocation is never
 * charged, the account is unanimous and wrong: the call refuses at the same
 * budget it always did, the reported figure is self-consistent, and the only
 * symptom is a machine that runs out of memory at a budget it was told it
 * would fit inside. That failure has happened here twice, and both times the
 * number that was wrong looked right.
 *
 * So this test does not ask the account. It wraps `malloc`, `calloc`,
 * `realloc` and `free`, measures the true live-byte peak of the call from the
 * allocator's side, and requires the reported figure to cover it.
 *
 * IT COVERS THE WHOLE PATH, which is the point: the backward field, the six
 * accumulator maps, both of their doubling paths, the directories the
 * doublings replace, and the six blocks emitted at the end. An allocation
 * added to any of them later without a charge fails this and nothing else.
 *
 * One charge it does NOT pin is the blocks', because that one provably cannot
 * set the peak -- see the note in `rowmap_emit`. It is kept as a guard against
 * a future change to the emit ordering, and it is honest to say here that
 * nothing tests it.
 *
 * The wrappers are armed only around the call under test. Frees of pointers
 * allocated before arming are ignored, so what is measured is this call's own
 * footprint and not the model's.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

/** The wide model, built once for the binary. See `build_wide`. */
static BeamModel *g_wide = NULL;

#include "test_utils.h"

/* ── The allocator's side of the story ───────────────────────── */

void *__real_malloc(size_t n);
void *__real_calloc(size_t n, size_t sz);
void *__real_realloc(void *p, size_t n);
void  __real_free(void *p);

/*
 * Pointer -> size for allocations made while armed. Open addressing, never
 * grown: the call under test holds a few dozen blocks at once and makes a few
 * hundred over its life, so 8,192 slots is two orders of magnitude of slack.
 * A full table is reported rather than silently wrapping, because a silent
 * wrap would understate the peak and pass.
 */
#define TRACK_SLOTS 8192u
static struct { void *p; size_t n; } track[TRACK_SLOTS];
static bool armed = false, overflowed = false;
static uint64_t live = 0, peak = 0;

static uint32_t slot_of(void *p) {
    return (uint32_t)(((uintptr_t)p >> 4) * 2654435761u) % TRACK_SLOTS;
}

static void note(void *p, size_t n) {
    if (!p) return;
    uint32_t h = slot_of(p);
    for (uint32_t i = 0; i < TRACK_SLOTS; ++i) {
        const uint32_t s = (h + i) % TRACK_SLOTS;
        if (!track[s].p) {
            track[s].p = p; track[s].n = n;
            live += n;
            if (live > peak) peak = live;
            return;
        }
    }
    overflowed = true;
}

/** Forget `p` and return what it was, or 0 if it was not ours. */
static size_t forget(void *p) {
    if (!p) return 0;
    uint32_t h = slot_of(p);
    for (uint32_t i = 0; i < TRACK_SLOTS; ++i) {
        const uint32_t s = (h + i) % TRACK_SLOTS;
        if (track[s].p == p) {
            const size_t n = track[s].n;
            /* Tombstone-free removal is wrong under linear probing, so the
             * slot keeps its key with a zero size and is reused by `note`
             * only after a full pass fails -- which the slack prevents. */
            track[s].p = NULL; track[s].n = 0;
            live -= n;
            return n;
        }
    }
    return 0;
}

void *__wrap_malloc(size_t n) {
    void *p = __real_malloc(n);
    if (armed) note(p, n);
    return p;
}

void *__wrap_calloc(size_t n, size_t sz) {
    void *p = __real_calloc(n, sz);
    if (armed) note(p, n * sz);
    return p;
}

void *__wrap_realloc(void *p, size_t n) {
    if (!armed) return __real_realloc(p, n);
    /*
     * The old block is live alongside the new one for the length of the copy
     * whenever `realloc` moves, so the peak has to see both. Charged that way
     * unconditionally: an in-place extension makes this an over-count of the
     * measurement, which can only make the test harder to pass, never easier.
     */
    const size_t was = forget(p);
    live += was; if (live > peak) peak = live;      /* both, for the copy */
    void *q = __real_realloc(p, n);
    live -= was;
    note(q, n);
    return q;
}

void __wrap_free(void *p) {
    if (armed) forget(p);
    __real_free(p);
}

static void arm(void) {
    memset(track, 0, sizeof(track));
    live = 0; peak = 0; overflowed = false;
    armed = true;
}

static uint64_t disarm(void) { armed = false; return peak; }

/* ── The model ───────────────────────────────────────────────── */

/*
 * The same wide model `test_expected_counts_budget.c` uses, and for the same
 * reason: it is the one that drives the accumulators past their starting
 * capacity, so the doubling paths -- where an uncharged allocation would hide
 * -- actually run. Nucleotides for the runtime reason given there: what these
 * tests need is many distinct rows, and what costs the time is `n_states`.
 */
#define N_WIDE 240
#define WIDE_MAX_LEN 24u
static char wide_buf[N_WIDE][WIDE_MAX_LEN + 1u];
static const char *wide_seq[N_WIDE];
static double wide_wt[N_WIDE];

static BeamModel *build_wide_once(uint32_t order) {
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
    cfg.order = order;
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
static BeamModel *build_wide(uint32_t order) {
    if (!g_wide) g_wide = build_wide_once(order);
    return g_wide;
}

/* ── The tests ───────────────────────────────────────────────── */

/**
 * The reported working set covers what the allocator actually handed out.
 *
 * `>=` and not `==` on purpose. The account charges a doubling's new array
 * before releasing the old, because `realloc` may move and hold both; when it
 * extends in place instead, the account has counted a copy that never
 * existed. Over-stating a memory figure is the safe direction and the one
 * this library takes everywhere. What is NOT allowed is the other direction,
 * and that is what the assertion pins.
 */
static void test_the_reported_working_set_covers_every_allocation(void) {
    BeamModel *m = build_wide(2u);
    ASSERT_MSG(m, "wide model did not build");
    beam_model_set_memory_budget(0);

    BeamExpectedCounts ec;
    arm();
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    const uint64_t measured = disarm();

    ASSERT_MSG(err == BEAM_OK, beam_error_message());
    ASSERT_MSG(!overflowed, "the allocation tracker ran out of slots, so the "
                            "measured peak is an undercount");
    ASSERT_MSG(measured > 0, "no allocation was seen at all, so the wrappers "
                             "are not reaching the library");

    static char msg[256];
    if (ec.working_set_bytes < measured) {
        snprintf(msg, sizeof msg,
                 "reported %.2f MB but the allocator handed out %.2f MB at "
                 "peak: %.2f MB is uncharged",
                 ec.working_set_bytes / 1e6, measured / 1e6,
                 (measured - ec.working_set_bytes) / 1e6);
        ASSERT_MSG(false, msg);
    }
    beam_model_expected_counts_free(&ec);
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

/**
 * And it is not covering it by being absurdly large.
 *
 * Without this the assertion above passes by reporting `UINT64_MAX`, which is
 * how a bound that protects nothing usually gets written. Twice the true peak
 * is loose enough for the realloc over-count and tight enough that a figure
 * with nothing behind it fails.
 */
static void test_the_reported_working_set_is_not_a_meaningless_ceiling(void) {
    BeamModel *m = build_wide(2u);
    ASSERT_MSG(m, "wide model did not build");
    beam_model_set_memory_budget(0);

    BeamExpectedCounts ec;
    arm();
    const BeamError err = beam_model_expected_counts(m, 0u, &ec);
    const uint64_t measured = disarm();
    ASSERT_MSG(err == BEAM_OK, beam_error_message());
    ASSERT_MSG(!overflowed, "the allocation tracker ran out of slots");

    static char msg[256];
    if (ec.working_set_bytes > 2u * measured) {
        snprintf(msg, sizeof msg,
                 "reported %.2f MB against a true peak of %.2f MB, which is "
                 "more than twice as much", ec.working_set_bytes / 1e6,
                 measured / 1e6);
        ASSERT_MSG(false, msg);
    }
    beam_model_expected_counts_free(&ec);
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

/**
 * The figure tracks the work, rather than being a constant that happens to be
 * large enough.
 *
 * Pooling over every live length reaches strictly more `(depth, context)`
 * pairs than any single length does, so its accumulators are strictly larger.
 * A reported figure that did not move between the two would satisfy both
 * assertions above and mean nothing.
 */
static void test_the_reported_figure_tracks_the_work(void) {
    BeamModel *m = build_wide(2u);
    ASSERT_MSG(m, "wide model did not build");
    beam_model_set_memory_budget(0);

    uint32_t shortest = 0;
    for (uint32_t L = 1; L <= m->max_len; ++L)
        if (beam_length_live(m, L)) { shortest = L; break; }
    ASSERT_MSG(shortest > 0, "the model has no live length");

    BeamExpectedCounts one, all;
    ASSERT_MSG(beam_model_expected_counts(m, shortest, &one) == BEAM_OK,
               beam_error_message());
    ASSERT_MSG(beam_model_expected_counts(m, 0u, &all) == BEAM_OK,
               beam_error_message());

    ASSERT_MSG(one.working_set_bytes > 0 && all.working_set_bytes > 0,
               "a call reported holding nothing at all");
    ASSERT_MSG(all.working_set_bytes > one.working_set_bytes,
               "pooling over every length reported the same working set as a "
               "single length, so the figure is not measuring anything");

    beam_model_expected_counts_free(&one);
    beam_model_expected_counts_free(&all);
    if (m != g_wide) beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("expected-count accounting\n");
    /*
     * ONE THREAD. These tests are about what the sweep allocates, not about
     * how it divides the work, and the models here are small enough that the
     * per-layer parallel regions cost several hundred times what they save:
     * the pooled sweep below is 0.02 s on one thread and 17.6 s on this
     * machine's default, which is a property of the model's size and not of
     * anything under test.
     */
    const int threads_before = beam_model_threads();
    beam_model_set_threads(1);

    RUN_TEST(test_the_reported_working_set_covers_every_allocation);
    RUN_TEST(test_the_reported_working_set_is_not_a_meaningless_ceiling);
    RUN_TEST(test_the_reported_figure_tracks_the_work);
    if (g_wide) beam_model_destroy(g_wide);
    beam_model_set_threads(threads_before);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
