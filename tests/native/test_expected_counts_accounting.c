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
 * IT COVERS THE WHOLE PATH, which is the point: the backward field, the two
 * accumulator maps, both of their doubling paths, the directories the
 * doublings replace, and the two blocks emitted at the end. An allocation
 * added to any of them later without a charge fails this and nothing else.
 * That the doubling paths really run on the model here is asserted, not
 * assumed.
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

/** The model, built once for the binary. See `build`. */
static BeamModel *g_model = NULL;

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

/*
 * What a freed slot holds. Emptying it to NULL would be wrong under linear
 * probing: a pointer that collided and was placed past it would stop being
 * found, its free would go unrecorded, and its bytes would stay "live" for the
 * rest of the measurement. So a lookup walks past this and only an insertion
 * reuses it. No allocation can be at address 1, so it is never a real key.
 */
#define TRACK_GONE ((void *)(uintptr_t)1u)

static uint32_t slot_of(void *p) {
    return (uint32_t)(((uintptr_t)p >> 4) * 2654435761u) % TRACK_SLOTS;
}

static void note(void *p, size_t n) {
    if (!p) return;
    uint32_t h = slot_of(p);
    for (uint32_t i = 0; i < TRACK_SLOTS; ++i) {
        const uint32_t s = (h + i) % TRACK_SLOTS;
        if (!track[s].p || track[s].p == TRACK_GONE) {
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
        if (!track[s].p) return 0;          /* the end of p's probe chain */
        if (track[s].p == p) {
            const size_t n = track[s].n;
            track[s].p = TRACK_GONE; track[s].n = 0;
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
 * The model `test_expected_counts_budget.c` uses, and for the same reason: it
 * drives the accumulators past their starting capacity, so the doubling paths
 * -- where an uncharged allocation would hide -- actually run.
 *
 * Under the hierarchical prior every string up to `max_length` has mass, so
 * the sweep reaches every `(depth, context)` the trellis can form and the row
 * count is fixed by the shape: at order 2 over 20 residues, one row at depth
 * 0, 20 at depth 1 and 400 at every depth after, 4,021 per map at
 * `max_length` 12, where `rowmap_init` starts with room for 512. Each map's rows double three times and its key
 * directory is rehashed three times. A separate wide nucleotide model used to
 * stand here only to make the maps grow; this one does that on its own.
 *
 * Sparse storage, the default: the accumulators do not depend on how the
 * tables are stored, and sparse is the form a per-repertoire model is in.
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

static BeamModel *build_once(void) {
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

/* Built once for the binary: every test wants the same model. */
static BeamModel *build(void) {
    if (!g_model) g_model = build_once();
    return g_model;
}

/** Rows `rowmap_init` gives a map before its first doubling. */
#define MAP_START_ROWS 512u

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
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
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
    /* A map that never outgrew its first allocation never ran a doubling
     * path, and those are the allocations this test most needs to have seen. */
    ASSERT_MSG(ec.block[0].n_rows > MAP_START_ROWS
               && ec.block[1].n_rows > MAP_START_ROWS,
               "a map never grew past its starting capacity, so the doubling "
               "paths were not measured");

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
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
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
    PASS();
}

/**
 * The figure tracks the work, rather than being a constant that happens to be
 * large enough.
 *
 * Pooling over every live length reaches every `(depth, context)` the trellis
 * can form, 4,021 per map here, where the shortest length reaches only the
 * rows at its own few depths -- and it sweeps the longest field rather than
 * the shortest. So its working set is strictly larger. A reported figure that
 * did not move between the two would satisfy both assertions above and mean
 * nothing.
 */
static void test_the_reported_figure_tracks_the_work(void) {
    BeamModel *m = build();
    ASSERT_MSG(m, "model did not build");
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
    PASS();
}

int main(void) {
    printf("expected-count accounting\n");
    /*
     * ONE THREAD, and not for speed: the allocation tracker above is a plain
     * global table with no locking, so a `note` and a `forget` racing on two
     * worker threads would corrupt `live` and the measured peak would be a
     * number with nothing behind it. Nothing in the sweep allocates from
     * inside a parallel region today, which is exactly the kind of fact that
     * stops being true without anyone noticing here. Pinning makes the
     * measurement sound rather than merely correct at the moment.
     */
    const int threads_before = beam_model_threads();
    beam_model_set_threads(1);

    RUN_TEST(test_the_reported_working_set_covers_every_allocation);
    RUN_TEST(test_the_reported_working_set_is_not_a_meaningless_ceiling);
    RUN_TEST(test_the_reported_figure_tracks_the_work);
    if (g_model) beam_model_destroy(g_model);
    beam_model_set_threads(threads_before);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
