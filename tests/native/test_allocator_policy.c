/**
 * @file test_allocator_policy.c
 * @brief A finished model must not cost many times its size in resident memory.
 *
 * glibc's mmap threshold is dynamic: freeing an mmap-ed block raises it, and
 * afterwards large allocations come from the heap, which can only be returned
 * to the OS from its top. One live allocation above a freed region pins all of
 * it. Measured before this was pinned, a 13.8 MB model held 450 MB resident
 * after building and 678 MB after one analytic -- and 348 MB after the model
 * itself was destroyed.
 *
 * Nothing leaks, so no leak checker sees it and no size accounting sees it
 * either: `beam_model_memory_bytes` reports what the model asked for, which
 * was right all along. Only resident memory shows it, which is why this test
 * reads /proc rather than any BEAM accessor.
 *
 * The bound is deliberately loose, and the fixture is sized so that it
 * separates the two cases. The ratio compares a fixed cost -- an order-4
 * sweep's scratch, which the ratchet pins -- to the model's size, so a model
 * that is too large hides the regression: the 60,000-sequence, length-20
 * fixture this used to build holds 236 MB under the hierarchical prior and
 * measured 1.8x with the policy and only 5.2x without it, passing either way.
 * This one measured 2.7x with the policy and 17.4x with it disabled
 * (BEAM_MMAP_THRESHOLD=0), so 8x sits between them with room for allocator
 * differences between machines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

static const char AA[] = "ACDEFGHIKLMNPQRSTVWY";
static uint64_t rs;
static uint32_t rnd(uint32_t n){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17;
                                 return (uint32_t)(rs % n); }

static long rss_bytes(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return -1;
    long total = 0, res = 0;
    if (fscanf(f, "%ld %ld", &total, &res) != 2) res = -1;
    fclose(f);
    return res < 0 ? -1 : res * (long)sysconf(_SC_PAGESIZE);
}

static BeamModel *build(uint32_t n_seqs) {
    rs = 88172645463325252ull;
    char **seqs = (char **)malloc(n_seqs * sizeof(char *));
    double *w = (double *)malloc(n_seqs * sizeof(double));
    if (!seqs || !w) { free(seqs); free(w); return NULL; }
    for (uint32_t k = 0; k < n_seqs; ++k) {
        const uint32_t L = 6u + rnd(5u);
        char *s = (char *)malloc(L + 3u);
        s[0] = 'C';
        for (uint32_t j = 0; j < L; ++j) s[1 + j] = AA[rnd(20u)];
        s[L + 1] = 'F'; s[L + 2] = 0;
        seqs[k] = s; w[k] = 1.0 + (double)rnd(40u);
    }
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 4u; cfg.max_length = 12u; cfg.sparse = true;
    BeamModel *m = beam_model_new(&cfg);
    if (m && (beam_model_observe(m, (const char **)seqs, w, n_seqs) != BEAM_OK
              || beam_model_finalize(m) != BEAM_OK)) {
        beam_model_destroy(m); m = NULL;
    }
    for (uint32_t k = 0; k < n_seqs; ++k) free(seqs[k]);
    free(seqs); free(w);
    return m;
}

/* The policy is applied unless the environment opted out. */
static void test_the_allocator_policy_is_applied(void) {
    const int applied = beam_applied_mmap_threshold();
#if defined(__GLIBC__)
    if (getenv("MALLOC_MMAP_THRESHOLD_") || getenv("BEAM_MMAP_THRESHOLD")) {
        printf("SKIP (overridden by the environment) ");
        PASS(); return;
    }
    ASSERT_MSG(applied > 0, "no mmap threshold was applied on glibc");
    ASSERT_MSG(applied == 128 * 1024,
               "the applied threshold is not glibc's static default");
#else
    ASSERT_MSG(applied == 0, "a threshold was applied off glibc");
#endif
    PASS();
}

/*
 * Resident memory must stay near the model's own size. Measured from BEFORE
 * the build so the harness's own footprint is not counted as the model's.
 */
static void test_resident_memory_stays_near_the_model_size(void) {
    const long before = rss_bytes();
    ASSERT_MSG(before > 0, "/proc/self/statm is unreadable");

    BeamModel *m = build(10000u);
    ASSERT_MSG(m != NULL, "model did not build");

    /* One analytic, because the old behaviour got dramatically worse here. */
    double h = 0;
    ASSERT_MSG(beam_model_hill_number(m, 1.0, &h) == BEAM_OK, "hill failed");

    const long after = rss_bytes();
    const uint64_t model = beam_model_memory_bytes(m);
    beam_model_destroy(m);

    ASSERT_MSG(after > 0, "/proc/self/statm is unreadable");
    ASSERT_MSG(model > 4u * 1024u * 1024u,
               "the model is too small for this to mean anything");

    const double grew = (double)(after - before);
    const double ratio = grew / (double)model;
    printf("\n      model %.1f MB, resident grew %.1f MB, ratio %.2fx  ",
           model / 1e6, grew / 1e6, ratio);
    if (ratio > 8.0)
        printf("\n      (about 17x on this fixture without the pinned threshold)  ");
    ASSERT_MSG(ratio <= 8.0,
               "resident memory is many times the model's own size");
    PASS();
}

int main(void) {
    printf("\nallocator policy\n");
    RUN_TEST(test_the_allocator_policy_is_applied);
    RUN_TEST(test_resident_memory_stays_near_the_model_size);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
