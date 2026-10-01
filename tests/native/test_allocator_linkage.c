/**
 * @file test_allocator_linkage.c
 * @brief The allocator policy must survive a link that never names it.
 *
 * The policy is applied by a constructor in memory/allocator.c. A constructor
 * only runs if its object file is linked, and a static archive contributes
 * only objects that resolve a symbol somebody referenced. allocator.c defines
 * the policy and `beam_applied_mmap_threshold`, and nothing else calls
 * either, so `libbeam.a` consumers dropped the object and got none of it:
 * measured, a model that holds 28.3 MB with the policy held 482.2 MB without.
 *
 * THIS FILE MUST NOT NAME ANY SYMBOL FROM memory/allocator.c. Referencing one
 * is what pulls the object in, which is exactly the condition under test --
 * `test_allocator_policy.c` calls `beam_applied_mmap_threshold` and therefore
 * cannot detect this, and did not. What it checks is that the policy WORKS;
 * what this checks is that it is THERE when nobody asks for it.
 *
 * The bound is deliberately loose, and the fixture is sized so that it
 * separates the two cases: the ratio compares a sweep's pinned scratch to the
 * model's size, and a model too large hides it. The 60,000-sequence,
 * length-20 fixture this used to build holds 236 MB under the hierarchical
 * prior and measured 5.2x with the policy disabled -- a pass. This one
 * measured 2.7x with the policy and 17.4x without it, so 8x separates them
 * with room for allocator differences.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "beam/common.h"
#include "beam/model.h"

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

static void test_the_policy_applies_without_being_named(void) {
    const long before = rss_bytes();
    ASSERT_MSG(before > 0, "/proc/self/statm is unreadable");

    rs = 88172645463325252ull;
    const uint32_t n = 10000u;
    char **seqs = (char **)malloc(n * sizeof(char *));
    double *w = (double *)malloc(n * sizeof(double));
    ASSERT_MSG(seqs && w, "out of memory");
    for (uint32_t k = 0; k < n; ++k) {
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
    const BeamError oe = m ? beam_model_observe(m, (const char **)seqs, w, n)
                           : BEAM_ERR_NULL_ARG;
    for (uint32_t k = 0; k < n; ++k) free(seqs[k]);
    free(seqs); free(w);
    if (!m || oe != BEAM_OK) { if (m) beam_model_destroy(m);
                               ASSERT_MSG(false, "model did not build"); }
    if (beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m); ASSERT_MSG(false, "finalize failed"); }

    double h = 0;
    const BeamError he = beam_model_hill_number(m, 1.0, &h);
    const long after = rss_bytes();
    const uint64_t model = beam_model_memory_bytes(m);
    beam_model_destroy(m);

    ASSERT_MSG(he == BEAM_OK, "hill failed");
    ASSERT_MSG(model > 4u * 1024u * 1024u, "the model is too small to mean anything");
    const double ratio = (double)(after - before) / (double)model;
    printf("\n      model %.1f MB, resident grew %.1f MB, ratio %.2fx  ",
           model / 1e6, (after - before) / 1e6, ratio);
    if (ratio > 8.0)
        printf("\n      (allocator.o was dropped from the link: about 17x "
               "is the unpolicied figure)  ");
    ASSERT_MSG(ratio <= 8.0, "the allocator policy did not reach this binary");
    PASS();
}

int main(void) {
    printf("\nallocator policy reaches a link that never names it\n");
    RUN_TEST(test_the_policy_applies_without_being_named);
    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
