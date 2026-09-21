/**
 * @file test_lazy_thaw.c
 * @brief Expanding counts one table at a time must build the same tables.
 *
 * `beam_model_finalize` used to expand all six count tables at once and hold
 * them for the whole build. On the foundation that is 29.6 MB of frozen counts
 * becoming 731.3 MB of mutable rows, alive while 714.2 MB of log tables
 * accumulate beside them: a measured peak of 1,993 MB for a model that ends up
 * occupying 807.
 *
 * It now expands ONE table, writes its log table, and destroys it before the
 * next -- but only for a model that arrives frozen and has no backoff, because
 * `beam_build_hierarchy` reads counts across tables and would find them gone.
 *
 * The risk is that the two paths quietly stop agreeing: the lazy one reads a
 * locally thawed table through a different pointer, and a mistake there gives
 * plausible numbers from the wrong rows. So the test finalizes the SAME model
 * both ways and compares every cell of every table. It also pins the thing the
 * first version of this change got wrong -- the loader's empty count
 * directories were left allocated, 29,563,416 B of them, because the
 * thaw-then-freeze cycle that used to dispose of them no longer ran.
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

static const char AA[] = "ACDEFGHIKLMNPQRSTVWY";
static uint64_t rs;
static uint32_t rnd(uint32_t n){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17;
                                 return (uint32_t)(rs % n); }

static BeamModel *build(uint32_t n_seqs, uint8_t order, uint32_t max_length) {
    rs = 88172645463325252ull;
    char **seqs = (char **)malloc(n_seqs * sizeof(char *));
    double *w = (double *)malloc(n_seqs * sizeof(double));
    if (!seqs || !w) { free(seqs); free(w); return NULL; }
    for (uint32_t k = 0; k < n_seqs; ++k) {
        const uint32_t L = 6u + rnd(8u);
        char *s = (char *)malloc(L + 3u);
        s[0] = 'C';
        for (uint32_t j = 0; j < L; ++j) s[1 + j] = AA[rnd(20u)];
        s[L + 1] = 'F'; s[L + 2] = 0;
        seqs[k] = s; w[k] = 1.0 + (double)rnd(40u);
    }
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order; cfg.max_length = max_length; cfg.sparse = true;
    BeamModel *m = beam_model_new(&cfg);
    if (m && (beam_model_observe(m, (const char **)seqs, w, n_seqs) != BEAM_OK
              || beam_model_finalize(m) != BEAM_OK)) {
        beam_model_destroy(m); m = NULL;
    }
    for (uint32_t k = 0; k < n_seqs; ++k) free(seqs[k]);
    free(seqs); free(w);
    return m;
}

static BeamSparseRows *lp_of(BeamModel *m, uint32_t i) {
    const uint32_t d = i / 3u, k = i % 3u;
    if (k == 0u) return &m->sp_lp[d];
    if (k == 1u) return &m->sp_lp[d];
    return &m->sp_lp[d];
}
static BeamSparseRows *cnt_of(BeamModel *m, uint32_t i) {
    const uint32_t d = i / 3u, k = i % 3u;
    if (k == 0u) return &m->sp_cnt[d];
    if (k == 1u) return &m->sp_cnt[d];
    return &m->sp_cnt[d];
}

/** Every cell of every finalized table, in dense-row order. */
static double *snapshot(BeamModel *m, uint64_t *n_out) {
    uint64_t cap = 0;
    for (uint32_t t = 0; t < 6u; ++t)
        cap += (uint64_t)lp_of(m, t)->n_dense_rows * m->n_residues;
    double *v = (double *)malloc((size_t)cap * sizeof(double));
    if (!v) return NULL;
    uint64_t n = 0;
    for (uint32_t t = 0; t < 6u; ++t) {
        const BeamSparseRows *tb = lp_of(m, t);
        for (uint32_t r = 0; r < tb->n_dense_rows; ++r) {
            const BeamLogRow lr = beam_sparse_log_row(tb, r);
            for (uint32_t x = 0; x < m->n_residues; ++x)
                v[n++] = lr.present ? beam_log_cell(&lr, x) : -INFINITY;
        }
    }
    *n_out = n;
    return v;
}

/*
 * The two paths must agree cell for cell. `build` already finalized once,
 * from mutable counts; finalizing again with the counts frozen takes the lazy
 * path, and thawing first takes the original one.
 */
static void test_lazy_and_eager_finalize_agree(void) {
    BeamModel *m = build(600u, 4u, 16u);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(m->counts_frozen, "counts are not frozen after finalize");

    /* Lazy: counts arrive frozen. */
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "lazy finalize failed");
    ASSERT_MSG(m->counts_frozen, "the lazy path left the counts thawed");
    uint64_t n_lazy = 0; double *lazy = snapshot(m, &n_lazy);
    ASSERT_MSG(lazy != NULL, "out of memory");

    /* Eager: thaw first, so finalize takes the original path. */
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "thaw failed");
    ASSERT_MSG(!m->counts_frozen, "thaw did not clear the flag");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "eager finalize failed");
    uint64_t n_eager = 0; double *eager = snapshot(m, &n_eager);
    ASSERT_MSG(eager != NULL, "out of memory");

    uint64_t bad = 0, checked = 0, finite = 0;
    if (n_lazy == n_eager)
        for (uint64_t i = 0; i < n_lazy; ++i) {
            checked++;
            if (lazy[i] > -INFINITY) finite++;
            if (memcmp(&lazy[i], &eager[i], sizeof(double)) != 0) bad++;
        }
    free(lazy); free(eager);
    beam_model_destroy(m);

    ASSERT_MSG(n_lazy == n_eager, "the two paths produced different table sizes");
    ASSERT_MSG(finite > 1000u, "too few live cells for this to mean anything");
    ASSERT_MSG(bad == 0u, "the lazy and eager paths disagree on a cell");
    PASS();
}

/*
 * The loader allocates a directory for every count table -- four bytes per
 * dense row, occupied or not. The thaw-then-freeze cycle used to dispose of
 * them; the lazy path does not run it, and the first version of this change
 * left 29,563,416 B of empty directory behind on the foundation. It showed up
 * as the model getting BIGGER, which no value test would have caught.
 *
 * It MUST go through a file. A model built in this process has already had
 * those tables destroyed by its first freeze, so there is nothing left to
 * leak and the check passes whether or not the cleanup runs -- which is
 * exactly what the first version of this test did, and it was worthless.
 * Only a loaded model has directories the loader itself allocated.
 */
static void test_a_loaded_model_keeps_no_count_directory(void) {
    BeamModel *src = build(600u, 4u, 16u);
    ASSERT_MSG(src != NULL, "model did not build");
    const char *path = "build/tests/lazy_thaw_roundtrip.beam";
    const BeamError se = beam_model_save(src, path);
    beam_model_destroy(src);
    ASSERT_MSG(se == BEAM_OK, "save failed");

    BeamModel *m = beam_model_load(path);
    remove(path);
    ASSERT_MSG(m != NULL, "load failed");
    ASSERT_MSG(m->counts_frozen, "a loaded model did not arrive frozen");

    uint64_t left = 0;
    for (uint32_t t = 0; t < 6u; ++t)
        left += beam_sparse_memory_bytes(cnt_of(m, t));
    const uint64_t total = beam_model_memory_bytes(m);
    /* The frozen counts must still be there -- releasing THOSE would be a
     * different and much worse bug than the one this guards. */
    const uint64_t frozen = beam_model_frozen_bytes(m);
    beam_model_destroy(m);

    if (left != 0u)
        printf("FAIL: %llu bytes of mutable count table survived the load\n",
               (unsigned long long)left);
    ASSERT_MSG(left == 0u, "a loaded model holds empty count directories");
    ASSERT_MSG(frozen > 0u, "the frozen counts were released too");
    ASSERT_MSG(total > frozen, "the model reports less than its own counts");
    PASS();
}

/*
 * The lazy path is gated on `cfg.sparse`, and `beam_config_ok` refuses sparse
 * storage with any non-zero kappa -- backoff can populate a row that was never
 * observed, and a sparse table has nowhere to put it. So a backoff model is
 * necessarily DENSE and necessarily takes the original path. This pins that:
 * if the config rule ever relaxes and sparse+backoff becomes constructible,
 * the lazy path would start reading counts that `beam_build_hierarchy` needs
 * across tables, and this test is where that shows up.
 */
static void test_a_backoff_model_is_dense_and_unaffected(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 4u; cfg.max_length = 16u; cfg.strict = false;

    cfg.sparse = true;
    BeamModel *bad = beam_model_new(&cfg);
    if (bad) { beam_model_destroy(bad);
               ASSERT_MSG(false, "sparse storage was accepted with backoff on"); }

    cfg.sparse = false;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "the dense backoff model did not construct");

    rs = 88172645463325252ull;
    const uint32_t n = 400u;
    char **seqs = (char **)malloc(n * sizeof(char *));
    double *w = (double *)malloc(n * sizeof(double));
    if (!seqs || !w) { free(seqs); free(w); beam_model_destroy(m);
                       ASSERT_MSG(false, "out of memory"); }
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t L = 6u + rnd(8u);
        char *s2 = (char *)malloc(L + 3u);
        s2[0] = 'C';
        for (uint32_t j = 0; j < L; ++j) s2[1 + j] = AA[rnd(20u)];
        s2[L + 1] = 'F'; s2[L + 2] = 0;
        seqs[k] = s2; w[k] = 1.0;
    }
    const BeamError oe = beam_model_observe(m, (const char **)seqs, w, n);
    for (uint32_t k = 0; k < n; ++k) free(seqs[k]);
    free(seqs); free(w);
    if (oe != BEAM_OK) { beam_model_destroy(m);
                         ASSERT_MSG(false, "observe failed"); }

    const BeamError f1 = beam_model_finalize(m);
    const BeamError f2 = f1 == BEAM_OK ? beam_model_finalize(m) : f1;
    const bool sparse_off = !m->cfg.sparse;
    beam_model_destroy(m);
    ASSERT_MSG(f1 == BEAM_OK, "the dense backoff model did not finalize");
    ASSERT_MSG(f2 == BEAM_OK, "it did not re-finalize");
    ASSERT_MSG(sparse_off, "a backoff model reported sparse storage");
    PASS();
}

int main(void) {
    printf("\nexpanding counts one table at a time\n");
    RUN_TEST(test_lazy_and_eager_finalize_agree);
    RUN_TEST(test_a_loaded_model_keeps_no_count_directory);
    RUN_TEST(test_a_backoff_model_is_dense_and_unaffected);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
