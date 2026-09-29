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
 * It now expands ONE direction's table, writes that direction's log tables --
 * the top level and every lower level of the hierarchical prior -- and
 * destroys it before the next. That is only sound because every level of the
 * prior is summed out of the SAME direction's counts; nothing one direction
 * builds reads the other's table. A sparse model that arrives frozen takes
 * this lazy path. A sparse model whose counts are mutable, and every dense
 * model, read the model's own thawed tables instead.
 *
 * The risk is that the paths quietly stop agreeing: the lazy one reads a
 * locally thawed table through a different pointer, and a mistake there gives
 * plausible numbers from the wrong rows. So the test finalizes the SAME model
 * both ways and compares every stored cell of every level, and then compares
 * the lazy sparse model against a dense one, which never takes the lazy path,
 * on every row the prior resolves. It also pins the thing the first version
 * of this change got wrong -- the loader's empty count directories were left
 * allocated, 29,563,416 B of them, because the thaw-then-freeze cycle that
 * used to dispose of them no longer ran.
 *
 * Order 3 rather than the default 4. Nothing here depends on the order, and
 * under the prior every state of the trellis is live, so each finalize's
 * partition functions cost about twenty times more at order 4 -- and this file
 * finalizes nine times.
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

#define ORDER   3u
#define MAX_LEN 16u     /* the corpus below spans lengths 8..15 */

static BeamModel *build(uint32_t n_seqs, bool sparse) {
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
    cfg.order = ORDER; cfg.max_length = MAX_LEN; cfg.sparse = sparse;
    BeamModel *m = beam_model_new(&cfg);
    if (m && (beam_model_observe(m, (const char **)seqs, w, n_seqs) != BEAM_OK
              || beam_model_finalize(m) != BEAM_OK)) {
        beam_model_destroy(m); m = NULL;
    }
    for (uint32_t k = 0; k < n_seqs; ++k) free(seqs[k]);
    free(seqs); free(w);
    return m;
}

/*
 * Every finalized table of one direction: the top level, then level 0 up to
 * `order - 1`. The lower levels are finalized state exactly as much as the top
 * one is -- an unobserved top row IS its nearest stored ancestor -- so a path
 * that got them wrong would change scores while leaving the top rows intact.
 */
static uint32_t n_tables(const BeamModel *m) { return BEAM_DIRS * (m->order + 1u); }
static const BeamSparseRows *table_of(const BeamModel *m, uint32_t i) {
    const uint32_t d = i / (m->order + 1u), lvl = i % (m->order + 1u);
    return lvl == 0u ? &m->sp_lp[d] : &m->sp_lo[d][lvl - 1u];
}

/*
 * What a sparse model stores: which rows each table holds, and the bits of
 * every cell of those rows. A row a table does not hold resolves to its
 * ancestor, which is a function of these, so comparing them compares every
 * answer the model can give.
 */
typedef struct {
    uint8_t *held;       /* one per dense row of every table, in order */
    double  *cells;      /* R per held row, in the same order          */
    uint64_t n_rows, n_held;
} Snapshot;

static void snapshot_free(Snapshot *s) { free(s->held); free(s->cells); }

static bool snapshot(const BeamModel *m, Snapshot *s) {
    memset(s, 0, sizeof(*s));
    uint64_t rows = 0, held = 0;
    for (uint32_t t = 0; t < n_tables(m); ++t) {
        const BeamSparseRows *tb = table_of(m, t);
        rows += tb->n_dense_rows;
        held += beam_sparse_n_rows(tb);
    }
    s->held  = (uint8_t *)malloc((size_t)rows);
    s->cells = (double *)malloc((size_t)(held ? held : 1u) * m->n_residues
                                * sizeof(double));
    if (!s->held || !s->cells) { snapshot_free(s); return false; }
    for (uint32_t t = 0; t < n_tables(m); ++t) {
        const BeamSparseRows *tb = table_of(m, t);
        for (uint32_t r = 0; r < tb->n_dense_rows; ++r) {
            const BeamLogRow lr = beam_sparse_log_row(tb, r);
            s->held[s->n_rows++] = lr.present;
            if (!lr.present) continue;
            if (s->n_held == held) { snapshot_free(s); return false; }
            beam_log_row_into(&lr, m->n_residues,
                              s->cells + s->n_held * m->n_residues);
            s->n_held++;
        }
    }
    return true;
}

/*
 * The two paths must agree cell for cell. `build` already finalized once,
 * from mutable counts; finalizing again with the counts frozen takes the lazy
 * path, and thawing first takes the original one.
 */
static void test_lazy_and_eager_finalize_agree(void) {
    BeamModel *m = build(600u, true);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(m->counts_frozen, "counts are not frozen after finalize");

    /* One stored top row per observed count row: the lazy path must not
     * lose any of them, and a table with none would compare nothing. */
    uint64_t observed = 0;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        observed += beam_model_frozen_table(m, d)->n_rows;

    /* Lazy: counts arrive frozen. */
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "lazy finalize failed");
    ASSERT_MSG(m->counts_frozen, "the lazy path left the counts thawed");
    Snapshot lazy, eager;
    ASSERT_MSG(snapshot(m, &lazy), "out of memory");
    uint64_t top_held = 0;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        top_held += beam_sparse_n_rows(&m->sp_lp[d]);
    double lz[MAX_LEN + 1u], lpi[MAX_LEN + 1u];
    memcpy(lz, m->log_z, sizeof lz);
    memcpy(lpi, m->log_pi, sizeof lpi);

    /* Eager: thaw first, so finalize takes the original path. */
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "thaw failed");
    ASSERT_MSG(!m->counts_frozen, "thaw did not clear the flag");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "eager finalize failed");
    if (!snapshot(m, &eager)) { snapshot_free(&lazy); beam_model_destroy(m);
                                ASSERT_MSG(false, "out of memory"); }

    const bool same_shape = lazy.n_rows == eager.n_rows
                         && lazy.n_held == eager.n_held;
    const bool same_rows = same_shape
        && memcmp(lazy.held, eager.held, (size_t)lazy.n_rows) == 0;
    const bool same_cells = same_rows
        && memcmp(lazy.cells, eager.cells,
                  (size_t)lazy.n_held * m->n_residues * sizeof(double)) == 0;
    const bool same_z = memcmp(lz, m->log_z, sizeof lz) == 0
                     && memcmp(lpi, m->log_pi, sizeof lpi) == 0;
    snapshot_free(&lazy); snapshot_free(&eager);
    beam_model_destroy(m);

    ASSERT_MSG(observed > 0u, "nothing was observed, so nothing is compared");
    ASSERT_MSG(top_held == observed,
               "the lazy path did not store one top row per observed row");
    ASSERT_MSG(same_shape, "the two paths produced different table sizes");
    ASSERT_MSG(same_rows, "the two paths stored different rows");
    ASSERT_MSG(same_cells, "the lazy and eager paths disagree on a cell");
    ASSERT_MSG(same_z, "the two paths gave different partition functions");
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
    BeamModel *src = build(600u, true);
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
    for (uint32_t d = 0; d < BEAM_DIRS; ++d)
        left += beam_sparse_memory_bytes(&m->sp_cnt[d]);
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
 * The lazy path is taken only by a sparse model, so a dense one is the control
 * that never takes it: `finalize` thaws both of its tables before building
 * anything, re-finalize included. Sparse and dense are the same model stored
 * two ways, so the dense one re-finalized from both thawed tables must give,
 * on EVERY row -- stored or resolved to an ancestor -- the bits the sparse one
 * gives after expanding its tables one at a time.
 *
 * This is where a level of the prior that needed the other direction's counts
 * would show up. It used to be where sparse storage with backoff would have,
 * when the config refused that pairing; every model now carries the prior and
 * sparse is the default, so the pairing is the ordinary case.
 */
static void test_a_dense_model_finalizes_eagerly_to_the_same_bits(void) {
    BeamModel *sp = build(400u, true);
    BeamModel *dn = build(400u, false);
    if (!sp || !dn) { beam_model_destroy(sp); beam_model_destroy(dn);
                      ASSERT_MSG(false, "the models did not build"); }

    /* Re-finalize both: the sparse one lazily, from its frozen counts; the
     * dense one by thawing both tables first. */
    const BeamError fs = beam_model_finalize(sp);
    const BeamError fd = beam_model_finalize(dn);
    const bool sparse_on = sp->cfg.sparse, sparse_off = !dn->cfg.sparse;

    uint64_t rows = 0, differ = 0;
    if (fs == BEAM_OK && fd == BEAM_OK) {
        const uint32_t R = sp->n_residues;
        double a[BEAM_MAX_RESIDUES], b[BEAM_MAX_RESIDUES];
        for (uint32_t d = 0; d < BEAM_DIRS; ++d)
            for (uint32_t depth = 0; depth < sp->max_len; ++depth)
                for (uint32_t c = 0; c < sp->n_ctx; ++c) {
                    const BeamLogRow ra = beam_lp_row(sp, d, depth, c);
                    const BeamLogRow rb = beam_lp_row(dn, d, depth, c);
                    beam_log_row_into(&ra, R, a);
                    beam_log_row_into(&rb, R, b);
                    if (!ra.present || !rb.present
                        || memcmp(a, b, (size_t)R * sizeof(double)) != 0)
                        differ++;
                    rows++;
                }
        for (uint32_t L = 0; L <= sp->max_len; ++L)
            if (memcmp(&sp->log_z[L], &dn->log_z[L], sizeof(double)) != 0
                || memcmp(&sp->log_pi[L], &dn->log_pi[L], sizeof(double)) != 0)
                differ++;
    }
    const uint64_t want_rows = (uint64_t)BEAM_DIRS * sp->max_len * sp->n_ctx;
    beam_model_destroy(sp); beam_model_destroy(dn);

    ASSERT_MSG(sparse_on && sparse_off, "the two storage modes were not built");
    ASSERT_MSG(fs == BEAM_OK, "the sparse model did not re-finalize");
    ASSERT_MSG(fd == BEAM_OK, "the dense model did not re-finalize");
    ASSERT_MSG(rows == want_rows, "not every row was compared");
    if (differ) printf("FAIL: %llu rows or lengths differ\n",
                       (unsigned long long)differ);
    ASSERT_MSG(differ == 0u,
               "the dense eager path and the sparse lazy path disagree");
    PASS();
}

int main(void) {
    printf("\nexpanding counts one table at a time\n");
    RUN_TEST(test_lazy_and_eager_finalize_agree);
    RUN_TEST(test_a_loaded_model_keeps_no_count_directory);
    RUN_TEST(test_a_dense_model_finalizes_eagerly_to_the_same_bits);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
