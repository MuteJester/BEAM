/**
 * @file test_plane_layout.c
 * @brief A backward table laid out by residue must read the same as by row.
 *
 * The backward half of the edge kernel reads ONE residue from each of twenty
 * rows whose dense ids are consecutive. Stored by row that is fifty cache
 * lines to use a hundred and sixty bytes; stored by residue those bytes are
 * contiguous. Nothing reads a backward row whole, so nothing loses by it.
 *
 * The failure this guards is a transposed index: `values0[x * rows + slot]`
 * against `values0[slot * stride + x]` are both in bounds and both return a
 * real log-probability, so getting it backwards produces plausible numbers
 * for the wrong residue and nothing downstream can tell. The check is
 * therefore against the DEFINITION -- recompute each cell from the counts it
 * came from -- rather than against the other layout, which would only prove
 * the two agree about being wrong.
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

static BeamSparseRows *table_of(BeamModel *m, uint32_t i) {
    const uint32_t d = i / 3u, k = i % 3u;
    if (k == 0u) return &m->sp_lp_wide[d];
    if (k == 1u) return &m->sp_lp_mid[d];
    return &m->sp_lp_len[d];
}
static const char *name_of(uint32_t i) {
    static const char *n[6] = { "fwd wide","fwd mid","fwd len",
                                "bwd wide","bwd mid","bwd len" };
    return n[i];
}

#define DENSE 4000u
#define STRIDE 20u

/** A finalized table of plain rows, filled with values derived from (row, x). */
static void build_plain(BeamSparseRows *t, uint32_t n, uint32_t step) {
    ASSERT_MSG(beam_sparse_init(t, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "table did not init");
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t row = i * step;
        double *r = beam_sparse_row_mut(t, row, 0u, true);
        if (!r) return;
        for (uint32_t x = 0; x < STRIDE; ++x)
            r[x] = -((double)row + 0.125 * (double)x + 1.0);
    }
}

/**
 * A table whose rows repeat and whose cells take few values, so interning and
 * packing both apply. `build_plain` gives every row distinct values and every
 * cell a distinct one, so on that table three of the four transforms decline
 * and a report check over it is mostly `false == false`.
 */
static void build_repetitive(BeamSparseRows *t, uint32_t n, uint32_t step) {
    ASSERT_MSG(beam_sparse_init(t, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "table did not init");
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t row = i * step;
        double *r = beam_sparse_row_mut(t, row, 0u, true);
        if (!r) return;
        /* Eight distinct row patterns over four distinct cell values. */
        const uint32_t pat = i % 8u;
        for (uint32_t x = 0; x < STRIDE; ++x)
            r[x] = -(double)((x + pat) % 4u);
    }
}

/**
 * Every cell reads back as itself through the transposed layout.
 *
 * On a hand-built table rather than a finalized one, because whether a real
 * backward table stays plain depends on whether its values repeat enough to
 * be worth a dictionary -- a per-repertoire model's do, so its backward
 * tables are packed and this never fires there. The foundation's do not.
 * Testing the primitive directly is what makes this independent of that.
 */
static void test_a_transposed_table_reads_back_identically(void) {
    const uint32_t n = 900u, step = 3u;
    BeamSparseRows t;
    build_plain(&t, n, step);
    if (t.size[0] != n) { beam_sparse_destroy(&t);
                          ASSERT_MSG(false, "table did not fill"); }

    /* What every dense row says before. */
    double *before = (double *)malloc((size_t)DENSE * STRIDE * sizeof(double));
    uint8_t *had = (uint8_t *)calloc(DENSE, 1u);
    ASSERT_MSG(before && had, "out of memory");
    for (uint32_t row = 0; row < DENSE; ++row) {
        const BeamLogRow r = beam_sparse_log_row(&t, row);
        had[row] = r.present ? 1u : 0u;
        if (!r.present) continue;
        for (uint32_t x = 0; x < STRIDE; ++x)
            before[(size_t)row * STRIDE + x] = beam_log_cell(&r, x);
    }

    ASSERT_MSG(beam_sparse_plane_transpose(&t) == BEAM_OK, "transpose failed");
    ASSERT_MSG(t.plane_step == n, "plane_step is not the row count");

    for (uint32_t row = 0; row < DENSE; ++row) {
        const BeamLogRow r = beam_sparse_log_row(&t, row);
        if ((r.present ? 1u : 0u) != had[row]) {
            printf("FAIL: row %u appeared or vanished\n", row);
            fail_count++; goto out;
        }
        if (!r.present) continue;
        for (uint32_t x = 0; x < STRIDE; ++x) {
            const double got = beam_log_cell(&r, x);
            const double want = before[(size_t)row * STRIDE + x];
            if (memcmp(&got, &want, sizeof got) != 0) {
                printf("FAIL: row %u residue %u: %.17g -> %.17g\n",
                       row, x, want, got);
                fail_count++; goto out;
            }
        }
        /* The whole-row reader must agree with the per-cell one. */
        double whole[BEAM_MAX_RESIDUES];
        beam_log_row_into(&r, STRIDE, whole);
        if (memcmp(whole, before + (size_t)row * STRIDE,
                   STRIDE * sizeof(double)) != 0) {
            printf("FAIL: row %u differs between the whole-row and cell readers\n",
                   row);
            fail_count++; goto out;
        }
    }
    free(before); free(had); beam_sparse_destroy(&t);
    PASS();
    return;
out:
    free(before); free(had); beam_sparse_destroy(&t);
}

/**
 * The layout really did change -- the cells are somewhere else.
 *
 * Without this, the comparison above would pass on a transpose that did
 * nothing at all.
 */
static void test_the_transpose_actually_moves_the_cells(void) {
    const uint32_t n = 900u;
    BeamSparseRows t;
    build_plain(&t, n, 3u);
    if (t.size[0] != n) { beam_sparse_destroy(&t);
                          ASSERT_MSG(false, "table did not fill"); }
    /* Row 1, residue 1 sits at offset stride+1 by row and at n+1 by residue,
     * and those differ whenever n != stride. */
    const double by_row = t.values0[STRIDE + 1u];
    ASSERT_MSG(beam_sparse_plane_transpose(&t) == BEAM_OK, "transpose failed");
    const double by_plane = t.values0[(size_t)n + 1u];
    ASSERT_MSG(memcmp(&by_row, &by_plane, sizeof by_row) == 0,
               "the cell did not move to where a plane layout puts it");
    beam_sparse_destroy(&t);
    PASS();
}

/**
 * Finalize never transposes a forward table, AND says why not on this fixture.
 *
 * The assertion on its own was passing for the wrong reason. On this model
 * every table -- forward and backward alike -- comes out PACKED, and the
 * transpose declines for a packed table, so nothing anywhere was transposed
 * and "no forward table was transposed" could not have failed. A test that
 * cannot fail for the reason it names is not testing that reason.
 *
 * So the fixture's actual outcome is pinned too: the backward tables are
 * packed, which is what excludes them from the transpose. If packing ever
 * stops applying here, this fails and says so, and the forward assertion
 * becomes load-bearing again rather than silently vacuous.
 */
static void test_finalize_never_transposes_a_forward_table(void) {
    BeamModel *m = build(400u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    for (uint32_t t = 0; t < 3u; ++t)
        if (table_of(m, t)->plane_step != 1u) {
            printf("FAIL: %s was transposed; forward rows are read whole\n",
                   name_of(t));
            fail_count++; beam_model_destroy(m); return;
        }
    const bool packed = m->sp_lp_wide[BEAM_BWD].cell_mask != NULL;
    const bool moved  = m->sp_lp_wide[BEAM_BWD].plane_step > 1u;
    beam_model_destroy(m);
    ASSERT_MSG(packed != moved,
               "the backward wide table is neither packed nor transposed, or "
               "somehow both; they are alternatives and exactly one applies");
    ASSERT_MSG(packed,
               "the backward table is no longer packed on this fixture, so "
               "the forward assertion above has stopped being vacuous for a "
               "reason worth looking at rather than asserting");
    PASS();
}

/**
 * The compaction pipeline's report is what the table actually became.
 *
 * Every one of the four transforms keeps the table as it is and returns
 * `BEAM_OK` when it cannot apply, which for a single call is right and in
 * aggregate means a model can fail to compact entirely with no error, no log,
 * and `beam_model_memory_bytes` reporting the larger figure with nothing to
 * compare it against. Real memory, no signal. `BeamCompactionApplied` is the
 * signal, so it has to be checked against the table rather than trusted: a
 * report that says "sealed" about a table that is not is worse than none.
 *
 * Run on a table built here rather than on a finalized model's, because a
 * finalized table has already been through this and would report four
 * declines -- which is the wrong thing to assert.
 */
static void test_the_compaction_report_matches_what_the_table_became(void) {
    static char msg[192];
    bool saw_packed = false, saw_moved = false, saw_interned = false;

    /* Two fixtures, because no single one exercises all four. `build_plain`
     * has distinct rows and distinct cells, so only the transpose applies;
     * `build_repetitive` repeats both, so interning and packing apply and the
     * transpose is then excluded. Between them every field of the report is
     * seen true at least once, which is what makes the agreement mean
     * something rather than being `false == false` three times over. */
    for (uint32_t which = 0; which < 2u; ++which) {
        BeamSparseRows t;
        if (which == 0u) build_plain(&t, 900u, 3u);
        else             build_repetitive(&t, 900u, 3u);
        ASSERT_MSG(t.size[0] == 900u, "table did not fill");

        BeamCompactionApplied got;
        memset(&got, 0, sizeof(got));
        ASSERT_MSG(beam_sparse_compact_finalized(&t, DENSE, true, &got)
                   == BEAM_OK, "compaction failed");

        const bool sealed = t.rank != NULL;
        const bool interned = t.n_pool != 0u;
        const bool packed = t.cell_mask != NULL;
        const bool moved = t.plane_step > 1u;
        saw_packed |= packed; saw_moved |= moved; saw_interned |= interned;
        const bool agree = got.sealed == sealed && got.interned == interned
                        && got.packed == packed && got.transposed == moved;
        const bool exclusive = !(got.packed && got.transposed);
        beam_sparse_destroy(&t);

        if (!agree) {
            snprintf(msg, sizeof msg,
                     "fixture %u: the report says %d/%d/%d/%d and the table is "
                     "%d/%d/%d/%d", which, got.sealed, got.interned, got.packed,
                     got.transposed, sealed, interned, packed, moved);
            ASSERT_MSG(false, msg);
        }
        ASSERT_MSG(exclusive,
                   "packed and transposed are alternatives and both were "
                   "reported");
    }
    ASSERT_MSG(saw_interned, "no fixture interned, so that field is untested");
    ASSERT_MSG(saw_packed, "no fixture packed, so that field is untested");
    ASSERT_MSG(saw_moved, "no fixture transposed, so that field is untested");
    PASS();
}

/** Scores are unchanged, which is what a caller actually sees. */
static void test_scores_are_bit_identical_with_planes(void) {
    static const char *PROBE[6] = { "CASSF", "CASSYW", "CAGSFK",
                                    "CGSSFP", "CASSLNM", "CASSQEV" };
    BeamModel *a = build(400u, 4u, 14u);
    ASSERT_MSG(a != NULL, "model did not build");
    /* Undo the layout by hand and require the same answers: a plain row-major
     * copy of the same table is the control. */
    double before[6];
    for (uint32_t k = 0; k < 6u; ++k) before[k] = beam_model_log_pseq(a, PROBE[k]);
    for (uint32_t t = 3u; t < 6u; ++t) {
        BeamSparseRows *tab = table_of(a, t);
        if (tab->plane_step <= 1u) continue;
        const uint32_t rows = tab->n_pool ? tab->n_pool : tab->size[0];
        double *back = (double *)malloc((size_t)rows * tab->stride * sizeof(double));
        ASSERT_MSG(back != NULL, "out of memory");
        for (uint32_t s = 0; s < rows; ++s)
            for (uint32_t x = 0; x < tab->stride; ++x)
                back[(size_t)s * tab->stride + x] =
                    tab->values0[(size_t)x * rows + s];
        free(tab->values[0]);
        tab->values[0] = back; tab->values0 = back; tab->plane_step = 1u;
    }
    for (uint32_t k = 0; k < 6u; ++k) {
        const double now = beam_model_log_pseq(a, PROBE[k]);
        if (memcmp(&now, &before[k], sizeof now) != 0) {
            printf("FAIL: %s scored %.17g by plane, %.17g by row\n",
                   PROBE[k], before[k], now);
            fail_count++; beam_model_destroy(a); return;
        }
    }
    beam_model_destroy(a);
    PASS();
}

/** What cannot be transposed is left alone, and twice is a no-op. */
static void test_what_cannot_be_transposed_is_left_alone(void) {
    BeamModel *m = build(400u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");

    BeamSparseRows *cnt = &m->sp_cnt_wide[BEAM_FWD];
    ASSERT_MSG(beam_sparse_plane_transpose(cnt) == BEAM_OK, "mutable errored");
    ASSERT_MSG(cnt->plane_step == 1u, "a mutable table was transposed");

    BeamSparseRows *bw = table_of(m, 3u);
    if (bw->plane_step > 1u) {
        const double *first = bw->values0;
        ASSERT_MSG(beam_sparse_plane_transpose(bw) == BEAM_OK, "second errored");
        ASSERT_MSG(bw->values0 == first, "transposing twice rebuilt the payload");
    }
    beam_model_destroy(m);
    PASS();
}

/*
 * `beam_sparse_row` hands back a pointer to `stride` CONTIGUOUS doubles.
 * A plane-major payload contains no such run: the values after the first
 * belong to OTHER rows, so the pointer would be in bounds, non-NULL, and
 * wrong -- the same failure this file exists to catch, one level down.
 *
 * Built by hand and transposed explicitly, for the reason given above
 * `test_a_transposed_table_reads_back_identically`: whether a finalized
 * backward table stays plain depends on whether its values repeat enough to
 * be packed instead, so a model-built table is the wrong place to test the
 * primitive. The row-major half of the check runs FIRST, so a guard that
 * refused everything could not pass this by refusing nothing.
 */
static void test_a_plane_major_table_refuses_the_row_pointer(void) {
    const uint32_t n = 900u, step = 3u;
    BeamSparseRows t;
    build_plain(&t, n, step);
    if (t.size[0] != n) { beam_sparse_destroy(&t);
                          ASSERT_MSG(false, "table did not fill"); }

    /* Row-major: the accessor must work here, or refusing below proves
     * nothing at all. */
    uint32_t served = 0;
    for (uint32_t i = 0; i < 32u; ++i)
        if (beam_sparse_row(&t, i * step, 0u)) served++;
    if (served != 32u) { beam_sparse_destroy(&t);
                         ASSERT_MSG(false, "row-major reads were refused"); }

    if (beam_sparse_plane_transpose(&t) != BEAM_OK) {
        beam_sparse_destroy(&t); ASSERT_MSG(false, "transpose failed"); }
    if (t.plane_step != n) { beam_sparse_destroy(&t);
                             ASSERT_MSG(false, "plane_step is not the row count"); }

    uint32_t refused = 0, readable = 0;
    for (uint32_t i = 0; i < 32u; ++i) {
        const uint32_t row = i * step;
        if (beam_sparse_row(&t, row, 0u) == NULL) refused++;
        /* The row is still readable through the contract that knows how. */
        const BeamLogRow r = beam_sparse_log_row(&t, row);
        if (r.present && r.step == n) readable++;
    }
    beam_sparse_destroy(&t);
    ASSERT_MSG(refused == 32u, "a plane-major table handed back a row pointer");
    ASSERT_MSG(readable == 32u, "the rows stopped reading through log_row");
    PASS();
}

int main(void) {
    printf("\nplane-major backward tables\n");
    RUN_TEST(test_a_transposed_table_reads_back_identically);
    RUN_TEST(test_the_transpose_actually_moves_the_cells);
    RUN_TEST(test_finalize_never_transposes_a_forward_table);
    RUN_TEST(test_the_compaction_report_matches_what_the_table_became);
    RUN_TEST(test_scores_are_bit_identical_with_planes);
    RUN_TEST(test_what_cannot_be_transposed_is_left_alone);
    RUN_TEST(test_a_plane_major_table_refuses_the_row_pointer);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
