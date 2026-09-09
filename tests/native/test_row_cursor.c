/**
 * @file test_row_cursor.c
 * @brief The bit-walking cursor must yield exactly the rows the predicate does.
 *
 * `beam_row_cursor_*` exists to answer "which rows in this range are occupied"
 * for sixty-four rows at a time instead of one. The failure it invites is
 * off-by-one at a word or block boundary: a range starting mid-word, ending
 * mid-word, or crossing the 256-row block seam where the bitmap is split
 * across separate 40-byte structs. Such a bug drops or invents a row and the
 * caller still produces a plausible number.
 *
 * So every check here is against the DEFINITION -- `beam_sparse_present` over
 * every row of the range, in order -- rather than against a second walk that
 * could share the same mistake. Both directory forms are exercised, because
 * the cursor has a separate path for each and only one of them uses `ctz`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define DENSE  4000u
#define STRIDE 20u

/** A finalized table occupying `row` whenever `keep(row)`. */
static void build_table(BeamSparseRows *t, int (*keep)(uint32_t)) {
    ASSERT_MSG(beam_sparse_init(t, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "table did not init");
    for (uint32_t row = 0; row < DENSE; ++row) {
        if (!keep(row)) continue;
        double *r = beam_sparse_row_mut(t, row, 0u, true);
        if (!r) return;
        for (uint32_t x = 0; x < STRIDE; ++x) r[x] = -(double)(row + x);
    }
}

static int keep_sparse(uint32_t r)   { return (r % 37u) == 0u; }
static int keep_dense(uint32_t r)    { return (r % 3u) != 0u; }
static int keep_all(uint32_t r)      { (void)r; return 1; }
static int keep_none(uint32_t r)     { (void)r; return 0; }
static int keep_edges(uint32_t r)    { return r == 0u || r == 63u || r == 64u
                                           || r == 255u || r == 256u
                                           || r == DENSE - 1u; }

/** Walk `[first,end)` with the cursor and with the predicate; compare. */
static int agrees(const BeamSparseRows *t, uint32_t first, uint32_t end) {
    uint32_t want[DENSE]; uint32_t n_want = 0;
    for (uint32_t r = first; r < end; ++r)
        if (beam_sparse_present(t, r)) want[n_want++] = r;

    BeamRowCursor c; beam_row_cursor_init(&c, t, first, end);
    uint32_t got, n_got = 0;
    while (beam_row_cursor_next(&c, &got)) {
        if (n_got >= n_want) { printf("  [%u,%u) yielded too many\n", first, end);
                               return 0; }
        if (got != want[n_got]) {
            printf("  [%u,%u) item %u: got %u, want %u\n",
                   first, end, n_got, got, want[n_got]);
            return 0;
        }
        n_got++;
    }
    if (n_got != n_want) {
        printf("  [%u,%u) yielded %u rows, predicate says %u\n",
               first, end, n_got, n_want);
        return 0;
    }
    return 1;
}

/* Ranges chosen to land on and between every boundary the cursor has. */
static int sweep_ranges(const BeamSparseRows *t, const char *what) {
    static const uint32_t pts[] = { 0u, 1u, 63u, 64u, 65u, 127u, 128u, 255u,
                                    256u, 257u, 511u, 512u, 1000u, 2047u,
                                    2048u, 3999u, 4000u };
    const uint32_t n = (uint32_t)(sizeof pts / sizeof *pts);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i; j < n; ++j)
            if (!agrees(t, pts[i], pts[j])) {
                printf("  (%s)\n", what); return 0;
            }
    return 1;
}

static void run_shape(int (*keep)(uint32_t), const char *what, bool seal) {
    BeamSparseRows t;
    build_table(&t, keep);
    if (seal) {
        if (beam_sparse_seal_rank(&t) != BEAM_OK) {
            beam_sparse_destroy(&t); ASSERT_MSG(false, "seal failed"); }
        if (!t.rank) {
            /* Sealing declines above its occupancy threshold; force the point
             * of the test rather than silently checking the other path. */
            beam_sparse_destroy(&t);
            printf("SKIP (too dense to seal) ");
            return;
        }
    }
    const int ok = sweep_ranges(&t, what);
    const bool ranked = t.rank != NULL;
    beam_sparse_destroy(&t);
    ASSERT_MSG(ok, what);
    ASSERT_MSG(seal ? ranked : !ranked, "the wrong directory form was tested");
}

static void test_the_cursor_matches_the_predicate_when_sealed(void) {
    run_shape(keep_sparse, "sparse, sealed", true);
    run_shape(keep_edges,  "boundary rows only, sealed", true);
    PASS();
}

static void test_the_cursor_matches_the_predicate_when_direct(void) {
    run_shape(keep_sparse, "sparse, direct", false);
    run_shape(keep_dense,  "dense, direct", false);
    run_shape(keep_all,    "every row, direct", false);
    run_shape(keep_none,   "no rows, direct", false);
    PASS();
}

/* An empty or inverted range must yield nothing rather than walk off. */
static void test_a_degenerate_range_yields_nothing(void) {
    BeamSparseRows t;
    build_table(&t, keep_sparse);
    uint32_t got, n = 0;
    BeamRowCursor c;
    beam_row_cursor_init(&c, &t, 100u, 100u);
    while (beam_row_cursor_next(&c, &got)) n++;
    beam_row_cursor_init(&c, &t, 0u, 0u);
    while (beam_row_cursor_next(&c, &got)) n++;
    beam_sparse_destroy(&t);
    ASSERT_MSG(n == 0u, "an empty range yielded a row");
    PASS();
}

/*
 * `beam_row_cursor_slot0` must equal the number of occupied rows before a
 * row, which on a finalized table IS that row's payload slot. Checked against
 * a running count, and against the payload the row actually holds.
 */
static void test_slot0_counts_the_rows_before_it(void) {
    BeamSparseRows t;
    build_table(&t, keep_sparse);
    if (beam_sparse_seal_rank(&t) != BEAM_OK || !t.rank) {
        beam_sparse_destroy(&t); ASSERT_MSG(false, "table did not seal"); }

    uint32_t before = 0, checked = 0;
    for (uint32_t r = 0; r < DENSE; ++r) {
        if (beam_row_cursor_slot0(&t, r) != before) {
            printf("  row %u: slot0 %u, counted %u\n",
                   r, beam_row_cursor_slot0(&t, r), before);
            beam_sparse_destroy(&t); ASSERT_MSG(false, "slot0 disagrees"); }
        if (beam_sparse_present(&t, r)) {
            const BeamLogRow lr = beam_sparse_log_row(&t, r);
            if (!lr.present || lr.dense[0] != -(double)r) {
                beam_sparse_destroy(&t);
                ASSERT_MSG(false, "slot0 does not name the row's own payload"); }
            before++; checked++;
        }
    }
    beam_sparse_destroy(&t);
    ASSERT_MSG(checked > 32u, "too few rows to be meaningful");
    PASS();
}

int main(void) {
    printf("\nthe occupied-row cursor\n");
    RUN_TEST(test_the_cursor_matches_the_predicate_when_sealed);
    RUN_TEST(test_the_cursor_matches_the_predicate_when_direct);
    RUN_TEST(test_a_degenerate_range_yields_nothing);
    RUN_TEST(test_slot0_counts_the_rows_before_it);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
