/**
 * @file test_sparse_reserve.c
 * @brief Telling a partition its size up front must change nothing but cost.
 *
 * Growth doubles, because an incremental filler discovers rows one at a time.
 * A caller that already knows the total should not pay for that: from nothing
 * to 1,846,623 rows doubles twenty-one times, copies every row already written
 * each time, and overshoots by up to a factor of two.
 *
 * `beam_sparse_reserve` is only worth having if the table it produces is
 * indistinguishable from the one growth produces. So these tests fill the same
 * rows both ways and require the same rows, the same slots and the same bytes
 * back -- and separately require that the reserved one never grew, because a
 * reserve that quietly fell back to doubling would pass every value check
 * while saving nothing.
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

#define DENSE 40000u
#define STRIDE 20u

/** Fill `n` ascending rows with a value derived from the row id. */
static void fill(BeamSparseRows *t, uint32_t n, uint32_t step) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t row = i * step;
        double *r = beam_sparse_row_mut(t, row, 0u, true);
        if (!r) return;
        for (uint32_t x = 0; x < STRIDE; ++x) r[x] = (double)row + 0.5 * x;
    }
}

static void test_a_reserved_table_holds_what_a_grown_one_holds(void) {
    const uint32_t n = 1500u, step = 7u;
    BeamSparseRows grown, reserved;
    ASSERT_MSG(beam_sparse_init(&grown, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "grown did not init");
    ASSERT_MSG(beam_sparse_init(&reserved, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "reserved did not init");

    fill(&grown, n, step);
    ASSERT_MSG(beam_sparse_reserve(&reserved, 0u, n) == BEAM_OK, "reserve failed");
    fill(&reserved, n, step);

    if (grown.size[0] != reserved.size[0]) {
        printf("FAIL: %u rows grown, %u reserved\n", grown.size[0], reserved.size[0]);
        fail_count++; goto out;
    }
    /* Every dense row: the same presence and the same bytes. */
    for (uint32_t row = 0; row < DENSE; ++row) {
        const BeamLogRow a = beam_sparse_log_row(&grown, row);
        const BeamLogRow b = beam_sparse_log_row(&reserved, row);
        if (a.present != b.present) {
            printf("FAIL: row %u present %d grown, %d reserved\n",
                   row, (int)a.present, (int)b.present);
            fail_count++; goto out;
        }
        if (!a.present) continue;
        for (uint32_t x = 0; x < STRIDE; ++x) {
            const double va = beam_log_cell(&a, x), vb = beam_log_cell(&b, x);
            if (memcmp(&va, &vb, sizeof va) != 0) {
                printf("FAIL: row %u residue %u: %.17g vs %.17g\n", row, x, va, vb);
                fail_count++; goto out;
            }
        }
    }
    /* Slots must match too: the finalized directory hands out slot k to the
     * k-th row added, and sealing later depends on exactly that. */
    for (uint32_t row = 0; row < DENSE; ++row)
        if (grown.index[row] != reserved.index[row]) {
            printf("FAIL: row %u slot %u grown, %u reserved\n",
                   row, grown.index[row], reserved.index[row]);
            fail_count++; goto out;
        }
    PASS();
out:
    beam_sparse_destroy(&grown); beam_sparse_destroy(&reserved);
}

/**
 * The reserve must actually reserve. Growth overshoots; a reserve must land
 * on the number asked for and stay there while the rows are added.
 */
static void test_a_reserved_partition_never_grows(void) {
    const uint32_t n = 1500u;
    BeamSparseRows t;
    ASSERT_MSG(beam_sparse_init(&t, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "did not init");
    ASSERT_MSG(beam_sparse_reserve(&t, 0u, n) == BEAM_OK, "reserve failed");
    if (t.capacity[0] != n) {
        printf("FAIL: reserved %u, capacity is %u\n", n, t.capacity[0]);
        fail_count++; beam_sparse_destroy(&t); return;
    }
    const double *base = t.values[0];
    fill(&t, n, 3u);
    if (t.capacity[0] != n) {
        printf("FAIL: capacity moved to %u while filling %u rows\n",
               t.capacity[0], n);
        fail_count++; beam_sparse_destroy(&t); return;
    }
    /* Same allocation throughout: no grow means no second buffer. */
    if (t.values[0] != base) {
        printf("FAIL: the payload was reallocated during the fill\n");
        fail_count++; beam_sparse_destroy(&t); return;
    }
    if (t.size[0] != n) {
        printf("FAIL: %u rows added, expected %u\n", t.size[0], n);
        fail_count++; beam_sparse_destroy(&t); return;
    }
    beam_sparse_destroy(&t);
    PASS();
}

/**
 * The detector detects: growth really does overshoot and reallocate, so the
 * test above is measuring something.
 */
static void test_growth_overshoots_and_reallocates(void) {
    const uint32_t n = 1500u;
    BeamSparseRows t;
    ASSERT_MSG(beam_sparse_init(&t, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "did not init");
    fill(&t, 1u, 3u);
    const double *first = t.values[0];
    fill(&t, n, 3u);
    ASSERT_MSG(t.capacity[0] > n,
               "growth did not overshoot; the reserve test proves nothing");
    ASSERT_MSG(t.values[0] != first,
               "growth did not reallocate; the reserve test proves nothing");
    beam_sparse_destroy(&t);
    PASS();
}

/** A reserve never shrinks, and a nonsensical one is refused. */
static void test_what_reserve_refuses_and_ignores(void) {
    BeamSparseRows t;
    ASSERT_MSG(beam_sparse_init(&t, DENSE, 1u, STRIDE, false) == BEAM_OK,
               "did not init");
    ASSERT_MSG(beam_sparse_reserve(&t, 0u, 0u) == BEAM_OK, "zero should be a no-op");
    ASSERT_MSG(t.capacity[0] == 0u, "zero reserved something");

    ASSERT_MSG(beam_sparse_reserve(&t, 0u, 900u) == BEAM_OK, "reserve failed");
    ASSERT_MSG(t.capacity[0] == 900u, "did not reserve 900");
    ASSERT_MSG(beam_sparse_reserve(&t, 0u, 100u) == BEAM_OK, "smaller errored");
    ASSERT_MSG(t.capacity[0] == 900u, "a smaller reserve shrank the partition");

    ASSERT_MSG(beam_sparse_reserve(&t, 0u, DENSE + 1u) != BEAM_OK,
               "reserved more rows than the table can address");
    ASSERT_MSG(beam_sparse_reserve(&t, 3u, 10u) != BEAM_OK,
               "reserved on a partition that does not exist");
    beam_sparse_destroy(&t);
    PASS();
}

int main(void) {
    printf("\nexact-capacity reserve\n");
    RUN_TEST(test_a_reserved_table_holds_what_a_grown_one_holds);
    RUN_TEST(test_a_reserved_partition_never_grows);
    RUN_TEST(test_growth_overshoots_and_reallocates);
    RUN_TEST(test_what_reserve_refuses_and_ignores);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
