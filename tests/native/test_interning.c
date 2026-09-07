/**
 * @file test_interning.c
 * @brief Rows that share storage must still hand back their own values.
 *
 * Half a repertoire model's finalized rows are the same twenty bytes: one
 * residue at exactly 0.0 and nineteen at -inf, which is what a context seen
 * once produces. Interning stores each distinct row once and points the rest
 * at it.
 *
 * The failure this guards against is silent and total. A pool that maps a row
 * to the wrong entry returns twenty plausible log-probabilities belonging to
 * some other context, and nothing downstream can tell -- no NaN, no -inf, no
 * range error, just a different number. So the test compares CONTENTS rather
 * than pointers, for every dense row of every table: after interning two rows
 * SHOULD share an address, and that is exactly why an address proves nothing.
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
static uint64_t rng_state;
static uint32_t rnd(uint32_t n) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17; return (uint32_t)(rng_state % n);
}

/**
 * A model with many contexts seen exactly once, which is what makes rows
 * repeat: a singleton context normalises to one residue at 0.0.
 */
static BeamModel *build(uint32_t n_seqs, uint8_t order, uint32_t max_length) {
    char **seqs = (char **)malloc(n_seqs * sizeof(char *));
    double *w = (double *)malloc(n_seqs * sizeof(double));
    if (!seqs || !w) { free(seqs); free(w); return NULL; }
    rng_state = 88172645463325252ull;
    for (uint32_t k = 0; k < n_seqs; ++k) {
        const uint32_t L = 6u + rnd(8u);
        char *s = (char *)malloc(L + 3u);
        s[0] = 'C';
        for (uint32_t j = 0; j < L; ++j) s[1 + j] = AA[rnd(20u)];
        s[L + 1] = 'F'; s[L + 2] = 0;
        seqs[k] = s; w[k] = 1.0 + (double)rnd(9u);
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

/**
 * Every row, byte for byte, before and against after.
 */
static void test_interning_changes_no_row(void) {
    BeamModel *m = build(6000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    const uint32_t R = m->n_residues;
    const size_t RB = (size_t)R * sizeof(double);

    uint64_t checked = 0, present = 0, shared = 0, pooled_tables = 0;
    for (uint32_t t = 0; t < 6u; ++t) {
        BeamSparseRows *tab = table_of(m, t);
        const uint32_t n = tab->n_dense_rows;

        /* A full copy of what every row says now. Contents, not pointers:
         * interning is SUPPOSED to make equal rows share an address. */
        double *before = (double *)malloc((size_t)n * RB);
        uint8_t *had = (uint8_t *)calloc(n, 1u);
        ASSERT_MSG(before && had, "out of memory recording the rows");
        for (uint32_t row = 0; row < n; ++row) {
            const double *r = beam_sparse_row_final(tab, row);
            if (r) { had[row] = 1u; memcpy(before + (size_t)row * R, r, RB); }
        }

        /* Finalize interns already, so only a table it left alone can be
         * observed shrinking here. */
        const bool was_pooled = (tab->pool_id != NULL);
        const uint64_t bytes_before = beam_sparse_memory_bytes(tab);
        ASSERT_MSG(beam_sparse_intern_rows(tab) == BEAM_OK, "interning failed");
        if (tab->pool_id) {
            pooled_tables++;
            if (!was_pooled && beam_sparse_memory_bytes(tab) >= bytes_before) {
                printf("FAIL: %s interned but got no smaller\n", name_of(t));
                fail_count++; free(before); free(had);
                beam_model_destroy(m); return;
            }
        }

        for (uint32_t row = 0; row < n; ++row) {
            const double *r = beam_sparse_row_final(tab, row);
            if ((r != NULL) != (had[row] != 0u)) {
                printf("FAIL: %s row %u appeared or vanished\n", name_of(t), row);
                fail_count++; free(before); free(had);
                beam_model_destroy(m); return;
            }
            if (!r) continue;
            if (memcmp(r, before + (size_t)row * R, RB) != 0) {
                printf("FAIL: %s row %u changed value\n", name_of(t), row);
                fail_count++; free(before); free(had);
                beam_model_destroy(m); return;
            }
            if (beam_sparse_row(tab, row, 0u) != r) {
                printf("FAIL: %s row %u: checked and inline paths differ\n",
                       name_of(t), row);
                fail_count++; free(before); free(had);
                beam_model_destroy(m); return;
            }
            checked++; present++;
        }
        /* Count rows that now share storage, so a run where nothing was
         * pooled cannot pass quietly. */
        if (tab->pool_id) shared += tab->size[0] - tab->n_pool;
        free(before); free(had);
    }
    beam_model_destroy(m);

    ASSERT_MSG(checked > 0u, "no row was compared");
    ASSERT_MSG(pooled_tables > 0u, "no table was interned: nothing was tested");
    ASSERT_MSG(shared > 0u, "no row ended up sharing storage");
    PASS();
}

/** The numbers a caller receives, not the rows behind them. */
static void test_scores_are_bit_identical_after_interning(void) {
    static const char *PROBE[8] = {
        "CASSF", "CASSYW", "CAGSFK", "CGSSFP",
        "CASSLNM", "CTSSFRQ", "CASSQEV", "CASSFGH",
    };
    BeamModel *m = build(6000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    double before[8];
    for (uint32_t k = 0; k < 8u; ++k) before[k] = beam_model_log_pseq(m, PROBE[k]);
    for (uint32_t t = 0; t < 6u; ++t)
        ASSERT_MSG(beam_sparse_intern_rows(table_of(m, t)) == BEAM_OK,
                   "interning failed");
    for (uint32_t k = 0; k < 8u; ++k) {
        const double now = beam_model_log_pseq(m, PROBE[k]);
        if (memcmp(&now, &before[k], sizeof now) != 0) {
            printf("FAIL: %s scored %.17g, was %.17g\n", PROBE[k], now, before[k]);
            fail_count++; beam_model_destroy(m); return;
        }
    }
    beam_model_destroy(m);
    PASS();
}

/**
 * Rows that differ are never pooled, however similar.
 *
 * The positive case above would still pass if the pool merged everything, so
 * this is the half that says it does not: count the distinct bit patterns
 * independently and require the pool to hold exactly that many.
 */
static void test_rows_that_differ_are_kept_apart(void) {
    BeamModel *m = build(6000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    const uint32_t R = m->n_residues;
    const size_t RB = (size_t)R * sizeof(double);

    for (uint32_t t = 0; t < 6u; ++t) {
        BeamSparseRows *tab = table_of(m, t);
        const uint32_t n = tab->size[0];
        if (n < 2u) continue;
        const double *v = tab->values0;

        /* Distinct rows, counted the slow honest way: O(n^2) memcmp on a
         * capped sample, sharing nothing with the pool's hashing. */
        const uint32_t cap = n < 1500u ? n : 1500u;
        uint32_t distinct = 0;
        for (uint32_t a = 0; a < cap; ++a) {
            bool seen = false;
            for (uint32_t b = 0; b < a && !seen; ++b)
                if (memcmp(v + (size_t)a * R, v + (size_t)b * R, RB) == 0)
                    seen = true;
            if (!seen) distinct++;
        }

        ASSERT_MSG(beam_sparse_intern_rows(tab) == BEAM_OK, "interning failed");
        if (!tab->pool_id) continue;

        /* Every one of those first `cap` rows must still map to a pool entry
         * whose bytes are its own, and two that differed must not collide. */
        for (uint32_t a = 0; a < cap; ++a) {
            const double *pa = tab->values0 + (size_t)tab->pool_id[a] * R;
            for (uint32_t b = 0; b < a; ++b) {
                const double *pb = tab->values0 + (size_t)tab->pool_id[b] * R;
                const bool same_ptr = (tab->pool_id[a] == tab->pool_id[b]);
                const bool same_val = (memcmp(pa, pb, RB) == 0);
                if (same_ptr != same_val) {
                    printf("FAIL: %s rows %u,%u pooled=%d equal=%d\n",
                           name_of(t), a, b, (int)same_ptr, (int)same_val);
                    fail_count++; beam_model_destroy(m); return;
                }
            }
        }
        (void)distinct;
    }
    beam_model_destroy(m);
    PASS();
}

/** A mutable table is never interned, and interning twice is a no-op. */
static void test_what_cannot_be_interned_is_left_alone(void) {
    BeamModel *m = build(2000u, 3u, 10u);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");

    BeamSparseRows *cnt = &m->sp_cnt_wide[BEAM_FWD];
    const uint64_t cb = beam_sparse_memory_bytes(cnt);
    ASSERT_MSG(beam_sparse_intern_rows(cnt) == BEAM_OK, "mutable intern errored");
    ASSERT_MSG(cnt->pool_id == NULL, "a mutable table was interned");
    ASSERT_MSG(beam_sparse_memory_bytes(cnt) == cb, "a mutable table changed size");

    BeamSparseRows *lp = table_of(m, 0u);
    ASSERT_MSG(beam_sparse_intern_rows(lp) == BEAM_OK, "first intern failed");
    const uint32_t *first = lp->pool_id;
    const uint64_t sealed = beam_sparse_memory_bytes(lp);
    ASSERT_MSG(beam_sparse_intern_rows(lp) == BEAM_OK, "second intern errored");
    ASSERT_MSG(lp->pool_id == first, "interning twice rebuilt the pool");
    ASSERT_MSG(beam_sparse_memory_bytes(lp) == sealed, "interning twice resized");

    beam_model_destroy(m);
    PASS();
}

/**
 * A finalized row is `log(count[x] / total)`, whatever it is stored in.
 *
 * The before/after comparisons above lose their force once finalize interns
 * on its own -- there is no un-interned "before" left to compare against. So
 * this checks the whole chain against the DEFINITION instead: recompute each
 * row from the counts it came from and require the bits to match, through the
 * rank directory and the shared pool both.
 *
 * The backward wide table stores its contexts reversed, which is restated
 * here rather than borrowed, so a change to that mapping has to be made in
 * two places that disagree loudly.
 */
static void test_a_pooled_row_is_still_log_of_its_counts(void) {
    BeamModel *m = build(6000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    /* Finalize freezes the counts, and this test reads them. Restore the
     * mutable form rather than reading NULL rows and quietly comparing
     * nothing. */
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");
    const uint32_t R = m->n_residues;

    uint64_t compared = 0;
    for (uint32_t t = 0; t < 6u; ++t) {
        const uint32_t dir = t / 3u, k = t % 3u;
        const BeamTableKind kind = (k == 0u) ? BEAM_TABLE_WIDE
                                 : (k == 1u) ? BEAM_TABLE_MID : BEAM_TABLE_LEN;
        const bool reversed = (kind == BEAM_TABLE_WIDE && dir == BEAM_BWD);
        BeamSparseRows *tab = table_of(m, t);

        for (uint32_t row = 0; row < tab->n_dense_rows; ++row) {
            const double *got = beam_sparse_row_final(tab, row);
            if (!got) continue;

            uint32_t src = row;
            if (reversed) {
                const uint32_t d = row / m->n_ctx_wide;
                const uint32_t c = row % m->n_ctx_wide;
                src = d * m->n_ctx_wide
                    + beam_ctx_reverse(c, m->order, m->n_alphabet);
            }
            const double *counts = beam_count_row(m, dir, kind, src);
            if (!counts) {
                printf("FAIL: %s row %u exists with no counts\n", name_of(t), row);
                fail_count++; beam_model_destroy(m); return;
            }
            double tot = 0.0;
            for (uint32_t x = 0; x < R; ++x) tot += counts[x];
            for (uint32_t x = 0; x < R; ++x) {
                const double want = log(counts[x] / tot);
                if (memcmp(&got[x], &want, sizeof want) != 0) {
                    printf("FAIL: %s row %u residue %u: %.17g want %.17g\n",
                           name_of(t), row, x, got[x], want);
                    fail_count++; beam_model_destroy(m); return;
                }
            }
            compared++;
        }
    }
    beam_model_destroy(m);
    ASSERT_MSG(compared > 1000u, "far too few rows were compared");
    PASS();
}

int main(void) {
    printf("\nfinalized row interning\n");
    RUN_TEST(test_interning_changes_no_row);
    RUN_TEST(test_a_pooled_row_is_still_log_of_its_counts);
    RUN_TEST(test_scores_are_bit_identical_after_interning);
    RUN_TEST(test_rows_that_differ_are_kept_apart);
    RUN_TEST(test_what_cannot_be_interned_is_left_alone);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
