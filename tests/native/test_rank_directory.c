/**
 * @file test_rank_directory.c
 * @brief A sealed directory answers exactly what the direct one answered.
 *
 * The rank form replaces a four-byte-per-dense-row array with about 1.25 bits
 * per row. It can do that only because a finalized table hands out slots in
 * ascending row order, so a row's slot IS the count of occupied rows before
 * it. If that ever stopped being true the lookup would not fail -- it would
 * quietly return a DIFFERENT row's twenty log-probabilities, and every score
 * built on it would be wrong by a plausible-looking amount.
 *
 * So the test is exhaustive rather than sampled: every dense row of every
 * finalized table, before and after sealing, must give the same answer --
 * the same NULL, or the same pointer to the same twenty doubles. On the
 * models here that is several million comparisons, which is cheap next to
 * what a silent mismatch would cost.
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
static const char NT[] = "ACGT";
static uint64_t rng_state = 88172645463325252ull;
static uint32_t rnd(uint32_t n) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17; return (uint32_t)(rng_state % n);
}

/**
 * Two regimes, because finalize now chooses between them.
 *
 * `sparse_donor` looks like a per-repertoire model: an amino-acid row space
 * far larger than anything a few thousand sequences can fill, so finalize
 * seals it. `saturated` uses the four-letter alphabet over a short trellis,
 * where the row space is small enough that the corpus fills most of it and
 * finalize leaves the direct directory alone.
 *
 * Both are needed. The sealed one is what per-repertoire work will actually
 * run; the unsealed one is the only way to test the transition itself, since
 * sealing is one-way and there is nothing to compare against once finalize
 * has already done it.
 */
static BeamModel *build_model(bool amino, uint32_t n_seqs, uint8_t order,
                              uint32_t max_length) {
    const char *pool = amino ? AA : NT;
    const uint32_t pool_n = amino ? 20u : 4u;
    char **seqs = (char **)malloc(n_seqs * sizeof(char *));
    double *w = (double *)malloc(n_seqs * sizeof(double));
    if (!seqs || !w) { free(seqs); free(w); return NULL; }
    rng_state = 88172645463325252ull;
    for (uint32_t k = 0; k < n_seqs; ++k) {
        const uint32_t L = (max_length > 6u) ? (5u + rnd(max_length - 5u))
                                             : (max_length - 1u);
        char *s = (char *)malloc(L + 1u);
        for (uint32_t j = 0; j < L; ++j) s[j] = pool[rnd(pool_n)];
        s[L] = 0;
        seqs[k] = s; w[k] = 1.0 + (double)rnd(50u);
    }
    BeamModelConfig cfg;
    beam_model_config_default_for(amino ? BEAM_ALPHABET_AA : BEAM_ALPHABET_NT,
                                  &cfg);
    cfg.order = order;
    cfg.max_length = max_length;
    cfg.sparse = true;
    BeamModel *m = beam_model_new(&cfg);
    if (m && (beam_model_observe(m, (const char **)seqs, w, n_seqs) != BEAM_OK
              || beam_model_finalize(m) != BEAM_OK)) {
        beam_model_destroy(m); m = NULL;
    }
    for (uint32_t k = 0; k < n_seqs; ++k) free(seqs[k]);
    free(seqs); free(w);
    return m;
}

/** A model finalize will have sealed. */
static BeamModel *sparse_donor(void) { return build_model(true, 4000u, 4u, 12u); }
/** A model finalize will have left on the direct directory. */
static BeamModel *saturated(void)    { return build_model(false, 20000u, 3u, 8u); }

/** Every finalized table of the model, so no direction or kind is skipped. */
static BeamSparseRows *table_of(BeamModel *m, uint32_t idx) {
    const uint32_t dir = idx / 3u, kind = idx % 3u;
    if (kind == 0u) return &m->sp_lp_wide[dir];
    if (kind == 1u) return &m->sp_lp_mid[dir];
    return &m->sp_lp_len[dir];
}
static const char *name_of(uint32_t idx) {
    static const char *n[6] = { "fwd wide", "fwd mid", "fwd len",
                                "bwd wide", "bwd mid", "bwd len" };
    return n[idx];
}

/** True when the threshold rule says this table should have been sealed. */
static bool should_be_sealed(const BeamSparseRows *t) {
    const uint32_t occ = t->size ? t->size[0] : 0u;
    return (uint64_t)occ * BEAM_RANK_SEAL_MAX_OCCUPANCY_DENOM
           <= (uint64_t)t->n_dense_rows;
}

/**
 * Finalize applies the threshold rule to every table, one at a time.
 *
 * Per table, not per model, and the difference is visible rather than
 * theoretical: a four-thousand-sequence donor fills under one percent of the
 * wide row space and about a fifth of the mid one, so the same model comes out
 * with a rank directory on one table and a direct directory on another. That
 * is the intended behaviour -- each table's lookup branches on its own
 * pointer -- and this is where it is written down.
 */
static void test_finalize_picks_the_representation_per_table(void) {
    uint32_t sealed = 0, direct = 0;
    for (int which = 0; which < 2; ++which) {
        BeamModel *m = which ? saturated() : sparse_donor();
        ASSERT_MSG(m != NULL, "model did not build");
        for (uint32_t t = 0; t < 6u; ++t) {
            const BeamSparseRows *tab = table_of(m, t);
            const bool want = should_be_sealed(tab);
            const bool got  = (tab->rank != NULL);
            if (got != want || (got ? tab->index != NULL : tab->rank != NULL)) {
                printf("FAIL: %s %s: %u/%u rows, sealed=%d wanted=%d\n",
                       which ? "saturated" : "donor", name_of(t),
                       tab->size ? tab->size[0] : 0u, tab->n_dense_rows,
                       (int)got, (int)want);
                fail_count++; beam_model_destroy(m); return;
            }
            if (got) sealed++; else direct++;
        }
        beam_model_destroy(m);
    }
    /* Both outcomes must actually occur, or the rule was never tested. */
    ASSERT_MSG(sealed > 0u, "nothing was sealed");
    ASSERT_MSG(direct > 0u, "nothing was left direct");
    PASS();
}

/**
 * The whole point, stated exhaustively: sealing changes no answer.
 */
static void test_sealing_changes_no_row(void) {
    uint64_t rows_checked = 0, rows_present = 0;
    uint32_t transitions = 0;
    /* The donor supplies the row volume -- millions of them, most absent --
     * and the saturated model supplies the tables that are still direct and
     * can therefore be watched making the transition. */
    for (int which = 0; which < 2; ++which) {
    BeamModel *m = which ? saturated() : sparse_donor();
    ASSERT_MSG(m != NULL, "model did not build");

    for (uint32_t t = 0; t < 6u; ++t) {
        BeamSparseRows *tab = table_of(m, t);
        const uint32_t n = tab->n_dense_rows;

        /* Record what the direct directory says for every row. */
        const double **before = (const double **)malloc((size_t)n * sizeof(double *));
        ASSERT_MSG(before != NULL, "out of memory recording the direct answers");
        for (uint32_t row = 0; row < n; ++row)
            before[row] = beam_sparse_row_final(tab, row);

        /*
         * Finalize has already applied both steps, so only a table it left
         * both direct AND un-interned can be watched making the transition.
         * An interned table declines -- correctly: its directory now names
         * pooled rows, which is exactly the ascending-slot property sealing
         * needs, so the two cannot be applied in that order.
         */
        const bool sealable = (tab->index != NULL) && (tab->n_pool == 0u);
        const uint64_t bytes_before = beam_sparse_memory_bytes(tab);
        const BeamError e = beam_sparse_seal_rank(tab);
        if (e != BEAM_OK) {
            printf("FAIL: %s did not seal (%d)\n", name_of(t), (int)e);
            fail_count++; free(before); beam_model_destroy(m); return;
        }
        if (sealable) {
            transitions++;
            if (tab->index != NULL || tab->rank == NULL) {
                printf("FAIL: %s stayed direct after a seal that should fire\n",
                       name_of(t));
                fail_count++; free(before); beam_model_destroy(m); return;
            }
            if (beam_sparse_memory_bytes(tab) >= bytes_before) {
                printf("FAIL: %s got no smaller (%llu -> %llu)\n", name_of(t),
                       (unsigned long long)bytes_before,
                       (unsigned long long)beam_sparse_memory_bytes(tab));
                fail_count++; free(before); beam_model_destroy(m); return;
            }
        } else if (tab->n_pool != 0u && tab->index != NULL && tab->rank != NULL) {
            printf("FAIL: %s is both sealed and folded\n", name_of(t));
            fail_count++; free(before); beam_model_destroy(m); return;
        }

        for (uint32_t row = 0; row < n; ++row) {
            const double *now = beam_sparse_row_final(tab, row);
            if (now != before[row]) {
                printf("FAIL: %s row %u: %p -> %p\n", name_of(t), row,
                       (const void *)before[row], (const void *)now);
                fail_count++; free(before); beam_model_destroy(m); return;
            }
            /* And the validating path must agree with the inline one. */
            if (beam_sparse_row(tab, row, 0u) != now) {
                printf("FAIL: %s row %u: checked and inline paths differ\n",
                       name_of(t), row);
                fail_count++; free(before); beam_model_destroy(m); return;
            }
            rows_checked++;
            if (now) rows_present++;
        }
        free(before);
    }
    beam_model_destroy(m);
    }

    ASSERT_MSG(rows_checked > 1000000u, "far too few rows were compared");
    ASSERT_MSG(rows_present > 0u, "no row was occupied: nothing was tested");
    ASSERT_MSG(transitions > 0u,
               "every table was sealed already: the transition was not tested");
    PASS();
}

/**
 * Scores are unchanged by sealing.
 *
 * The row comparison above is the stronger statement, but it is about
 * pointers. This one is about the numbers a caller actually receives, and it
 * is what would catch a lookup that returned the right row through the wrong
 * arithmetic.
 */
static void test_scores_are_bit_identical_after_sealing(void) {
    static const char *PROBE[8] = {
        "CASSF", "CASSYW", "CAGSFK", "CGSSFP",
        "CASSLNM", "CTSSFRQ", "CASSQEV", "CASSFGH",
    };
    BeamModel *m = saturated();
    ASSERT_MSG(m != NULL, "model did not build");

    double before[8];
    for (uint32_t k = 0; k < 8u; ++k)
        before[k] = beam_model_log_pseq(m, PROBE[k]);

    for (uint32_t t = 0; t < 6u; ++t)
        ASSERT_MSG(beam_sparse_seal_rank(table_of(m, t)) == BEAM_OK,
                   "a table did not seal");

    for (uint32_t k = 0; k < 8u; ++k) {
        const double now = beam_model_log_pseq(m, PROBE[k]);
        if (memcmp(&now, &before[k], sizeof now) != 0) {
            printf("FAIL: %s scored %.17g, was %.17g\n",
                   PROBE[k], now, before[k]);
            fail_count++; beam_model_destroy(m); return;
        }
    }
    beam_model_destroy(m);
    PASS();
}

/**
 * `beam_layer_max` reads the directory directly, so it has to keep working
 * when the directory is a bitmap.
 */
static void test_layer_max_is_unchanged_after_sealing(void) {
    BeamModel *m = saturated();
    ASSERT_MSG(m != NULL, "model did not build");

    /* Bounded by the MODEL's max_length, not by a literal. `beam_layer_max`
     * takes `row < n_dense_rows` as a caller guarantee and the finalized
     * lookup does not re-check it, so a longer length is undefined behaviour
     * rather than a -inf: it indexes the length table past its end. */
    const uint32_t ML = m->max_len;
    double before[BEAM_MAX_LENGTH * BEAM_MAX_LENGTH];
    uint32_t n = 0, finite = 0;
    for (uint32_t len = 1; len <= ML; ++len)
        for (uint32_t i = 0; i < len; ++i) {
            before[n] = beam_layer_max(m, i, len);
            if (before[n] != -INFINITY) finite++;
            n++;
        }
    ASSERT_MSG(finite > 0u, "every layer was -inf: nothing was tested");

    for (uint32_t t = 0; t < 6u; ++t)
        ASSERT_MSG(beam_sparse_seal_rank(table_of(m, t)) == BEAM_OK,
                   "a table did not seal");

    uint32_t k = 0;
    for (uint32_t len = 1; len <= ML; ++len)
        for (uint32_t i = 0; i < len; ++i, ++k) {
            const double now = beam_layer_max(m, i, len);
            if (memcmp(&now, &before[k], sizeof now) != 0) {
                printf("FAIL: layer_max len=%u layer=%u %.17g -> %.17g\n",
                       len, i, before[k], now);
                fail_count++; beam_model_destroy(m); return;
            }
        }
    beam_model_destroy(m);
    PASS();
}

/**
 * A mutable table must not be sealed, and sealing twice must not corrupt.
 *
 * Both are reported OK rather than as errors: sealing is an optimisation a
 * caller may apply blindly, so "already sealed" and "cannot be sealed" are
 * outcomes, not faults. What would be a fault is doing it anyway.
 */
static void test_what_cannot_be_sealed_is_left_alone(void) {
    BeamModel *m = saturated();
    ASSERT_MSG(m != NULL, "model did not build");
    /* Finalize freezes the counts, which releases the mutable tables this
     * case is about. Restore them so there is a mutable table to refuse. */
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");

    /* A count table is mutable. */
    BeamSparseRows *cnt = &m->sp_cnt_wide[BEAM_FWD];
    const uint64_t cnt_bytes = beam_sparse_memory_bytes(cnt);
    ASSERT_MSG(beam_sparse_seal_rank(cnt) == BEAM_OK, "mutable seal errored");
    ASSERT_MSG(cnt->rank == NULL, "a mutable table was sealed");
    ASSERT_MSG(cnt->index != NULL, "a mutable table lost its directory");
    ASSERT_MSG(beam_sparse_memory_bytes(cnt) == cnt_bytes,
               "a mutable table changed size");

    BeamSparseRows *lp = table_of(m, 0u);
    ASSERT_MSG(beam_sparse_seal_rank(lp) == BEAM_OK, "first seal failed");
    const BeamRankBlock *first = lp->rank;
    const uint64_t sealed_bytes = beam_sparse_memory_bytes(lp);
    ASSERT_MSG(beam_sparse_seal_rank(lp) == BEAM_OK, "second seal errored");
    ASSERT_MSG(lp->rank == first, "sealing twice rebuilt the directory");
    ASSERT_MSG(beam_sparse_memory_bytes(lp) == sealed_bytes,
               "sealing twice changed the size");

    beam_model_destroy(m);
    PASS();
}

/**
 * The invariant the form rests on is CHECKED, not assumed.
 *
 * Corrupt one slot so the table no longer numbers its rows in ascending
 * order, and require the seal to refuse and leave the table exactly as it
 * was. Without this the guard in `beam_sparse_seal_rank` is untested code
 * standing between a build change and silently wrong probabilities.
 */
static void test_a_table_numbered_out_of_order_is_refused(void) {
    BeamModel *m = saturated();
    ASSERT_MSG(m != NULL, "model did not build");

    /* Whichever table finalize left direct -- not table 0 by assumption,
     * because which tables clear the threshold depends on the corpus. */
    BeamSparseRows *lp = NULL;
    for (uint32_t t = 0; t < 6u && !lp; ++t)
        if (table_of(m, t)->index) lp = table_of(m, t);
    if (!lp) {
        beam_model_destroy(m);
        ASSERT_MSG(false, "no table was left direct: nothing to corrupt");
    }

    uint32_t victim = UINT32_MAX;
    for (uint32_t row = 0; row < lp->n_dense_rows; ++row)
        if (lp->index[row] != BEAM_ROW_MISSING && lp->index[row] > 0u) {
            victim = row; break;
        }
    ASSERT_MSG(victim != UINT32_MAX, "no occupied row to corrupt");

    const uint32_t saved = lp->index[victim];
    lp->index[victim] = saved + 7u;          /* no longer the running count */

    const BeamError e = beam_sparse_seal_rank(lp);
    ASSERT_MSG(e != BEAM_OK, "an out-of-order table was sealed anyway");
    ASSERT_MSG(lp->rank == NULL, "a refused seal left a rank directory");
    ASSERT_MSG(lp->index != NULL, "a refused seal freed the directory");
    ASSERT_MSG(lp->index[victim] == saved + 7u, "a refused seal edited the table");

    lp->index[victim] = saved;
    beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("\nrank directory\n");
    RUN_TEST(test_finalize_picks_the_representation_per_table);
    RUN_TEST(test_sealing_changes_no_row);
    RUN_TEST(test_scores_are_bit_identical_after_sealing);
    RUN_TEST(test_layer_max_is_unchanged_after_sealing);
    RUN_TEST(test_what_cannot_be_sealed_is_left_alone);
    RUN_TEST(test_a_table_numbered_out_of_order_is_refused);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
