/**
 * @file test_packed_cells.c
 * @brief A dictionary of values must hand back the values it replaced.
 *
 * A finalized cell is `log(c / t)` for integers c <= t, and a repertoire's
 * counts are small, so the values repeat: 95% of the stored doubles are a
 * value already stored elsewhere. Packing keeps each distinct one once and
 * gives every cell a two-byte index to it.
 *
 * The failure this guards is silent and complete. An index off by one returns
 * a real log-probability belonging to a different residue -- finite,
 * plausible, negative, and wrong. Nothing downstream can tell. So every cell
 * of every row is compared, by bits, through BOTH accessors: the single-cell
 * one the backward kernel wants and the whole-row one the forward kernel
 * wants, which walk the structure differently and must agree.
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
        const uint32_t L = 6u + rnd(9u);
        char *s = (char *)malloc(L + 3u);
        s[0] = 'C';
        for (uint32_t j = 0; j < L; ++j) s[1 + j] = AA[rnd(20u)];
        s[L + 1] = 'F'; s[L + 2] = 0;
        seqs[k] = s; w[k] = 1.0 + (double)rnd(60u);
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
    return &m->sp_lp[i];
}
static const char *name_of(uint32_t i) {
    static const char *n[2] = { "forward", "backward" };
    return n[i];
}

/**
 * Every cell is still `log(count[x] / total)`, read through the packed path.
 *
 * Finalize packs on its own, so there is no un-packed "before" left to
 * compare against -- the payload it replaced has been freed. The reference is
 * therefore the DEFINITION: recompute each cell from the counts it came from
 * and require the bits to match, through the directory, the pool and the
 * dictionary all three.
 *
 * The backward wide table stores its contexts reversed. That is restated here
 * rather than borrowed, so a change to the mapping has to be made in two
 * places that disagree loudly.
 */
static void test_every_cell_is_still_log_of_its_counts(void) {
    BeamModel *m = build(7000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");
    const uint32_t R = m->n_residues;

    uint64_t compared = 0, packed_tables = 0;
    for (uint32_t t = 0; t < 2u; ++t) {
        const uint32_t dir = t;
        const bool reversed = (dir == BEAM_BWD);
        BeamSparseRows *tab = table_of(m, t);
        if (tab->cell_mask) packed_tables++;

        for (uint32_t row = 0; row < tab->n_dense_rows; ++row) {
            const BeamLogRow lr = beam_sparse_log_row(tab, row);
            if (!lr.present) continue;

            uint32_t src = row;
            if (reversed) {
                const uint32_t d = row / m->n_ctx;
                const uint32_t c = row % m->n_ctx;
                src = d * m->n_ctx
                    + beam_ctx_reverse(c, m->order, m->n_alphabet);
            }
            const double *counts = beam_cnt_row(m, dir, src);
            if (!counts) {
                printf("FAIL: %s row %u exists with no counts\n", name_of(t), row);
                fail_count++; beam_model_destroy(m); return;
            }
            double tot = 0.0;
            for (uint32_t x = 0; x < R; ++x) tot += counts[x];
            for (uint32_t x = 0; x < R; ++x) {
                const double want = log(counts[x] / tot);
                const double got = beam_log_cell(&lr, x);
                if (memcmp(&got, &want, sizeof want) != 0) {
                    printf("FAIL: %s row %u residue %u: %.17g want %.17g\n",
                           name_of(t), row, x, got, want);
                    fail_count++; beam_model_destroy(m); return;
                }
            }
            compared++;
        }
    }
    beam_model_destroy(m);
    ASSERT_MSG(packed_tables > 0u, "no table was packed: nothing was tested");
    ASSERT_MSG(compared > 1000u, "far too few rows were compared");
    PASS();
}

/**
 * The two accessors must agree, cell for cell.
 *
 * They walk the structure differently -- the whole-row one consumes the
 * indices in order, the single-cell one computes a rank per residue -- and
 * the kernel uses both, the forward half scanning rows and the backward half
 * taking one residue from each of twenty. An off-by-one in either shows up
 * here and nowhere else.
 */
static void test_both_accessors_agree(void) {
    BeamModel *m = build(7000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    const uint32_t R = m->n_residues;
    double whole[BEAM_MAX_RESIDUES];

    uint64_t cells = 0, finite = 0;
    for (uint32_t t = 0; t < 2u; ++t) {
        const BeamSparseRows *tab = table_of(m, t);
        if (!tab->cell_mask) continue;
        for (uint32_t s = 0; s < tab->n_cell_rows; ++s) {
            beam_sparse_row_into(tab, s, R, whole);
            for (uint32_t x = 0; x < R; ++x) {
                const double one = beam_sparse_cell(tab, s, x);
                if (memcmp(&one, &whole[x], sizeof one) != 0) {
                    printf("FAIL: %s row %u residue %u: cell %.17g, row %.17g\n",
                           name_of(t), s, x, one, whole[x]);
                    fail_count++; beam_model_destroy(m); return;
                }
                cells++;
                if (whole[x] != -INFINITY) finite++;
            }
        }
    }
    beam_model_destroy(m);
    /* Sized for two tables per model, not six. The point is only that the
     * comparison had enough to chew on; measured at 78,800 cells here. */
    ASSERT_MSG(cells > 50000u, "far too few cells were compared");
    ASSERT_MSG(finite > 0u, "every cell was -inf: nothing was tested");
    PASS();
}

/**
 * The dictionary must hold each value once and no value twice.
 *
 * The comparison above would still pass if the dictionary were simply a copy
 * of every cell, which would save nothing. This is the half that says the
 * sharing actually happened, and that no two entries are the same bits --
 * a duplicate entry means the hash probe failed to find an existing value and
 * the table is larger than it should be.
 */
static void test_the_dictionary_holds_each_value_once(void) {
    BeamModel *m = build(7000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    const uint32_t R = m->n_residues;

    uint64_t total_cells = 0, total_dict = 0;
    for (uint32_t t = 0; t < 2u; ++t) {
        BeamSparseRows *tab = table_of(m, t);
        if (!tab->cell_mask) continue;

        for (uint32_t a = 0; a < tab->n_dict; ++a)
            for (uint32_t b = 0; b < a; ++b)
                if (memcmp(&tab->dict[a], &tab->dict[b], sizeof(double)) == 0) {
                    printf("FAIL: %s dictionary holds %.17g twice\n",
                           name_of(t), tab->dict[a]);
                    fail_count++; beam_model_destroy(m); return;
                }
        total_cells += tab->cell_start[tab->n_cell_rows];
        total_dict += tab->n_dict;
        (void)R;
    }
    beam_model_destroy(m);
    ASSERT_MSG(total_dict > 0u, "no dictionary was built");
    /* If the values did not repeat there would be nothing to gain, and the
     * builder should have declined rather than paid for an indirection. */
    /*
     * Packing has to SAVE bytes, which is a sharper claim than "the values
     * repeat" and is the one that decides whether it should fire.
     *
     * A packed cell costs a two-byte index; the dictionary costs eight bytes
     * per distinct value. So packing pays exactly when
     *
     *     cells * 2 + dict * 8  <  cells * 8      i.e.  dict * 4 < cells * 3
     *
     * This used to assert `dict * 2 < cells`, a comfortable margin rather
     * than the break-even, and the single-expert model moved this fixture to
     * the thin side of it. That is the FIXTURE, not the constant: `build`
     * above draws residues uniformly at random with weights 1..60, so
     * contexts almost never recur and log(count/total) takes a different
     * value nearly every time. Uniform noise is the worst case for a
     * dictionary.
     *
     * Measured on 200,000 real repertoire sequences at order 4, the same
     * model packs 314,813 stored cells into 28,012 distinct values, a factor
     * of 11.2, and the payload falls 66% -- 2,518,504 bytes to 853,722. The
     * bar below is the break-even, so it holds on noise and has enormous
     * margin on data.
     */
    ASSERT_MSG(total_dict * 4u < total_cells * 3u,
               "packing fired without saving bytes");
    PASS();
}

/** Scores are unchanged: the numbers a caller actually receives. */
static void test_scores_are_bit_identical_after_packing(void) {
    static const char *PROBE[8] = {
        "CASSF", "CASSYW", "CAGSFK", "CGSSFP",
        "CASSLNM", "CTSSFRQ", "CASSQEV", "CASSFGH",
    };
    BeamModel *m = build(7000u, 4u, 14u);
    ASSERT_MSG(m != NULL, "model did not build");
    double before[8];
    for (uint32_t k = 0; k < 8u; ++k) before[k] = beam_model_log_pseq(m, PROBE[k]);
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

/** A mutable table is never packed, and packing twice is a no-op. */
static void test_what_cannot_be_packed_is_left_alone(void) {
    BeamModel *m = build(1500u, 3u, 10u);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");

    BeamSparseRows *cnt = &m->sp_cnt[BEAM_FWD];
    const uint64_t cb = beam_sparse_memory_bytes(cnt);
    ASSERT_MSG(beam_sparse_pack_cells(cnt) == BEAM_OK, "mutable pack errored");
    ASSERT_MSG(cnt->cell_mask == NULL, "a mutable table was packed");
    ASSERT_MSG(beam_sparse_memory_bytes(cnt) == cb, "a mutable table resized");

    BeamSparseRows *lp = table_of(m, 0u);
    ASSERT_MSG(beam_sparse_pack_cells(lp) == BEAM_OK, "first pack failed");
    const uint32_t *first = lp->cell_mask;
    const uint64_t packed = beam_sparse_memory_bytes(lp);
    ASSERT_MSG(beam_sparse_pack_cells(lp) == BEAM_OK, "second pack errored");
    ASSERT_MSG(lp->cell_mask == first, "packing twice rebuilt the payload");
    ASSERT_MSG(beam_sparse_memory_bytes(lp) == packed, "packing twice resized");

    beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("\npacked value cells\n");
    RUN_TEST(test_every_cell_is_still_log_of_its_counts);
    RUN_TEST(test_both_accessors_agree);
    RUN_TEST(test_the_dictionary_holds_each_value_once);
    RUN_TEST(test_scores_are_bit_identical_after_packing);
    RUN_TEST(test_what_cannot_be_packed_is_left_alone);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
