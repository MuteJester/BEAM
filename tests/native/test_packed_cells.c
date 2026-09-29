/**
 * @file test_packed_cells.c
 * @brief A dictionary of values must hand back the values it replaced.
 *
 * A finalized cell is `log((c + alpha * P_parent(x)) / (t + alpha))`. The
 * residues a row never saw all take `alpha * P_parent(x) / (t + alpha)`, the
 * same for every sibling under that parent with the same total, so when the
 * counts are small integers -- a model of distinct sequences, each weighed
 * once -- the values repeat heavily. Packing keeps each distinct one once and
 * gives every cell a two-byte index to it.
 *
 * The failure this guards is silent and complete. An index off by one returns
 * a real log-probability belonging to a different residue -- finite,
 * plausible, negative, and wrong. Nothing downstream can tell. So every cell
 * of every row is compared, by bits, against the prior recomputed from the
 * counts, and through BOTH accessors: the single-cell one the backward kernel
 * wants and the whole-row one the forward kernel wants, which walk the
 * structure differently and must agree. The lower levels of the prior are
 * packed the same way and are checked with the top tables.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#define BEAM_TEST_PRIOR_REFERENCE
#include "test_utils.h"

static const char AA[] = "ACDEFGHIKLMNPQRSTVWY";
static uint64_t rs;
static uint32_t rnd(uint32_t n){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17;
                                 return (uint32_t)(rs % n); }

#define N_SEQS  1500u
#define ORDER   3u
#define MAX_LEN 16u

/*
 * Random CDR3-shaped sequences, each weighed ONCE. The unit weight is what
 * makes packing apply: with the weights drawn from 1..60 instead, as this
 * fixture used to be, no table packs at order 2, 3 or 4 (measured on 7,000
 * sequences), because the totals stop repeating and so do the values. Order
 * 3 puts the top tables on the rank directory and leaves the lower levels on
 * the direct one, so both directory forms sit under a packed payload here.
 */
static BeamModel *build(bool sparse) {
    rs = 88172645463325252ull;
    char **seqs = (char **)calloc(N_SEQS, sizeof(char *));
    double *w = (double *)malloc(N_SEQS * sizeof(double));
    BeamModel *m = NULL;
    if (!seqs || !w) goto out;
    for (uint32_t k = 0; k < N_SEQS; ++k) {
        const uint32_t L = 6u + rnd(9u);
        char *s = (char *)malloc(L + 3u);
        if (!s) goto out;
        s[0] = 'C';
        for (uint32_t j = 0; j < L; ++j) s[1 + j] = AA[rnd(20u)];
        s[L + 1] = 'F'; s[L + 2] = 0;
        seqs[k] = s; w[k] = 1.0;
    }
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = ORDER; cfg.max_length = MAX_LEN; cfg.sparse = sparse;
    m = beam_model_new(&cfg);
    if (m && (beam_model_observe(m, (const char **)seqs, w, N_SEQS) != BEAM_OK
              || beam_model_finalize(m) != BEAM_OK)) {
        beam_model_destroy(m); m = NULL;
    }
out:
    if (seqs) for (uint32_t k = 0; k < N_SEQS; ++k) free(seqs[k]);
    free(seqs); free(w);
    return m;
}

/*
 * Every finalized table: the two top tables, then each direction's lower
 * levels of the prior, which finalize compacts the same way.
 */
static uint32_t n_tables(const BeamModel *m) { return 2u + 2u * m->order; }

static BeamSparseRows *table_of(BeamModel *m, uint32_t i, uint32_t *dir,
                                uint32_t *level) {
    if (i < 2u) { *dir = i; *level = m->order; return &m->sp_lp[i]; }
    const uint32_t j = i - 2u;
    *dir = j / m->order; *level = j % m->order;
    return &m->sp_lo[*dir][*level];
}

static const char *name_of(BeamModel *m, uint32_t i) {
    static char buf[48];
    uint32_t dir = 0, level = 0;
    table_of(m, i, &dir, &level);
    snprintf(buf, sizeof buf, "%s level %u%s",
             dir == BEAM_FWD ? "forward" : "backward", level,
             level == m->order ? " (top)" : "");
    return buf;
}

/**
 * Every cell is still the prior of its counts, read through the packed path.
 *
 * Finalize packs on its own, so there is no un-packed "before" left to
 * compare against -- the payload it replaced has been freed. The reference is
 * therefore the DEFINITION: recompute each cell from the counts it came from
 * and require the bits to match, through the directory, the pool and the
 * dictionary all three.
 *
 * The backward tables store their contexts reversed. That is restated here
 * rather than borrowed, so a change to the mapping has to be made in two
 * places that disagree loudly.
 */
static void test_every_cell_is_still_the_prior_of_its_counts(void) {
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");
    const uint32_t R = m->n_residues;

    uint64_t compared = 0, packed_rows = 0;
    const uint32_t packed_tops = (m->sp_lp[BEAM_FWD].cell_mask != NULL)
                               + (m->sp_lp[BEAM_BWD].cell_mask != NULL);
    for (uint32_t dir = 0; dir < 2u; ++dir)
        for (uint32_t d = 0; d < m->max_len; ++d) {
            PriorRef pr;
            ASSERT_MSG(prior_ref_build(&pr, m, dir, d), "out of memory");
            for (uint32_t t = 0; t < n_tables(m); ++t) {
                uint32_t tdir, level;
                const BeamSparseRows *tab = table_of(m, t, &tdir, &level);
                if (tdir != dir) continue;
                const uint32_t width = pr.width[level];
                for (uint32_t j = 0; j < width; ++j) {
                    const uint32_t row = d * width + j;
                    const BeamLogRow lr = beam_sparse_log_row(tab, row);
                    if (!lr.present) continue;
                    /* The counts' packing is most-distant-first both ways;
                     * the finalized backward rows are nearest-first. */
                    const uint32_t c = dir == BEAM_BWD
                        ? beam_ctx_reverse(j, level, m->n_alphabet) : j;
                    double want[BEAM_MAX_RESIDUES];
                    bool observed;
                    if (level == m->order) {
                        observed = prior_ref_top_log(&pr, c, want);
                    } else {
                        prior_ref_level_log(&pr, level, c, want);
                        observed = pr.tot[level][c] > 0.0;
                    }
                    if (!observed) {
                        printf("FAIL: %s row %u is stored but nothing was "
                               "observed there\n", name_of(m, t), row);
                        fail_count++; prior_ref_free(&pr);
                        beam_model_destroy(m); return;
                    }
                    for (uint32_t x = 0; x < R; ++x) {
                        const double got = beam_log_cell(&lr, x);
                        if (memcmp(&got, &want[x], sizeof got) != 0) {
                            printf("FAIL: %s row %u residue %u: %.17g want %.17g\n",
                                   name_of(m, t), row, x, got, want[x]);
                            fail_count++; prior_ref_free(&pr);
                            beam_model_destroy(m); return;
                        }
                    }
                    compared++;
                    if (tab->cell_mask) packed_rows++;
                }
            }
            prior_ref_free(&pr);
        }
    beam_model_destroy(m);
    ASSERT_MSG(packed_tops == 2u, "a top table was not packed: the fixture no "
               "longer does what its comment says");
    ASSERT_MSG(packed_rows > 10000u, "far too few packed rows were compared");
    ASSERT_MSG(compared > 10000u, "far too few rows were compared");
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
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    const uint32_t R = m->n_residues;
    double whole[BEAM_MAX_RESIDUES];

    uint64_t cells = 0, finite = 0;
    for (uint32_t t = 0; t < n_tables(m); ++t) {
        uint32_t dir, level;
        const BeamSparseRows *tab = table_of(m, t, &dir, &level);
        if (!tab->cell_mask) continue;
        for (uint32_t s = 0; s < tab->n_cell_rows; ++s) {
            beam_sparse_row_into(tab, s, R, whole);
            for (uint32_t x = 0; x < R; ++x) {
                const double one = beam_sparse_cell(tab, s, x);
                if (memcmp(&one, &whole[x], sizeof one) != 0) {
                    printf("FAIL: %s row %u residue %u: cell %.17g, row %.17g\n",
                           name_of(m, t), s, x, one, whole[x]);
                    fail_count++; beam_model_destroy(m); return;
                }
                cells++;
                if (isfinite(whole[x])) finite++;
            }
        }
    }
    beam_model_destroy(m);
    /* The point is only that the comparison had enough to chew on; 641,280
     * cells here. Every one must be finite: under the prior
     * no finalized cell is -inf, so a packed -inf is a lost value. */
    ASSERT_MSG(cells > 100000u, "far too few cells were compared");
    ASSERT_MSG(finite == cells, "a packed cell read back as -inf or NaN");
    PASS();
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
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
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");

    uint64_t total_cells = 0, total_dict = 0;
    for (uint32_t t = 0; t < n_tables(m); ++t) {
        uint32_t dir, level;
        BeamSparseRows *tab = table_of(m, t, &dir, &level);
        if (!tab->cell_mask) continue;

        /* By bits, sorted, rather than pairwise: the dictionaries here run
         * to tens of thousands of entries. */
        uint64_t *bits = (uint64_t *)malloc((size_t)tab->n_dict * sizeof *bits);
        if (!bits) { beam_model_destroy(m); ASSERT_MSG(false, "out of memory"); }
        for (uint32_t a = 0; a < tab->n_dict; ++a)
            memcpy(&bits[a], &tab->dict[a], sizeof bits[a]);
        qsort(bits, tab->n_dict, sizeof *bits, cmp_u64);
        for (uint32_t a = 1; a < tab->n_dict; ++a)
            if (bits[a] == bits[a - 1u]) {
                double v; memcpy(&v, &bits[a], sizeof v);
                printf("FAIL: %s dictionary holds %.17g twice\n",
                       name_of(m, t), v);
                fail_count++; free(bits); beam_model_destroy(m); return;
            }
        free(bits);
        total_cells += tab->cell_start[tab->n_cell_rows];
        total_dict += tab->n_dict;
    }
    beam_model_destroy(m);
    ASSERT_MSG(total_dict > 0u, "no dictionary was built");
    /*
     * Packing has to SAVE bytes, which is a sharper claim than "the values
     * repeat" and is the one that decides whether it should fire.
     *
     * A packed cell costs a two-byte index; the dictionary costs eight bytes
     * per distinct value. So packing pays exactly when
     *
     *     cells * 2 + dict * 8  <  cells * 8      i.e.  dict * 4 < cells * 3
     *
     * The bar is the break-even rather than a margin because the margin is
     * the fixture's, not the packer's: this one stores 641,280 cells in
     * 154,592 distinct values, about four to one, while random weights would
     * leave nothing to pack at all.
     */
    ASSERT_MSG(total_dict * 4u < total_cells * 3u,
               "packing fired without saving bytes");
    PASS();
}

/**
 * The numbers a caller receives are the dense model's, which packs nothing.
 *
 * `sparse = false` is the same model materialised row by row, bit for bit,
 * and a dense top table is never packed. So any score the packed model gives
 * that the dense one does not is an index read in the wrong place. The
 * unseen probes are there to reach the packed lower levels, where an
 * unobserved context is looked up.
 */
static void test_scores_are_those_of_the_dense_model(void) {
    static const char *PROBE[10] = {
        "CASSF", "CASSYW", "CAGSFK", "CGSSFP", "CASSLNM",
        "CTSSFRQ", "CASSQEV", "CASSFGH", "WWWWWWWWWWWWWWWW", "CPDKFRQEVHGF",
    };
    BeamModel *sp = build(true);
    BeamModel *dn = build(false);
    if (!sp || !dn) { beam_model_destroy(sp); beam_model_destroy(dn);
                      ASSERT_MSG(false, "model did not build"); }
    const bool packed = sp->sp_lp[BEAM_FWD].cell_mask != NULL
                     && sp->sp_lp[BEAM_BWD].cell_mask != NULL;
    for (uint32_t k = 0; k < 10u; ++k) {
        const double a = beam_model_log_pseq(sp, PROBE[k]);
        const double b = beam_model_log_pseq(dn, PROBE[k]);
        if (memcmp(&a, &b, sizeof a) != 0 || !isfinite(a)) {
            printf("FAIL: %s scored %.17g packed, %.17g dense\n", PROBE[k], a, b);
            fail_count++; beam_model_destroy(sp); beam_model_destroy(dn); return;
        }
    }
    beam_model_destroy(sp); beam_model_destroy(dn);
    ASSERT_MSG(packed, "the top tables were not packed: nothing was tested");
    PASS();
}

/** A mutable table is never packed, and packing twice is a no-op. */
static void test_what_cannot_be_packed_is_left_alone(void) {
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");

    BeamSparseRows *cnt = &m->sp_cnt[BEAM_FWD];
    const uint64_t cb = beam_sparse_memory_bytes(cnt);
    ASSERT_MSG(beam_sparse_pack_cells(cnt) == BEAM_OK, "mutable pack errored");
    ASSERT_MSG(cnt->cell_mask == NULL, "a mutable table was packed");
    ASSERT_MSG(beam_sparse_memory_bytes(cnt) == cb, "a mutable table resized");

    BeamSparseRows *lp = &m->sp_lp[BEAM_FWD];
    ASSERT_MSG(lp->cell_mask != NULL, "finalize did not pack the forward table");
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
    RUN_TEST(test_every_cell_is_still_the_prior_of_its_counts);
    RUN_TEST(test_both_accessors_agree);
    RUN_TEST(test_the_dictionary_holds_each_value_once);
    RUN_TEST(test_scores_are_those_of_the_dense_model);
    RUN_TEST(test_what_cannot_be_packed_is_left_alone);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
