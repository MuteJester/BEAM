/**
 * @file test_interning.c
 * @brief Rows that share storage must still hand back their own values.
 *
 * Interning stores each distinct finalized row once and points the rest at
 * it. Under the hierarchical prior a row is its counts blended with its
 * parent's row, so two rows are byte-identical exactly when they saw the same
 * counts under the same parent -- siblings that differ only in the residue the
 * parent drops. Random sequences almost never give that: 6,000 of them at
 * order 4 leave none of about 40,000 rows sharing. The fixture below is built
 * so that most rows do, which is what makes interning fire here at all.
 *
 * The failure this guards against is silent and total. A pool that maps a row
 * to the wrong entry returns twenty plausible log-probabilities belonging to
 * some other context, and nothing downstream can tell -- no NaN, no -inf, no
 * range error, just a different number. So the test compares CONTENTS rather
 * than pointers, for every row of every finalized table, the lower levels of
 * the prior included: after interning two rows SHOULD share an address, and
 * that is exactly why an address proves nothing.
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
static uint64_t rng_state;
static uint32_t rnd(uint32_t n) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17; return (uint32_t)(rng_state % n);
}

#define N_PAIRED   400u    /* every (start, step) pair over twenty residues */
#define PAIRED_LEN 12u
#define N_NOISE    150u
#define ORDER      3u

/**
 * A corpus whose rows repeat, plus N_NOISE random sequences whose rows do not.
 *
 * Each paired sequence alternates a free residue e_i with f(e_i), f a fixed
 * permutation, and the free residues step through the alphabet as
 * e_i = start + i * step. Every (start, step) pair is spelled once, so each
 * context ending (.., e_{i-1}, f(e_{i-1}), e_i) is seen exactly once and
 * predicts f(e_i): a count that depends on the nearest residue alone, as do
 * its parents' rows, so the contexts that differ further back are the same
 * bytes -- in both directions, since f is invertible. The noise perturbs the
 * contexts it touches, so the pool also holds many distinct rows and a wrong
 * pool entry has somewhere wrong to point. Every weight is one, so the counts
 * are small integers and the reference in test_utils.h is exact.
 */
static BeamModel *build(bool sparse) {
    const uint32_t n_seqs = N_PAIRED + N_NOISE;
    char **seqs = (char **)calloc(n_seqs, sizeof(char *));
    double *w = (double *)malloc(n_seqs * sizeof(double));
    BeamModel *m = NULL;
    if (!seqs || !w) goto out;
    rng_state = 88172645463325252ull;
    for (uint32_t k = 0; k < n_seqs; ++k) {
        char *s = (char *)malloc(PAIRED_LEN + 1u);
        if (!s) goto out;
        if (k < N_PAIRED) {
            const uint32_t start = k % 20u, step = k / 20u;
            for (uint32_t i = 0; i < PAIRED_LEN / 2u; ++i) {
                const uint32_t e = (start + i * step) % 20u;
                s[2u * i] = AA[e];
                s[2u * i + 1u] = AA[(7u * e + 3u) % 20u];   /* 7 is prime to 20 */
            }
            s[PAIRED_LEN] = 0;
        } else {
            const uint32_t L = 6u + rnd(PAIRED_LEN - 5u);
            for (uint32_t j = 0; j < L; ++j) s[j] = AA[rnd(20u)];
            s[L] = 0;
        }
        seqs[k] = s; w[k] = 1.0;
    }
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = ORDER; cfg.max_length = PAIRED_LEN; cfg.sparse = sparse;
    m = beam_model_new(&cfg);
    if (m && (beam_model_observe(m, (const char **)seqs, w, n_seqs) != BEAM_OK
              || beam_model_finalize(m) != BEAM_OK)) {
        beam_model_destroy(m); m = NULL;
    }
out:
    if (seqs) for (uint32_t k = 0; k < n_seqs; ++k) free(seqs[k]);
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

/** A whole finalized row, however the table stores it. */
static void read_row(const BeamSparseRows *t, uint32_t row, uint32_t R,
                     double *out, bool *present) {
    const BeamLogRow r = beam_sparse_log_row(t, row);
    *present = r.present;
    if (!r.present) return;
    for (uint32_t x = 0; x < R; ++x) out[x] = beam_log_cell(&r, x);
}

/*
 * What row `row` of a level-`level` finalized table should hold, from the
 * definition; false when nothing was observed there. Finalized rows are
 * packed most-distant-first forward and nearest-first backward, and the
 * reference takes the counts' most-distant-first packing, so the backward
 * digits are reversed here -- restated rather than borrowed, so a change to
 * the mapping has to be made in two places that disagree loudly. `pr` must
 * be built for this row's direction and depth.
 */
static bool reference_row(const PriorRef *pr, uint32_t level, uint32_t row,
                          double *out) {
    uint32_t c = row % pr->width[level];
    if (pr->dir == BEAM_BWD) c = beam_ctx_reverse(c, level, pr->m->n_alphabet);
    if (level == pr->order) return prior_ref_top_log(pr, c, out);
    prior_ref_level_log(pr, level, c, out);
    return pr->tot[level][c] > 0.0;
}

/**
 * Interning again changes no row, and something WAS interned.
 *
 * Every row, byte for byte, before and against after. Finalize interns
 * already, so a second call must leave the rows exactly as they read; the
 * counts at the end are what stop a run where nothing was pooled from
 * passing quietly.
 */
static void test_interning_changes_no_row(void) {
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    const uint32_t R = m->n_residues;
    const size_t RB = (size_t)R * sizeof(double);

    uint64_t checked = 0, shared = 0, pooled_tables = 0, pooled_tops = 0;
    for (uint32_t t = 0; t < n_tables(m); ++t) {
        uint32_t dir, level;
        BeamSparseRows *tab = table_of(m, t, &dir, &level);
        const uint32_t n = tab->n_dense_rows;

        /* A full copy of what every row says now. Contents, not pointers:
         * interning is SUPPOSED to make equal rows share an address. */
        double *before = (double *)malloc((size_t)n * RB);
        uint8_t *had = (uint8_t *)calloc(n, 1u);
        if (!before || !had) { free(before); free(had); beam_model_destroy(m);
                               ASSERT_MSG(false, "out of memory recording the rows"); }
        for (uint32_t row = 0; row < n; ++row) {
            bool present = false;
            read_row(tab, row, R, before + (size_t)row * R, &present);
            had[row] = present ? 1u : 0u;
        }

        if (beam_sparse_intern_rows(tab) != BEAM_OK) {
            free(before); free(had); beam_model_destroy(m);
            ASSERT_MSG(false, "interning failed");
        }

        double now[BEAM_MAX_RESIDUES];
        for (uint32_t row = 0; row < n; ++row) {
            bool present = false;
            read_row(tab, row, R, now, &present);
            if (present != (had[row] != 0u)) {
                printf("FAIL: %s row %u appeared or vanished\n", name_of(m, t), row);
                fail_count++; free(before); free(had);
                beam_model_destroy(m); return;
            }
            if (!present) continue;
            if (memcmp(now, before + (size_t)row * R, RB) != 0) {
                printf("FAIL: %s row %u changed value\n", name_of(m, t), row);
                fail_count++; free(before); free(had);
                beam_model_destroy(m); return;
            }
            checked++;
        }
        if (tab->n_pool) {
            pooled_tables++;
            if (level == m->order) pooled_tops++;
            shared += tab->size[0] - tab->n_pool;
        }
        free(before); free(had);
    }
    beam_model_destroy(m);

    ASSERT_MSG(checked > 1000u, "far too few rows were compared");
    ASSERT_MSG(pooled_tables > 0u, "no table was interned: nothing was tested");
    ASSERT_MSG(pooled_tops == 2u, "a top table was not interned, so the fixture "
               "no longer does what its comment says");
    ASSERT_MSG(shared > 0u, "no row ended up sharing storage");
    PASS();
}

/**
 * Every finalized row is the prior computed from its counts, whatever it is
 * stored in.
 *
 * The before/after comparison above loses its force once finalize interns on
 * its own -- there is no un-interned "before" left to compare against. So
 * this checks the whole chain against the DEFINITION instead: recompute each
 * row from the counts it came from and require the bits to match, through
 * the rank directory and the shared pool both. A row the table does not hold
 * must be one nothing was observed at, and the resolver must hand back its
 * nearest observed ancestor, which it reads from the interned lower levels.
 */
static void test_a_pooled_row_is_still_the_prior_of_its_counts(void) {
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    /* Finalize freezes the counts, and this test reads them. Restore the
     * mutable form rather than reading NULL rows and quietly comparing
     * nothing. */
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");
    const uint32_t R = m->n_residues;

    uint64_t stored = 0, resolved = 0, in_pool = 0;
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
                    double got[BEAM_MAX_RESIDUES], want[BEAM_MAX_RESIDUES];
                    bool present = false;
                    read_row(tab, row, R, got, &present);
                    const bool observed = reference_row(&pr, level, row, want);
                    if (present != observed) {
                        printf("FAIL: %s row %u is %s but was %s\n",
                               name_of(m, t), row,
                               present ? "stored" : "absent",
                               observed ? "observed" : "never observed");
                        fail_count++; prior_ref_free(&pr);
                        beam_model_destroy(m); return;
                    }
                    if (present) {
                        stored++;
                        if (tab->n_pool) in_pool++;
                    } else if (level == m->order) {
                        /* An absent top row is read through the resolver,
                         * which is what every kernel does. */
                        const BeamLogRow lr = beam_lp_row(m, dir, d, j);
                        for (uint32_t x = 0; x < R; ++x)
                            got[x] = beam_log_cell(&lr, x);
                        resolved++;
                    } else {
                        continue;
                    }
                    for (uint32_t x = 0; x < R; ++x)
                        if (memcmp(&got[x], &want[x], sizeof want[x]) != 0) {
                            printf("FAIL: %s row %u residue %u: %.17g want %.17g\n",
                                   name_of(m, t), row, x, got[x], want[x]);
                            fail_count++; prior_ref_free(&pr);
                            beam_model_destroy(m); return;
                        }
                }
            }
            prior_ref_free(&pr);
        }
    beam_model_destroy(m);
    ASSERT_MSG(stored > 1000u, "far too few stored rows were compared");
    ASSERT_MSG(in_pool > 1000u, "far too few interned rows were compared");
    ASSERT_MSG(resolved > 1000u, "far too few resolved rows were compared");
    PASS();
}

/**
 * The numbers a caller receives are the dense model's, which shares nothing.
 *
 * `sparse = false` is the same model materialised row by row, bit for bit,
 * and a dense top table is never interned. So any score the interned model
 * gives that the dense one does not is a pool entry read in the wrong place --
 * including through the lower levels, which is where an unobserved context
 * is looked up; the unseen probes below are there to take that path.
 */
static void test_scores_are_those_of_the_dense_model(void) {
    static const char *PROBE[10] = {
        "ADAFAH", "CPDKFR", "CASSYW", "CAGSFK", "CGSSFP",
        "CASSLNM", "CTSSFRQ", "CASSQEV", "WWWWWWWWWWWW", "ADCFDHEKGMHP",
    };
    BeamModel *sp = build(true);
    BeamModel *dn = build(false);
    if (!sp || !dn) { beam_model_destroy(sp); beam_model_destroy(dn);
                      ASSERT_MSG(false, "model did not build"); }
    uint32_t pooled = 0;
    for (uint32_t t = 0; t < n_tables(sp); ++t) {
        uint32_t dir, level;
        if (table_of(sp, t, &dir, &level)->n_pool) pooled++;
    }
    for (uint32_t k = 0; k < 10u; ++k) {
        const double a = beam_model_log_pseq(sp, PROBE[k]);
        const double b = beam_model_log_pseq(dn, PROBE[k]);
        if (memcmp(&a, &b, sizeof a) != 0 || !isfinite(a)) {
            printf("FAIL: %s scored %.17g interned, %.17g dense\n", PROBE[k], a, b);
            fail_count++; beam_model_destroy(sp); beam_model_destroy(dn); return;
        }
    }
    beam_model_destroy(sp); beam_model_destroy(dn);
    ASSERT_MSG(pooled > 0u, "nothing was interned: nothing was tested");
    PASS();
}

/**
 * Rows that differ are never pooled, however similar.
 *
 * The positive case above would still pass if the pool merged everything, so
 * this is the half that says it does not. Over a sample of every interned
 * table's stored rows, two rows read back equal exactly when the definition
 * says they are equal -- and the sample must contain both kinds of pair, or
 * one half of that claim was never put to the test.
 */
static void test_rows_that_differ_are_kept_apart(void) {
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");
    const uint32_t R = m->n_residues;
    const size_t RB = (size_t)R * sizeof(double);
    enum { CAP = 240 };
    double (*got)[BEAM_MAX_RESIDUES] = malloc(CAP * sizeof *got);
    double (*want)[BEAM_MAX_RESIDUES] = malloc(CAP * sizeof *want);
    if (!got || !want) { free(got); free(want); beam_model_destroy(m);
                         ASSERT_MSG(false, "out of memory"); }

    uint64_t equal_pairs = 0, distinct_pairs = 0;
    for (uint32_t t = 0; t < n_tables(m); ++t) {
        uint32_t dir, level;
        const BeamSparseRows *tab = table_of(m, t, &dir, &level);
        if (!tab->n_pool) continue;
        /* Spread the sample over the whole table, every depth included,
         * rather than taking its first rows. */
        const uint32_t step = tab->size[0] / CAP + 1u;
        const uint32_t width = tab->n_dense_rows / m->max_len;
        uint32_t n = 0, seen = 0, depth_built = UINT32_MAX;
        PriorRef pr; memset(&pr, 0, sizeof pr);
        for (uint32_t row = 0; row < tab->n_dense_rows && n < CAP; ++row) {
            bool present = false;
            read_row(tab, row, R, got[n], &present);
            if (!present || (seen++ % step) != 0u) continue;
            if (row / width != depth_built) {
                prior_ref_free(&pr);
                depth_built = row / width;
                if (!prior_ref_build(&pr, m, dir, depth_built)) {
                    free(got); free(want); beam_model_destroy(m);
                    ASSERT_MSG(false, "out of memory");
                }
            }
            reference_row(&pr, level, row, want[n]);
            n++;
        }
        prior_ref_free(&pr);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < i; ++j) {
                const bool same_got = memcmp(got[i], got[j], RB) == 0;
                const bool same_want = memcmp(want[i], want[j], RB) == 0;
                if (same_got != same_want) {
                    printf("FAIL: %s: two rows read back %s but should be %s\n",
                           name_of(m, t), same_got ? "equal" : "different",
                           same_want ? "equal" : "different");
                    fail_count++; free(got); free(want);
                    beam_model_destroy(m); return;
                }
                if (same_want) equal_pairs++; else distinct_pairs++;
            }
    }
    free(got); free(want);
    beam_model_destroy(m);
    ASSERT_MSG(equal_pairs > 100u, "too few equal pairs: sharing went untested");
    ASSERT_MSG(distinct_pairs > 1000u, "too few distinct pairs were compared");
    PASS();
}

/** A mutable table is never interned, and interning twice is a no-op. */
static void test_what_cannot_be_interned_is_left_alone(void) {
    BeamModel *m = build(true);
    ASSERT_MSG(m != NULL, "model did not build");
    ASSERT_MSG(beam_model_thaw_counts(m) == BEAM_OK, "counts did not thaw");

    BeamSparseRows *cnt = &m->sp_cnt[BEAM_FWD];
    const uint64_t cb = beam_sparse_memory_bytes(cnt);
    ASSERT_MSG(beam_sparse_intern_rows(cnt) == BEAM_OK, "mutable intern errored");
    ASSERT_MSG(cnt->pool_id == NULL && cnt->n_pool == 0u,
               "a mutable table was interned");
    ASSERT_MSG(beam_sparse_memory_bytes(cnt) == cb, "a mutable table changed size");

    BeamSparseRows *lp = &m->sp_lp[BEAM_FWD];
    ASSERT_MSG(lp->n_pool != 0u, "the forward table was not interned by finalize");
    const uint32_t *first = lp->pool_id;
    const uint32_t n_pool = lp->n_pool;
    const uint64_t bytes = beam_sparse_memory_bytes(lp);
    ASSERT_MSG(beam_sparse_intern_rows(lp) == BEAM_OK, "second intern errored");
    ASSERT_MSG(lp->pool_id == first && lp->n_pool == n_pool,
               "interning twice rebuilt the pool");
    ASSERT_MSG(beam_sparse_memory_bytes(lp) == bytes, "interning twice resized");

    beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("\nfinalized row interning\n");
    RUN_TEST(test_interning_changes_no_row);
    RUN_TEST(test_a_pooled_row_is_still_the_prior_of_its_counts);
    RUN_TEST(test_scores_are_those_of_the_dense_model);
    RUN_TEST(test_rows_that_differ_are_kept_apart);
    RUN_TEST(test_what_cannot_be_interned_is_left_alone);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
