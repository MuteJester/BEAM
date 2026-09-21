/**
 * @file test_frozen_counts.c
 * @brief Freezing counts must lose nothing, including what is not an integer.
 *
 * The frozen form keeps only the cells that are present, in the narrowest
 * type that holds them exactly, and implies the rest. Two things can go wrong
 * and neither announces itself: a cell can come back as a different number,
 * or a cell that was absent can come back present (or the reverse), which
 * changes a row's normalising total and therefore every probability derived
 * from it.
 *
 * So the standard here is bit patterns, not equality: `2.0` and `2.0` compare
 * equal whether one arrived through a `uint8_t` or not, but `-0.0` and `0.0`
 * do too, and only one of them is right. Every assertion below uses memcmp.
 *
 * The integer tiers are the easy case. The cases that matter are the ones
 * BEAM's contract allows and this corpus does not happen to contain:
 * fractional abundances, which `observe` accepts, and posterior
 * pseudo-counts, which are `kappa * normalise(prior)` and fractional in
 * general. Those must fall through to binary64 and survive unchanged.
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

/* The partition a thawed row belongs to, supplied by the caller the way the
 * model will. One table per direction now, so the partition IS the depth. */
typedef struct { const BeamModel *m; } PartCtx;
static uint32_t part_of(uint32_t row, void *ctx) {
    const PartCtx *c = (const PartCtx *)ctx;
    return beam_part_for_row(c->m, row);
}

static const char AA[] = "ACDEFGHIKLMNPQRSTVWY";
static uint64_t rs;
static uint32_t rnd(uint32_t n) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs % n);
}

/** `weights == NULL` means integer abundances; otherwise they are used. */
static BeamModel *build(uint32_t n_seqs, uint8_t order, uint32_t max_length,
                        bool fractional) {
    rs = 88172645463325252ull;
    char **seqs = (char **)malloc(n_seqs * sizeof(char *));
    double *w = (double *)malloc(n_seqs * sizeof(double));
    if (!seqs || !w) { free(seqs); free(w); return NULL; }
    for (uint32_t k = 0; k < n_seqs; ++k) {
        const uint32_t L = 5u + rnd(7u);
        char *s = (char *)malloc(L + 3u);
        s[0] = 'C';
        for (uint32_t j = 0; j < L; ++j) s[1 + j] = AA[rnd(20u)];
        s[L + 1] = 'F'; s[L + 2] = 0;
        seqs[k] = s;
        /* Fractional weights are what force the binary64 escape hatch. */
        w[k] = fractional ? (0.5 + (double)rnd(1000u) / 7.0)
                          : (1.0 + (double)rnd(400000u));
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

static BeamSparseRows *count_table(BeamModel *m, uint32_t i) {
    return &m->sp_cnt[i];
}
static const char *name_of(uint32_t i) {
    static const char *n[2] = { "forward", "backward" };
    return n[i];
}

/**
 * Freeze, thaw, and require every cell of every dense row to be the same
 * bits -- including the cells that were zero and are now implied by a mask.
 */
static bool round_trips(BeamModel *m, const char *what, uint64_t *saved,
                        uint64_t *before_total, uint64_t *after_total) {
    const uint32_t R = m->n_residues;
    /* Finalize freezes on its own now, so restore the mutable form: this
     * file is about the conversion itself, exercised in both directions. */
    if (beam_model_thaw_counts(m) != BEAM_OK) {
        printf("FAIL: %s could not thaw the model's counts\n", what);
        fail_count++; return false;
    }
    for (uint32_t t = 0; t < 2u; ++t) {
        BeamSparseRows *src = count_table(m, t);
        const uint32_t nr = beam_count_n_rows(m);

        BeamFrozenCounts f;
        const BeamError e = beam_frozen_freeze(src, src->n_parts, &f);
        if (e != BEAM_OK) {
            printf("FAIL: %s %s did not freeze (%d)\n", what, name_of(t), (int)e);
            fail_count++; return false;
        }
        *before_total += beam_sparse_memory_bytes(src);
        *after_total  += beam_frozen_memory_bytes(&f);

        BeamSparseRows back;
        PartCtx ctx = { m };
        const BeamError te = beam_frozen_thaw(&f, nr, src->n_parts,
                                              part_of, &ctx, &back);
        if (te != BEAM_OK) {
            printf("FAIL: %s %s did not thaw (%d)\n", what, name_of(t), (int)te);
            fail_count++; beam_frozen_destroy(&f); return false;
        }

        for (uint32_t row = 0; row < nr; ++row) {
            const uint32_t p = beam_part_for_row(m, row);
            const double *a = beam_sparse_row(src, row, p);
            const double *b = beam_sparse_row(&back, row, p);
            if ((a != NULL) != (b != NULL)) {
                printf("FAIL: %s %s row %u appeared or vanished\n",
                       what, name_of(t), row);
                fail_count++; beam_frozen_destroy(&f);
                beam_sparse_destroy(&back); return false;
            }
            if (!a) continue;
            if (memcmp(a, b, (size_t)R * sizeof(double)) != 0) {
                for (uint32_t x = 0; x < R; ++x)
                    if (memcmp(&a[x], &b[x], sizeof(double)) != 0)
                        printf("FAIL: %s %s row %u residue %u: %.17g -> %.17g\n",
                               what, name_of(t), row, x, a[x], b[x]);
                fail_count++; beam_frozen_destroy(&f);
                beam_sparse_destroy(&back); return false;
            }
        }
        *saved += 1u;
        beam_frozen_destroy(&f);
        beam_sparse_destroy(&back);
    }
    return true;
}

static void test_integer_counts_round_trip(void) {
    BeamModel *m = build(4000u, 4u, 12u, false);
    ASSERT_MSG(m != NULL, "model did not build");
    uint64_t tables = 0, before = 0, after = 0;
    const bool ok = round_trips(m, "integer", &tables, &before, &after);
    beam_model_destroy(m);
    if (!ok) return;
    ASSERT_MSG(tables == 2u, "not every table was checked");
    ASSERT_MSG(after < before, "freezing made the counts no smaller");
    printf("[%.1f MB -> %.1f MB] ", before / 1e6, after / 1e6);
    PASS();
}

/**
 * The case the corpus does not contain and the contract allows.
 *
 * `beam_model_observe` takes arbitrary positive doubles, so a row can hold
 * values no integer tier can represent. Those must reach binary64 and come
 * back bit-for-bit, or fractional abundances would be silently rounded.
 */
static void test_fractional_counts_round_trip(void) {
    BeamModel *m = build(3000u, 4u, 12u, true);
    ASSERT_MSG(m != NULL, "model did not build");
    uint64_t tables = 0, before = 0, after = 0;
    const bool ok = round_trips(m, "fractional", &tables, &before, &after);
    beam_model_destroy(m);
    if (!ok) return;
    ASSERT_MSG(tables == 2u, "not every table was checked");
    printf("[%.1f MB -> %.1f MB] ", before / 1e6, after / 1e6);
    PASS();
}

/**
 * A posterior contributes `kappa * normalise(prior)`, which is fractional
 * even when both models were built from whole counts. This is the path that
 * would break an integer-only representation on a model a user really has.
 */
static void test_posterior_counts_round_trip(void) {
    BeamModel *m = build(2500u, 4u, 12u, false);
    BeamModel *prior = build(2500u, 4u, 12u, false);
    ASSERT_MSG(m != NULL && prior != NULL, "models did not build");
    ASSERT_MSG(beam_model_posterior(m, prior, 100.0) == BEAM_OK,
               "posterior failed");
    uint64_t tables = 0, before = 0, after = 0;
    const bool ok = round_trips(m, "posterior", &tables, &before, &after);
    beam_model_destroy(m); beam_model_destroy(prior);
    if (!ok) return;
    ASSERT_MSG(tables == 2u, "not every table was checked");
    printf("[%.1f MB -> %.1f MB] ", before / 1e6, after / 1e6);
    PASS();
}

/** The rule the boundary table below was built with. */
static uint32_t part_by_seven(uint32_t row, void *ctx) {
    (void)ctx;
    return (row / 7u) % 4u;
}

/**
 * Every tier boundary, exactly, plus the values that must refuse an integer
 * tier altogether.
 *
 * Built by hand because a corpus will not reliably contain 2^32 or 2^53, and
 * those are precisely where a width choice goes wrong: 255 and 256 straddle
 * `u8`, 65,535 and 65,536 straddle `u16`, 4,294,967,295 and 4,294,967,296
 * straddle `u32`, and 2^53 is the last integer a double holds exactly. The
 * fractions alongside them must drag their whole row to binary64 rather than
 * being truncated into a narrower neighbour's tier.
 */
static void test_every_tier_boundary(void) {
    /* Four groups, one per tier, each straddling that tier's boundary. A row
     * must take the narrowest type that holds ALL of its present cells, so a
     * single member of a wider group has to widen the whole row. */
    static const double TIER[4][4] = {
        { 1.0,        200.0,      255.0,           7.0 },   /* u8      */
        { 256.0,      65535.0,    1.0,             9.0 },   /* u16     */
        { 65536.0,    4294967295.0, 3.0,        1000.0 },   /* u32     */
        { 4294967296.0, 9007199254740992.0,          /* 2^32, 2^53     */
          0.5,        1.0 / 3.0 },                          /* f64     */
    };
    /* The width each group's largest value demands: 255 needs 8 bits,
     * 65,535 needs 16, 4,294,967,295 needs 32, and the fractional group must
     * fall to the escape however large its integers are. */
    static const uint32_t WANT_WIDTH[4] = { 8u, 16u, 32u,
                                            BEAM_FROZEN_WIDTH_F64 };
    const uint32_t R = 20u, NR = 64u, NROWS = 1024u;

    BeamSparseRows t;
    ASSERT_MSG(beam_sparse_init(&t, NROWS, 4u, R, true) == BEAM_OK,
               "table did not init");
    /* Row r holds a rotation of the values, so every tier and every mask
     * shape occurs and no two rows have the same absent cells. */
    for (uint32_t r = 0; r < NR; ++r) {
        const uint32_t id = r * 7u;
        double *row = beam_sparse_row_mut(&t, id, part_by_seven(id, NULL), true);
        if (!row) { beam_sparse_destroy(&t);
                    ASSERT_MSG(false, "row did not allocate"); }
        const uint32_t want = r % 4u;
        for (uint32_t x = 0; x < R; ++x)
            row[x] = ((x + r) % 3u == 0u) ? 0.0 : TIER[want][(x + r) % 4u];
    }

    BeamFrozenCounts f;
    if (beam_frozen_freeze(&t, 4u, &f) != BEAM_OK) {
        beam_sparse_destroy(&t); ASSERT_MSG(false, "freeze failed");
    }
    BeamSparseRows back;
    if (beam_frozen_thaw(&f, NROWS, 4u, part_by_seven, NULL, &back) != BEAM_OK) {
        beam_frozen_destroy(&f); beam_sparse_destroy(&t);
        ASSERT_MSG(false, "thaw failed");
    }

    uint32_t seen_tiers = 0u;
    for (uint32_t r = 0; r < NR; ++r) {
        const uint32_t id = r * 7u, p = part_by_seven(id, NULL);
        const double *a = beam_sparse_row(&t, id, p);
        const double *b = beam_sparse_row(&back, id, p);
        if (!a || !b) {
            printf("FAIL: boundary row %u missing after the round trip\n", id);
            fail_count++; goto done;
        }
        for (uint32_t x = 0; x < R; ++x)
            if (memcmp(&a[x], &b[x], sizeof(double)) != 0) {
                printf("FAIL: boundary row %u residue %u: %.17g -> %.17g\n",
                       id, x, a[x], b[x]);
                fail_count++; goto done;
            }
        /* Frozen rows are sorted by dense id and the ids ascend with r, so
         * frozen entry r is this row. The width must be the one the row's
         * largest value needs -- or the escape, for a row an integer cannot
         * hold. */
        const uint32_t w = (f.mask_width[r] >> BEAM_FROZEN_WIDTH_SHIFT)
                         & BEAM_FROZEN_WIDTH_MASK;
        const uint32_t want_w = WANT_WIDTH[r % 4u];
        if (w != want_w) {
            printf("FAIL: boundary row %u took width %u, wanted %u\n",
                   id, w, want_w);
            fail_count++; goto done;
        }
        seen_tiers |= 1u << (r % 4u);
    }
    /* All four tiers must actually have occurred, or the boundaries that
     * matter were never encoded. */
    if (seen_tiers != 0xFu) {
        printf("FAIL: only groups 0x%x occurred; wanted all four\n", seen_tiers);
        fail_count++; goto done;
    }
    beam_frozen_destroy(&f); beam_sparse_destroy(&back); beam_sparse_destroy(&t);
    PASS();
    return;
done:
    beam_frozen_destroy(&f); beam_sparse_destroy(&back); beam_sparse_destroy(&t);
}

/**
 * Every lifecycle operation, on a model whose counts are frozen.
 *
 * This is the test that guards the integration rather than the encoding. A
 * consumer that forgets to thaw does not crash: `beam_count_row` returns NULL
 * for every row and the caller's `if (!values) continue;` skips it, so a save
 * writes an empty model and an add adds nothing. Nothing announces that. The
 * only reliable check is to run each operation on a frozen model and compare
 * against the same operation on one that was never frozen.
 *
 * Finalize freezes, so every model here arrives frozen; the control is thawed
 * by hand first. Both must agree to the bit.
 */
static void test_every_lifecycle_operation_survives_frozen_counts(void) {
    static const char *PROBE[6] = { "CASSF", "CASSYW", "CAGSFK",
                                    "CGSSFP", "CASSLNM", "CASSQEV" };
    const char *step = "";
    for (int op = 0; op < 4; ++op) {
        BeamModel *frozen = build(2500u, 4u, 12u, false);
        BeamModel *thawed = build(2500u, 4u, 12u, false);
        BeamModel *other  = build(2000u, 4u, 12u, false);
        if (!frozen || !thawed || !other) {
            printf("FAIL: models did not build\n"); fail_count++; return;
        }
        /* The control keeps its counts mutable throughout. */
        if (beam_model_thaw_counts(thawed) != BEAM_OK) {
            printf("FAIL: control did not thaw\n"); fail_count++; return;
        }
        if (!frozen->counts_frozen) {
            printf("FAIL: finalize did not freeze the counts\n");
            fail_count++; return;
        }

        BeamError ea = BEAM_OK, eb = BEAM_OK;
        switch (op) {
        case 0: step = "observe then re-finalize";
            ea = beam_model_observe(frozen, PROBE, NULL, 6u);
            eb = beam_model_observe(thawed, PROBE, NULL, 6u);
            if (ea == BEAM_OK) ea = beam_model_finalize(frozen);
            if (eb == BEAM_OK) eb = beam_model_finalize(thawed);
            break;
        case 1: step = "add";
            ea = beam_model_add(frozen, other);
            eb = beam_model_add(thawed, other);
            break;
        case 2: step = "subtract then add back";
            ea = beam_model_subtract(frozen, other);
            eb = beam_model_subtract(thawed, other);
            break;
        case 3: step = "posterior";
            ea = beam_model_posterior(frozen, other, 100.0);
            eb = beam_model_posterior(thawed, other, 100.0);
            break;
        default: break;
        }
        if (ea != eb) {
            printf("FAIL: %s returned %d frozen, %d thawed\n",
                   step, (int)ea, (int)eb);
            fail_count++; goto cleanup;
        }
        if (ea != BEAM_OK) goto cleanup;   /* both refused the same way */

        if (beam_model_finalize(frozen) != BEAM_OK
            || beam_model_finalize(thawed) != BEAM_OK) {
            printf("FAIL: %s: re-finalize failed\n", step);
            fail_count++; goto cleanup;
        }
        for (uint32_t k = 0; k < 2u; ++k) {
            const double a = beam_model_log_pseq(frozen, PROBE[k]);
            const double b = beam_model_log_pseq(thawed, PROBE[k]);
            if (memcmp(&a, &b, sizeof a) != 0) {
                printf("FAIL: %s: %s scored %.17g frozen, %.17g thawed\n",
                       step, PROBE[k], a, b);
                fail_count++; goto cleanup;
            }
        }
        /* The counts themselves, not only what they imply. */
        if (beam_model_n_nonzero_cells(frozen)
            != beam_model_n_nonzero_cells(thawed)) {
            printf("FAIL: %s: %llu nonzero cells frozen, %llu thawed\n", step,
                   (unsigned long long)beam_model_n_nonzero_cells(frozen),
                   (unsigned long long)beam_model_n_nonzero_cells(thawed));
            fail_count++; goto cleanup;
        }
    cleanup:
        beam_model_destroy(frozen); beam_model_destroy(thawed);
        beam_model_destroy(other);
        if (fail_count) return;
    }
    PASS();
}

/**
 * Saving reads the frozen counts directly, so the bytes it writes must be the
 * ones it would have written from the mutable form -- byte for byte, because
 * the saved-model baseline is compared as bytes.
 *
 * Two regions are skipped, and only these two. The provenance window holds
 * three clock readings, so two builds of identical data differ there by
 * construction; the trailing CRC covers that window and so differs with it.
 * `beam_model_file_provenance_span` is what says where the first one is, so
 * this test and `check.py` skip the same bytes for the same stated reason
 * rather than each hardcoding an offset. Everything between them is decided
 * by the corpus and is still compared.
 */
static void test_saving_a_frozen_model_writes_the_same_bytes(void) {
    BeamModel *frozen = build(2500u, 4u, 12u, false);
    BeamModel *thawed = build(2500u, 4u, 12u, false);
    ASSERT_MSG(frozen && thawed, "models did not build");
    ASSERT_MSG(beam_model_thaw_counts(thawed) == BEAM_OK, "control did not thaw");
    ASSERT_MSG(frozen->counts_frozen, "finalize did not freeze");

    const char *pa = "/tmp/beam_frozen_a.beam", *pb = "/tmp/beam_frozen_b.beam";
    ASSERT_MSG(beam_model_save(frozen, pa) == BEAM_OK, "frozen save failed");
    ASSERT_MSG(beam_model_save(thawed, pb) == BEAM_OK, "thawed save failed");

    size_t prov_off = 0, prov_len = 0;
    beam_model_file_provenance_span(&prov_off, &prov_len);

    FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb");
    ASSERT_MSG(fa && fb, "saved files did not open");
    /* The CRC is the last four bytes and is not otherwise delimited, so the
     * length has to be known before the walk to know where to stop. */
    fseek(fa, 0, SEEK_END); const long na = ftell(fa); fseek(fa, 0, SEEK_SET);
    fseek(fb, 0, SEEK_END); const long nb = ftell(fb); fseek(fb, 0, SEEK_SET);
    int same = na == nb; long n = 0, compared = 0;
    const long tail = na - (long)sizeof(uint32_t);
    for (; same; ++n) {
        const int ca = fgetc(fa), cb = fgetc(fb);
        if (ca == EOF || cb == EOF) { same = ca == cb; break; }
        if (n >= (long)prov_off && n < (long)(prov_off + prov_len)) continue;
        if (n >= tail) continue;
        if (ca != cb) { same = 0; break; }
        compared++;
    }
    fclose(fa); fclose(fb);
    remove(pa); remove(pb);
    beam_model_destroy(frozen); beam_model_destroy(thawed);
    if (na != nb) { printf("FAIL: saved files differ in length: %ld vs %ld\n",
                           na, nb); fail_count++; return; }
    if (!same) { printf("FAIL: saved bytes differ at offset %ld\n", n);
                 fail_count++; return; }
    ASSERT_MSG(compared > 1000, "the saved files were suspiciously small");
    /* A skip list that swallowed the payload would pass the comparison
     * above. The window is three 64-bit fields and the CRC is one 32-bit
     * word, and nothing else may be exempt. */
    ASSERT_MSG(prov_len == 24u, "the provenance window is not three int64s");
    ASSERT_MSG(compared == na - (long)prov_len - (long)sizeof(uint32_t),
               "more bytes were skipped than provenance and the checksum");
    PASS();
}

/**
 * Rows are identified whichever way is smaller, and both ways round-trip.
 *
 * A bitmap costs one bit per DENSE row however few are occupied; a sorted
 * list costs four bytes per OCCUPIED row. They cross at one row in
 * thirty-two, and both sides are real: the foundation occupies 57.90% and the
 * bitmap is 18x smaller, while a small model occupies a fraction of a percent
 * and a bitmap is several times LARGER -- which is what this originally
 * shipped as, and what the saved-bytes baseline caught by growing 40%.
 */
static void test_rows_are_identified_the_cheaper_way(void) {
    /* Sparse: a large row space barely touched -> a list must win.
     * Saturated: a small row space well filled -> a bitmap must win. */
    struct { const char *label; BeamModel *m; } cases[2];
    /* The helper emits sequences of seven to thirteen residues, so
     * max_length must admit them. Order 4 over a barely-touched row space
     * against order 2 over a much smaller one, well filled. */
    struct { uint32_t n; uint8_t order; uint32_t ml; } cfg[2] =
        { { 200u, 4u, 14u }, { 20000u, 2u, 14u } };
    cases[0].label = "sparse";
    cases[0].m = build(cfg[0].n, cfg[0].order, cfg[0].ml, false);
    cases[1].label = "saturated";
    cases[1].m = build(cfg[1].n, cfg[1].order, cfg[1].ml, false);

    uint32_t seen_bitmap = 0, seen_list = 0;
    for (uint32_t c = 0; c < 2u; ++c) {
        BeamModel *m = cases[c].m;
        if (!m) { printf("FAIL: %s did not build\n", cases[c].label);
                  fail_count++; goto out; }
        if (beam_model_thaw_counts(m) != BEAM_OK) {
            printf("FAIL: %s did not thaw\n", cases[c].label);
            fail_count++; goto out;
        }
        for (uint32_t t = 0; t < 2u; ++t) {
                BeamSparseRows *src = count_table(m, t);
            BeamFrozenCounts f;
            if (beam_frozen_freeze(src, src->n_parts, &f) != BEAM_OK) {
                printf("FAIL: %s %s did not freeze\n", cases[c].label, name_of(t));
                fail_count++; goto out;
            }
            if (f.n_rows == 0u) { beam_frozen_destroy(&f); continue; }

            /* Exactly one form, and it must be the smaller one. */
            const bool bm = (f.present != NULL);
            if (bm == (f.row_id != NULL)) {
                printf("FAIL: %s %s has %s form\n", cases[c].label, name_of(t),
                       bm ? "both" : "neither");
                fail_count++; beam_frozen_destroy(&f); goto out;
            }
            const uint64_t as_bitmap =
                (uint64_t)beam_frozen_bitwords(f.n_dense_rows) * sizeof(uint64_t);
            const uint64_t as_list = (uint64_t)f.n_rows * sizeof(uint32_t);
            if (bm != (as_bitmap < as_list)) {
                printf("FAIL: %s %s chose %s: bitmap %llu, list %llu\n",
                       cases[c].label, name_of(t), bm ? "bitmap" : "list",
                       (unsigned long long)as_bitmap,
                       (unsigned long long)as_list);
                fail_count++; beam_frozen_destroy(&f); goto out;
            }
            if (bm) seen_bitmap++; else seen_list++;

            /* And whichever it chose, the rows come back. */
            BeamSparseRows back;
            PartCtx ctx = { m };
            if (beam_frozen_thaw(&f, beam_count_n_rows(m), src->n_parts,
                                 part_of, &ctx, &back) != BEAM_OK) {
                printf("FAIL: %s %s did not thaw back\n",
                       cases[c].label, name_of(t));
                fail_count++; beam_frozen_destroy(&f); goto out;
            }
            for (uint32_t row = 0; row < beam_count_n_rows(m); ++row) {
                const uint32_t pp = beam_part_for_row(m, row);
                const double *a = beam_sparse_row(src, row, pp);
                const double *b = beam_sparse_row(&back, row, pp);
                if ((a != NULL) != (b != NULL)
                    || (a && memcmp(a, b, (size_t)m->n_residues * sizeof(double)))) {
                    printf("FAIL: %s %s row %u differs after the round trip\n",
                           cases[c].label, name_of(t), row);
                    fail_count++; beam_frozen_destroy(&f);
                    beam_sparse_destroy(&back); goto out;
                }
            }
            beam_frozen_destroy(&f); beam_sparse_destroy(&back);
        }
    }
    if (!seen_bitmap || !seen_list) {
        printf("FAIL: only %s occurred; both forms must be exercised\n",
               seen_bitmap ? "bitmaps" : "lists");
        fail_count++; goto out;
    }
    beam_model_destroy(cases[0].m); beam_model_destroy(cases[1].m);
    PASS();
    return;
out:
    beam_model_destroy(cases[0].m); beam_model_destroy(cases[1].m);
}

int main(void) {
    printf("\nfrozen counts\n");
    RUN_TEST(test_integer_counts_round_trip);
    RUN_TEST(test_fractional_counts_round_trip);
    RUN_TEST(test_posterior_counts_round_trip);
    RUN_TEST(test_every_tier_boundary);
    RUN_TEST(test_rows_are_identified_the_cheaper_way);
    RUN_TEST(test_every_lifecycle_operation_survives_frozen_counts);
    RUN_TEST(test_saving_a_frozen_model_writes_the_same_bytes);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
