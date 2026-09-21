/**
 * @file test_model.c
 * @brief Tests for BEAM — Ensemble of Dual Graph Experts.
 *
 * The checks that carry weight here are the ones that would fail if the two
 * anchors were confused with each other:
 *
 *   - the context packings are asserted against hand-computed indices, in both
 *     directions, because swapping them yields a model that trains, scores and
 *     normalises perfectly while being wrong. Reading the DAG state forwards
 *     instead of backwards leaves every other test in this file green;
 *   - the direct scorer and the DAG's own edge weights are required to agree,
 *     because otherwise the analytics would describe a different model from
 *     the one being scored;
 *   - the partition function is checked by exhaustive enumeration, which is
 *     the only test here that cannot be passed by accident.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/crc32c.h"
#include "beam/crc32c.h"

/*
 * The context packings and the DAG's edge weights are internal, and they are
 * exactly the things this file exists to pin down: a swapped packing produces
 * a model that trains, normalises and scores perfectly while being wrong, and
 * no public entry point can distinguish it. Reaching into the implementation
 * header is the price of testing them at all.
 */
#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define A   BEAM_AA_RADIX
#define R   BEAM_AA_RESIDUES
#define NON BEAM_AA_NONE

/* ── A small, CDR3-shaped training corpus ────────────────────── */

#define N_TRAIN 12
static const char *TRAIN[N_TRAIN] = {
    "CASSLGEQYF",  "CASSLGEQFF",  "CASSQGEQYF",  "CASSPGEQYF",
    "CASSLAEQYF",  "CASSLGETQYF", "CASSVGEQYF",  "CASSLGNTIYF",
    "CASSLREQYF",  "CASSLGEQYFF", "CASSDGEQYF",  "CASSLGEQY",
};
static const double WEIGHT[N_TRAIN] = {
    50, 30, 20, 15, 12, 25, 18, 22, 9, 7, 11, 6
};

/** True when the thread-local error context carries a non-empty message. */
static bool beam_model_error_is_reported(void) {
    return beam_error_message()[0] != '\0';
}

/* ── Injected allocation failure ─────────────────────────────── */

/*
 * `finalize` builds a complete replacement table set before releasing the one
 * the model already has, so that a failure part-way through leaves a model
 * that still scores rather than one stripped of its tables with `finalized`
 * clear. That is the whole reason the function is shaped the way it is, and
 * nothing exercises it: allocation does not fail on a machine with 125 GB free.
 *
 * The link line wraps `malloc` (see the Makefile) so a test can arm the Nth
 * call to fail. The wrapper is inert until armed, so every other test in this
 * binary allocates normally.
 */
void *__real_malloc(size_t n);
static long g_alloc_fail_at = -1;      /* -1 = never fail */

void *__wrap_malloc(size_t n) {
    if (g_alloc_fail_at >= 0) {
        if (g_alloc_fail_at == 0) return NULL;
        g_alloc_fail_at--;
    }
    return __real_malloc(n);
}

static BeamModel *build(uint8_t order, uint8_t max_length) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = max_length;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/* The open world, asked for explicitly.
 *
 * `build` above takes the defaults, and the default is closed-world: the
 * model holds only what TRAIN spells. A few properties below belong to the
 * smoother rather than to the grammar -- a finite score for an unseen ending,
 * a support of 20^L, a posterior that can move a score it would otherwise
 * leave at -inf -- and those need backoff switched on. */
static BeamModel *build_open(uint8_t order, uint8_t max_length) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = max_length;
    cfg.strict = false;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

static const char *STRICT_TRAIN[] = { "CASSF", "CASSY", "CTTGW" };

static BeamModel *build_strict_mode(bool sparse) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 2;
    cfg.max_length = 6;
    cfg.strict = true;
    cfg.sparse = sparse;

    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, STRICT_TRAIN, NULL, 3) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/* ── Configuration ───────────────────────────────────────────── */

static void test_defaults_are_the_measured_ones(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    ASSERT_MSG(c.order == 4, "default order should be 4");
    ASSERT_MSG(c.weight_forward == c.weight_backward,
               "the two anchors are weighted equally by default");
    ASSERT_MSG(!c.sparse, "the low-level C default remains dense");
    PASS();
}

static void test_provenance_is_recorded_and_aggregated(void) {
    /* Three facts a finished model cannot recompute about itself. The build
     * time has to AGGREGATE: a model observed in chunks must report what the
     * whole build cost, not what the last chunk did. */
    const char *s[] = {"CASSLGEQYF", "CASSQDTQYF", "CASSLGTEAFF"};
    double w[] = {3.0, 1.0, 2.0};
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.max_length = 12u; c.order = 3u;
    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m != NULL, "model did not build");

    int64_t created = 0, obs0 = 0, fin0 = 0;
    ASSERT_MSG(beam_model_provenance(m, &created, &obs0, &fin0) == BEAM_OK,
               "provenance");
    ASSERT_MSG(created > 1600000000, "a creation stamp from this century");
    ASSERT_MSG(fin0 == 0, "nothing has been finalized yet");

    ASSERT_MSG(beam_model_observe(m, s, w, 3u) == BEAM_OK, "observe");
    int64_t obs1 = 0;
    beam_model_provenance(m, NULL, &obs1, NULL);
    ASSERT_MSG(beam_model_observe(m, s, w, 3u) == BEAM_OK, "observe again");
    int64_t obs2 = 0;
    beam_model_provenance(m, NULL, &obs2, NULL);
    ASSERT_MSG(obs2 >= obs1, "a second observe cannot shrink the total");

    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");
    int64_t created2 = 0, fin1 = 0;
    beam_model_provenance(m, &created2, NULL, &fin1);
    ASSERT_MSG(created2 == created, "the creation stamp does not move");
    ASSERT_MSG(fin1 > 0, "finalizing was timed");

    /* NULL means "do not want it", not "no answer". */
    ASSERT_MSG(beam_model_provenance(m, NULL, NULL, NULL) == BEAM_OK,
               "every output is optional");
    beam_model_destroy(m);
    PASS();
}

static void test_depth_occupancy_sums_to_the_nonzero_cells(void) {
    /* The per-depth breakdown and the scalar count are two readings of the
     * same tables, so they must agree exactly. That is what makes the
     * breakdown usable as a denominator: it is not an estimate. */
    const char *s[] = {"CASSLGEQYF", "CASSQDTQYF", "CASSLGTEAFF"};
    double w[] = {3.0, 1.0, 2.0};
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.max_length = 12u; c.order = 3u;
    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m != NULL, "model did not build");

    uint64_t rows[2u * 12u], cells[2u * 12u];
    ASSERT_MSG(beam_model_depth_occupancy(m, rows, cells) == BEAM_ERR_NOT_BUILT,
               "an unfinalized model has no tables to count");

    ASSERT_MSG(beam_model_observe(m, s, w, 3u) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");
    ASSERT_MSG(beam_model_depth_occupancy(m, rows, cells) == BEAM_OK,
               "depth occupancy");

    uint64_t tr = 0, tc = 0;
    for (uint32_t i = 0; i < 2u * 12u; ++i) {
        tr += rows[i];
        tc += cells[i];
        ASSERT_MSG(cells[i] >= rows[i],
                   "an occupied row holds at least one live cell");
        ASSERT_MSG(cells[i] <= rows[i] * 20u,
                   "and at most one per residue");
    }
    ASSERT_MSG(tc == beam_model_n_nonzero_cells(m),
               "the per-depth cells must sum to n_nonzero_cells exactly");
    ASSERT_MSG(tr > 0, "the model observed something");

    /* No sequence is longer than max_length, so the deepest depth a
     * length-12 model can reach is 11 and slot 11 of each direction is the
     * last that may be occupied. */
    ASSERT_MSG(rows[11u] <= tr && rows[12u + 11u] <= tr, "in range");
    beam_model_destroy(m);
    PASS();
}

static void test_single_anchor_weights_are_accepted(void) {
    /* The anchor weights SHOULD sum to 1 and are deliberately not refused
     * when they do not. Scoring one anchor's chain alone is the only way to
     * separate a partition function below 1 into its two causes: the two
     * chains disagreeing, and the closed world leaking mass at rows it never
     * observed. A refusal here would make that unmeasurable. */
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.weight_forward = 1.0; c.weight_backward = 0.0;
    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m != NULL, "a single-anchor configuration must build");
    beam_model_destroy(m);
    beam_model_config_default(&c);
    c.order = 5;
    ASSERT_MSG(beam_model_new(&c) == NULL, "order 5 should be rejected");
    PASS();
}

static void test_depth_cap_is_clamped(void) {
    /* max_length 12 still admits every training sequence, but sits below the
     * default depth cap of 15. */
    BeamModel *m = build(4, 12);
    ASSERT_MSG(m != NULL, "a max_length below the default depth cap must work");
    beam_model_destroy(m);
    PASS();
}

static void test_sparse_requires_strict(void) {
    /* An omitted count row is a dead probability row only in the closed
     * world. The open world's add-one gives every row mass, including rows
     * nothing was ever observed at, so a sparse table would have nowhere to
     * put them and the scores would silently change. */
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.strict = false;
    c.sparse = true;
    ASSERT_MSG(beam_model_new(&c) == NULL,
               "the open world cannot omit zero-count rows");

    c.strict = true;

    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m != NULL, "strict zero-backoff sparse storage must be valid");
    ASSERT_MSG(beam_model_config(m)->sparse,
               "the resolved C configuration must report sparse storage");
    beam_model_destroy(m);
    PASS();
}

/* ── Sparse row storage, directly ───────────────────────────── */

static void test_sparse_rows_boundaries_and_erase(void) {
    BeamSparseRows rows = {0};
    ASSERT_MSG(beam_sparse_init(&rows, 8u, 2u, R, true) == BEAM_OK,
               "initialise mutable sparse rows");
    ASSERT_MSG(beam_sparse_n_rows(&rows) == 0u, "the store starts empty");
    for (uint32_t row = 0; row < 8u; ++row)
        ASSERT_MSG(beam_sparse_row(&rows, row, row & 1u) == NULL,
                   "every directory entry starts missing");

    double *r0 = beam_sparse_row_mut(&rows, 0u, 0u, true);
    ASSERT_MSG(r0 != NULL, "the first logical row is insertable");
    for (uint32_t x = 0; x < R; ++x)
        ASSERT_MSG(r0[x] == 0.0, "a new payload row is exactly zeroed");
    r0[3] = 3.25;

    double *r2 = beam_sparse_row_mut(&rows, 2u, 0u, true);
    double *r7 = beam_sparse_row_mut(&rows, 7u, 1u, true);
    ASSERT_MSG(r2 && r7, "interior and final logical rows are insertable");
    r2[4] = 24.0;
    r7[19] = 719.0;
    ASSERT_MSG(beam_sparse_n_rows(&rows) == 3u,
               "occupancy counts rows rather than cells");
    ASSERT_MSG(beam_sparse_row(&rows, 0u, 0u)[3] == 3.25,
               "the first boundary row round-trips");
    ASSERT_MSG(beam_sparse_row(&rows, 7u, 1u)[19] == 719.0,
               "the final boundary row round-trips");
    ASSERT_MSG(beam_sparse_row(&rows, 7u, 0u) == NULL,
               "a mutable row cannot be read through the wrong partition");
    ASSERT_MSG(beam_sparse_row_mut(&rows, 8u, 1u, true) == NULL,
               "one-past-the-end insertion is refused");
    ASSERT_MSG(beam_sparse_row_mut(&rows, 0u, 2u, true) == NULL,
               "one-past-the-end partition is refused");

    /* Row 2 is not the only row in partition 0. Erasing it swap-moves row 0,
     * so this checks both the deletion and the moved row's directory repair. */
    ASSERT_MSG(beam_sparse_erase(&rows, 2u, 0u), "erase an occupied row");
    ASSERT_MSG(beam_sparse_row(&rows, 2u, 0u) == NULL,
               "an erased row becomes missing");
    ASSERT_MSG(beam_sparse_row(&rows, 0u, 0u)[3] == 3.25,
               "swap-delete rewrites the moved row's directory entry");
    ASSERT_MSG(beam_sparse_n_rows(&rows) == 2u,
               "occupancy shrinks after erase");
    ASSERT_MSG(!beam_sparse_erase(&rows, 2u, 0u),
               "erasing a missing row is a clean no-op");

    const uint64_t before = beam_sparse_memory_bytes(&rows);
    beam_sparse_shrink(&rows);
    ASSERT_MSG(beam_sparse_memory_bytes(&rows) <= before,
               "shrink never increases allocation");
    ASSERT_MSG(beam_sparse_row(&rows, 0u, 0u)[3] == 3.25
               && beam_sparse_row(&rows, 7u, 1u)[19] == 719.0,
               "shrinking preserves every surviving value");
    beam_sparse_destroy(&rows);

    BeamSparseRows finalized = {0};
    ASSERT_MSG(beam_sparse_init(&finalized, 8u, 1u, R, false) == BEAM_OK,
               "initialise immutable finalized rows");
    double *last = beam_sparse_row_mut(&finalized, 7u, 0u, true);
    ASSERT_MSG(last != NULL, "finalization may populate an immutable store");
    last[0] = -7.0;
    ASSERT_MSG(!beam_sparse_erase(&finalized, 7u, 0u),
               "finalized rows cannot be erased");
    ASSERT_MSG(beam_sparse_row(&finalized, 7u, 0u)[0] == -7.0,
               "a refused erase leaves immutable data intact");
    beam_sparse_destroy(&finalized);
    PASS();
}

static void test_sparse_growth_failure_is_atomic(void) {
    BeamSparseRows rows = {0};
    ASSERT_MSG(beam_sparse_init(&rows, 16u, 1u, R, true) == BEAM_OK,
               "initialise sparse rows");
    for (uint32_t row = 0; row < 8u; ++row) {
        double *values = beam_sparse_row_mut(&rows, row, 0u, true);
        ASSERT_MSG(values != NULL, "fill the initial payload capacity");
        values[row] = (double)row + 0.5;
    }

    g_alloc_fail_at = 0;
    double *failed = beam_sparse_row_mut(&rows, 8u, 0u, true);
    g_alloc_fail_at = -1;
    ASSERT_MSG(failed == NULL, "the injected payload growth must fail");
    ASSERT_MSG(beam_sparse_n_rows(&rows) == 8u,
               "failed growth cannot publish a row");
    ASSERT_MSG(beam_sparse_row(&rows, 8u, 0u) == NULL,
               "the failed row's directory entry stays missing");
    for (uint32_t row = 0; row < 8u; ++row)
        ASSERT_MSG(beam_sparse_row(&rows, row, 0u)[row]
                   == (double)row + 0.5,
                   "failed growth leaves existing payloads untouched");

    g_alloc_fail_at = 1;  /* values succeeds; the parallel row-id map fails */
    failed = beam_sparse_row_mut(&rows, 8u, 0u, true);
    g_alloc_fail_at = -1;
    ASSERT_MSG(failed == NULL && beam_sparse_n_rows(&rows) == 8u,
               "a row-id allocation failure cannot half-commit growth");
    ASSERT_MSG(beam_sparse_row_mut(&rows, 8u, 0u, true) != NULL,
               "the table remains growable after an allocation failure");

    beam_sparse_destroy(&rows);
    PASS();
}

static void test_sparse_strict_matches_dense_bit_for_bit(void) {
    BeamModel *dense = build_strict_mode(false);
    BeamModel *sparse = build_strict_mode(true);
    ASSERT_MSG(dense && sparse, "build dense and sparse strict models");
    ASSERT_MSG(!dense->cfg.sparse && sparse->cfg.sparse,
               "the two models must use different storage backends");
    ASSERT_MSG(!sparse->tables_all_finite,
               "omitted sparse rows are semantically dead, not finite");

    const char *probe[] = {
        "CASSF", "CASSY", "CTTGW", "CASSW", "WWWWW", "CTTGF"
    };
    for (uint32_t i = 0; i < 6u; ++i)
        ASSERT_MSG(beam_model_log_pseq(dense, probe[i])
                   == beam_model_log_pseq(sparse, probe[i]),
                   "dense and sparse scalar scores must be bit-identical");
    ASSERT_MSG(beam_model_log_pseq(sparse, "CASSW") == -INFINITY,
               "an omitted transition remains exactly impossible");

    /* One table per direction: max_length depth slots, A^order contexts,
     * twenty residues. Was three tables of three different shapes. */
    {
        const BeamModelConfig *dc = beam_model_config(dense);
        uint64_t want = 2ull * dc->max_length * 20ull;
        for (uint32_t j = 0; j < dc->order; ++j) want *= A;
        ASSERT_MSG(beam_model_n_cells(dense) == want,
                   "n_cells is the logical dense capacity");
    }
    ASSERT_MSG(beam_model_n_cells(sparse) == beam_model_n_cells(dense),
               "storage must not change logical capacity");
    ASSERT_MSG(beam_model_n_nonzero_cells(sparse)
               == beam_model_n_nonzero_cells(dense),
               "cell occupancy is independent of row storage");
    ASSERT_MSG(beam_model_memory_bytes(dense)
               >= 20u * beam_model_memory_bytes(sparse),
               "a tiny strict corpus must be substantially smaller sparse");

    double hd = 0.0, hs = 0.0;
    ASSERT_MSG(beam_model_entropy(dense, &hd) == BEAM_OK
               && beam_model_entropy(sparse, &hs) == BEAM_OK,
               "compute both entropies");
    ASSERT_MSG(hd == hs, "entropy must be bit-identical");
    for (uint32_t q = 0; q <= 2u; ++q) {
        double dd = 0.0, ds = 0.0;
        ASSERT_MSG(beam_model_hill_number(dense, (double)q, &dd) == BEAM_OK
                   && beam_model_hill_number(sparse, (double)q, &ds)
                      == BEAM_OK,
                   "compute both Hill numbers");
        ASSERT_MSG(dd == ds, "Hill numbers must be bit-identical");
    }
    ASSERT_MSG(beam_model_path_count(dense, 5u)
               == beam_model_path_count(sparse, 5u),
               "path counts must be bit-identical");

    double nd = 0.0, ed = 0.0, ns = 0.0, es = 0.0;
    uint32_t ld[6] = {0}, ls[6] = {0};
    ASSERT_MSG(beam_model_topology(dense, 5u, &nd, &ed, ld) == BEAM_OK
               && beam_model_topology(sparse, 5u, &ns, &es, ls) == BEAM_OK,
               "compute both topologies");
    ASSERT_MSG(nd == ns && ed == es, "topology totals must be identical");
    for (uint32_t i = 0; i < 6u; ++i)
        ASSERT_MSG(ld[i] == ls[i], "topology layers must be identical");

    const uint32_t bins = 32u;
    double dc[32] = {0}, dg[32] = {0}, sc[32] = {0}, sg[32] = {0};
    double d_spacing = 0.0, s_spacing = 0.0;
    ASSERT_MSG(beam_model_spectrum_pair(dense, 5u, bins, 0.0,
                                             dc, dg, &d_spacing) == BEAM_OK
               && beam_model_spectrum_pair(sparse, 5u, bins, 0.0,
                                                sc, sg, &s_spacing) == BEAM_OK,
               "compute both spectra");
    ASSERT_MSG(d_spacing == s_spacing, "spectrum grids must be identical");
    for (uint32_t i = 0; i < bins; ++i)
        ASSERT_MSG(dc[i] == sc[i] && dg[i] == sg[i],
                   "both spectrum measures must be bit-identical");

    beam_model_destroy(dense);
    beam_model_destroy(sparse);
    PASS();
}

/* ── The DAG state is wider than the wide expert ─────────────── */

static void test_the_state_is_exactly_one_context_wide(void) {
    /* The state holds exactly the residues one context needs, no more.
     *
     * It used to be max(order, 3), because a second expert of fixed order 3
     * had a context the state had to cover whatever the wide expert's order
     * was; a narrower state would have made the order-3 index read a
     * neighbouring cell. With one table per direction there is nothing to be
     * wider than, and an order-2 model now carries a 2-residue state. */
    BeamModel *m2 = build(2, 27);
    BeamModel *m4 = build(4, 27);
    ASSERT_MSG(m2 && m4, "models should build");
    ASSERT_MSG(beam_model_n_states(m2) == A * A,
               "order 2 carries a 2-residue state");
    ASSERT_MSG(beam_model_n_states(m4) == A * A * A * A,
               "order 4 carries a 4-residue state");
    beam_model_destroy(m2);
    beam_model_destroy(m4);
    PASS();
}

/* ── Training bookkeeping ────────────────────────────────────── */

static void test_observe_counts_and_rejects(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    BeamModel *m = beam_model_new(&cfg);
    const char *seqs[] = { "CASSLGEQYF", "CASSXGEQYF", "",
                           "CASSLGEQYFCASSLGEQYFCASSLGEQYFCASSLGEQYF" };
    const double w[] = { 3.0, 1.0, 1.0, 1.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, w, 4) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_n_observed(m) == 1, "only one usable sequence");
    ASSERT_MSG(beam_model_n_skipped(m) == 3, "X, empty and over-long skip");
    ASSERT_MSG(beam_model_total_weight(m) == 3.0, "weight");
    beam_model_destroy(m);
    PASS();
}

/* ── Context packing, against hand-computed indices ──────────── */

/*
 * Residue indices, from the table in build.c:
 *   A=0  C=1  D=2  E=3  F=4  G=5  H=6  I=7  K=8  L=9
 *   M=10 N=11 P=12 Q=13 R=14 S=15 T=16 V=17 W=18 Y=19   NONE=20
 *
 * "CASSL" is therefore { 1, 0, 15, 15, 9 }, and it is deliberately not a
 * palindrome: every index below changes if the two directions are swapped.
 */
static const uint8_t CASSL[5] = { 1, 0, 15, 15, 9 };

static void test_forward_context_packing(void) {
    /* c_j = s[pos - j], packed most-distant-first, so the index is
     * c_k * A^(k-1) + ... + c_1, sentinel-padded below zero. */

    /* pos 4 (L), order 3: c_3 = s[1] = A = 0, c_2 = s[2] = S = 15,
     *                     c_1 = s[3] = S = 15  ->  0*441 + 15*21 + 15 */
    ASSERT_MSG(beam_ctx_forward(CASSL, 4, 3u, A) == 0u * 441u + 15u * 21u + 15u,
               "forward order-3 context at position 4");

    /* pos 1 (A), order 3: two slots fall off the front and pad with NONE.
     *   c_3 = NONE = 20, c_2 = NONE = 20, c_1 = s[0] = C = 1 */
    ASSERT_MSG(beam_ctx_forward(CASSL, 1, 3u, A) == 20u * 441u + 20u * 21u + 1u,
               "forward order-3 context at position 1 is sentinel-padded");

    /* pos 0, order 2: nothing precedes it at all. */
    ASSERT_MSG(beam_ctx_forward(CASSL, 0, 2u, A) == 20u * 21u + 20u,
               "forward order-2 context at position 0 is all sentinel");

    /* A shorter context is a SUFFIX of a longer one, which is the property the
     * DAG relies on when it reads every forward order out of one state. */
    ASSERT_MSG(beam_ctx_forward(CASSL, 4, 2u, A)
               == beam_ctx_forward(CASSL, 4, 3u, A) % 441u,
               "a shorter forward context must be a suffix of a longer one");
    PASS();
}

static void test_backward_context_packing(void) {
    /* c_j = s[pos + j], packed most-distant-first, sentinel-padded past len. */

    /* pos 0 (C), order 3: c_3 = s[3] = S = 15, c_2 = s[2] = S = 15,
     *                     c_1 = s[1] = A = 0  ->  15*441 + 15*21 + 0 */
    ASSERT_MSG(beam_ctx_backward_counts(CASSL, 0, 5u, 3u, A)
               == 15u * 441u + 15u * 21u + 0u,
               "backward order-3 context at position 0");

    /* pos 4 (L) is the last residue: every right-context slot is off the end. */
    ASSERT_MSG(beam_ctx_backward_counts(CASSL, 4, 5u, 3u, A)
               == 20u * 441u + 20u * 21u + 20u,
               "backward order-3 context at the C terminus is all sentinel");

    /* pos 3 (S), order 2: c_2 = off the end = 20, c_1 = s[4] = L = 9. */
    ASSERT_MSG(beam_ctx_backward_counts(CASSL, 3, 5u, 2u, A) == 20u * 21u + 9u,
               "backward order-2 context one in from the C terminus");

    /*
     * The load-bearing assertion: at the same position the two directions read
     * DIFFERENT residues, so they cannot be each other's mirror image. A model
     * that swapped them would still train and still normalise.
     */
    ASSERT_MSG(beam_ctx_forward(CASSL, 1, 2u, A) == 20u * 21u + 1u
               && beam_ctx_backward_counts(CASSL, 1, 5u, 2u, A) == 15u * 21u + 15u,
               "the two directions must disagree at the same position");
    PASS();
}

static void test_backward_context_from_state_matches_the_sequence(void) {
    /*
     * `beam_ctx_backward_counts_from_state` is the one place the forward/backward
     * asymmetry lives, and the only place the digits come out of a state in
     * reverse. Two independent checks:
     *
     *   1. hand-computed indices for a known state;
     *   2. agreement with `beam_ctx_backward_counts` read off the sequence the
     *      state came from, which is the invariant the DAG actually needs.
     */
    const uint32_t K = 4u;                       /* state order */

    /* State holding "CASS" = s[i-4..i-1], oldest most significant. */
    const uint32_t state = 1u * 9261u + 0u * 441u + 15u * 21u + 15u;
    const uint32_t x = 9u;                       /* L, the residue emitted now */

    /* slot[0]=C(scored) slot[1]=A slot[2]=S slot[3]=S slot[4]=x=L, and
     * c_j = slot[j] packed most-distant-first. */
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 4u, A)
               == 9u * 9261u + 15u * 441u + 15u * 21u + 0u,
               "order-4 backward context out of a state");
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 3u, A)
               == 15u * 441u + 15u * 21u + 0u,
               "order-3 backward context out of a state");
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 2u, A)
               == 15u * 21u + 0u,
               "order-2 backward context out of a state");

    /*
     * Shortening the context drops the MOST DISTANT residue, which for the
     * backward direction is `x` — the residue just emitted — and for the
     * forward direction is the oldest one in the state. Both therefore read as
     * a modulo of their own longer form, and that is not what distinguishes
     * them; what does is which residue leaves.
     */
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 3u, A)
               == beam_ctx_backward_counts_from_state(state, x, K, 4u, A) % 9261u,
               "shortening a backward context must drop x, the most distant");
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 3u, A)
               == beam_ctx_backward_counts_from_state(state, 0u, K, 3u, A),
               "at order 3 the emitted residue is outside the context entirely");
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 4u, A)
               != beam_ctx_backward_counts_from_state(state, 0u, K, 4u, A),
               "at order 4 the emitted residue IS the most distant slot");

    /* The state is "CASS" and x is "L", so the scored residue is CASSL[0] and
     * its right context is exactly what the sequence form reports. */
    for (uint32_t k = 2u; k <= 3u; ++k) {
        ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, k, A)
                   == beam_ctx_backward_counts(CASSL, 0, 5u, k, A),
                   "the state form and the sequence form must agree");
    }

    /*
     * The distinguishing property, and the one that would silently break the
     * model: a FORWARD context index is a suffix of the state index, because
     * the state is packed oldest-first and the forward context runs the same
     * way. A backward context is not — the digits come out reversed. Reusing
     * the forward trick here would produce a model that trains, normalises and
     * scores perfectly while reading its right-hand context backwards.
     */
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 3u, A) != state % 9261u,
               "a backward context must not be a suffix of the state index");
    ASSERT_MSG(beam_ctx_backward_counts_from_state(state, x, K, 2u, A) != state % 441u,
               "the order-2 backward context is not a suffix of the state either");
    PASS();
}

static void test_backward_wide_context_matches_the_sequence(void) {
    /*
     * The finalized C-anchored WIDE table is the one table packed
     * NEAREST-first, so that a trellis sweep reads it front to back instead of
     * scattering half a gigabyte of lookups. Two helpers produce that index —
     * one from a sequence, one from a DAG state — and if they ever disagree
     * the model scores one thing and analyses another, with the partition
     * function normalising perfectly either way.
     *
     * Checked here against hand-computed indices, against each other, and
     * against the reversal of the common packing, which is what
     * `finalize_direction` applies when it relocates each row.
     */
    const uint32_t K = 4u;
    const uint32_t SK = 9261u;                   /* A^(K-1) */

    /* State holding "CASS", emitting "L": scored = C, c_1..c_4 = A,S,S,L. */
    const uint32_t state = 1u * 9261u + 0u * 441u + 15u * 21u + 15u;
    const uint32_t x = 9u;

    ASSERT_MSG(beam_ctx_backward_finalized_from_state(state, x, SK, K, 4u, A)
               == ((0u * A + 15u) * A + 15u) * A + 9u,
               "order-4 nearest-first backward context out of a state");
    /* Nearest-first makes the full-width index plain arithmetic on the state's
     * low digits, which is the whole point of the packing. */
    ASSERT_MSG(beam_ctx_backward_finalized_from_state(state, x, SK, K, 4u, A)
               == (state % SK) * A + x,
               "a full-width backward wide context is (state % shift) * A + x");
    /* There is no narrower context to ask for: the state holds exactly the
     * residues one context needs, so `order` and `state_order` are equal and
     * the full-width form above is the only one reachable. The narrowing
     * branch existed for a second expert of fixed order 3 and is gone with
     * it. */
    ASSERT_MSG(beam_ctx_backward_finalized_from_state(state, x, SK, K, K, A)
               != beam_ctx_backward_finalized_from_state(state, 0u, SK, K, K, A),
               "the emitted residue IS the last digit, so it must move the "
               "context; equality here would mean x is being dropped");

    /* The state is "CASS" and x is "L", so this is CASSL scored at position 0.
     *
     * Only at the full width, because that is the only width reachable: the
     * state holds exactly the residues one context needs. The sequence form
     * still takes a `k`, so the reversal identity below is checked across
     * widths even though the state form is not. */
    ASSERT_MSG(beam_ctx_backward_finalized_from_state(state, x, SK, K, K, A)
               == beam_ctx_backward_finalized(CASSL, 0, 5u, K, A),
               "the state form and the sequence form must agree");
    for (uint32_t k = 2u; k <= 4u; ++k) {
        /* The packing is exactly the common one with its digits reversed,
         * which is the relabelling finalize applies to the table. */
        ASSERT_MSG(beam_ctx_backward_finalized(CASSL, 0, 5u, k, A)
                   == beam_ctx_reverse(beam_ctx_backward_counts(CASSL, 0, 5u, k, A), k, A),
                   "nearest-first is most-distant-first with the digits reversed");
    }

    /* Reversal must be an involution on every index a table can hold, or the
     * relocation in finalize would drop rows on the floor. */
    for (uint32_t c = 0; c < 21u * 21u * 21u; ++c)
        ASSERT_MSG(beam_ctx_reverse(beam_ctx_reverse(c, 3u, A), 3u, A) == c,
                   "reversing an order-3 context twice must be the identity");
    PASS();
}

/**
 * A CRC-32C reference derived from the polynomial, with no table and no
 * hardware.
 *
 * The point of computing it here is that it shares nothing with the library.
 * The test this replaces compared the two library implementations against each
 * OTHER, which cannot distinguish "both right" from "both wrong the same way"
 * -- and they came from one polynomial constant that appeared in two files.
 */
static uint32_t crc32c_by_hand(uint32_t crc, const uint8_t *p, size_t len) {
    uint32_t c = ~crc;
    for (size_t i = 0; i < len; ++i) {
        c ^= p[i];
        for (int b = 0; b < 8; ++b)
            c = (c >> 1) ^ ((c & 1u) ? 0x82F63B78u : 0u);
    }
    return ~c;
}

/**
 * Both CRC paths agree with the polynomial, at every alignment and length.
 *
 * `beam_crc32c_update` takes the SSE4.2 instruction where the CPU has one, and
 * `beam_crc32c_software` is the table. On this machine the first is the
 * hardware path, so without the second the fallback would go unexercised until
 * someone ran BEAM on a machine without the instruction -- which, since the
 * Python binding releases the GIL around save and load, is also the only
 * machine where the table's initialisation ever mattered.
 *
 * The sweep covers offsets 0-8 against every length up to 300, because the
 * unaligned head and the short tail are handled separately in the fast path
 * and that is exactly where a hand-written CRC goes wrong.
 */
/**
 * A top-k search cannot hold more than the budget by splitting the check.
 *
 * `top_k_one` charges the backward field and `beam_best_completions` charges
 * its layer of edge weights, and the layer is allocated INSIDE the field's
 * lifetime -- so the peak is the sum. While each checked the whole budget on
 * its own, a figure between `max(field, layer)` and `field + layer` admitted a
 * search that then held the sum: measured at 46.87 MB against a 33.65 MB
 * budget on an order-4 model, 39% over.
 *
 * The budget here is computed from the two figures rather than guessed, and
 * the test asserts both sides: at a budget that admits each part separately
 * but not both, the search must refuse; with room for the sum, it must answer.
 * Without the second half this would pass on a build that refused everything.
 */
static void test_a_top_k_search_cannot_split_its_way_past_the_budget(void) {
    BeamModel *m = build(4, 12);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t n_states = beam_model_n_states(m);
    const uint32_t n_res = beam_model_n_residues(m);
    uint32_t longest = 0;
    for (uint32_t L = 1; L <= 12u; ++L)
        if (isfinite(beam_model_log_length_prior(m, L))) longest = L;
    ASSERT_MSG(longest > 0, "the model generates nothing");

    const uint64_t field = ((uint64_t)longest + 1u) * n_states * sizeof(double)
                         + ((uint64_t)longest + 1u) * sizeof(double *)
                         + (uint64_t)n_res * sizeof(double);
    const uint64_t layer = (uint64_t)n_states * n_res * sizeof(double)
                         + n_states;
    const uint64_t each  = field > layer ? field : layer;
    ASSERT_MSG(each < field + layer, "the two parts are not distinguishable");

    const uint64_t before = beam_model_memory_budget();
    char buf[8 * 16];
    double lp[8];
    uint32_t n = 0;

    /* Above either part alone, below their sum. */
    beam_model_set_memory_budget(each + (field + layer - each) / 2u);
    const BeamError split = beam_model_top_k(m, 8u, 0u, buf, lp, &n);

    /* And with room for the sum it still works, so the refusal above is the
     * budget binding and not the search being broken. */
    beam_model_set_memory_budget(0);
    uint32_t n2 = 0;
    const BeamError ample = beam_model_top_k(m, 8u, 0u, buf, lp, &n2);

    beam_model_set_memory_budget(before);
    beam_model_destroy(m);

    ASSERT_MSG(split == BEAM_ERR_BUDGET,
               "a budget between one part and the sum admitted a search that "
               "holds the sum");
    ASSERT_MSG(ample == BEAM_OK && n2 > 0,
               "the search fails even with the whole budget available");
    PASS();
}

/**
 * The merge buffer is refused before it is allocated.
 *
 * `k * n_len * sizeof(Found)` had no guard of any kind. The Python binding
 * validates `k` and budgets the two buffers IT allocates, which says nothing
 * about this one, and a C caller has nothing between `k` and the `malloc` at
 * all. At the sizes a caller can legitimately ask for -- ten million is a
 * plausible "give me everything" -- that is a gigabyte and a half taken
 * without the budget hearing about it.
 *
 * The output buffers here are deliberately small: a correct implementation
 * refuses before it writes anything, and if that stops being true this test
 * smashes the stack rather than passing quietly.
 */
static void test_the_top_k_merge_buffer_is_refused_before_it_is_taken(void) {
    BeamModel *m = build(3, 12);
    ASSERT_MSG(m != NULL, "model built");
    const uint64_t before = beam_model_memory_budget();

    char buf[64];
    double lp[4];
    uint32_t n = 0;
    /* 10M results x at least one length x 16 bytes each is >= 160 MB. */
    beam_model_set_memory_budget(64u << 20);
    const BeamError huge = beam_model_top_k(m, 10000000u, 0u, buf, lp, &n);

    beam_model_set_memory_budget(before);
    uint32_t n2 = 0;
    const BeamError small = beam_model_top_k(m, 4u, 0u, buf, lp, &n2);
    beam_model_destroy(m);

    ASSERT_MSG(huge == BEAM_ERR_BUDGET,
               "a ten-million-result merge buffer was allocated without the "
               "budget being consulted");
    ASSERT_MSG(small == BEAM_OK && n2 > 0,
               "an ordinary search stopped working");
    PASS();
}

static void test_both_crc_paths_match_the_polynomial(void) {
    uint8_t buf[300];
    for (size_t i = 0; i < sizeof buf; ++i) buf[i] = (uint8_t)(i * 37u + 11u);
    for (size_t off = 0; off < 9; ++off) {
        for (size_t len = 0; len + off <= sizeof buf; ++len) {
            const uint32_t want = crc32c_by_hand(0x1234u, buf + off, len);
            ASSERT_MSG(beam_crc32c_update(0x1234u, buf + off, len) == want,
                       "the dispatching CRC-32C disagrees with the polynomial");
            ASSERT_MSG(beam_crc32c_software(0x1234u, buf + off, len) == want,
                       "the table CRC-32C disagrees with the polynomial");
        }
    }
    PASS();
}

/**
 * The literal table is what the polynomial generates, all 256 entries.
 *
 * It became a literal so that it needs no initialisation and cannot be raced;
 * the cost of that is 256 constants nobody reads. This regenerates them. It
 * reaches the table through `beam_crc32c_software` on single bytes, because
 * `crc32c_table[i]` for a zero starting CRC is exactly the entry.
 */
static void test_crc32c_table_matches_the_polynomial(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t want = i;
        for (int j = 0; j < 8; ++j)
            want = (want >> 1) ^ ((want & 1u) ? 0x82F63B78u : 0u);
        /*
         * One byte through the table path is
         *     ~( table[0xFF ^ byte] ^ 0x00FFFFFF )
         * because the running CRC starts at ~0, so feeding `0xFF ^ i` and
         * undoing that recovers entry `i` exactly. Reading the entry through
         * the shipped function rather than the array keeps this a test of what
         * callers actually reach.
         */
        const uint8_t byte = (uint8_t)(0xFFu ^ i);
        const uint32_t got = (~beam_crc32c_software(0u, &byte, 1)) ^ 0x00FFFFFFu;
        ASSERT_MSG(got == want, "a CRC-32C table entry does not match the "
                                "polynomial it was generated from");
    }
    PASS();
}

static void test_threads_setting_round_trips(void) {
    const int before = beam_model_threads();
    beam_model_set_threads(3);
    ASSERT_MSG(beam_model_threads() == 3, "the cap must be readable back");
    ASSERT_MSG(beam_model_effective_threads() == 3,
               "with a cap set, that is what a region would start");
    beam_model_set_threads(-5);
    ASSERT_MSG(beam_model_threads() == 0,
               "a negative count means the default, not a negative cap");
    ASSERT_MSG(beam_model_effective_threads() >= 1,
               "the default must be at least one thread");
    beam_model_set_threads(before);
    PASS();
}

static void test_results_do_not_depend_on_the_thread_count(void) {
    /*
     * Every reduction in the implementation is either order-independent (a
     * maximum) or taken over a fixed number of chunks, so the answers must not
     * move when the thread count does. This is the test that would catch a
     * `reduction(+:)` slipping in.
     */
    const int before = beam_model_threads();
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");

    double h1 = 0.0, h4 = 0.0, d1 = 0.0, d4 = 0.0, s1 = 0.0, s4 = 0.0;
    double dep1[BEAM_MAX_LENGTH_CEILING] = {0}, dep4[BEAM_MAX_LENGTH_CEILING] = {0};

    beam_model_set_threads(1);
    ASSERT_MSG(beam_model_entropy(m, &h1) == BEAM_OK, "entropy at 1 thread");
    ASSERT_MSG(beam_model_hill_number(m, 2.0, &d1) == BEAM_OK, "hill at 1");
    s1 = beam_model_path_count_total(m);
    ASSERT_MSG(beam_model_surprisal_by_depth(m, 1.0, dep1, NULL) == BEAM_OK,
               "by_depth at 1 thread");

    beam_model_set_threads(4);
    ASSERT_MSG(beam_model_entropy(m, &h4) == BEAM_OK, "entropy at 4 threads");
    ASSERT_MSG(beam_model_hill_number(m, 2.0, &d4) == BEAM_OK, "hill at 4");
    s4 = beam_model_path_count_total(m);
    ASSERT_MSG(beam_model_surprisal_by_depth(m, 1.0, dep4, NULL) == BEAM_OK,
               "by_depth at 4 threads");

    ASSERT_MSG(h1 == h4, "the entropy must not depend on the thread count");
    ASSERT_MSG(d1 == d4, "a Hill number must not depend on the thread count");
    ASSERT_MSG(s1 == s4, "the path count must not depend on the thread count");
    /* Bounded by what the model actually writes, not by the buffer, which is
     * now sized for the longest sequence any alphabet could ask for. */
    for (uint32_t i = 0; i < beam_model_config(m)->max_length; ++i)
        ASSERT_MSG(dep1[i] == dep4[i],
                   "the depth profile must not depend on the thread count");

    beam_model_set_threads(before);
    beam_model_destroy(m);
    PASS();
}

/* ── Fused analytics must equal the unfused ones ─────────────── */

static void test_spectrum_pair_equals_two_spectra(void) {
    /*
     * The paired form exists to halve the cost of the pipeline's most-used
     * call, and the ONLY thing that makes it a legitimate substitute is that
     * it returns the same numbers. Not close: the same. Both measures ride one
     * transport with one shift and one pair of splitting fractions, so every
     * arithmetic operation is the one the single-measure form performed.
     */
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const uint32_t bins = 256u;
    double *want_c = (double *)calloc(bins, sizeof(double));
    double *want_g = (double *)calloc(bins, sizeof(double));
    double *got_c  = (double *)calloc(bins, sizeof(double));
    double *got_g  = (double *)calloc(bins, sizeof(double));
    ASSERT_MSG(want_c && want_g && got_c && got_g, "alloc");

    double ds = 0.0, ds_pair = 0.0;
    ASSERT_MSG(beam_model_spectrum(m, true, 0, bins, 0.0, want_c, &ds)
               == BEAM_OK, "counting spectrum");
    ASSERT_MSG(beam_model_spectrum(m, false, 0, bins, ds, want_g, NULL)
               == BEAM_OK, "generated spectrum");
    ASSERT_MSG(beam_model_spectrum_pair(m, 0, bins, ds, got_c, got_g,
                                            &ds_pair) == BEAM_OK, "pair");
    ASSERT_MSG(ds_pair == ds, "the pair must land on the grid it was given");
    for (uint32_t b = 0; b < bins; ++b) {
        ASSERT_MSG(got_c[b] == want_c[b], "counting weights must be identical");
        ASSERT_MSG(got_g[b] == want_g[b], "generated weights must be identical");
    }

    /* And at one length, where the ratio per bin is what the discovery-curve
     * arithmetic actually consumes. */
    memset(want_c, 0, bins * sizeof(double));
    memset(want_g, 0, bins * sizeof(double));
    memset(got_c, 0, bins * sizeof(double));
    memset(got_g, 0, bins * sizeof(double));
    const uint32_t L = 10u;
    ASSERT_MSG(beam_model_spectrum(m, true, L, bins, ds, want_c, NULL)
               == BEAM_OK, "counting spectrum at one length");
    ASSERT_MSG(beam_model_spectrum(m, false, L, bins, ds, want_g, NULL)
               == BEAM_OK, "generated spectrum at one length");
    ASSERT_MSG(beam_model_spectrum_pair(m, L, bins, ds, got_c, got_g, NULL)
               == BEAM_OK, "pair at one length");
    for (uint32_t b = 0; b < bins; ++b) {
        ASSERT_MSG(got_c[b] == want_c[b], "counting must be identical at L");
        ASSERT_MSG(got_g[b] == want_g[b], "generated must be identical at L");
    }
    free(want_c); free(want_g); free(got_c); free(got_g);
    beam_model_destroy(m);
    PASS();
}

static void test_spectra_by_length_equal_one_at_a_time(void) {
    /*
     * Same standard as the pair: the fused sweep replaces 27 separate ones and
     * has to return what they returned, row for row. The rows must also sum to
     * the pooled spectrum, which is what makes them a decomposition of it
     * rather than 27 unrelated histograms.
     */
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const uint32_t bins = 256u;
    const uint32_t W = beam_model_config(m)->max_length;

    double ds = 0.0;
    double *pooled = (double *)calloc(bins, sizeof(double));
    double *rows   = (double *)calloc((size_t)(W + 1u) * bins, sizeof(double));
    double *one    = (double *)calloc(bins, sizeof(double));
    double *acc    = (double *)calloc(bins, sizeof(double));
    ASSERT_MSG(pooled && rows && one && acc, "alloc");

    ASSERT_MSG(beam_model_spectrum(m, false, 0, bins, 0.0, pooled, &ds)
               == BEAM_OK, "pooled spectrum");
    ASSERT_MSG(beam_model_spectra_by_length(m, false, 0, bins, ds, rows, NULL)
               == BEAM_OK, "spectra by length");

    for (uint32_t L = 1; L <= W; ++L) {
        memset(one, 0, bins * sizeof(double));
        const bool live = isfinite(beam_model_log_length_prior(m, L));
        if (!live) {
            for (uint32_t b = 0; b < bins; ++b)
                ASSERT_MSG(rows[(size_t)L * bins + b] == 0.0,
                           "a length the model cannot generate must be zero");
            continue;
        }
        ASSERT_MSG(beam_model_spectrum(m, false, L, bins, ds, one, NULL)
                   == BEAM_OK, "single-length spectrum");
        for (uint32_t b = 0; b < bins; ++b) {
            ASSERT_MSG(rows[(size_t)L * bins + b] == one[b],
                       "a fused row must equal the single-length sweep");
            acc[b] += one[b];
        }
    }
    for (uint32_t b = 0; b < bins; ++b)
        ASSERT_MSG(fabs(acc[b] - pooled[b]) < 1e-15,
                   "the rows must decompose the pooled spectrum");

    /*
     * The memory budget changes the SCHEDULE, not the answer -- and below the
     * size of a single sweep it refuses rather than running anyway.
     *
     * This used to assert that a budget of one byte still returned BEAM_OK,
     * pinning the old promise that "one length always runs, however large".
     * That promise was never kept: `positional.c` and the query layer refused
     * above the budget from the same setter, so a call that worked at 4 GB
     * failed at 1 MB in exactly the paths the documentation said it could not.
     * The refusing half was kept, because one sweep reaches 6.6 GB on a
     * nucleotide model at the ceiling order and a budget that cannot say no to
     * that protects nothing.
     */
    const uint64_t before = beam_model_memory_budget();
    double *serial = (double *)calloc((size_t)(W + 1u) * bins, sizeof(double));
    ASSERT_MSG(serial != NULL, "alloc");

    /* Far below the 4 GB default, so concurrency drops, but comfortably above
     * one sweep of this order-3 model: the answer must not move. */
    beam_model_set_memory_budget(256ull << 20);
    ASSERT_MSG(beam_model_spectra_by_length(m, false, 0, bins, ds, serial,
                                                NULL) == BEAM_OK,
               "a reduced budget still runs, one length at a time");
    for (size_t i = 0; i < (size_t)(W + 1u) * bins; ++i)
        ASSERT_MSG(serial[i] == rows[i],
                   "the memory budget must not change the answer");

    /* Below one sweep: refused, and refused with a diagnosable message rather
     * than a bare code. The budget is process-global and ASSERT_MSG returns
     * from the middle of a test, so it is restored BEFORE the assertions --
     * leaving it starved made later, unrelated tests fail with "build". */
    beam_model_set_memory_budget(1u);
    const BeamError starved = beam_model_spectra_by_length(m, false, 0, bins,
                                                           ds, serial, NULL);
    const int named = strstr(beam_error_message(), "memory budget") != NULL;
    beam_model_set_memory_budget(before);

    ASSERT_MSG(starved != BEAM_OK,
               "a budget below a single sweep must refuse, not run anyway");
    ASSERT_MSG(named, "and the refusal must name the budget as the cause");
    ASSERT_MSG(beam_model_memory_budget() == before, "budget restored");

    free(pooled); free(rows); free(one); free(acc); free(serial);
    beam_model_destroy(m);
    PASS();
}

static void test_length_derivatives_mix_to_the_pooled_moments(void) {
    /*
     * The per-length jets are only useful if they are the components the
     * pooled moments are made of, so that is what is checked: the tilted
     * shares in slot 0 must sum to 1, and mixing each order's row by them must
     * reproduce `surprisal_moments` exactly. At q = 1 the shares are the
     * length prior, which pins the normalisation independently.
     */
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const uint32_t W = beam_model_config(m)->max_length;
    const uint32_t order = 3u, NM = order + 1u;

    double *rows = (double *)calloc((size_t)(W + 1u) * NM, sizeof(double));
    double mu[4] = {0};
    ASSERT_MSG(rows != NULL, "alloc");

    for (uint32_t t = 0; t < 2; ++t) {
        const double q = t == 0 ? 1.0 : 0.7;
        ASSERT_MSG(beam_model_length_derivatives(m, q, order, rows) == BEAM_OK,
                   "length derivatives");
        ASSERT_MSG(beam_model_surprisal_moments(m, q, order, mu) == BEAM_OK,
                   "pooled moments");
        double share = 0.0;
        for (uint32_t L = 0; L <= W; ++L) share += rows[(size_t)L * NM];
        ASSERT_MSG(fabs(share - 1.0) < 1e-12,
                   "the tilted length shares must sum to 1");
        for (uint32_t k = 1; k < NM; ++k) {
            double mix = 0.0;
            for (uint32_t L = 0; L <= W; ++L)
                mix += rows[(size_t)L * NM] * rows[(size_t)L * NM + k];
            ASSERT_MSG(fabs(mix - mu[k]) <= 1e-9 * fabs(mu[k]) + 1e-12,
                       "mixing the per-length moments must give the pooled one");
        }
        if (t == 0) {
            for (uint32_t L = 1; L <= W; ++L) {
                const double lp = beam_model_log_length_prior(m, L);
                const double want = isfinite(lp) ? exp(lp) : 0.0;
                ASSERT_MSG(fabs(rows[(size_t)L * NM] - want) < 1e-12,
                           "at q = 1 the shares are the length prior");
            }
        }
    }
    free(rows);
    beam_model_destroy(m);
    PASS();
}

static void test_posterior_shrinks_towards_the_prior(void) {
    /*
     * The update is a Dirichlet-multinomial one, so it has two ends that can
     * be checked without trusting the middle: kappa = 0 must change nothing at
     * all, and a very large kappa must drive a thin model's scores towards the
     * prior's. In between, the identity that pins it is that adding kappa
     * pseudo-counts of a model to ITSELF is the same as scaling its counts,
     * which leaves every normalised row — and therefore every score — where it
     * was, up to the backoff's own kappa.
     */
    BeamModel *prior = build_open(3, 27);
    ASSERT_MSG(prior != NULL, "build");

    /* A thin individual: one sequence, so its own counts say almost nothing. */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3; cfg.max_length = 27;
    /* Open, like the prior: under the closed-world default a model
     * holding one sequence scores the probe -inf, and "moved closer to
     * the prior" is not a statement about infinity. */
    cfg.strict = false;
    BeamModel *thin = beam_model_new(&cfg);
    const char *one[] = { "CASSLGEQYF" };
    ASSERT_MSG(thin && beam_model_observe(thin, one, NULL, 1) == BEAM_OK,
               "observe one");
    ASSERT_MSG(beam_model_finalize(thin) == BEAM_OK, "finalize thin");

    const char *probe = "CASSLGGEQFF";
    const double before = beam_model_log_pseq(thin, probe);
    const double want   = beam_model_log_pseq(prior, probe);

    /* kappa = 0 leaves the counts alone. */
    ASSERT_MSG(beam_model_posterior(thin, prior, 0.0) == BEAM_OK, "kappa 0");
    ASSERT_MSG(beam_model_finalize(thin) == BEAM_OK, "refinalize");
    ASSERT_MSG(beam_model_log_pseq(thin, probe) == before,
               "kappa = 0 must be a no-op");

    /* A heavy prior pulls the thin model towards it. */
    ASSERT_MSG(beam_model_posterior(thin, prior, 1000.0) == BEAM_OK, "kappa");
    ASSERT_MSG(beam_model_is_finalized(thin) == false,
               "posterior must clear finalized, as add and subtract do");
    ASSERT_MSG(beam_model_finalize(thin) == BEAM_OK, "refinalize");
    const double after = beam_model_log_pseq(thin, probe);
    ASSERT_MSG(fabs(after - want) < fabs(before - want),
               "a heavy prior must move the score towards the prior's");

    /* The sequence counters are observations, not pseudo-counts. */
    ASSERT_MSG(beam_model_n_observed(thin) == 1,
               "a pseudo-count is not an observed sequence");

    /* Shape and argument checks. */
    BeamModel *wide = build_open(4, 27);
    ASSERT_MSG(wide != NULL, "build order 4");
    ASSERT_MSG(beam_model_posterior(thin, wide, 1.0) == BEAM_ERR_VARIANT_MISMATCH,
               "a shape mismatch must be refused");
    ASSERT_MSG(beam_model_posterior(thin, prior, -1.0)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "a negative kappa must be refused");
    ASSERT_MSG(beam_model_posterior(NULL, prior, 1.0) == BEAM_ERR_NULL_ARG,
               "a NULL model must be refused");

    beam_model_destroy(wide);
    beam_model_destroy(thin);
    beam_model_destroy(prior);
    PASS();
}

static void test_posterior_of_a_model_with_itself_keeps_its_scores(void) {
    /*
     * Adding kappa pseudo-counts of a model's own belief back into it scales
     * every row by the same factor as long as the row's own mass is scaled
     * too — which is not exactly true here, because kappa is a fixed budget
     * spread by the row's SHARE rather than proportional to the row's mass.
     * What must hold is weaker and still worth pinning: the operation is
     * stable, the model still finalizes, and the scores move by far less than
     * folding in a different model's belief does.
     */
    BeamModel *a = build(3, 27);
    BeamModel *b = build(3, 27);
    ASSERT_MSG(a && b, "build");
    const char *probe = "CASSLGEQYF";
    const double before = beam_model_log_pseq(a, probe);

    ASSERT_MSG(beam_model_posterior(a, b, 5.0) == BEAM_OK, "self posterior");
    ASSERT_MSG(beam_model_finalize(a) == BEAM_OK, "refinalize");
    const double after = beam_model_log_pseq(a, probe);
    ASSERT_MSG(isfinite(after), "the posterior must still score");
    ASSERT_MSG(fabs(after - before) < 0.5,
               "folding a model into itself must barely move its scores");
    beam_model_destroy(a);
    beam_model_destroy(b);
    PASS();
}

static void test_finalize_rolls_back_on_allocation_failure(void) {
    /*
     * Fail the Nth allocation inside `finalize` for a range of N and demand
     * that the model come out of it unchanged: still finalized, still scoring
     * the same number. An error that left the model unusable would be worse
     * than the allocation failure itself, because the counts are the only copy
     * there is and the caller may have just subtracted a cohort out of them.
     *
     * The exact N at which each internal allocation happens is not the point
     * and is not asserted; what is asserted is that whenever finalize refuses,
     * it refuses cleanly. The final check that at least one N actually failed
     * is what stops this from passing vacuously if the wrapper is ever
     * disconnected.
     */
    const int before_threads = beam_model_threads();
    beam_model_set_threads(1);          /* one thread, so N is meaningful */

    BeamModel *m = build_strict_mode(false);
    ASSERT_MSG(m != NULL, "build");
    const char *probe = STRICT_TRAIN[0];
    const double want = beam_model_log_pseq(m, probe);
    ASSERT_MSG(isfinite(want), "the probe must score before the experiment");

    int n_failed = 0;
    for (long n = 0; n < 24; ++n) {
        g_alloc_fail_at = n;
        const BeamError e = beam_model_finalize(m);
        g_alloc_fail_at = -1;
        if (e != BEAM_OK) {
            n_failed++;
            ASSERT_MSG(beam_model_is_finalized(m),
                       "a failed finalize must leave the model finalized");
            ASSERT_MSG(beam_model_log_pseq(m, probe) == want,
                       "a failed finalize must leave the scores untouched");
            ASSERT_MSG(beam_model_error_is_reported(),
                       "a failed finalize must say what ran out");
        } else {
            ASSERT_MSG(beam_model_log_pseq(m, probe) == want,
                       "a successful re-finalize must reproduce the scores");
        }
    }
    ASSERT_MSG(n_failed > 0,
               "the injected failure must actually reach finalize, or this "
               "test proves nothing");

    /* And the model is still usable afterwards, not merely unbroken. */
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize after the storm");
    ASSERT_MSG(beam_model_log_pseq(m, probe) == want, "scores survive");

    beam_model_destroy(m);
    beam_model_set_threads(before_threads);
    PASS();
}

/* ── The scorer and the DAG must describe the same model ─────── */

static void test_layer_weights_reproduce_the_scorer(void) {
    /*
     * `beam_model_log_pseq` walks the sequence directly; the partition
     * function, the spectrum, the sampler and the top-k all consume
     * `beam_layer_weights` plus `beam_tail_weights` instead. If the two
     * ever diverge the analytics describe a different model from the one being
     * scored, and nothing else in this file would notice.
     *
     * Walk the trellis along one known sequence and add the edge weights up.
     */
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");

    const char *seq = "CASSLGEQYF";
    uint8_t buf[BEAM_MAX_LENGTH_CEILING];
    uint32_t len = 0;
    ASSERT_MSG(beam_encode(m, seq, buf, &len), "encode");

    const uint32_t S  = beam_model_n_states(m);
    const uint32_t SK = S / A;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    ASSERT_MSG(w && tail, "scratch");

    /* The walk starts in the all-sentinel state. */
    uint32_t state = S - 1u;
    double raw = 0.0;
    for (uint32_t i = 0; i < len; i++) {
        beam_layer_weights(m, i, len, w, NULL);
        raw += w[(size_t)state * R + buf[i]];
        state = (state % SK) * A + buf[i];
    }
    beam_tail_weights(m, len, tail);
    raw += tail[state];
    free(w); free(tail);

    const double want = raw + beam_model_log_length_prior(m, len)
                            - beam_model_log_partition(m, len);
    const double got = beam_model_log_pseq(m, seq);
    ASSERT_MSG(isfinite(got) && isfinite(want), "both must be finite");
    ASSERT_MSG(fabs(want - got) < 1e-9,
               "the DAG's edge weights must sum to the scorer's own answer");

    beam_model_destroy(m);
    PASS();
}

/* ── Scoring ─────────────────────────────────────────────────── */

static void test_scores_are_finite_and_ordered(void) {
    BeamModel *m = build_open(4, 27);
    ASSERT_MSG(m != NULL, "build");
    const double seen = beam_model_log_pseq(m, "CASSLGEQYF");
    const double rare = beam_model_log_pseq(m, "CASSLGEQYW");
    ASSERT_MSG(isfinite(seen) && isfinite(rare), "both scoreable");
    ASSERT_MSG(seen > rare, "a trained sequence must beat an unseen ending");
    beam_model_destroy(m);
    PASS();
}

static void test_c_anchor_penalises_a_bad_terminus(void) {
    /* This is the property a single left-to-right chain cannot have: by the
     * final position it has no strong opinion, while the C-anchored expert
     * has a very strong one. */
    BeamModel *m = build(4, 27);
    ASSERT_MSG(m != NULL, "build");
    const double good = beam_model_log_pseq(m, "CASSLGEQYF");
    const double bad = beam_model_log_pseq(m, "CASSLGEQYW");
    ASSERT_MSG(good - bad > log(100.0),
               "swapping the terminal residue should cost orders of magnitude");
    beam_model_destroy(m);
    PASS();
}

static void test_rejects_unusable_sequences(void) {
    BeamModel *m = build(4, 27);
    ASSERT_MSG(m != NULL, "build");
    ASSERT_MSG(beam_model_log_pseq(m, "CASSBGEQYF") == -INFINITY, "not an AA");
    ASSERT_MSG(beam_model_log_pseq(m, "") == -INFINITY, "empty");
    ASSERT_MSG(beam_model_log_pseq(m, "CASSLG") == -INFINITY,
               "a length never observed has prior zero");
    beam_model_destroy(m);
    PASS();
}

static void test_batch_matches_scalar(void) {
    BeamModel *m = build(4, 27);
    ASSERT_MSG(m != NULL, "build");
    double out[N_TRAIN];
    ASSERT_MSG(beam_model_log_pseq_batch(m, TRAIN, N_TRAIN, out) == BEAM_OK,
               "batch");
    for (uint32_t i = 0; i < N_TRAIN; i++) {
        ASSERT_MSG(fabs(out[i] - beam_model_log_pseq(m, TRAIN[i])) < 1e-12,
                   "batch must match scalar exactly");
    }
    beam_model_destroy(m);
    PASS();
}

static void test_explain_reconstructs_the_score(void) {
    BeamModel *m = build(4, 27);
    ASSERT_MSG(m != NULL, "build");
    const char *seq = "CASSLGEQYF";
    double fwd[BEAM_MAX_LENGTH_CEILING], bwd[BEAM_MAX_LENGTH_CEILING];
    uint32_t len = 0;
    ASSERT_MSG(beam_model_explain(m, seq, fwd, bwd, &len) == BEAM_OK, "explain");
    ASSERT_MSG(len == strlen(seq), "length");

    const BeamModelConfig *c = beam_model_config(m);
    double raw = 0.0;
    for (uint32_t i = 0; i < len; i++)
        raw += c->weight_forward * fwd[i] + c->weight_backward * bwd[i];
    const double want = raw + beam_model_log_length_prior(m, len)
                            - beam_model_log_partition(m, len);
    ASSERT_MSG(fabs(want - beam_model_log_pseq(m, seq)) < 1e-9,
               "the decomposition must reconstruct the score, not approximate it");

    /* Every training sequence starts with C, so the N-anchored expert must be
     * essentially certain at position 0. The mirror claim about the C anchor
     * at the last position needs a real corpus and is checked in the Python
     * suite, not against twelve hand-written strings. */
    ASSERT_MSG(fwd[0] > -0.05, "N anchor should be near-certain at position 0");
    beam_model_destroy(m);
    PASS();
}

/* ── The partition function, by exhaustive enumeration ───────── */

static void test_partition_normalises(void) {
    /* Sum P(s | L) over EVERY sequence of length 3, all 8,000 of them. Any
     * error in a context packing, in the backward lag, or in the sink weights
     * shows up here and nowhere cheaper. */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3;
    cfg.max_length = 5;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "new");
    const char *tiny[] = { "CAS", "CAF", "GAS", "CQS", "CAS", "SAS" };
    const double tw[] = { 5, 3, 2, 4, 1, 2 };
    ASSERT_MSG(beam_model_observe(m, tiny, tw, 6) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    static const char AAs[] = "ACDEFGHIKLMNPQRSTVWY";
    const double logpi = beam_model_log_length_prior(m, 3);
    double total = 0.0;
    char buf[4] = { 0, 0, 0, 0 };
    for (int a = 0; a < 20; a++)
      for (int b = 0; b < 20; b++)
        for (int c = 0; c < 20; c++) {
            buf[0] = AAs[a]; buf[1] = AAs[b]; buf[2] = AAs[c];
            const double lp = beam_model_log_pseq(m, buf);
            if (isfinite(lp)) total += exp(lp - logpi);
        }
    ASSERT_MSG(fabs(total - 1.0) < 1e-9, "sum over all 8000 length-3 sequences");
    beam_model_destroy(m);
    PASS();
}

static void test_length_prior_normalises(void) {
    BeamModel *m = build(4, 27);
    ASSERT_MSG(m != NULL, "build");
    double total = 0.0;
    for (uint32_t L = 0; L <= 27; L++) {
        const double lp = beam_model_log_length_prior(m, L);
        if (isfinite(lp)) total += exp(lp);
    }
    ASSERT_MSG(fabs(total - 1.0) < 1e-12, "length prior must sum to 1");
    beam_model_destroy(m);
    PASS();
}

/* ── Support size ────────────────────────────────────────────── */

static void test_support_is_all_of_sequence_space(void) {
    /* Documented, not desired, and the reason path_count carries a warning:
     * hierarchical backoff leaves every cell positive, so nothing is dead.
     * This number describes the smoother, not the grammar, and must never be
     * quoted as a diversity. */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3;
    cfg.max_length = 5;
    cfg.strict = false;   /* the support IS 20^L only with backoff on */
    BeamModel *m = beam_model_new(&cfg);
    const char *tiny[] = { "CAS", "CAF", "GAS", "CQS" };
    const double tw[] = { 5, 3, 2, 4 };
    beam_model_observe(m, tiny, tw, 4);
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");
    const double want = 20.0 * 20.0 * 20.0;
    ASSERT_MSG(fabs(beam_model_path_count(m, 3) - want) < 1e-6,
               "the support at length 3 is exactly 20^3");
    beam_model_destroy(m);
    PASS();
}


/* ── The live structure: its invariants and its cache ────────── */

/**
 * Build a corpus-shaped model at a given order and smoothing regime.
 *
 * Strict here means what it means everywhere: no add-one, no probability
 * floor, all five backoff strengths zero. That is the regime where cells go
 * to -inf, and therefore the ONLY regime in which the frontier's edge masks
 * can be anything other than all ones -- which is why every test below runs
 * both regimes rather than the default one.
 */
static BeamModel *build_regime(uint8_t order, uint8_t max_length,
                                  bool strict, const char *const *seqs,
                                  const double *w, uint32_t n) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = max_length;
    cfg.strict = strict;
    if (strict) {
        cfg.sparse = true;
    }
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, seqs, w, n) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/**
 * The frontier's own claims, checked against the tables it read them from.
 *
 * The claim that matters is the edge mask: `path_count` and `topology` no
 * longer look at a single weight VALUE, they read the twenty bits the
 * frontier distilled while it was discovering reachability. A wrong bit on a
 * state that carries no mass changes no published number today, so nothing
 * downstream can catch it -- `beam_live_selfcheck` recomputes every one of
 * them from `beam_state_weights` and compares.
 *
 * ORDERS 2, 3 AND 4, both regimes. Order 4 is the default and is where every
 * reference value in this project was captured, which is exactly why it
 * cannot be the only order tested: the last defect in this area was verified
 * exhaustively over all 194,481 states at order 4 and drove every partition
 * function to -inf at order 2.
 */
static void test_live_frontier_invariants_hold_at_every_order(void) {
    static const char *SEQ[] = {
        "CASSLGEQYF", "CASSLGEQFF", "CASSQGEQYF", "CASSPGEQYF",
        "CASSLAEQYF", "CASSLGETQYF", "CASSVGEQYF", "CASSLGNTIYF",
        "CASSLREQYF", "CASSDGEQYF", "CASSLGEQY", "CTTGWF",
    };
    for (uint8_t order = 2; order <= 4; ++order) {
        for (int strict = 0; strict <= 1; ++strict) {
            BeamModel *m = build_regime(order, 12, strict != 0, SEQ, NULL, 12);
            ASSERT_MSG(m != NULL, "build");
            int seen = 0;
            for (uint32_t L = 1; L <= 12; ++L) {
                if (!beam_length_live(m, L)) continue;
                seen++;
                const int rc = beam_live_selfcheck(m, L);
                if (rc != 0) {
                    printf("FAIL: order %u %s length %u: selfcheck %d\n",
                           (unsigned)order, strict ? "strict" : "open",
                           (unsigned)L, rc);
                    fail_count++;
                    beam_model_destroy(m);
                    return;
                }
            }
            ASSERT_MSG(seen > 0, "the model must generate some length");
            beam_model_destroy(m);
        }
    }
    PASS();
}

/**
 * The cache answers the same thing the uncached build does.
 *
 * Run every length twice: the first call fills the cache, the second reads
 * it. Exact `double` equality, because a cache that returns a slightly
 * different number is not a cache.
 */
static void test_live_cache_returns_identical_answers(void) {
    static const char *SEQ[] = { "CASSF", "CASSY", "CTTGW", "CASSW" };
    for (uint8_t order = 2; order <= 4; ++order) {
        for (int strict = 0; strict <= 1; ++strict) {
            BeamModel *m = build_regime(order, 6, strict != 0, SEQ, NULL, 4);
            ASSERT_MSG(m != NULL, "build");
            for (uint32_t L = 1; L <= 6; ++L) {
                if (!beam_length_live(m, L)) continue;
                const double cold = beam_model_path_count(m, L);
                const double warm = beam_model_path_count(m, L);
                ASSERT_MSG(cold == warm, "cached path_count must be identical");
                double n0 = 0.0, e0 = 0.0, n1 = 0.0, e1 = 0.0;
                ASSERT_MSG(beam_model_topology(m, L, &n0, &e0, NULL) == BEAM_OK,
                           "topology");
                ASSERT_MSG(beam_model_topology(m, L, &n1, &e1, NULL) == BEAM_OK,
                           "topology again");
                ASSERT_MSG(n0 == n1 && e0 == e1,
                           "cached topology must be identical");
            }
            ASSERT_MSG(beam_live_cache_entries(m) > 0,
                       "something should have been cached");
            beam_model_destroy(m);
        }
    }
    PASS();
}

/**
 * Every mutation drops the cache, and the answer afterwards is the answer of
 * a model built that way from the start.
 *
 * The second half is what makes this a real test. Checking that the cache
 * emptied only proves a pointer was freed; checking that `path_count` now
 * reports what an independently built model reports proves the DP did not
 * quietly keep answering questions about the model as it used to be.
 */
static void test_live_cache_is_dropped_by_every_mutation(void) {
    static const char *A_SEQ[] = { "CASSLGEQYF", "CASSQGEQYF", "CASSLAEQYF" };
    static const char *B_SEQ[] = { "CTTGWEQYF", "CASSLGEQYF" };
    const uint32_t LEN = 10u;

    /* observe */
    {
        BeamModel *m = build_regime(4, 12, true, A_SEQ, NULL, 3);
        ASSERT_MSG(m != NULL, "build");
        (void)beam_model_path_count(m, LEN);
        ASSERT_MSG(beam_live_cache_entries(m) > 0, "cache should be warm");
        ASSERT_MSG(beam_model_observe(m, B_SEQ, NULL, 2) == BEAM_OK, "observe");
        ASSERT_MSG(beam_live_cache_entries(m) == 0,
                   "observe must drop the live cache");
        ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "refinalize");
        const char *both[] = { "CASSLGEQYF", "CASSQGEQYF", "CASSLAEQYF",
                               "CTTGWEQYF", "CASSLGEQYF" };
        BeamModel *ref = build_regime(4, 12, true, both, NULL, 5);
        ASSERT_MSG(ref != NULL, "reference build");
        ASSERT_MSG(beam_model_path_count(m, LEN)
                   == beam_model_path_count(ref, LEN),
                   "after observe the count must match a fresh build");
        beam_model_destroy(m);
        beam_model_destroy(ref);
    }
    /* add, then subtract back */
    {
        BeamModel *m = build_regime(4, 12, true, A_SEQ, NULL, 3);
        BeamModel *o = build_regime(4, 12, true, B_SEQ, NULL, 2);
        ASSERT_MSG(m && o, "build");
        const double before = beam_model_path_count(m, LEN);
        ASSERT_MSG(beam_live_cache_entries(m) > 0, "cache should be warm");
        ASSERT_MSG(beam_model_add(m, o) == BEAM_OK, "add");
        ASSERT_MSG(beam_live_cache_entries(m) == 0,
                   "add must drop the live cache");
        ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "refinalize");
        const double grown = beam_model_path_count(m, LEN);
        ASSERT_MSG(beam_model_subtract(m, o) == BEAM_OK, "subtract");
        ASSERT_MSG(beam_live_cache_entries(m) == 0,
                   "subtract must drop the live cache");
        ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "refinalize");
        ASSERT_MSG(beam_model_path_count(m, LEN) == before,
                   "add then subtract must restore the count exactly");
        ASSERT_MSG(grown >= before, "adding sequences cannot shrink the support");
        beam_model_destroy(m);
        beam_model_destroy(o);
    }
    /* posterior */
    {
        BeamModel *m = build_regime(4, 12, true, A_SEQ, NULL, 3);
        BeamModel *p = build_regime(4, 12, true, B_SEQ, NULL, 2);
        ASSERT_MSG(m && p, "build");
        (void)beam_model_path_count(m, LEN);
        ASSERT_MSG(beam_live_cache_entries(m) > 0, "cache should be warm");
        ASSERT_MSG(beam_model_posterior(m, p, 100.0) == BEAM_OK, "posterior");
        ASSERT_MSG(beam_live_cache_entries(m) == 0,
                   "posterior must drop the live cache");
        beam_model_destroy(m);
        beam_model_destroy(p);
    }
    /* re-finalize on its own, and compact */
    {
        BeamModel *m = build_regime(4, 12, true, A_SEQ, NULL, 3);
        ASSERT_MSG(m != NULL, "build");
        (void)beam_model_path_count(m, LEN);
        ASSERT_MSG(beam_live_cache_entries(m) > 0, "cache should be warm");
        ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "refinalize");
        ASSERT_MSG(beam_live_cache_entries(m) == 0,
                   "finalize must drop the live cache");
        (void)beam_model_path_count(m, LEN);
        ASSERT_MSG(beam_live_cache_entries(m) > 0, "cache should be warm");
        ASSERT_MSG(beam_model_compact(m) == BEAM_OK, "compact");
        ASSERT_MSG(beam_live_cache_entries(m) == 0,
                   "compact must drop the live cache");
        beam_model_destroy(m);
    }
    PASS();
}

/**
 * Over the memory bound the cache declines, and every answer is unchanged.
 *
 * The fallback is the code that ran before the cache existed, so this is the
 * test that the two paths have not drifted apart.
 */
static void test_live_cache_respects_the_memory_bound(void) {
    static const char *SEQ[] = { "CASSLGEQYF", "CASSQGEQYF", "CASSLAEQYF",
                                 "CTTGWEQYF" };
    const uint64_t saved = beam_model_memory_budget();
    BeamModel *m = build_regime(4, 12, true, SEQ, NULL, 4);
    ASSERT_MSG(m != NULL, "build");

    double cached[13], nodes[13], edges[13];
    for (uint32_t L = 1; L <= 12; ++L) {
        cached[L] = beam_length_live(m, L)
                  ? beam_model_path_count(m, L) : 0.0;
        nodes[L] = edges[L] = 0.0;
        if (beam_length_live(m, L))
            (void)beam_model_topology(m, L, &nodes[L], &edges[L], NULL);
    }
    ASSERT_MSG(beam_live_cache_entries(m) > 0, "cache should be warm");
    ASSERT_MSG(beam_live_cache_bytes(m) > 0, "a warm cache holds bytes");

    /*
     * Two properties, and they can no longer be starved with one number.
     *
     * The cache cap is a quarter of the budget, and the budget now also
     * REFUSES a sweep too large to fit. On this model an entry averages
     * 0.26 MB, so the cache declines only below about a 1.05 MB budget, while
     * one path-counting sweep needs 5.45 MB. There is no budget at which the
     * cache declines and that sweep still runs -- and there should not be:
     * at 1 MB, refusing a 5.45 MB sweep is the whole point of the setting.
     *
     * So the fallback and the decline are exercised separately. Dropping the
     * cache forces the build path, which IS the code that ran before the cache
     * existed; and a budget that starves the cap is checked against topology,
     * which needs 0.78 MB and therefore still fits.
     */
    int ok_count = 1, ok_topo = 1, ok_ret = 1;

    /* The fallback path, at a budget where everything runs. */
    for (uint32_t L = 1; L <= 12; ++L) {
        if (!beam_length_live(m, L)) continue;
        beam_live_cache_drop(m);
        if (beam_model_path_count(m, L) != cached[L]) ok_count = 0;
        beam_live_cache_drop(m);
        double n = 0.0, e = 0.0;
        if (beam_model_topology(m, L, &n, &e, NULL) != BEAM_OK) ok_ret = 0;
        else if (n != nodes[L] || e != edges[L]) ok_topo = 0;
    }

    /* The decline, at a budget below the cap but above a topology sweep. */
    beam_model_set_memory_budget(1000000u);   /* cap 250 kB < a 256 kB entry */
    beam_live_cache_drop(m);
    int ok_uncached = 1, ok_empty = 1;
    for (uint32_t L = 1; L <= 12; ++L) {
        if (!beam_length_live(m, L)) continue;
        double n = 0.0, e = 0.0;
        if (beam_model_topology(m, L, &n, &e, NULL) != BEAM_OK
            || n != nodes[L] || e != edges[L]) ok_uncached = 0;
    }
    /*
     * The invariant is that the cache never exceeds its cap, not that it is
     * necessarily empty: entries vary in size, and a budget low enough to
     * exclude every one of them is also too low to run the topology sweep this
     * loop needs. Asserting "empty" was asserting an accident of these two
     * entries averaging 256 kB against a 250 kB cap.
     */
    if (beam_live_cache_bytes(m) > beam_model_memory_budget() / 4u)
        ok_empty = 0;

    /*
     * Restored BEFORE the assertions. The budget is process-global, and an
     * ASSERT_MSG returns from the middle of the test: leaving it starved made
     * eight later, unrelated tests fail with "build", which is how this was
     * found.
     */
    beam_model_set_memory_budget(saved);

    ASSERT_MSG(ok_count, "the uncached path must give the identical count");
    ASSERT_MSG(ok_ret, "topology");
    ASSERT_MSG(ok_topo, "the uncached path must give the identical topology");
    ASSERT_MSG(ok_uncached,
               "declining to cache must not change the topology");
    ASSERT_MSG(ok_empty, "the cache must stay within its cap");
    ASSERT_MSG(beam_model_memory_budget() == saved, "budget restored");
    beam_model_destroy(m);
    PASS();
}

/* ── Hill numbers ────────────────────────────────────────────── */

static void test_hill_numbers(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    double h = 0.0, d1 = 0.0;
    ASSERT_MSG(beam_model_entropy(m, &h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_model_hill_number(m, 1.0, &d1) == BEAM_OK, "hill 1");
    ASSERT_MSG(fabs(d1 - exp(h)) < 1e-6 * d1, "D1 must equal exp(H)");

    double prev = INFINITY;
    const double alphas[] = { 0.0, 0.5, 1.0, 1.5, 2.0 };
    for (int i = 0; i < 5; i++) {
        double d = 0.0;
        ASSERT_MSG(beam_model_hill_number(m, alphas[i], &d) == BEAM_OK, "hill");
        ASSERT_MSG(d <= prev * (1.0 + 1e-9), "Hill numbers decrease in alpha");
        prev = d;
    }
    beam_model_destroy(m);
    PASS();
}

/* ── Spectrum ────────────────────────────────────────────────── */

static void test_spectrum_totals_and_identity(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const uint32_t bins = 512;
    double *wc = (double *)calloc(bins, sizeof(double));
    double *wg = (double *)calloc(bins, sizeof(double));
    double ds = 0.0;
    ASSERT_MSG(beam_model_spectrum(m, true, 0, bins, 0.0, wc, &ds) == BEAM_OK,
               "counting spectrum");
    ASSERT_MSG(beam_model_spectrum(m, false, 0, bins, ds, wg, NULL) == BEAM_OK,
               "generated spectrum");

    double sc = 0.0, sg = 0.0, identity = 0.0;
    for (uint32_t b = 0; b < bins; b++) {
        sc += wc[b];
        sg += wg[b];
        if (wc[b] > 0.0 && wg[b] > 0.0) identity += wg[b];
    }
    const double d0 = beam_model_path_count_total(m);
    ASSERT_MSG(fabs(sc - d0) < 1e-6 * d0,
               "the counting measure must integrate to the support size");
    ASSERT_MSG(fabs(sg - 1.0) < 1e-9,
               "the generated measure must integrate to 1");
    ASSERT_MSG(fabs(identity - 1.0) < 1e-9,
               "every bin carrying probability must also carry sequences");
    /* Mass must not reach the top of the grid: the grid is sized from the
     * accumulator, which can exceed the largest surprisal. */
    ASSERT_MSG(wc[bins - 1] == 0.0 && wc[bins - 2] == 0.0,
               "spectrum overflowed its grid");
    free(wc); free(wg);
    beam_model_destroy(m);
    PASS();
}

static void test_accumulator_max_ignores_an_earlier_threads_error(void) {
    /*
     * A regression, and a subtle one. `beam_model_accumulator_max` runs its
     * per-length sweeps in parallel and asks `beam_last_error()` after each
     * one whether that sweep failed. `beam_set_error` is thread-local and
     * OpenMP reuses its threads, so a code left on a worker by an UNRELATED
     * earlier call was read back as this sweep's own failure: the function
     * returned 0.0 and reported "a per-length sweep failed" when every sweep
     * had succeeded.
     *
     * It only fired when the work reached a worker other than the caller,
     * which made it intermittent -- it surfaced as eight unrelated spectrum
     * tests failing in a full suite run and passing on a re-run of the same
     * command.
     *
     * The stale code is planted here rather than waited for. Anything that
     * reports a per-item refusal inside a parallel region leaves one behind;
     * what matters is that a later, successful call is not affected by it.
     */
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const double clean = beam_model_accumulator_max(m);
    ASSERT_MSG(clean > 0.0 && beam_last_error() == BEAM_OK,
               "the undisturbed call must succeed");

#ifdef _OPENMP
    #pragma omp parallel
    { beam_set_error(BEAM_ERR_INVALID_SEQUENCE, "stale, from an earlier call"); }
#endif

    const double after = beam_model_accumulator_max(m);
    beam_model_destroy(m);
    ASSERT_MSG(beam_last_error() == BEAM_OK,
               "a stale error on a worker was reported as this call's failure");
    ASSERT_MSG(after == clean,
               "a stale error on a worker changed the answer");
    PASS();
}

static void test_accumulator_max_is_not_the_surprisal(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const double hi = beam_model_accumulator_max(m);
    ASSERT_MSG(hi > 0.0, "accumulator must reach somewhere");

    const uint32_t bins = 512;
    double *w = (double *)calloc(bins, sizeof(double));
    double ds = 0.0;
    ASSERT_MSG(beam_model_spectrum(m, true, 0, bins, 0.0, w, &ds) == BEAM_OK,
               "spectrum");
    uint32_t last = 0;
    for (uint32_t b = 0; b < bins; b++) if (w[b] > 0.0) last = b;
    /* Linear splitting can push a sliver one bin past the exact value per
     * layer, so the occupied range may run up to max_length+1 bins beyond the
     * bound. What must NOT happen is mass reaching the top of the grid, which
     * would mean it fell off. */
    const uint32_t slack = beam_model_config(m)->max_length + 1u;
    ASSERT_MSG((double)last * ds <= hi + (double)slack * ds,
               "surprisal ran far past the accumulator bound");
    ASSERT_MSG(w[bins - 1] == 0.0, "mass reached the top of the grid");
    free(w);
    beam_model_destroy(m);
    PASS();
}

/* ── Generation ──────────────────────────────────────────────── */

static void test_sampling(void) {
    BeamModel *m = build(4, 27);
    ASSERT_MSG(m != NULL, "build");
    const uint32_t n = 200, stride = 28;
    char *buf = (char *)calloc((size_t)n * stride + 1, 1);
    uint32_t *lens = (uint32_t *)calloc(n, sizeof(uint32_t));
    BeamRng rng;
    beam_rng_seed(&rng, 42);
    ASSERT_MSG(beam_model_sample(m, n, 10, &rng, buf, lens) == BEAM_OK, "sample");
    for (uint32_t i = 0; i < n; i++) {
        const char *s = buf + (size_t)i * stride;
        ASSERT_MSG(strlen(s) == 10, "fixed-length draws must have that length");
        ASSERT_MSG(isfinite(beam_model_log_pseq(m, s)),
                   "every draw must be scoreable by the model that made it");
    }
    free(buf); free(lens);
    beam_model_destroy(m);
    PASS();
}

static void test_sampling_is_reproducible(void) {
    BeamModel *m = build(4, 27);
    ASSERT_MSG(m != NULL, "build");
    const uint32_t n = 50, stride = 28;
    char *a = (char *)calloc((size_t)n * stride + 1, 1);
    char *b = (char *)calloc((size_t)n * stride + 1, 1);
    BeamRng r1, r2;
    beam_rng_seed(&r1, 7);
    beam_rng_seed(&r2, 7);
    beam_model_sample(m, n, 0, &r1, a, NULL);
    beam_model_sample(m, n, 0, &r2, b, NULL);
    ASSERT_MSG(memcmp(a, b, (size_t)n * stride) == 0, "same seed, same draws");
    free(a); free(b);
    beam_model_destroy(m);
    PASS();
}

/* ── Counts are additive ─────────────────────────────────────── */

static void test_subtract_is_the_additive_inverse(void) {
    /* Holding a cohort out of a foundation model must be a subtraction rather
     * than a recount of the corpus. */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    BeamModel *full = beam_model_new(&cfg);
    BeamModel *tail = beam_model_new(&cfg);
    BeamModel *head = beam_model_new(&cfg);
    beam_model_observe(full, TRAIN, WEIGHT, N_TRAIN);
    beam_model_observe(tail, TRAIN + 8, WEIGHT + 8, N_TRAIN - 8);
    beam_model_observe(head, TRAIN, WEIGHT, 8);
    beam_model_finalize(head);
    ASSERT_MSG(beam_model_subtract(full, tail) == BEAM_OK, "subtract");
    ASSERT_MSG(beam_model_finalize(full) == BEAM_OK, "refinalize");

    for (uint32_t i = 0; i < 8; i++) {
        const double a = beam_model_log_pseq(full, TRAIN[i]);
        const double b = beam_model_log_pseq(head, TRAIN[i]);
        ASSERT_MSG(fabs(a - b) < 1e-9, "subtracting must equal never adding");
    }
    beam_model_destroy(full);
    beam_model_destroy(tail);
    beam_model_destroy(head);
    PASS();
}

static void test_subtract_restores_the_sequence_counters(void) {
    /*
     * `add` adjusts total_weight, n_observed AND n_skipped, so the inverse has
     * to move all three. Adjusting only the weight left every
     * leave-one-cohort-out workflow reporting a corpus size that never shrank.
     */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 2;
    BeamModel *a = beam_model_new(&cfg);
    BeamModel *b = beam_model_new(&cfg);
    ASSERT_MSG(a && b, "new");

    /* "CASSXGEQYF" holds an X, so it lands in n_skipped rather than n_observed
     * and the skip counter has to round-trip too. */
    const char *extra[] = { "CASSLGEQYF", "CASSVGEQYF", "CASSXGEQYF" };
    const double ew[] = { 4.0, 5.0, 6.0 };
    beam_model_observe(a, TRAIN, WEIGHT, N_TRAIN);
    beam_model_observe(b, extra, ew, 3);

    const uint64_t n0 = beam_model_n_observed(a);
    const uint64_t s0 = beam_model_n_skipped(a);
    const double   w0 = beam_model_total_weight(a);

    ASSERT_MSG(beam_model_add(a, b) == BEAM_OK, "add");
    ASSERT_MSG(beam_model_n_observed(a) == n0 + beam_model_n_observed(b),
               "add must carry the observed count");
    ASSERT_MSG(beam_model_n_skipped(a) == s0 + beam_model_n_skipped(b),
               "add must carry the skipped count");

    ASSERT_MSG(beam_model_subtract(a, b) == BEAM_OK, "subtract");
    ASSERT_MSG(beam_model_n_observed(a) == n0,
               "subtract must restore the observed count exactly");
    ASSERT_MSG(beam_model_n_skipped(a) == s0,
               "subtract must restore the skipped count exactly");
    ASSERT_MSG(fabs(beam_model_total_weight(a) - w0) < 1e-9,
               "subtract must restore the total weight");

    beam_model_destroy(a);
    beam_model_destroy(b);
    PASS();
}

static void test_subtract_refuses_to_empty_the_model(void) {
    /*
     * The counts are the only copy there is. A subtraction that removes
     * everything used to succeed and leave a model whose re-finalize could
     * only fail, with the tables already freed and nothing to roll back to.
     * It must be refused up front, with the model left exactly as it was.
     */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 2;
    BeamModel *a = beam_model_new(&cfg);
    BeamModel *same = beam_model_new(&cfg);
    ASSERT_MSG(a && same, "new");
    beam_model_observe(a, TRAIN, WEIGHT, N_TRAIN);
    beam_model_observe(same, TRAIN, WEIGHT, N_TRAIN);
    ASSERT_MSG(beam_model_finalize(a) == BEAM_OK, "finalize");

    const double before = beam_model_log_pseq(a, TRAIN[0]);
    const uint64_t n0 = beam_model_n_observed(a);

    ASSERT_MSG(beam_model_subtract(a, same) != BEAM_OK,
               "subtracting an identical model must be refused");
    ASSERT_MSG(beam_model_error_is_reported(),
               "the refusal must carry a message");
    ASSERT_MSG(beam_model_n_observed(a) == n0,
               "a refused subtraction must not touch the counters");
    ASSERT_MSG(beam_model_log_pseq(a, TRAIN[0]) == before,
               "a refused subtraction must leave the model scoring as before");

    beam_model_destroy(a);
    beam_model_destroy(same);
    PASS();
}

static void test_failures_carry_a_message(void) {
    /*
     * Every BEAM failure used to reach Python as a bare code, so a caller was
     * told "invalid argument" and nothing else. `beam_error_message()` has to
     * come back non-empty and mention something the caller can act on.
     */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 99u;                           /* past every alphabet's max */
    beam_clear_error();
    ASSERT_MSG(beam_model_new(&cfg) == NULL, "bad config must be refused");
    ASSERT_MSG(strstr(beam_error_message(), "order") != NULL,
               "the config refusal must name the field it refused");

    beam_model_config_default(&cfg);
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "new");
    beam_clear_error();
    ASSERT_MSG(beam_model_finalize(m) == BEAM_ERR_EMPTY_INPUT,
               "finalizing an unobserved model must fail");
    ASSERT_MSG(strstr(beam_error_message(), "observed") != NULL,
               "the finalize refusal must say nothing was observed");

    beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN);
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");
    beam_clear_error();
    double out = 0.0;
    ASSERT_MSG(beam_model_hill_number(NULL, 2.0, &out) != BEAM_OK, "null model");
    ASSERT_MSG(strstr(beam_error_message(), "NULL") != NULL,
               "a null argument must be named");

    beam_clear_error();
    ASSERT_MSG(beam_model_topology(m, 3u, NULL, NULL, NULL) != BEAM_OK,
               "a length the model never generates must be refused");
    ASSERT_MSG(strstr(beam_error_message(), "length") != NULL,
               "the length refusal must say which length");

    beam_model_destroy(m);
    PASS();
}

/* ── Persistence ─────────────────────────────────────────────── */

static void test_save_load_round_trip(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const char *path = "build/test_model.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save");
    BeamModel *back = beam_model_load(path);
    ASSERT_MSG(back != NULL, "load");
    ASSERT_MSG(beam_model_n_observed(back) == beam_model_n_observed(m),
               "observed count");
    ASSERT_MSG(beam_model_config(back)->order
               == beam_model_config(m)->order, "order");
    for (uint32_t i = 0; i < N_TRAIN; i++) {
        ASSERT_MSG(fabs(beam_model_log_pseq(back, TRAIN[i])
                        - beam_model_log_pseq(m, TRAIN[i])) < 1e-12,
                   "round trip must be exact");
    }
    beam_model_destroy(back);
    beam_model_destroy(m);
    remove(path);
    PASS();
}

/*
 * Rewrite a format-3 file as an older format, in place.
 *
 * The format-3 header appends eight bytes -- the alphabet and a reserved word
 * -- to what formats 1 and 2 wrote, so an older header is exactly a prefix of
 * a newer one and downgrading is a splice: drop those eight bytes, relabel
 * the format, recompute the CRC. That is what lets the backward-compatible
 * reader be pinned without carrying binary fixtures that nothing regenerates.
 */
static bool rewrite_as_legacy_format(const char *path, uint32_t format) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    const long end = ftell(f);
    rewind(f);
    unsigned char *b = (unsigned char *)malloc((size_t)end);
    if (!b || fread(b, 1, (size_t)end, f) != (size_t)end) {
        free(b); fclose(f); return false;
    }
    fclose(f);

    const size_t tail = 2u * sizeof(uint32_t);   /* alphabet + reserved */
    const size_t v2_bytes = 136u;
    memcpy(b + sizeof(uint32_t), &format, sizeof format);
    memmove(b + v2_bytes, b + v2_bytes + tail,
            (size_t)end - v2_bytes - tail);
    const long shorter = end - (long)tail;

    const uint32_t crc = beam_crc32c_update(0u, b,
                                         (size_t)shorter - sizeof(uint32_t));
    memcpy(b + shorter - (long)sizeof crc, &crc, sizeof crc);

    f = fopen(path, "wb");
    if (!f) { free(b); return false; }
    const bool ok = fwrite(b, 1, (size_t)shorter, f) == (size_t)shorter;
    fclose(f);
    free(b);
    return ok;
}

/* Both formats that predate the alphabet field must still load, and must load
 * as amino acid -- which is what they are, and what a zeroed field says. */
static void test_legacy_format_files_are_refused_by_name(void) {
    BeamModel *m = build(2, 12);
    ASSERT_MSG(m != NULL, "build");

    const uint32_t legacy[] = { 1u, 2u };
    for (unsigned k = 0; k < 2u; ++k) {
        const char *path = "build/test_legacy_format.tmp";
        ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save current format");
        ASSERT_MSG(rewrite_as_legacy_format(path, legacy[k]),
                   "rewrite as a legacy format");

        /* Refused, and the message must say which format and why. A file
         * written before this change carries a CLAMPED depth index, so which
         * depth an observation came from was discarded at training time and
         * no transformation recovers it. Reading it anyway would produce
         * wrong payload extents and then a checksum failure, which blames an
         * intact file for the reader's mistake. */
        beam_clear_error();
        BeamModel *back = beam_model_load(path);
        ASSERT_MSG(back == NULL, "a legacy file must be refused");
        ASSERT_MSG(strstr(beam_error_message(), "three experts") != NULL,
                   "the refusal must say what the file was written by");
        ASSERT_MSG(strstr(beam_error_message(), "checksum") == NULL
                   && strstr(beam_error_message(), "corrupt") == NULL,
                   "an intact legacy file must not be called corrupt");
        continue;
        ASSERT_MSG(beam_model_alphabet(back) == BEAM_ALPHABET_AA,
                   "a file written before the alphabet field is amino acid");
        ASSERT_MSG(!beam_model_config(back)->sparse,
                   "legacy dense storage stays dense");
        for (uint32_t i = 0; i < 3u; ++i)
            ASSERT_MSG(beam_model_log_pseq(back, STRICT_TRAIN[i])
                       == beam_model_log_pseq(m, STRICT_TRAIN[i]),
                       "a legacy score must round-trip bit-identically");
        beam_model_destroy(back);
        remove(path);
    }
    beam_model_destroy(m);
    PASS();
}

static void test_corrupt_file_is_rejected(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const char *path = "build/test_corrupt.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save");
    beam_model_destroy(m);

    FILE *f = fopen(path, "r+b");
    ASSERT_MSG(f != NULL, "reopen");
    fseek(f, 512, SEEK_SET);
    unsigned char byte = 0;
    if (fread(&byte, 1, 1, f) == 1) {
        byte ^= 0xFF;
        fseek(f, 512, SEEK_SET);
        fwrite(&byte, 1, 1, f);
    }
    fclose(f);
    ASSERT_MSG(beam_model_load(path) == NULL, "a bad CRC must be refused");
    remove(path);
    PASS();
}

/* ── Compaction ──────────────────────────────────────────────── */

static void test_compact_keeps_scores(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const double before = beam_model_log_pseq(m, TRAIN[0]);
    const uint64_t mem = beam_model_memory_bytes(m);
    ASSERT_MSG(beam_model_compact(m) == BEAM_OK, "compact");
    ASSERT_MSG(beam_model_memory_bytes(m) < mem, "memory must fall");
    ASSERT_MSG(beam_model_log_pseq(m, TRAIN[0]) == before,
               "compaction must not change a single score");
    ASSERT_MSG(beam_model_save(m, "build/nope.tmp") != BEAM_OK,
               "a compacted model has no counts to save");
    beam_model_destroy(m);
    PASS();
}

/* ── Mellin transform ────────────────────────────────────────── */

static void test_log_mellin_endpoints(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    double at0 = 0.0, at1 = 0.0;
    ASSERT_MSG(beam_model_log_mellin(m, 0.0, &at0) == BEAM_OK, "q=0");
    ASSERT_MSG(beam_model_log_mellin(m, 1.0, &at1) == BEAM_OK, "q=1");
    /* q=0 counts every supported sequence once, so it must reproduce the
     * path-count DP, which shares no code with the tilted sweep. */
    const double d0 = beam_model_path_count_total(m);
    ASSERT_MSG(fabs(at0 - log(d0)) < 1e-9, "q=0 must be the log support size");
    ASSERT_MSG(fabs(at1) < 1e-9, "q=1 must be zero: the model is normalised");
    beam_model_destroy(m);
    PASS();
}

static void test_hill_matches_mellin(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const double alphas[] = { 0.5, 2.0, 3.0 };
    for (int i = 0; i < 3; i++) {
        double lm = 0.0, d = 0.0;
        ASSERT_MSG(beam_model_log_mellin(m, alphas[i], &lm) == BEAM_OK, "mellin");
        ASSERT_MSG(beam_model_hill_number(m, alphas[i], &d) == BEAM_OK, "hill");
        const double want = exp(lm / (1.0 - alphas[i]));
        ASSERT_MSG(fabs(d - want) < 1e-6 * want, "D_alpha must follow from M(alpha)");
    }
    beam_model_destroy(m);
    PASS();
}

/* ── Surprisal moments ───────────────────────────────────────── */

static void test_surprisal_mean_is_the_entropy(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    double mu[2] = {0}, h = 0.0;
    ASSERT_MSG(beam_model_surprisal_moments(m, 1.0, 1u, mu) == BEAM_OK, "moments");
    ASSERT_MSG(beam_model_entropy(m, &h) == BEAM_OK, "entropy");
    ASSERT_MSG(fabs(mu[0] - 1.0) < 1e-12, "the zeroth moment is 1");
    ASSERT_MSG(fabs(mu[1] - h) < 1e-6 * fabs(h),
               "at q=1 the mean surprisal IS the entropy");
    beam_model_destroy(m);
    PASS();
}

static void test_surprisal_moments_by_enumeration(void) {
    /* Exhaustive over every length-3 sequence: the jet DP against a direct
     * weighted sum, which is the only way to know the binomial bookkeeping is
     * right rather than merely self-consistent. */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3; cfg.max_length = 4;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "new");
    const char *tiny[] = { "CAS", "CAF", "GAS", "CQS", "SAF", "CAY" };
    const double tw[] = { 9, 4, 3, 6, 2, 5 };
    beam_model_observe(m, tiny, tw, 6);
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    static const char AAs[] = "ACDEFGHIKLMNPQRSTVWY";
    double wsum = 0.0, acc[4] = {0};
    char buf[4] = {0};
    for (int a = 0; a < 20; a++)
      for (int b = 0; b < 20; b++)
        for (int c = 0; c < 20; c++) {
            buf[0] = AAs[a]; buf[1] = AAs[b]; buf[2] = AAs[c];
            const double lp = beam_model_log_pseq(m, buf);
            if (!isfinite(lp)) continue;
            const double w = exp(lp), u = -lp;
            wsum += w;
            double up = 1.0;
            for (int k = 0; k < 4; k++) { acc[k] += w * up; up *= u; }
        }
    double mu[4] = {0};
    ASSERT_MSG(beam_model_surprisal_moments(m, 1.0, 3u, mu) == BEAM_OK, "moments");
    for (int k = 1; k < 4; k++) {
        const double want = acc[k] / wsum;
        ASSERT_MSG(fabs(mu[k] - want) < 1e-7 * fabs(want),
                   "raw moment must match the exhaustive sum");
    }
    beam_model_destroy(m);
    PASS();
}

static void test_cumulants_follow_from_moments(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    double mu[4] = {0}, k[3] = {0};
    ASSERT_MSG(beam_model_surprisal_moments(m, 1.0, 3u, mu) == BEAM_OK, "moments");
    ASSERT_MSG(beam_model_surprisal_cumulants(m, 1.0, 3u, k) == BEAM_OK, "cumulants");
    ASSERT_MSG(fabs(k[0] - mu[1]) < 1e-9 * fabs(mu[1]), "k1 = mean");
    const double var = mu[2] - mu[1] * mu[1];
    ASSERT_MSG(fabs(k[1] - var) < 1e-7 * fabs(var), "k2 = variance");
    const double third = mu[3] - 3.0 * mu[1] * mu[2] + 2.0 * mu[1] * mu[1] * mu[1];
    ASSERT_MSG(fabs(k[2] - third) < 1e-5 * fabs(third), "k3 = third central moment");
    ASSERT_MSG(k[1] > 0.0, "variance must be positive");
    beam_model_destroy(m);
    PASS();
}

/* ── Surprisal range ─────────────────────────────────────────── */

static void test_surprisal_range_by_enumeration(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3; cfg.max_length = 4;
    BeamModel *m = beam_model_new(&cfg);
    const char *tiny[] = { "CAS", "CAF", "GAS", "CQS" };
    const double tw[] = { 9, 4, 3, 6 };
    beam_model_observe(m, tiny, tw, 4);
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    double lo = 0.0, hi = 0.0;
    ASSERT_MSG(beam_model_surprisal_range(m, &lo, &hi) == BEAM_OK, "range");

    static const char AAs[] = "ACDEFGHIKLMNPQRSTVWY";
    double seen_lo = INFINITY, seen_hi = -INFINITY;
    char buf[4] = {0};
    for (int a = 0; a < 20; a++)
      for (int b = 0; b < 20; b++)
        for (int c = 0; c < 20; c++) {
            buf[0] = AAs[a]; buf[1] = AAs[b]; buf[2] = AAs[c];
            const double lp = beam_model_log_pseq(m, buf);
            if (!isfinite(lp)) continue;
            if (-lp < seen_lo) seen_lo = -lp;
            if (-lp > seen_hi) seen_hi = -lp;
        }
    ASSERT_MSG(fabs(lo - seen_lo) < 1e-9, "minimum surprisal must be attained");
    ASSERT_MSG(fabs(hi - seen_hi) < 1e-9, "maximum surprisal must be attained");
    beam_model_destroy(m);
    PASS();
}

/* ── Per-length marginals ────────────────────────────────────── */

static void test_length_marginals_decompose(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    const uint32_t n = (uint32_t)beam_model_config(m)->max_length + 1u;
    double *c = (double *)calloc(n, sizeof(double));
    double *g = (double *)calloc(n, sizeof(double));
    double *u = (double *)calloc(n, sizeof(double));
    double *v = (double *)calloc(n, sizeof(double));
    ASSERT_MSG(beam_model_length_marginals(m, c, g, u, v) == BEAM_OK, "marginals");

    double sc = 0.0, sg = 0.0, h = 0.0;
    for (uint32_t L = 0; L < n; L++) { sc += c[L]; sg += g[L]; h += g[L] * u[L]; }
    const double d0 = beam_model_path_count_total(m);
    ASSERT_MSG(fabs(sc - d0) < 1e-6 * d0, "counting must sum to the support size");
    ASSERT_MSG(fabs(sg - 1.0) < 1e-12, "generated must sum to 1");

    /* The entropy is the length-weighted mean of the per-length means. If the
     * per-length numbers were wrong this identity would not hold. */
    double H = 0.0;
    ASSERT_MSG(beam_model_entropy(m, &H) == BEAM_OK, "entropy");
    ASSERT_MSG(fabs(h - H) < 1e-6 * fabs(H),
               "per-length means must reconstruct the entropy");

    /* Law of total variance: the within-length spread plus the spread of the
     * per-length means must be the global variance. */
    double within = 0.0, between = 0.0;
    for (uint32_t L = 0; L < n; L++) {
        within  += g[L] * v[L];
        between += g[L] * (u[L] - H) * (u[L] - H);
    }
    double k[2] = {0};
    ASSERT_MSG(beam_model_surprisal_cumulants(m, 1.0, 2u, k) == BEAM_OK, "cumulants");
    ASSERT_MSG(fabs(within + between - k[1]) < 1e-5 * fabs(k[1]),
               "law of total variance must hold across the length mixture");
    free(c); free(g); free(u); free(v);
    beam_model_destroy(m);
    PASS();
}

/* ── Top k ───────────────────────────────────────────────────── */

static void test_top_k_by_enumeration(void) {
    /* The claim is exactness, not a beam, so it is checked against a full
     * enumeration. A beam would silently drop a sequence whose prefix looked
     * unpromising and give no sign of having done so. */
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3; cfg.max_length = 4;
    /* Open world: the check below enumerates all 20^3 length-3 sequences and
     * asserts none beats the K-th, which is only a statement about the whole
     * space while backoff keeps every cell positive. Under the closed-world
     * default this corpus of five sequences cannot even produce K=12. */
    cfg.strict = false;
    BeamModel *m = beam_model_new(&cfg);
    const char *tiny[] = { "CAS", "CAF", "GAS", "CQS", "SAF" };
    const double tw[] = { 9, 4, 3, 6, 2 };
    beam_model_observe(m, tiny, tw, 5);
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const uint32_t K = 12, stride = 5;
    char *out = (char *)calloc((size_t)K * stride + 1, 1);
    double *lp = (double *)calloc(K, sizeof(double));
    uint32_t n = 0;
    ASSERT_MSG(beam_model_top_k(m, K, 3, out, lp, &n) == BEAM_OK, "top_k");
    ASSERT_MSG(n == K, "should find K sequences");

    /* Descending, and every reported score must be the model's own. */
    for (uint32_t i = 0; i < n; i++) {
        const char *s = out + (size_t)i * stride;
        ASSERT_MSG(fabs(beam_model_log_pseq(m, s) - lp[i]) < 1e-9,
                   "reported score must equal log_pseq");
        if (i) ASSERT_MSG(lp[i] <= lp[i - 1] + 1e-12, "must be descending");
    }

    /* No length-3 sequence may beat the K-th. */
    static const char AAs[] = "ACDEFGHIKLMNPQRSTVWY";
    uint32_t better = 0;
    char buf[4] = {0};
    for (int a = 0; a < 20; a++)
      for (int b = 0; b < 20; b++)
        for (int c = 0; c < 20; c++) {
            buf[0] = AAs[a]; buf[1] = AAs[b]; buf[2] = AAs[c];
            const double v = beam_model_log_pseq(m, buf);
            if (isfinite(v) && v > lp[n - 1] + 1e-12) better++;
        }
    ASSERT_MSG(better < n, "no sequence outside the top-k may beat its last member");
    free(out); free(lp);
    beam_model_destroy(m);
    PASS();
}

static void test_top_sequence_attains_the_minimum_surprisal(void) {
    BeamModel *m = build(3, 27);
    ASSERT_MSG(m != NULL, "build");
    double lo = 0.0, hi = 0.0;
    ASSERT_MSG(beam_model_surprisal_range(m, &lo, &hi) == BEAM_OK, "range");
    char out[64] = {0};
    double lp = 0.0;
    uint32_t n = 0;
    ASSERT_MSG(beam_model_top_k(m, 1, 0, out, &lp, &n) == BEAM_OK, "top_k");
    ASSERT_MSG(n == 1, "one result");
    /* Links two independent DPs: best-first search and the min-plus sweep. */
    ASSERT_MSG(fabs(-lp - lo) < 1e-9,
               "the most probable sequence must sit at the minimum surprisal");
    beam_model_destroy(m);
    PASS();
}

/* ── Count addition ──────────────────────────────────────────── */

static void test_add_then_subtract_is_identity(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    BeamModel *a = beam_model_new(&cfg);
    BeamModel *b = beam_model_new(&cfg);
    BeamModel *ref = beam_model_new(&cfg);
    beam_model_observe(a, TRAIN, WEIGHT, 8);
    beam_model_observe(ref, TRAIN, WEIGHT, 8);
    beam_model_observe(b, TRAIN + 8, WEIGHT + 8, N_TRAIN - 8);
    beam_model_finalize(ref);

    ASSERT_MSG(beam_model_add(a, b) == BEAM_OK, "add");
    ASSERT_MSG(beam_model_subtract(a, b) == BEAM_OK, "subtract");
    ASSERT_MSG(beam_model_finalize(a) == BEAM_OK, "finalize");
    for (uint32_t i = 0; i < 8; i++) {
        ASSERT_MSG(fabs(beam_model_log_pseq(a, TRAIN[i])
                        - beam_model_log_pseq(ref, TRAIN[i])) < 1e-9,
                   "add then subtract must be the identity");
    }
    beam_model_destroy(a);
    beam_model_destroy(b);
    beam_model_destroy(ref);
    PASS();
}

static void test_add_equals_counting_together(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    BeamModel *a = beam_model_new(&cfg);
    BeamModel *b = beam_model_new(&cfg);
    BeamModel *whole = beam_model_new(&cfg);
    beam_model_observe(a, TRAIN, WEIGHT, 5);
    beam_model_observe(b, TRAIN + 5, WEIGHT + 5, N_TRAIN - 5);
    beam_model_observe(whole, TRAIN, WEIGHT, N_TRAIN);
    beam_model_finalize(whole);

    ASSERT_MSG(beam_model_add(a, b) == BEAM_OK, "add");
    ASSERT_MSG(beam_model_finalize(a) == BEAM_OK, "finalize");
    ASSERT_MSG(beam_model_n_observed(a) == beam_model_n_observed(whole),
               "counts must agree");
    for (uint32_t i = 0; i < N_TRAIN; i++) {
        ASSERT_MSG(fabs(beam_model_log_pseq(a, TRAIN[i])
                        - beam_model_log_pseq(whole, TRAIN[i])) < 1e-9,
                   "counting in shards must equal counting together");
    }
    beam_model_destroy(a);
    beam_model_destroy(b);
    beam_model_destroy(whole);
    PASS();
}

/* ── Runner ──────────────────────────────────────────────────── */

/* ── Positional marginals over a wide dynamic range ──────────── */

/**
 * Every row of a clamped marginal is a distribution, on a trellis whose live
 * states span thousands of nats.
 *
 * Regression for a defect found by review. The forward sweep carried `alpha`
 * linearly and renormalised it once per layer by the largest edge weight in
 * that layer -- and no single factor per layer can hold a trellis that wide.
 * The states that underflowed to zero were skipped from then on, so rows came
 * back summing to 5.9e-264 while the call reported BEAM_OK. Six of sixty rows
 * were lost here, including PINNED positions, whose row must be one-hot.
 *
 * Nothing exotic is needed to reach it: this is `beam_model_config_default()`
 * with the alphabet and the length changed. The sweep now carries logarithms,
 * as the backward half always did.
 */
static void test_clamped_marginals_survive_a_wide_dynamic_range(void) {
    enum { NSEQ = 64, LEN = 60 };
    static char store[NSEQ][72];
    static const char *seqs[NSEQ];
    static double ab[NSEQ];
    /* Two periodic dialects of differing length, so the trellis is deep and
     * its paths are very unequally weighted. */
    for (uint32_t s = 0; s < NSEQ; ++s) {
        const uint32_t n = 54u + (s % 11u);
        const char *cyc = (s % 2u) ? "ACGT" : "AACCGGTT";
        const uint32_t clen = (s % 2u) ? 4u : 8u;
        for (uint32_t i = 0; i < n; ++i)
            store[s][i] = cyc[(i + s) % clen];
        store[s][n] = '\0';
        seqs[s] = store[s];
        ab[s] = 1.0 + (double)(s % 7u);
    }

    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.alphabet = BEAM_ALPHABET_NT;
    cfg.order = 3;
    cfg.max_length = 70;
    cfg.strict = false;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "nucleotide model built");
    ASSERT_MSG(beam_model_observe(m, seqs, ab, NSEQ) == BEAM_OK, "observed");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalized");

    const uint32_t NR = BEAM_NT_RESIDUES;
    double *out = (double *)malloc((size_t)LEN * NR * sizeof(double));
    uint8_t *clamp = (uint8_t *)malloc(LEN);
    ASSERT_MSG(out && clamp, "alloc");
    memset(clamp, BEAM_RESIDUE_INVALID, LEN);
    clamp[7]  = 0u;   /* A */
    clamp[52] = 2u;   /* G */
    clamp[54] = 1u;   /* C */

    double logz = 0.0;
    ASSERT_MSG(beam_model_positional_marginals(m, LEN, clamp, out, &logz)
               == BEAM_OK, "clamped marginals");
    /* Thousands of nats down was reachable when an unobserved cell could be
     * floored at 1e-300. The open world is now a Laplace add-one that bottoms
     * out at 1/(n + R), so the deepest a clamped mass goes is set by how many
     * positions are pinned rather than by a floor artefact. The claim that
     * matters is that the mass is real and far below the mode, and that the
     * log-space path returns it at all. */
    ASSERT_MSG(isfinite(logz) && logz < -5.0,
               "the clamped mass is real and below the mode");

    for (uint32_t i = 0; i < LEN; ++i) {
        double s = 0.0;
        for (uint32_t x = 0; x < NR; ++x) s += out[(size_t)i * NR + x];
        ASSERT_MSG(fabs(s - 1.0) < 1e-9, "every row is a distribution");
    }
    /*
     * A pinned row is one-hot: that is the correct conditional, and the rows
     * that used to come back as 5.9e-264 were pinned ones. Sixty layers of
     * accumulation put these about 1.5e-12 from one, which is rounding; the
     * defect this guards against was 264 orders of magnitude away.
     */
    ASSERT_MSG(fabs(out[(size_t)7 * NR + 0] - 1.0) < 1e-9, "x7 is A");
    ASSERT_MSG(fabs(out[(size_t)52 * NR + 2] - 1.0) < 1e-9, "x52 is G");
    ASSERT_MSG(fabs(out[(size_t)54 * NR + 1] - 1.0) < 1e-9, "x54 is C");
    for (uint32_t x = 1; x < NR; ++x)
        ASSERT_MSG(out[(size_t)7 * NR + x] == 0.0,
                   "and carries nothing anywhere else");

    free(out); free(clamp);
    beam_model_destroy(m);
    PASS();
}

/* ── Tilted sweeps at a negative q ────────────────────────────── */

/**
 * A negative tilt must not overflow the forward sweeps.
 *
 * Regression for a defect found by review, in shipped code. The tilted sweeps
 * rescaled each layer by `q * wmax`, where `wmax` is the largest UNTILTED
 * weight. For `q < 0` the largest tilted weight sits at the SMALLEST untilted
 * one, so `q * wmax` is a minimum instead: every `exp(q * w - qmax)` became an
 * `exp` of a large positive number and the layer overflowed to infinity.
 *
 * The trigger is ordinary -- the open world with backoff off, where unobserved
 * cells sit at `ln(1e-300)` and any `|q| > 1.03` overflows. It was not a loud
 * failure: `log_mellin(-5)` returned `-inf` and `hill_number(-5)` returned
 * `0.0`, both with BEAM_OK, and at `q = -1.5` the answer came back FINITE and
 * wrong, which is worse than either.
 *
 * The reference is the model scoring its own support, one sequence at a time,
 * which shares no sweep with the quantity under test.
 */
static void test_a_negative_tilt_does_not_overflow_the_sweeps(void) {
    static const char *NT_TRAIN[10] = {
        "ACGTA", "ACGTT", "AGGTA", "ACCTA", "ACGAA",
        "TCGTA", "ACGTG", "ACGTC", "AGGTT", "TCGAA",
    };
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.alphabet = BEAM_ALPHABET_NT;
    cfg.order = 3; cfg.max_length = 5; cfg.strict = false;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "model built");
    ASSERT_MSG(beam_model_observe(m, NT_TRAIN, NULL, 10) == BEAM_OK, "observed");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalized");

    const uint32_t NR = BEAM_NT_RESIDUES;
    const BeamAlphabetInfo *ai = beam_alphabet_info(BEAM_ALPHABET_NT);
    const double qs[5] = {-0.5, -1.5, -2.0, -5.0, -24.0};

    for (uint32_t k = 0; k < 5u; ++k) {
        const double q = qs[k];
        /* ln sum_s P(s)^q over the whole support, by enumeration. */
        double mx = -INFINITY, acc = 0.0;
        char buf[8];
        uint8_t idx[8];
        for (uint32_t len = 1u; len <= 5u; ++len) {
            for (uint32_t i = 0; i < len; ++i) idx[i] = 0u;
            buf[len] = '\0';
            for (;;) {
                for (uint32_t i = 0; i < len; ++i) buf[i] = ai->idx_to_char[idx[i]];
                const double lp = beam_model_log_pseq(m, buf);
                if (isfinite(lp)) {
                    const double v = q * lp;
                    if (v > mx) { acc *= exp(mx - v); mx = v; acc += 1.0; }
                    else acc += exp(v - mx);
                }
                uint32_t j = len;
                while (j-- > 0u) {
                    if (++idx[j] < NR) break;
                    idx[j] = 0u;
                    if (j == 0u) goto next_q;
                }
            }
next_q:     ;
        }
        const double want = mx + log(acc);
        double got = 0.0;
        ASSERT_MSG(beam_model_log_mellin(m, q, &got) == BEAM_OK, "log_mellin");
        ASSERT_MSG(isfinite(got),
                   "a negative tilt has a finite answer and must not return "
                   "-inf with a success code");
        ASSERT_MSG(fabs(got - want) < 1e-9 * fabs(want),
                   "and it is the answer the model's own scores give");
    }
    beam_model_destroy(m);
    PASS();
}

/* ── A zero anchor weight ─────────────────────────────────────── */

/**
 * `weight_forward = 0` must not put NaNs in the edge weights.
 *
 * Regression for a defect found by review. A dead cell mixes to `-inf`, and
 * the forward half formed `weight_forward * mix`, so a weight of zero made
 * `0 * -inf` a NaN rather than the `-inf` it must be. The BACKWARD half was
 * already guarded, and that asymmetry is what gave the defect away.
 *
 * Nothing rejected the weight -- the config validates only `>= 0` -- and the
 * NaNs were latent here, because every sweep iterates reached states and
 * skipped them. On a larger corpus they were not latent: `finalize` reported
 * that the model could generate nothing. Measured on this model before the
 * fix: 608,573 NaN edge weights of 1,666,980.
 */
/*
 * Both storage modes, because the first version of this test checked only one
 * and the defect survived in the other for exactly that reason.
 *
 * `beam_model_config_default` sets `sparse = false`, so building from the
 * default exercised the DENSE kernel -- which the fix guarded. The Python
 * layer selects sparse for the default closed world, so the path users
 * actually run was the unguarded one, and it still produced 115 NaN edge
 * weights of 3,519,180 on this model. A test that takes the default
 * configuration tests the default configuration, not the shipped one.
 */
static void zero_forward_weight_one_mode(bool sparse) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3; cfg.max_length = 12;
    cfg.sparse = sparse;
    cfg.strict = sparse ? true : cfg.strict;   /* sparse is valid only strict */
    cfg.weight_forward = 0.0; cfg.weight_backward = 1.0;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "a zero anchor weight is a legal configuration");
    ASSERT_MSG(beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) == BEAM_OK, "obs");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK,
               "and finalizes rather than reporting the model generates nothing");

    /* `A`, `R` and `NON` are macros in this file, so the locals are named
     * around them. */
    const uint32_t n_st = beam_model_n_states(m);
    const uint32_t n_re = beam_model_n_residues(m);
    double *w = (double *)malloc((size_t)n_st * n_re * sizeof(double));
    ASSERT_MSG(w != NULL, "alloc");
    long nans = 0;
    for (uint32_t i = 0; i < 10u; ++i) {
        beam_layer_weights(m, i, 10u, w, NULL);
        for (size_t k = 0; k < (size_t)n_st * n_re; ++k)
            if (isnan(w[k])) nans++;
    }
    ASSERT_MSG(nans == 0, "no edge weight is a NaN");
    ASSERT_MSG(isfinite(beam_model_log_pseq(m, TRAIN[0])),
               "and a training sequence still scores");
    free(w);
    beam_model_destroy(m);
}

static void test_a_zero_forward_weight_leaves_no_nan_edge_weights(void) {
    zero_forward_weight_one_mode(false);   /* dense  */
    zero_forward_weight_one_mode(true);    /* sparse */
    PASS();
}

int main(void) {
    printf("\n=== BEAM (Ensemble of Dual Graph Experts) ===\n");

    RUN_TEST(test_defaults_are_the_measured_ones);
    RUN_TEST(test_provenance_is_recorded_and_aggregated);
    RUN_TEST(test_depth_occupancy_sums_to_the_nonzero_cells);
    RUN_TEST(test_single_anchor_weights_are_accepted);
    RUN_TEST(test_sparse_requires_strict);
    RUN_TEST(test_sparse_rows_boundaries_and_erase);
    RUN_TEST(test_sparse_growth_failure_is_atomic);
    RUN_TEST(test_sparse_strict_matches_dense_bit_for_bit);
    RUN_TEST(test_the_state_is_exactly_one_context_wide);
    RUN_TEST(test_observe_counts_and_rejects);

    RUN_TEST(test_forward_context_packing);
    RUN_TEST(test_backward_context_packing);
    RUN_TEST(test_backward_context_from_state_matches_the_sequence);
    RUN_TEST(test_backward_wide_context_matches_the_sequence);
    RUN_TEST(test_a_top_k_search_cannot_split_its_way_past_the_budget);
    RUN_TEST(test_the_top_k_merge_buffer_is_refused_before_it_is_taken);
    RUN_TEST(test_both_crc_paths_match_the_polynomial);
    RUN_TEST(test_crc32c_table_matches_the_polynomial);
    RUN_TEST(test_threads_setting_round_trips);
    RUN_TEST(test_results_do_not_depend_on_the_thread_count);
    RUN_TEST(test_layer_weights_reproduce_the_scorer);

    RUN_TEST(test_scores_are_finite_and_ordered);
    RUN_TEST(test_c_anchor_penalises_a_bad_terminus);
    RUN_TEST(test_rejects_unusable_sequences);
    RUN_TEST(test_batch_matches_scalar);
    RUN_TEST(test_explain_reconstructs_the_score);

    RUN_TEST(test_partition_normalises);
    RUN_TEST(test_length_prior_normalises);

    RUN_TEST(test_support_is_all_of_sequence_space);
    RUN_TEST(test_live_frontier_invariants_hold_at_every_order);
    RUN_TEST(test_live_cache_returns_identical_answers);
    RUN_TEST(test_live_cache_is_dropped_by_every_mutation);
    RUN_TEST(test_live_cache_respects_the_memory_bound);
    RUN_TEST(test_hill_numbers);

    RUN_TEST(test_spectrum_totals_and_identity);
    RUN_TEST(test_accumulator_max_is_not_the_surprisal);
    RUN_TEST(test_accumulator_max_ignores_an_earlier_threads_error);

    RUN_TEST(test_sampling);
    RUN_TEST(test_sampling_is_reproducible);

    RUN_TEST(test_subtract_is_the_additive_inverse);
    RUN_TEST(test_subtract_restores_the_sequence_counters);
    RUN_TEST(test_subtract_refuses_to_empty_the_model);
    RUN_TEST(test_failures_carry_a_message);
    RUN_TEST(test_save_load_round_trip);
    RUN_TEST(test_legacy_format_files_are_refused_by_name);
    RUN_TEST(test_corrupt_file_is_rejected);
    RUN_TEST(test_compact_keeps_scores);

    RUN_TEST(test_log_mellin_endpoints);
    RUN_TEST(test_hill_matches_mellin);
    RUN_TEST(test_surprisal_mean_is_the_entropy);
    RUN_TEST(test_surprisal_moments_by_enumeration);
    RUN_TEST(test_cumulants_follow_from_moments);
    RUN_TEST(test_surprisal_range_by_enumeration);
    RUN_TEST(test_length_marginals_decompose);
    RUN_TEST(test_top_k_by_enumeration);
    RUN_TEST(test_top_sequence_attains_the_minimum_surprisal);
    RUN_TEST(test_add_then_subtract_is_identity);
    RUN_TEST(test_add_equals_counting_together);
    RUN_TEST(test_spectrum_pair_equals_two_spectra);
    RUN_TEST(test_spectra_by_length_equal_one_at_a_time);
    RUN_TEST(test_length_derivatives_mix_to_the_pooled_moments);
    RUN_TEST(test_posterior_shrinks_towards_the_prior);
    RUN_TEST(test_posterior_of_a_model_with_itself_keeps_its_scores);
    RUN_TEST(test_finalize_rolls_back_on_allocation_failure);
    RUN_TEST(test_clamped_marginals_survive_a_wide_dynamic_range);
    RUN_TEST(test_a_negative_tilt_does_not_overflow_the_sweeps);
    RUN_TEST(test_a_zero_forward_weight_leaves_no_nan_edge_weights);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
