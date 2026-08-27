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

#define A   BEAM_ALPHABET
#define R   BEAM_RESIDUES
#define NON BEAM_NONE

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
    cfg.depth_cap = 5;
    cfg.max_length = 6;
    cfg.strict = true;
    cfg.sparse = sparse;
    cfg.kappa[0] = cfg.kappa[1] = cfg.kappa[2] = 0.0;
    cfg.kappa_wide = 0.0;
    cfg.kappa_len = 0.0;

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
    ASSERT_MSG(c.depth_cap == 15, "default depth cap should be 15");
    ASSERT_MSG(fabs(c.coef_wide + c.coef_mid + c.coef_len - 1.0) < 1e-12,
               "coefficients must sum to 1");
    ASSERT_MSG(c.weight_forward == c.weight_backward,
               "the two anchors are weighted equally by default");
    ASSERT_MSG(!c.sparse, "the low-level C default remains dense");
    PASS();
}

static void test_rejects_uncalibrated_coefficients(void) {
    /* The coefficient sum is a cross-length calibration knob, not a free
     * parameter, so an off-1 sum must be refused rather than silently
     * mis-calibrating global ranking. */
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.coef_wide = 0.7;
    ASSERT_MSG(beam_model_new(&c) == NULL, "sum 1.1 should be rejected");
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
    ASSERT_MSG(beam_model_config(m)->depth_cap < 12, "cap should be clamped");
    beam_model_destroy(m);
    PASS();
}

static void test_sparse_requires_strict_zero_backoff(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.strict = true;
    c.sparse = true;
    /* Backoff has to be asked for explicitly now: the default is the closed
     * world, which is exactly the regime sparse storage is valid in, so the
     * defaults alone no longer produce the combination being rejected. */
    c.kappa[0] = c.kappa[1] = c.kappa[2] = 20.0;
    c.kappa_wide = 20.0;
    c.kappa_len = 200.0;
    ASSERT_MSG(beam_model_new(&c) == NULL,
               "positive backoff cannot omit zero-count rows");

    c.kappa[0] = c.kappa[1] = c.kappa[2] = 0.0;
    c.kappa_wide = 0.0;
    c.kappa_len = 0.0;
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
    ASSERT_MSG(beam_sparse_init(&rows, 8u, 2u, true) == BEAM_OK,
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
    ASSERT_MSG(beam_sparse_init(&finalized, 8u, 1u, false) == BEAM_OK,
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
    ASSERT_MSG(beam_sparse_init(&rows, 16u, 1u, true) == BEAM_OK,
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

    ASSERT_MSG(beam_model_n_cells(dense) == 3069360u,
               "n_cells is the logical dense capacity");
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

static void test_state_covers_the_mid_expert(void) {
    /* The mid expert is order 3, so the state can never hold fewer than 3
     * residues however narrow the wide expert is. A state of 21^2 would make
     * the order-3 context index read a neighbouring cell. */
    BeamModel *m2 = build(2, 27);
    BeamModel *m4 = build(4, 27);
    ASSERT_MSG(m2 && m4, "models should build");
    ASSERT_MSG(beam_model_n_states(m2) == A * A * A,
               "order 2 must still carry a 3-residue state");
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
    ASSERT_MSG(beam_ctx_forward(CASSL, 4, 3u) == 0u * 441u + 15u * 21u + 15u,
               "forward order-3 context at position 4");

    /* pos 1 (A), order 3: two slots fall off the front and pad with NONE.
     *   c_3 = NONE = 20, c_2 = NONE = 20, c_1 = s[0] = C = 1 */
    ASSERT_MSG(beam_ctx_forward(CASSL, 1, 3u) == 20u * 441u + 20u * 21u + 1u,
               "forward order-3 context at position 1 is sentinel-padded");

    /* pos 0, order 2: nothing precedes it at all. */
    ASSERT_MSG(beam_ctx_forward(CASSL, 0, 2u) == 20u * 21u + 20u,
               "forward order-2 context at position 0 is all sentinel");

    /* A shorter context is a SUFFIX of a longer one, which is the property the
     * DAG relies on when it reads every forward order out of one state. */
    ASSERT_MSG(beam_ctx_forward(CASSL, 4, 2u)
               == beam_ctx_forward(CASSL, 4, 3u) % 441u,
               "a shorter forward context must be a suffix of a longer one");
    PASS();
}

static void test_backward_context_packing(void) {
    /* c_j = s[pos + j], packed most-distant-first, sentinel-padded past len. */

    /* pos 0 (C), order 3: c_3 = s[3] = S = 15, c_2 = s[2] = S = 15,
     *                     c_1 = s[1] = A = 0  ->  15*441 + 15*21 + 0 */
    ASSERT_MSG(beam_ctx_backward(CASSL, 0, 5u, 3u)
               == 15u * 441u + 15u * 21u + 0u,
               "backward order-3 context at position 0");

    /* pos 4 (L) is the last residue: every right-context slot is off the end. */
    ASSERT_MSG(beam_ctx_backward(CASSL, 4, 5u, 3u)
               == 20u * 441u + 20u * 21u + 20u,
               "backward order-3 context at the C terminus is all sentinel");

    /* pos 3 (S), order 2: c_2 = off the end = 20, c_1 = s[4] = L = 9. */
    ASSERT_MSG(beam_ctx_backward(CASSL, 3, 5u, 2u) == 20u * 21u + 9u,
               "backward order-2 context one in from the C terminus");

    /*
     * The load-bearing assertion: at the same position the two directions read
     * DIFFERENT residues, so they cannot be each other's mirror image. A model
     * that swapped them would still train and still normalise.
     */
    ASSERT_MSG(beam_ctx_forward(CASSL, 1, 2u) == 20u * 21u + 1u
               && beam_ctx_backward(CASSL, 1, 5u, 2u) == 15u * 21u + 15u,
               "the two directions must disagree at the same position");
    PASS();
}

static void test_backward_context_from_state_matches_the_sequence(void) {
    /*
     * `beam_ctx_backward_from_state` is the one place the forward/backward
     * asymmetry lives, and the only place the digits come out of a state in
     * reverse. Two independent checks:
     *
     *   1. hand-computed indices for a known state;
     *   2. agreement with `beam_ctx_backward` read off the sequence the
     *      state came from, which is the invariant the DAG actually needs.
     */
    const uint32_t K = 4u;                       /* state order */

    /* State holding "CASS" = s[i-4..i-1], oldest most significant. */
    const uint32_t state = 1u * 9261u + 0u * 441u + 15u * 21u + 15u;
    const uint32_t x = 9u;                       /* L, the residue emitted now */

    /* slot[0]=C(scored) slot[1]=A slot[2]=S slot[3]=S slot[4]=x=L, and
     * c_j = slot[j] packed most-distant-first. */
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 4u)
               == 9u * 9261u + 15u * 441u + 15u * 21u + 0u,
               "order-4 backward context out of a state");
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 3u)
               == 15u * 441u + 15u * 21u + 0u,
               "order-3 backward context out of a state");
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 2u)
               == 15u * 21u + 0u,
               "order-2 backward context out of a state");

    /*
     * Shortening the context drops the MOST DISTANT residue, which for the
     * backward direction is `x` — the residue just emitted — and for the
     * forward direction is the oldest one in the state. Both therefore read as
     * a modulo of their own longer form, and that is not what distinguishes
     * them; what does is which residue leaves.
     */
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 3u)
               == beam_ctx_backward_from_state(state, x, K, 4u) % 9261u,
               "shortening a backward context must drop x, the most distant");
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 3u)
               == beam_ctx_backward_from_state(state, 0u, K, 3u),
               "at order 3 the emitted residue is outside the context entirely");
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 4u)
               != beam_ctx_backward_from_state(state, 0u, K, 4u),
               "at order 4 the emitted residue IS the most distant slot");

    /* The state is "CASS" and x is "L", so the scored residue is CASSL[0] and
     * its right context is exactly what the sequence form reports. */
    for (uint32_t k = 2u; k <= 3u; ++k) {
        ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, k)
                   == beam_ctx_backward(CASSL, 0, 5u, k),
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
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 3u) != state % 9261u,
               "a backward context must not be a suffix of the state index");
    ASSERT_MSG(beam_ctx_backward_from_state(state, x, K, 2u) != state % 441u,
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

    ASSERT_MSG(beam_ctx_backward_wide_from_state(state, x, SK, K, 4u)
               == ((0u * A + 15u) * A + 15u) * A + 9u,
               "order-4 nearest-first backward context out of a state");
    /* Nearest-first makes the full-width index plain arithmetic on the state's
     * low digits, which is the whole point of the packing. */
    ASSERT_MSG(beam_ctx_backward_wide_from_state(state, x, SK, K, 4u)
               == (state % SK) * A + x,
               "a full-width backward wide context is (state % shift) * A + x");
    /* A narrower context drops the MOST DISTANT residue, which here is x. */
    ASSERT_MSG(beam_ctx_backward_wide_from_state(state, x, SK, K, 3u)
               == (0u * A + 15u) * A + 15u,
               "order-3 nearest-first backward context out of a state");
    ASSERT_MSG(beam_ctx_backward_wide_from_state(state, x, SK, K, 3u)
               == beam_ctx_backward_wide_from_state(state, 0u, SK, K, 3u),
               "below full width the emitted residue is outside the context");

    /* The state is "CASS" and x is "L", so this is CASSL scored at position 0. */
    for (uint32_t k = 2u; k <= 4u; ++k) {
        ASSERT_MSG(beam_ctx_backward_wide_from_state(state, x, SK, K, k)
                   == beam_ctx_backward_wide(CASSL, 0, 5u, k),
                   "the state form and the sequence form must agree");
        /* And the packing is exactly the common one with its digits reversed,
         * which is the relabelling finalize applies to the table. */
        ASSERT_MSG(beam_ctx_backward_wide(CASSL, 0, 5u, k)
                   == beam_ctx_reverse(beam_ctx_backward(CASSL, 0, 5u, k), k),
                   "nearest-first is most-distant-first with the digits reversed");
    }

    /* Reversal must be an involution on every index a table can hold, or the
     * relocation in finalize would drop rows on the floor. */
    for (uint32_t c = 0; c < 21u * 21u * 21u; ++c)
        ASSERT_MSG(beam_ctx_reverse(beam_ctx_reverse(c, 3u), 3u) == c,
                   "reversing an order-3 context twice must be the identity");
    PASS();
}

static void test_crc_matches_the_reference(void) {
    /*
     * io.c checksums a gigabyte of counts with a hardware CRC-32C where the
     * CPU has one. Same polynomial and same convention as the shared software
     * implementation, so the two must agree byte for byte — including on the
     * unaligned heads and short tails the fast path handles separately, which
     * is exactly where a hand-written CRC goes wrong.
     */
    uint8_t buf[300];
    for (size_t i = 0; i < sizeof buf; ++i) buf[i] = (uint8_t)(i * 37u + 11u);
    for (size_t off = 0; off < 9; ++off) {
        for (size_t len = 0; len + off <= sizeof buf; ++len) {
            const uint32_t want = beam_crc32c_update(0x1234u, buf + off, len);
            const uint32_t got  = beam_model_crc32c(0x1234u, buf + off, len);
            ASSERT_MSG(got == want, "hardware CRC-32C must match the reference");
        }
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
    double dep1[BEAM_MAX_LENGTH] = {0}, dep4[BEAM_MAX_LENGTH] = {0};

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
    for (uint32_t i = 0; i < BEAM_MAX_LENGTH; ++i)
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

    /* Lowering the memory budget changes only how many lengths run at once. */
    const uint64_t before = beam_model_memory_budget();
    beam_model_set_memory_budget(1u);
    double *serial = (double *)calloc((size_t)(W + 1u) * bins, sizeof(double));
    ASSERT_MSG(serial != NULL, "alloc");
    ASSERT_MSG(beam_model_spectra_by_length(m, false, 0, bins, ds, serial,
                                                NULL) == BEAM_OK, "budget 1");
    for (size_t i = 0; i < (size_t)(W + 1u) * bins; ++i)
        ASSERT_MSG(serial[i] == rows[i],
                   "the memory budget must not change the answer");
    beam_model_set_memory_budget(before);

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
    uint8_t buf[BEAM_MAX_LENGTH];
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
    double fwd[BEAM_MAX_LENGTH], bwd[BEAM_MAX_LENGTH];
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
    cfg.depth_cap = (uint8_t)(max_length - 1u);
    cfg.strict = strict;
    if (strict) {
        cfg.kappa[0] = cfg.kappa[1] = cfg.kappa[2] = 0.0;
        cfg.kappa_wide = 0.0;
        cfg.kappa_len = 0.0;
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

    beam_model_memory_budget();
    beam_model_set_memory_budget(4u);       /* a quarter of this is 1 byte */
    beam_live_cache_drop(m);
    for (uint32_t L = 1; L <= 12; ++L) {
        if (!beam_length_live(m, L)) continue;
        ASSERT_MSG(beam_model_path_count(m, L) == cached[L],
                   "the uncached path must give the identical count");
        double n = 0.0, e = 0.0;
        ASSERT_MSG(beam_model_topology(m, L, &n, &e, NULL) == BEAM_OK,
                   "topology");
        ASSERT_MSG(n == nodes[L] && e == edges[L],
                   "the uncached path must give the identical topology");
    }
    ASSERT_MSG(beam_live_cache_entries(m) == 0,
               "nothing may be cached under a 1-byte bound");
    beam_model_set_memory_budget(saved);
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
    cfg.coef_wide = 0.7;                       /* sum 1.1: uncalibrated */
    beam_clear_error();
    ASSERT_MSG(beam_model_new(&cfg) == NULL, "bad config must be refused");
    ASSERT_MSG(strstr(beam_error_message(), "coef") != NULL,
               "the config refusal must name the coefficients");

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

static void test_dense_v1_file_still_loads(void) {
    /* Format 2 changed only the sparse payload. A dense format-2 body is the
     * exact legacy layout, so relabel it as v1 and recompute the CRC to pin
     * the backward-compatible reader without carrying a large binary fixture. */
    BeamModel *m = build(2, 12);
    ASSERT_MSG(m != NULL, "build");
    const char *path = "build/test_dense_v1.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save dense v2");

    FILE *f = fopen(path, "rb");
    ASSERT_MSG(f != NULL, "open saved file");
    ASSERT_MSG(fseek(f, 0, SEEK_END) == 0, "seek end");
    const long end = ftell(f);
    ASSERT_MSG(end > (long)(2u * sizeof(uint32_t)), "file has header and CRC");
    rewind(f);
    unsigned char *bytes = (unsigned char *)malloc((size_t)end);
    ASSERT_MSG(bytes != NULL, "allocate file buffer");
    ASSERT_MSG(fread(bytes, 1, (size_t)end, f) == (size_t)end, "read file");
    fclose(f);

    const uint32_t v1 = 1u;
    memcpy(bytes + sizeof(uint32_t), &v1, sizeof v1);
    const uint32_t crc = beam_model_crc32c(0u, bytes,
                                         (size_t)end - sizeof(uint32_t));
    memcpy(bytes + (size_t)end - sizeof crc, &crc, sizeof crc);
    f = fopen(path, "wb");
    ASSERT_MSG(f != NULL, "rewrite legacy fixture");
    ASSERT_MSG(fwrite(bytes, 1, (size_t)end, f) == (size_t)end,
               "write legacy fixture");
    fclose(f);
    free(bytes);

    BeamModel *back = beam_model_load(path);
    ASSERT_MSG(back != NULL, "load dense v1");
    ASSERT_MSG(!beam_model_config(back)->sparse,
               "legacy dense storage stays dense");
    for (uint32_t i = 0; i < 3u; ++i)
        ASSERT_MSG(beam_model_log_pseq(back, STRICT_TRAIN[i])
                   == beam_model_log_pseq(m, STRICT_TRAIN[i]),
                   "legacy dense score must round-trip bit-identically");
    beam_model_destroy(back);
    beam_model_destroy(m);
    remove(path);
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

int main(void) {
    printf("\n=== BEAM (Ensemble of Dual Graph Experts) ===\n");

    RUN_TEST(test_defaults_are_the_measured_ones);
    RUN_TEST(test_rejects_uncalibrated_coefficients);
    RUN_TEST(test_depth_cap_is_clamped);
    RUN_TEST(test_sparse_requires_strict_zero_backoff);
    RUN_TEST(test_sparse_rows_boundaries_and_erase);
    RUN_TEST(test_sparse_growth_failure_is_atomic);
    RUN_TEST(test_sparse_strict_matches_dense_bit_for_bit);
    RUN_TEST(test_state_covers_the_mid_expert);
    RUN_TEST(test_observe_counts_and_rejects);

    RUN_TEST(test_forward_context_packing);
    RUN_TEST(test_backward_context_packing);
    RUN_TEST(test_backward_context_from_state_matches_the_sequence);
    RUN_TEST(test_backward_wide_context_matches_the_sequence);
    RUN_TEST(test_crc_matches_the_reference);
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

    RUN_TEST(test_sampling);
    RUN_TEST(test_sampling_is_reproducible);

    RUN_TEST(test_subtract_is_the_additive_inverse);
    RUN_TEST(test_subtract_restores_the_sequence_counters);
    RUN_TEST(test_subtract_refuses_to_empty_the_model);
    RUN_TEST(test_failures_carry_a_message);
    RUN_TEST(test_save_load_round_trip);
    RUN_TEST(test_dense_v1_file_still_loads);
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

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
