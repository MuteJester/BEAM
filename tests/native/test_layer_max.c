/**
 * @file test_layer_max.c
 * @brief `beam_layer_max` against the definition it is an optimisation of.
 *
 * `beam_layer_max` exists so a caller can rescale a layer without building
 * it. Its contract is exact: the value it returns must be the maximum edge
 * weight over EVERY state of the layer, to the last bit -- three callers
 * divide by it (`beam_compute_log_z` and both Hill-number sweeps), and
 * `diversity.c` documents what happened the day a maximum was taken over a
 * subset instead: the open-world Hill numbers moved at every order.
 *
 * So the reference here is not another implementation. It is the definition:
 * ask `beam_state_weights` for every state and take the maximum. That keeps
 * the test honest across changes to how the shortcut finds its rows.
 *
 * This matters because the shortcut's row source is changing. It used to walk
 * the finalized slot -> row map; it now walks the directory slice that holds
 * the layer's depth bucket. Both reach the same rows, and this file is what
 * says so in values rather than in a comment.
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


/* A corpus wide enough that the wide table is genuinely sparse: many
 * contexts, and lengths that do not all share a depth bucket. */
#define N_TRAIN 14
static const char *TRAIN[N_TRAIN] = {
    "CASSF",   "CASSY",   "CAGSF",  "CGSSF",   "CASSL",  "CTSSF",  "CASSQ",
    "CASSFGH", "CASSYWM", "CAGSFK", "CGSSFPD", "CASSLN", "CTSSFR", "CASSQEV",
};
static const double WEIGHT[N_TRAIN] = {
    40, 25, 18, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2,
};

static BeamModel *build(uint8_t order, uint32_t max_length, bool sparse) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = max_length;
    cfg.sparse = sparse;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/**
 * The definition: the largest edge weight anywhere in the layer.
 *
 * Deliberately the slow way -- every state, no row map, no directory, no
 * frontier -- because a reference that shared the shortcut's reasoning would
 * only prove the shortcut agrees with itself.
 */
static double reference_layer_max(const BeamModel *m, uint32_t i,
                                  uint32_t len) {
    const uint32_t S = m->n_states;
    const uint32_t R = m->n_residues;
    double row[BEAM_MAX_RESIDUES];
    double hi = -INFINITY;
    for (uint32_t s = 0; s < S; ++s) {
        beam_state_weights(m, i, len, s, row);
        for (uint32_t x = 0; x < R; ++x)
            if (row[x] > hi) hi = row[x];
    }
    return hi;
}

/** Bit-identical, or `-inf` on both sides. */
static bool same_double(double a, double b) {
    return memcmp(&a, &b, sizeof a) == 0;
}

static void check_every_layer(const char *what, uint8_t order,
                              uint32_t max_length, bool sparse) {
    BeamModel *m = build(order, max_length, sparse);
    ASSERT_MSG(m != NULL, "model did not build");

    uint32_t checked = 0, finite = 0;
    for (uint32_t len = 1; len <= max_length; ++len) {
        for (uint32_t i = 0; i < len; ++i) {
            const double got = beam_layer_max(m, i, len);
            const double want = reference_layer_max(m, i, len);
            if (!same_double(got, want)) {
                printf("FAIL: %s len=%u layer=%u got %.17g want %.17g\n",
                       what, len, i, got, want);
                fail_count++;
                beam_model_destroy(m);
                return;
            }
            checked++;
            if (got != -INFINITY) finite++;
        }
    }
    beam_model_destroy(m);

    /* A run where every layer was dead would pass the comparison above
     * without exercising anything. */
    ASSERT_MSG(checked > 0u, "no layer was checked at all");
    ASSERT_MSG(finite > 0u, "every layer came back -inf: nothing was tested");
    PASS();
}

static void test_sparse_order_4_matches_the_definition(void) {
    check_every_layer("sparse order 4", 4u, 8u, true);
}

static void test_sparse_order_2_matches_the_definition(void) {
    /* `stride = S / n_ctx` is 1 only when order == state_order. At a
     * lower wide order one context backs several states, and the shortcut
     * has to visit all of them. */
    check_every_layer("sparse order 2", 2u, 7u, true);
}

static void test_dense_matches_the_definition(void) {
    /* The dense branch builds the whole layer; included so the two branches
     * are pinned to one reference rather than to each other. */
    check_every_layer("dense order 3", 3u, 6u, false);
}

/**
 * A depth bucket with no occupied row at all.
 *
 * The shortcut slices the directory by depth bucket, so a bucket the corpus
 * never reached is the case where it must find nothing and say `-inf` rather
 * than run off the slice. Short sequences under a high `depth_cap` leave the
 * deep buckets empty, which is how a real per-repertoire model looks.
 */
static void test_a_depth_bucket_with_no_rows_is_minus_infinity(void) {
    static const char *SHORT[4] = { "CASF", "CASY", "CAGF", "CGSF" };
    static const double W[4] = { 4, 3, 2, 1 };

    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3u;
    cfg.max_length = 18u;
    cfg.sparse = true;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "model did not allocate");
    if (beam_model_observe(m, SHORT, W, 4u) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        ASSERT_MSG(false, "model did not build");
    }

    /* Nothing longer than four residues was ever seen, so every bucket from
     * five on is empty. Both the shortcut and the definition must agree that
     * those layers carry no edge. */
    uint32_t empty_seen = 0;
    for (uint32_t len = 6; len <= 18u; ++len) {
        for (uint32_t i = 5; i < len; ++i) {
            const double got = beam_layer_max(m, i, len);
            const double want = reference_layer_max(m, i, len);
            if (!same_double(got, want)) {
                printf("FAIL: empty bucket len=%u layer=%u got %.17g want %.17g\n",
                       len, i, got, want);
                fail_count++;
                beam_model_destroy(m);
                return;
            }
            if (got == -INFINITY) empty_seen++;
        }
    }
    beam_model_destroy(m);
    ASSERT_MSG(empty_seen > 0u, "no empty bucket was reached: nothing was tested");
    PASS();
}

/**
 * The log-Z thin-frontier path asks one traversal for two things: the exact
 * all-state maximum and complete rows for a marked subset. Pin both halves of
 * that contract to the old, independent state-at-a-time definition.
 */
static void test_sparse_combined_max_and_marked_rows(void) {
    BeamModel *m = build(4u, 8u, true);
    ASSERT_MSG(m != NULL, "model did not build");

    const uint32_t S = m->n_states;
    const uint32_t R = m->n_residues;
    const uint32_t i = 4u, len = 7u;
    uint8_t *alive = (uint8_t *)calloc(S, sizeof(uint8_t));
    double *out = (double *)malloc((size_t)S * R * sizeof(double));
    if (!alive || !out) {
        free(alive);
        free(out);
        beam_model_destroy(m);
        ASSERT_MSG(false, "combined-path buffers did not allocate");
    }

    /* Include definitely live rows plus arbitrary states whose forward-wide
     * row may be absent, since both must be fully written. */
    double reference[BEAM_MAX_RESIDUES];
    uint32_t finite_marked = 0u;
    for (uint32_t s = 0; s < S && finite_marked < 24u; ++s) {
        beam_state_weights(m, i, len, s, reference);
        bool finite = false;
        for (uint32_t x = 0; x < R; ++x)
            if (reference[x] != -INFINITY) { finite = true; break; }
        if (finite) {
            alive[s] = 1u;
            finite_marked++;
        }
    }
    const uint32_t step = S / 19u;
    for (uint32_t s = 0; s < S; s += step) alive[s] = 1u;
    alive[S - 1u] = 1u;
    memset(out, 0xa5, (size_t)S * R * sizeof(double));

    double got_max = -INFINITY;
    beam_layer_weights_live(m, i, len, out, &got_max, alive);
    const double want_max = reference_layer_max(m, i, len);
    if (!same_double(got_max, want_max)) {
        printf("FAIL: combined max got %.17g want %.17g\n",
               got_max, want_max);
        fail_count++;
        free(alive);
        free(out);
        beam_model_destroy(m);
        return;
    }

    for (uint32_t s = 0; s < S; ++s) {
        if (!alive[s]) continue;
        beam_state_weights(m, i, len, s, reference);
        const double *got = out + (size_t)s * R;
        if (memcmp(got, reference, (size_t)R * sizeof(double)) != 0) {
            printf("FAIL: combined marked row state=%u\n", s);
            fail_count++;
            free(alive);
            free(out);
            beam_model_destroy(m);
            return;
        }
    }

    free(alive);
    free(out);
    beam_model_destroy(m);
    ASSERT_MSG(finite_marked > 0u, "no finite marked row was exercised");
    PASS();
}

int main(void) {
    printf("\nbeam_layer_max\n");
    RUN_TEST(test_sparse_order_4_matches_the_definition);
    RUN_TEST(test_sparse_order_2_matches_the_definition);
    RUN_TEST(test_dense_matches_the_definition);
    RUN_TEST(test_a_depth_bucket_with_no_rows_is_minus_infinity);
    RUN_TEST(test_sparse_combined_max_and_marked_rows);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
