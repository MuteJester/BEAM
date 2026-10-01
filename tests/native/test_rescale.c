/**
 * @file test_rescale.c
 * @brief The one invariant the linear forward sweeps depend on.
 *
 * Every linear sweep divides its accumulator by the layer maximum and carries
 * the log of the scale. A state whose accumulator is far enough below that
 * maximum divides to exactly 0.0, and every sweep here reads 0.0 as "no mass",
 * so the path stops there and the total comes back short. Nothing about the
 * answer looks wrong afterwards, which is why this is a test rather than a
 * comment: the invariant is INVISIBLE in the published numbers.
 *
 * Two halves, and both are needed. The first checks that the detector detects,
 * because a guard that cannot fire proves nothing about the sweeps it guards.
 * The second runs the sweeps over a matrix of configurations and requires that
 * nothing is ever lost. Under the hierarchical prior every string over the
 * alphabet carries mass, so every state of every layer is live, and each layer
 * divides all of them by its one maximum. The spread a layer has to hold
 * grows with the length, which is why the matrix reaches the longest length
 * each alphabet is run at: 27 for amino acids, 90 for nucleotides.
 *
 * Each configuration runs on the default sparse storage, and the cheap ones
 * on dense storage as well; the two are the same model bit for bit and reach
 * their rows by different code. Amino-acid order 4 is not in the matrix: all
 * 160,000 of its residue contexts are live at every layer past the fourth,
 * and the eight analytics below cost minutes at its default length. What the
 * rescale depends on is the length and the spread of the weights, not the
 * width of a layer, so the long amino-acid case is order 3 at 27.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"

#include "beam/mixture.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"


/* ── Does the detector detect? ───────────────────────────────── */

static void test_the_guard_notices_a_value_that_vanishes(void) {
    beam_rescale_flush_reset();

    /* Ordinary division, nothing lost. */
    ASSERT_MSG(beam_rescale(1.0, 4.0) == 0.25, "plain division");
    ASSERT_MSG(beam_rescale(0.0, 4.0) == 0.0, "an empty state is not a loss");
    ASSERT_MSG(beam_rescale_flushes() == 0ull, "counted a loss that was not one");

    /* A state carrying mass, divided by a scale that annihilates it. */
    const double lost = beam_rescale(1e-320, 1e300);
    ASSERT_MSG(lost == 0.0, "the quotient really does vanish");
    ASSERT_MSG(beam_rescale_flushes() == 1ull, "the guard did not notice");

    beam_rescale_flush_reset();
    ASSERT_MSG(beam_rescale_flushes() == 0ull, "reset does not reset");
    PASS();
}

static void test_the_guard_divides_exactly_as_the_operator_does(void) {
    /*
     * Bit-identical, not merely close. These sweeps are pinned to the last bit
     * by verification/check.py, so a guard that reached the quotient any other
     * way -- multiplying by a reciprocal is the usual temptation -- would move
     * every recorded number in the library at once.
     */
    static const double vals[] = {1.0, 3.0, 1e-8, 7.13, 1e300, 4.9e-324, 0.0};
    static const double scales[] = {1.0, 3.0, 7.0, 1e-9, 1e17, 0.1};
    for (unsigned i = 0; i < sizeof vals / sizeof *vals; ++i)
        for (unsigned j = 0; j < sizeof scales / sizeof *scales; ++j) {
            const double a = beam_rescale(vals[i], scales[j]);
            const double b = vals[i] / scales[j];
            if (memcmp(&a, &b, sizeof a) != 0) {
                beam_rescale_flush_reset();
                ASSERT_MSG(0, "the guard is not bit-identical to `/`");
            }
        }
    beam_rescale_flush_reset();
    PASS();
}


/* ── Do the sweeps ever lose a state? ────────────────────────── */

static char pool_aa[] = "ACDEFGHIKLMNPQRSTVWY";
static char pool_nt[] = "ACGT";

static BeamModel *build(bool aa, uint8_t order, uint8_t max_len, bool sparse,
                        uint32_t lo, uint32_t hi, uint32_t n) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    if (!aa) beam_model_config_default_for(BEAM_ALPHABET_NT, &cfg);
    cfg.order = order;
    cfg.max_length = max_len;
    cfg.sparse = sparse;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;

    const char *pool = aa ? pool_aa : pool_nt;
    const unsigned np = aa ? 20u : 4u;
    char **buf = (char **)malloc((size_t)n * sizeof(char *));
    const char **ptr = (const char **)malloc((size_t)n * sizeof(char *));
    double *w = (double *)malloc((size_t)n * sizeof(double));
    if (!buf || !ptr || !w) {
        free(buf); free((void *)ptr); free(w);
        beam_model_destroy(m); return NULL;
    }
    unsigned s = 12345u;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t L = lo + (s % (hi - lo + 1u));
        s = s * 1103515245u + 12345u;
        buf[i] = (char *)malloc(L + 1u);
        for (uint32_t j = 0; j < L; ++j) {
            buf[i][j] = pool[s % np];
            s = s * 1103515245u + 12345u;
        }
        buf[i][L] = '\0';
        ptr[i] = buf[i];
        w[i] = 1.0 + (double)(s % 30u);
    }
    BeamError e = beam_model_observe(m, ptr, w, n);
    if (e == BEAM_OK) e = beam_model_finalize(m);
    for (uint32_t i = 0; i < n; ++i) free(buf[i]);
    free(buf); free((void *)ptr); free(w);
    if (e != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

/**
 * Every analytic that reaches a linear rescale, on one model.
 *
 * `partition` is reached by all of them, `tilted_log_z` by the Hill and Mellin
 * calls, `expected_raw` by the entropy, `jet_sweep` by the moments, and the
 * surprisal sweep by the range. Returns the number of states lost.
 */
static unsigned long long lost_by(const BeamModel *m) {
    double v = 0.0, mom[5] = {0}, lo = 0.0, hi = 0.0;
    const uint32_t depths = beam_model_config(m)->max_length;
    double *by_depth = (double *)calloc(depths ? depths : 1u, sizeof(double));
    if (!by_depth) return 1ull;          /* fail loudly rather than skip */

    beam_rescale_flush_reset();
    (void)beam_model_hill_number(m, 2.0, &v);          /* tilted_log_z    */
    (void)beam_model_entropy(m, &v);                   /* expected_raw    */
    (void)beam_model_log_mellin(m, 0.5, &v);           /* tilted_log_z    */
    (void)beam_model_log_mellin(m, 4.0, &v);           /* tilted_log_z    */
    (void)beam_model_surprisal_moments(m, 1.0, 4u, mom);  /* jet_sweep    */
    (void)beam_model_surprisal_moments(m, 3.0, 4u, mom);  /* jet_sweep    */
    (void)beam_model_surprisal_range(m, &lo, &hi);
    /*
     * `surprisal_by_depth`, not `surprisal_range`: the guarded sweep in
     * math/surprisal.c belongs to the former, and the latter lives in
     * math/extrema.c entirely. Calling the wrong one left that site
     * unexercised, which a mutation of its divisor found and "zero flushes"
     * could not have.
     */
    (void)beam_model_surprisal_by_depth(m, 1.0, by_depth, &v);
    free(by_depth);
    return beam_rescale_flushes();
}

static void check_config(const char *label, bool aa, uint8_t order,
                         uint8_t max_len, bool sparse, uint32_t lo,
                         uint32_t hi, uint32_t n) {
    BeamModel *m = build(aa, order, max_len, sparse, lo, hi, n);
    if (!m) { printf("FAIL: %s did not build\n", label); fail_count++; return; }
    const unsigned long long lost = lost_by(m);
    beam_model_destroy(m);
    if (lost != 0ull) {
        printf("FAIL: %s lost %llu state(s) to the rescale\n", label, lost);
        fail_count++;
        return;
    }
}

static void test_no_sparse_sweep_loses_a_state(void) {
    /*
     * Every cell is positive, so a layer holds live states many nats apart,
     * and the longer the sequence the further apart they drift. If any change
     * to the prior, the length range or the tilt ever pushes one of them past
     * what a double can hold beside its layer's maximum, this is what says so.
     * nt order 6 is cut from its default length of 81 to 24: every one of its
     * 15,625 residue contexts is live at every layer, and the length the
     * rescale is most exposed to is already covered by order 3 at 90.
     */
    const int before = fail_count;
    check_config("aa order 2 len 12", true, 2, 12, true, 6, 11, 2000);
    check_config("aa order 3 len 27", true, 3, 27, true, 11, 25, 1500);
    check_config("nt order 6 len 24", false, 6, 24, true, 10, 22, 2000);
    check_config("nt order 3 len 90", false, 3, 90, true, 40, 88, 1200);
    if (fail_count != before) return;
    PASS();
}

static void test_no_dense_sweep_loses_a_state(void) {
    const int before = fail_count;
    check_config("aa order 2 len 12 dense", true, 2, 12, false, 6, 11, 2000);
    check_config("nt order 3 len 90 dense", false, 3, 90, false, 40, 88, 1200);
    if (fail_count != before) return;
    PASS();
}

static void test_the_mixture_sweep_loses_nothing_either(void) {
    /* The product sweep rescales the same way and is guarded the same way,
     * and multiplies two members' weights, so its layers spread twice as far.
     * At the default amino-acid length. */
    BeamModel *a = build(true, 3, 27, true, 11, 25, 2000);
    BeamModel *b = build(true, 3, 27, true, 11, 25, 1500);
    ASSERT_MSG(a && b, "a model did not build");
    const BeamModel *pair[2] = {a, b};
    beam_rescale_flush_reset();
    double lip = 0.0;
    const BeamError e = beam_mixture_log_inner_product_by_length(pair, 2, NULL,
                                                                 &lip);
    const unsigned long long lost = beam_rescale_flushes();
    beam_model_destroy(a); beam_model_destroy(b);
    ASSERT_MSG(e == BEAM_OK, "the inner product failed");
    ASSERT_MSG(lost == 0ull, "the product sweep lost a state to the rescale");
    PASS();
}


int main(void) {
    printf("\n==============================================================\n");
    printf("  The per-layer rescale invariant\n");
    printf("==============================================================\n");

    printf("\n[the detector]\n");
    RUN_TEST(test_the_guard_notices_a_value_that_vanishes);
    RUN_TEST(test_the_guard_divides_exactly_as_the_operator_does);

    printf("\n[no sweep loses a state that carries mass]\n");
    RUN_TEST(test_no_sparse_sweep_loses_a_state);
    RUN_TEST(test_no_dense_sweep_loses_a_state);
    RUN_TEST(test_the_mixture_sweep_loses_nothing_either);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
