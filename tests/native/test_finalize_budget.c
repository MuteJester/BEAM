/**
 * @file test_finalize_budget.c
 * @brief Finalize divides the memory budget like every other by-length driver.
 *
 * The partition functions are a per-length dynamic program run concurrently,
 * and each sweep holds a layer of edge weights -- 31 MB on an order-4
 * amino-acid model, more at the alphabet's ceiling order. Every other
 * by-length driver in the library asks `beam_budget_concurrency` how many may
 * run at once. This one ran at `beam_threads()` regardless, so the budget
 * described a bound the largest allocation in a build did not observe.
 *
 * Two things are pinned here, and only the second is about memory.
 *
 * The first is that concurrency is a PERFORMANCE knob and never a correctness
 * one: the same model finalized under a budget that admits two lengths at a
 * time and one that admits sixteen must produce identical bits. That is the
 * property a reader needs in order to believe the budget is safe to lower.
 *
 * The second is that a refusal survives. `beam_set_error` is thread-local, so
 * a refusal raised inside an OpenMP worker reached the caller as a bare code
 * and the message naming the figure and the remedy was replaced by "a
 * partition function could not be computed". Refusing on the calling thread
 * is what keeps it.
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

#define N_TRAIN 14
static const char *TRAIN[N_TRAIN] = {
    "CASSF",   "CASSY",   "CAGSF",  "CGSSF",   "CASSL",  "CTSSF",  "CASSQ",
    "CASSFGH", "CASSYWM", "CAGSFK", "CGSSFPD", "CASSLN", "CTSSFR", "CASSQEV",
};
static const double WEIGHT[N_TRAIN] = { 40,25,18,15,12,10,9,8,7,6,5,4,3,2 };

static BeamModel *build(uint32_t max_length) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 4u;
    cfg.max_length = max_length;
    cfg.sparse = true;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/** One sweep's scratch for this model's shape. */
static uint64_t sweep_bytes(const BeamModel *m) {
    return beam_logz_sweep_bytes(m->n_states, m->n_residues, m->state_shift);
}

/**
 * A narrower budget changes how many lengths run at once, and nothing else.
 */
static void test_a_narrow_budget_finalizes_to_the_same_bits(void) {
    const uint64_t original = beam_model_memory_budget();

    BeamModel *wide = build(12u);
    ASSERT_MSG(wide != NULL, "the model did not build");
    const uint32_t n = wide->n_len_slots;
    double *want = (double *)malloc((size_t)n * sizeof(double));
    ASSERT_MSG(want != NULL, "out of memory");
    memcpy(want, wide->log_z, (size_t)n * sizeof(double));
    const uint64_t per = sweep_bytes(wide);

    /* Room for exactly two sweeps: enough to run, few enough that the loop
     * really is narrower than the thread count. */
    beam_model_set_memory_budget(per * 2u);
    if (beam_budget_concurrency(per, n) > 2u) {
        beam_model_set_memory_budget(original);
        beam_model_destroy(wide); free(want);
        ASSERT_MSG(false, "the budget did not narrow the loop; nothing tested");
    }
    BeamModel *narrow = build(12u);
    beam_model_set_memory_budget(original);
    ASSERT_MSG(narrow != NULL, "the model did not build under a narrow budget");

    for (uint32_t L = 0; L < n; ++L)
        if (memcmp(&want[L], &narrow->log_z[L], sizeof(double)) != 0) {
            printf("FAIL: log_z[%u] %.17g wide, %.17g narrow\n",
                   L, want[L], narrow->log_z[L]);
            fail_count++;
            beam_model_destroy(wide); beam_model_destroy(narrow); free(want);
            return;
        }
    /* And the scores those partition functions normalise. */
    for (uint32_t k = 0; k < N_TRAIN; ++k) {
        const double a = beam_model_log_pseq(wide, TRAIN[k]);
        const double b = beam_model_log_pseq(narrow, TRAIN[k]);
        if (memcmp(&a, &b, sizeof a) != 0) {
            printf("FAIL: %s scored %.17g wide, %.17g narrow\n", TRAIN[k], a, b);
            fail_count++;
            beam_model_destroy(wide); beam_model_destroy(narrow); free(want);
            return;
        }
    }
    beam_model_destroy(wide); beam_model_destroy(narrow); free(want);
    PASS();
}

/**
 * A budget too small for even one sweep is refused, with the message that
 * says what to do about it.
 *
 * The figure and the remedy come from `beam_budget_refuse`. Before the driver
 * refused on its own thread, this arrived as "a partition function could not
 * be computed" -- true, and useless.
 */
static void test_a_budget_too_small_refuses_and_says_why(void) {
    const uint64_t original = beam_model_memory_budget();

    BeamModel *sized = build(12u);
    ASSERT_MSG(sized != NULL, "the model did not build");
    const uint64_t per = sweep_bytes(sized);
    beam_model_destroy(sized);

    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 4u; cfg.max_length = 12u; cfg.sparse = true;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "the model did not allocate");
    if (beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) != BEAM_OK) {
        beam_model_destroy(m);
        ASSERT_MSG(false, "observation failed");
    }

    beam_clear_error();
    beam_model_set_memory_budget(per / 2u);          /* one does not fit */
    const BeamError e = beam_model_finalize(m);
    const char *msg = beam_error_message();
    char kept[512];
    snprintf(kept, sizeof kept, "%s", msg ? msg : "");
    beam_model_set_memory_budget(original);
    beam_model_destroy(m);

    ASSERT_MSG(e != BEAM_OK, "a budget that fits nothing was accepted");
    ASSERT_MSG(kept[0] != '\0', "refused with no message at all");
    /* The message must carry the figure and the way out, not merely report
     * that something failed. */
    ASSERT_MSG(strstr(kept, "partition-function trellis") != NULL,
               "the message does not name what refused");
    ASSERT_MSG(strstr(kept, "set_memory_budget") != NULL,
               "the message does not say how to proceed");
    PASS();
}

/**
 * The sizing the driver divides by is the sizing the sweep allocates from.
 *
 * Two copies of that arithmetic would drift, and the drift that matters is a
 * driver admitting more lengths than the sweep will accept -- which shows up
 * as a refusal from inside a worker, exactly what this file exists to stop.
 */
static void test_the_driver_and_the_sweep_agree_on_the_size(void) {
    BeamModel *m = build(12u);
    ASSERT_MSG(m != NULL, "the model did not build");
    const uint64_t per = sweep_bytes(m);
    const uint64_t original = beam_model_memory_budget();

    /* Exactly one sweep's worth must admit exactly one length and no more,
     * and must still finalize. One byte less must refuse. */
    beam_model_set_memory_budget(per);
    const uint32_t one = beam_budget_concurrency(per, m->n_len_slots);
    beam_model_set_memory_budget(per - 1u);
    const uint32_t none = beam_budget_concurrency(per, m->n_len_slots);
    beam_model_set_memory_budget(original);
    beam_model_destroy(m);

    ASSERT_MSG(one == 1u, "a budget of exactly one sweep did not admit one");
    ASSERT_MSG(none == 0u, "a budget one byte short still admitted a sweep");
    PASS();
}

int main(void) {
    printf("\nfinalize under the memory budget\n");
    RUN_TEST(test_a_narrow_budget_finalizes_to_the_same_bits);
    RUN_TEST(test_a_budget_too_small_refuses_and_says_why);
    RUN_TEST(test_the_driver_and_the_sweep_agree_on_the_size);

    printf("\n  %d passed, %d failed\n\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
