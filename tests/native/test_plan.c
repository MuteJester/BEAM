/**
 * @file test_plan.c
 * @brief Tests for the parallel thread-count policy.
 *
 * The policy is a pure function of an explicit input struct, which is the
 * whole point of its design: every property below is decided by arithmetic on
 * values this file supplies, so nothing here times anything, reads the
 * machine, or depends on how busy the box is. A parallelism test that needs a
 * stopwatch is a test people learn to ignore.
 *
 * The expected values are written out by hand rather than computed with the
 * same expression the policy uses. Reusing the expression would make the
 * suite agree with a transposed clamp instead of catching it.
 *
 * The `available / 2` rule is measured, not chosen: on sixteen cores the
 * optimum/free ratio came out 0.50 at three of four induced-load points, and
 * the floor of one is what makes a saturated machine correct. The numbers are
 * in .private/parallelism/phase0/induced_load.txt.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "beam/common.h"
#include "beam/model.h"        /* beam_model_set_threads, _effective_threads */
#include "beam/model_state.h"  /* beam_threads */
#include "beam/budget.h"       /* beam_budget_concurrency */
#include "beam/plan.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "test_utils.h"

/* ── Helpers ─────────────────────────────────────────────────── */

/*
 * A baseline every test mutates one field of. Auto by default -- no explicit
 * cap, no OMP_NUM_THREADS, nothing binding -- so each test isolates the one
 * clause it is about.
 */
static BeamPlanIn base(void) {
    BeamPlanIn in;
    memset(&in, 0, sizeof in);
    in.user_cap = 0;        /* auto */
    in.omp_env = 0;         /* caller set no OMP_NUM_THREADS */
    in.mem_cap = 0;         /* unmeasured, i.e. not binding */
    in.items = 0;           /* unknown, i.e. not binding */
    in.cores = 16;
    in.available = 16;
    return in;
}

/* ── The measured rule ───────────────────────────────────────── */

static void test_auto_plans_half_the_available_cores(void) {
    /* Hand-written, from the measured table: optimum/free == 0.50. */
    const int avail[] = { 16, 12, 8, 4, 2 };
    const int want[]  = {  8,  6, 4, 2, 1 };
    for (size_t i = 0; i < sizeof avail / sizeof *avail; ++i) {
        BeamPlanIn in = base();
        in.available = avail[i];
        const int got = beam_plan_threads(&in);
        if (got != want[i]) {
            char msg[128];
            snprintf(msg, sizeof msg,
                     "available=%d should plan %d threads, planned %d",
                     avail[i], want[i], got);
            beam_test_fail(msg);
            return;
        }
    }
    PASS();
}

static void test_a_saturated_machine_plans_one_thread(void) {
    /*
     * The case that refuted every constant rule: ig05, 48 cores, zero free,
     * where one thread was 3814x faster than the core count. The floor is
     * what produces it, so it is tested as behaviour and not as an edge case.
     */
    BeamPlanIn in = base();
    in.cores = 48;
    in.available = 0;
    ASSERT_MSG(beam_plan_threads(&in) == 1,
               "no free cores must plan exactly one thread");
    in.available = 1;
    ASSERT_MSG(beam_plan_threads(&in) == 1,
               "one free core must plan exactly one thread");
    PASS();
}

static void test_the_plan_never_drops_below_one(void) {
    /* Zero would ask OpenMP for an empty team; negative is nonsense. */
    const int avails[] = { -8, -1, 0, 1 };
    for (size_t i = 0; i < sizeof avails / sizeof *avails; ++i) {
        BeamPlanIn in = base();
        in.available = avails[i];
        ASSERT_MSG(beam_plan_threads(&in) >= 1,
                   "the plan must never be below one thread");
    }
    PASS();
}

static void test_more_free_cores_never_plans_fewer_threads(void) {
    /*
     * Monotonicity is the property that makes the rule explicable to a user:
     * freeing up the machine can never make BEAM ask for less of it.
     */
    int prev = 0;
    for (int a = 0; a <= 256; ++a) {
        BeamPlanIn in = base();
        in.cores = 256;
        in.available = a;
        const int got = beam_plan_threads(&in);
        if (got < prev) {
            char msg[128];
            snprintf(msg, sizeof msg,
                     "available=%d planned %d, less than %d at available=%d",
                     a, got, prev, a - 1);
            beam_test_fail(msg);
            return;
        }
        prev = got;
    }
    PASS();
}

/* ── Explicit caller intent wins ─────────────────────────────── */

static void test_an_explicit_cap_is_honoured_exactly(void) {
    /*
     * Decision 6, and it is what keeps a pinned benchmark reproducible: the
     * paper's supplementary performance table states its fits used 16 BEAM
     * threads and one thread respectively, so those runs must reproduce
     * whatever the machine looks like.
     */
    BeamPlanIn in = base();
    in.user_cap = 16;
    in.available = 0;           /* a saturated machine must not reduce it */
    ASSERT_MSG(beam_plan_threads(&in) == 16,
               "an explicit cap of 16 must plan 16 even with no free cores");
    in.user_cap = 1;
    in.available = 64;          /* an idle machine must not raise it */
    ASSERT_MSG(beam_plan_threads(&in) == 1,
               "an explicit cap of 1 must plan 1 even on an idle machine");
    PASS();
}

static void test_an_omp_env_setting_is_not_overridden(void) {
    /*
     * OMP_NUM_THREADS is the caller expressing intent through OpenMP. The
     * policy stands aside for it exactly as it does for an explicit cap;
     * silently ignoring it would make BEAM the one library on the machine
     * that disregards it.
     */
    BeamPlanIn in = base();
    in.omp_env = 1;
    in.cores = 16;
    in.available = 2;
    ASSERT_MSG(beam_plan_threads(&in) == 16,
               "with OMP_NUM_THREADS set the runtime's count must stand");
    PASS();
}

/* ── Binding caps ────────────────────────────────────────────── */

static void test_the_memory_cap_is_never_exceeded(void) {
    /*
     * This one is memory safety, not latency. A region's scratch is allocated
     * for the planned count (math/spectrum.c:402 sizes wk->scr by wk->nth),
     * so a plan wider than the budget allowed is a heap overflow, and
     * widening a budget of zero turns a clean refusal into an allocation
     * failure.
     */
    for (uint32_t cap = 1; cap <= 8; ++cap) {
        BeamPlanIn in = base();
        in.available = 64;
        in.cores = 64;
        in.mem_cap = cap;
        const int got = beam_plan_threads(&in);
        if (got > (int)cap) {
            char msg[128];
            snprintf(msg, sizeof msg,
                     "mem_cap=%u was exceeded: planned %d", cap, got);
            beam_test_fail(msg);
            return;
        }
    }
    PASS();
}

static void test_an_explicit_cap_does_not_override_the_memory_cap(void) {
    /* Decision 6 is about availability. Memory safety still binds. */
    BeamPlanIn in = base();
    in.user_cap = 32;
    in.mem_cap = 4;
    in.available = 64;
    ASSERT_MSG(beam_plan_threads(&in) <= 4,
               "an explicit cap must not widen the memory cap");
    PASS();
}

static void test_no_more_threads_than_work_items(void) {
    /* Team members with no iteration to run cost a barrier and nothing else. */
    for (uint64_t n = 1; n <= 8; ++n) {
        BeamPlanIn in = base();
        in.available = 64;
        in.cores = 64;
        in.items = n;
        const int got = beam_plan_threads(&in);
        if (got > (int)n) {
            char msg[128];
            snprintf(msg, sizeof msg,
                     "items=%llu but planned %d threads",
                     (unsigned long long)n, got);
            beam_test_fail(msg);
            return;
        }
    }
    PASS();
}

/* ── Purity and degenerate input ─────────────────────────────── */

static void test_the_same_input_always_plans_the_same_count(void) {
    /*
     * Purity is what lets the resolved plan be cached for a call's duration,
     * and that caching is required: beam_threads() is read once and used both
     * to size buffers and to set num_threads, so two reads that disagree
     * would size a buffer for one team and run another.
     */
    BeamPlanIn in = base();
    in.available = 11;
    in.mem_cap = 7;
    in.items = 5;
    const int first = beam_plan_threads(&in);
    for (int i = 0; i < 100; ++i) {
        ASSERT_MSG(beam_plan_threads(&in) == first,
                   "the policy must be a pure function of its input");
    }
    PASS();
}

static void test_degenerate_input_is_defined(void) {
    /* A library must not crash or return nonsense on a zeroed struct. */
    BeamPlanIn in;
    memset(&in, 0, sizeof in);
    ASSERT_MSG(beam_plan_threads(&in) == 1,
               "an all-zero input must plan exactly one thread");

    in = base();
    in.cores = 0;
    ASSERT_MSG(beam_plan_threads(&in) >= 1, "cores=0 must still plan >= 1");

    in = base();
    in.available = 1000000;
    in.cores = 16;
    ASSERT_MSG(beam_plan_threads(&in) >= 1,
               "an absurd availability must still plan >= 1");
    PASS();
}

static void test_availability_above_the_core_count_is_clamped(void) {
    /*
     * A machine cannot have more free cores than cores. If the availability
     * signal says otherwise -- a misparsed cgroup quota, a `max` read as a
     * number -- the plan must not act on it. Without this clamp a bad parse
     * asks OpenMP for hundreds of thousands of threads.
     */
    BeamPlanIn in = base();
    in.cores = 16;
    in.available = 1000000;
    ASSERT_MSG(beam_plan_threads(&in) <= 16,
               "availability above the core count must be clamped to cores");
    PASS();
}

static void test_a_null_input_plans_one_thread(void) {
    /* Serial is the only safe answer when there is nothing to decide from. */
    ASSERT_MSG(beam_plan_threads(NULL) == 1,
               "a NULL input must plan exactly one thread");
    PASS();
}


/* ── Resolving the plan once per call ────────────────────────── */

/*
 * These are the tests that matter most in this file, and the reason is not
 * latency. A resolved thread count is read once and then used BOTH to size
 * scratch and to set num_threads: math/spectrum.c caches it in wk->nth, sizes
 * wk->scr by it at line 402, and names it in four clauses. If two reads inside
 * one call could disagree, a team would run wider than the buffer allocated
 * for it. So stability inside a scope is a memory-safety property.
 */

static void test_beam_threads_is_stable_inside_a_plan_scope(void) {
    BeamPlanIn in = base();
    in.available = 12;                      /* plans 6 */
    const int token = beam_plan_enter(&in);
    const int first = beam_threads();
    int ok = (first == 6);
    /* Change the inputs underneath the scope: the active plan must not move. */
    in.available = 2;
    in.user_cap = 1;
    for (int i = 0; i < 100 && ok; ++i) ok = (beam_threads() == first);
    beam_plan_leave(token);
    ASSERT_MSG(ok, "beam_threads() must not change inside one plan scope");
    PASS();
}

static void test_leaving_a_plan_scope_restores_the_previous_value(void) {
    const int before = beam_threads();
    BeamPlanIn in = base();
    in.available = 4;                       /* plans 2 */
    const int token = beam_plan_enter(&in);
    const int inside = beam_threads();
    beam_plan_leave(token);
    const int after = beam_threads();
    ASSERT_MSG(inside == 2, "a plan of 2 must be visible inside its scope");
    ASSERT_MSG(after == before,
               "leaving a scope must restore the previous thread count");
    PASS();
}

static void test_plan_scopes_nest(void) {
    /*
     * An entry point may call another. Without save-and-restore the inner
     * leave would clear the outer plan, and the rest of the outer call would
     * silently run at a different width than the buffers it had sized.
     */
    const int before = beam_threads();
    BeamPlanIn outer = base();
    outer.available = 16;                   /* plans 8 */
    const int t_outer = beam_plan_enter(&outer);
    const int saw_outer = beam_threads();

    BeamPlanIn inner = base();
    inner.available = 4;                    /* plans 2 */
    const int t_inner = beam_plan_enter(&inner);
    const int saw_inner = beam_threads();
    beam_plan_leave(t_inner);
    const int saw_outer_again = beam_threads();
    beam_plan_leave(t_outer);
    const int after = beam_threads();

    ASSERT_MSG(saw_outer == 8, "the outer plan must be 8");
    ASSERT_MSG(saw_inner == 2, "the inner plan must be 2");
    ASSERT_MSG(saw_outer_again == 8,
               "leaving the inner scope must restore the outer plan");
    ASSERT_MSG(after == before, "leaving both scopes must restore the original");
    PASS();
}

static void test_with_no_plan_active_nothing_changes(void) {
    /*
     * Everything above this task is inert: with no scope entered,
     * beam_threads() must still be the cap, or the runtime's maximum when no
     * cap is set. This is what lets the policy land without moving a number.
     */
    beam_model_set_threads(5);
    ASSERT_MSG(beam_threads() == 5, "with a cap of 5 and no plan, expect 5");
    beam_model_set_threads(0);
    ASSERT_MSG(beam_threads() == beam_model_effective_threads(),
               "with no cap and no plan, expect the runtime's maximum");
    PASS();
}

static void test_effective_threads_reports_the_cap_not_the_plan(void) {
    /*
     * beam_model_effective_threads() is public API. It answers "what is
     * configured", and the plan answers "what will run"; conflating them would
     * break a user's introspection and the recipe for reproducing the paper's
     * pinned runs. The two questions get two functions.
     */
    beam_model_set_threads(8);
    BeamPlanIn in = base();
    in.available = 4;                       /* plans 2, well below the cap */
    const int token = beam_plan_enter(&in);
    const int eff = beam_model_effective_threads();
    const int thr = beam_threads();
    beam_plan_leave(token);
    beam_model_set_threads(0);
    ASSERT_MSG(eff == 8, "effective_threads() must still report the cap of 8");
    ASSERT_MSG(thr == 2, "beam_threads() must report the active plan of 2");
    PASS();
}

static void test_budget_concurrency_follows_the_active_plan(void) {
    /*
     * Seventeen num_threads clauses take their count from
     * beam_budget_concurrency, which caps by the thread count internally. If
     * it kept reading the cap rather than the plan, those regions would ignore
     * the policy entirely and the work would be half done.
     */
    BeamPlanIn in = base();
    in.available = 4;                       /* plans 2 */
    const int token = beam_plan_enter(&in);
    /* per_item of 1 byte, so the budget is not what binds here. */
    const uint32_t conc = beam_budget_concurrency(1u, 8u);
    beam_plan_leave(token);
    char msg[128];
    snprintf(msg, sizeof msg,
             "with a plan of 2 over 8 items, concurrency should be 2, got %u",
             conc);
    ASSERT_MSG(conc == 2u, msg);
    PASS();
}

static void test_a_plan_does_not_reach_another_thread(void) {
    /*
     * The plan is thread-local ON PURPOSE, and this test pins that choice
     * rather than leaving it to be rediscovered.
     *
     * A process-global plan would be seen by OpenMP workers, which sounds
     * useful, but two BEAM calls on two Python threads would then overwrite
     * each other's plan mid-call -- exactly the buffer-versus-team mismatch
     * this task exists to prevent. Thread-local cannot do that.
     *
     * Workers do not need it: num_threads is evaluated by the thread that
     * ENCOUNTERS the pragma, and every non-pragma read in the library caches
     * the value on the master before any region starts.
     */
#ifdef _OPENMP
    if (omp_get_max_threads() < 2) BEAM_SKIP("needs at least two OpenMP threads");
    /*
     * A cap of 7 so that "the worker fell back" is distinguishable from "the
     * worker inherited the plan". The first version of this test asserted only
     * that the worker did not see 2, the plan's value, and passed by luck when
     * run directly: under the suite runner, which caps each suite at two
     * threads, the fallback was ALSO 2, the assertion compared 2 against 2 and
     * the test failed for the right reason by accident. A test whose two
     * outcomes can collide onto one number is not testing anything.
     */
    beam_model_set_threads(7);
    BeamPlanIn in = base();
    in.available = 4;                       /* plans 2, which is never 7 */
    const int token = beam_plan_enter(&in);
    const int master_saw = beam_threads();
    int worker_saw = -1;
    #pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 1) worker_saw = beam_threads();
    }
    beam_plan_leave(token);
    beam_model_set_threads(0);              /* restored before any assert */
    ASSERT_MSG(master_saw == 2,
               "the thread that entered the scope must see the plan, 2");
    ASSERT_MSG(worker_saw == 7,
               "a worker must fall back to the cap, 7, not inherit the plan");
    PASS();
#else
    BEAM_SKIP("no OpenMP in this build");
#endif
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    beam_test_init(argc, argv);
    printf("\nBEAM Unit Tests - Thread-count policy\n");
    printf("==============================================================\n\n");

    RUN_TEST(test_auto_plans_half_the_available_cores);
    RUN_TEST(test_a_saturated_machine_plans_one_thread);
    RUN_TEST(test_the_plan_never_drops_below_one);
    RUN_TEST(test_more_free_cores_never_plans_fewer_threads);
    RUN_TEST(test_an_explicit_cap_is_honoured_exactly);
    RUN_TEST(test_an_omp_env_setting_is_not_overridden);
    RUN_TEST(test_the_memory_cap_is_never_exceeded);
    RUN_TEST(test_an_explicit_cap_does_not_override_the_memory_cap);
    RUN_TEST(test_no_more_threads_than_work_items);
    RUN_TEST(test_the_same_input_always_plans_the_same_count);
    RUN_TEST(test_degenerate_input_is_defined);
    RUN_TEST(test_availability_above_the_core_count_is_clamped);
    RUN_TEST(test_a_null_input_plans_one_thread);

    RUN_TEST(test_beam_threads_is_stable_inside_a_plan_scope);
    RUN_TEST(test_leaving_a_plan_scope_restores_the_previous_value);
    RUN_TEST(test_plan_scopes_nest);
    RUN_TEST(test_with_no_plan_active_nothing_changes);
    RUN_TEST(test_effective_threads_reports_the_cap_not_the_plan);
    RUN_TEST(test_budget_concurrency_follows_the_active_plan);
    RUN_TEST(test_a_plan_does_not_reach_another_thread);

    printf("\n==============================================================\n");
    return beam_test_finish();
}
