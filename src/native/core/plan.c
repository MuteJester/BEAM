/**
 * @file plan.c
 * @brief The parallel thread-count policy.
 *
 * One function, no globals, no OpenMP calls. The caller gathers the inputs;
 * this file only decides. See plan.h for why the default stopped being the
 * core count and what each clause's authority is.
 */
#include "beam/plan.h"

/*
 * Half the free cores.
 *
 * Measured rather than chosen. On sixteen cores, with a known number of
 * CPU-bound competitors inducing a known number of busy cores, the optimum
 * thread count against free cores came out:
 *
 *     free 16 -> 8    free 12 -> 6    free 8 -> 6    free 4 -> 2
 *
 * which is a ratio of 0.50 at three of the four points, and 0.75 at the
 * fourth. Planning free/2 there costs 12% (4.623 ms against 4.122 ms); that
 * is the error one constant makes across the range, and it is small against
 * the 25x to 461x that asking for every core costs at the same load levels.
 *
 * The divisor is named rather than inlined so that the one number this policy
 * contains is greppable, and so the test that pins it reads the same way.
 */
#define BEAM_PLAN_HEADROOM_DIVISOR 2

int beam_plan_threads(const BeamPlanIn *in) {
    /* Serial is the only safe answer when there is nothing to decide from. */
    if (!in) return 1;

    int n;

    if (in->user_cap > 0) {
        /* Clause 1: the caller has decided. Availability does not revise it. */
        n = in->user_cap;
    } else if (in->omp_env) {
        /* Clause 2: OMP_NUM_THREADS is the same intent through OpenMP. The
         * runtime has already folded it into the core count the caller sees,
         * so standing aside means using that. */
        n = in->cores > 0 ? in->cores : 1;
    } else {
        /* Clause 3: half the free cores.
         *
         * `available` is clamped to `cores` first, because a machine cannot
         * have more free cores than cores and a misparsed cgroup quota (a
         * `max` read as a number, say) would otherwise be acted on -- asking
         * OpenMP for hundreds of thousands of threads. */
        int avail = in->available;
        if (avail < 0) avail = 0;
        if (in->cores > 0 && avail > in->cores) avail = in->cores;
        n = avail / BEAM_PLAN_HEADROOM_DIVISOR;
    }

    /* Clause 4: memory safety binds, and is never widened -- not even by an
     * explicit cap. A plan wider than the budget allowed is a buffer sized
     * for one team and a wider team running against it. */
    if (in->mem_cap > 0u && n > (int)in->mem_cap) n = (int)in->mem_cap;

    /* Clause 5: no more threads than there is work to divide. The cast is
     * safe because it only happens when `items` is already below `n`. */
    if (in->items > 0u && (uint64_t)n > in->items) n = (int)in->items;

    /* Clause 6. Zero would ask OpenMP for an empty team. */
    return n < 1 ? 1 : n;
}
