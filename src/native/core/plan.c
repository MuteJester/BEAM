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
 * The free cores, capped by how wide this workload actually scales.
 *
 * Two different limits, and both are needed. Availability bounds the plan from
 * below -- a team wider than the free cores thrashes, which is the 3814x
 * finding. BEAM's own parallel efficiency bounds it from above: measured on an
 * idle box across four calls and two model sizes, eight threads captures
 * 93-100% of each call's own best speedup, and more than that buys nothing
 * because the regions stop scaling, not because the machine runs out.
 *
 * Fitted against the measured optima, with availability from the live signal:
 *
 *     available   optimum   min(available, 8)   available / 2
 *            14         8                   8               7
 *            10         6                   8               6
 *             6         6                   6               3
 *             1         2                   1               1
 *
 * An earlier version of this file halved availability instead. That was
 * calibrated against free cores counted as (cores - competitors), which
 * ignores the two or three other runnable entities any real machine has, so
 * it halved a number that was already contention-aware -- conservative twice,
 * and 33% and 69% off the optimum at the two busiest points. The ceiling form
 * is worst by 8%.
 *
 * Note that `min(cores, 8)` WITHOUT the availability term was measured at 186x
 * worse than serial on a saturated node. The two terms are not alternatives.
 *
 * The ceiling is the one number here that other core counts could move: eight
 * is where scaling flattened on sixteen cores, and a machine with far more
 * cores might flatten later. Confirming it elsewhere is the open measurement.
 */
#define BEAM_PLAN_MAX_WIDTH 8

int beam_plan_threads(const BeamPlanIn *in) {
    /* Serial is the only safe answer when there is nothing to decide from. */
    if (!in) return 1;

    int n;

    if (in->user_cap > 0) {
        /* Clause 1: the caller has decided. Availability does not revise it. */
        n = in->user_cap;
    } else if (in->omp_env) {
        /* Clause 2: OMP_NUM_THREADS is the same intent through OpenMP, so the
         * runtime's own answer stands.
         *
         * `omp_max`, NOT `cores`: omp_get_num_procs() ignores OMP_NUM_THREADS,
         * so reading `cores` here would answer OMP_NUM_THREADS=2 with sixteen
         * threads -- overriding the caller with the core count. */
        n = in->omp_max > 0 ? in->omp_max : (in->cores > 0 ? in->cores : 1);
    } else {
        /* Clause 3: the free cores, capped by where scaling flattens.
         *
         * `available` is clamped to `cores` first, because a machine cannot
         * have more free cores than cores and a misparsed cgroup quota (a
         * `max` read as a number, say) would otherwise be acted on -- asking
         * OpenMP for hundreds of thousands of threads. */
        int avail = in->available;
        if (avail < 0) avail = 0;
        if (in->cores > 0 && avail > in->cores) avail = in->cores;
        n = avail < BEAM_PLAN_MAX_WIDTH ? avail : BEAM_PLAN_MAX_WIDTH;
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
