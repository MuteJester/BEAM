/**
 * @file plan.h
 * @brief The parallel thread-count policy: one pure decision, named once.
 *
 * BEAM has 93 parallel regions, and until this file the question "how many
 * threads" was answered in 92 clauses from three different sources: the
 * explicit cap, the memory budget, and local variables derived from either.
 * This header makes every one of those an INPUT to a single pure function, so
 * that the decision can be unit-tested exhaustively with no stopwatch and
 * read in one place.
 *
 * Why a pure function and not a helper that reads the machine itself: a
 * parallelism rule that consults globals can only be tested by timing it, and
 * a timing test fails when the box is busy, which teaches people to ignore
 * the suite. Everything the rule needs arrives in BeamPlanIn.
 *
 * Why the default is not the core count. Measured on three machines: asking
 * for one thread per core costs between 1.7x and 3814x the optimum, and the
 * measurement goes bimodal there (process-to-process spread up to 427x), so
 * the old default is unpredictable as well as slow. No constant fraction of
 * the core count fixes it either -- half the cores is 301x worse than serial
 * on a saturated 48-core node. The governing quantity is FREE cores, and the
 * optimum is about half of them. See .private/parallelism/ for the data.
 */
#ifndef BEAM_INTERNAL_PLAN_H
#define BEAM_INTERNAL_PLAN_H

#include <stdint.h>

/**
 * Everything the policy is allowed to consider, gathered by the caller.
 *
 * Read ONCE per public API call and held immutable for that call's duration.
 * That is not an optimisation: a resolved count is used both to size scratch
 * and to set `num_threads` (math/spectrum.c caches it in `wk->nth`, sizes
 * `wk->scr` by it, and names it in four clauses), so two reads that disagreed
 * would allocate for one team and then run a wider one.
 */
typedef struct {
    int      user_cap;   /**< Explicit cap from set_threads(); 0 means auto. */
    int      omp_env;    /**< 1 when the caller set OMP_NUM_THREADS.         */
    uint32_t mem_cap;    /**< Budget concurrency; 0 means "not binding".     */
    uint64_t items;      /**< Work items available to split; 0 if unknown.   */
    int      cores;      /**< Cores present, from omp_get_num_procs().       */
    int      available;  /**< Estimated free cores.                          */
} BeamPlanIn;

/**
 * How many threads a region should ask for. Never below 1; 1 means serial.
 *
 * The clauses, in the order they apply and with the authority each carries:
 *
 *  1. An explicit `user_cap` is honoured exactly. The caller has decided, and
 *     a library that quietly ignores a direct instruction cannot be reasoned
 *     about -- it is also what keeps a pinned benchmark reproducible.
 *  2. Otherwise, `OMP_NUM_THREADS` set by the caller stands. It is the same
 *     intent expressed through OpenMP, and BEAM should not be the one library
 *     on the machine that disregards it.
 *  3. Otherwise, the available cores, capped by where this workload stops
 *     scaling. Both limits measured, neither chosen.
 *  4. `mem_cap` then binds and is never widened. Exceeding it is a memory
 *     fault rather than a slowdown, and widening a cap of zero would turn a
 *     clean refusal into an allocation failure.
 *  5. `items` binds: a team member with no iteration costs a barrier and
 *     contributes nothing.
 *  6. Floor at one.
 *
 * `in` may be NULL, which plans one thread: serial is the only safe answer
 * when there is nothing to decide from.
 */
int beam_plan_threads(const BeamPlanIn *in);

/* ── The plan in force, for one call on one thread ───────────── */

/**
 * Resolve the plan for one public API call and make it current on THIS
 * thread. Returns a token to hand to `beam_plan_leave`; scopes nest, so an
 * entry point may call another without the inner one clearing the outer plan.
 *
 * Resolve ONCE per call and hold it. That is not an efficiency choice: a
 * resolved count is used both to size scratch and to set `num_threads`
 * (math/spectrum.c caches it in `wk->nth`, sizes `wk->scr` by it at line 402,
 * and names it in four clauses), so two reads that disagreed would allocate
 * for one team and then run a wider one.
 *
 * The current plan is THREAD-LOCAL, deliberately. A process-global one would
 * be visible to OpenMP workers, which sounds useful, but two BEAM calls on two
 * threads would then overwrite each other's plan mid-call -- the same
 * buffer-versus-team mismatch, arrived at from the other direction. Workers do
 * not need it: `num_threads` is evaluated by the thread that ENCOUNTERS the
 * pragma, and every non-pragma read caches the value on the master before any
 * region starts.
 */
int  beam_plan_enter(const BeamPlanIn *in);

/** Restore the plan that `beam_plan_enter` replaced. */
void beam_plan_leave(int token);

/**
 * Threads the next parallel region should use: the plan in force on this
 * thread if there is one, otherwise the cap from `beam_model_set_threads`,
 * otherwise the OpenMP runtime's own maximum. Always 1 without OpenMP.
 *
 * Every `omp parallel` in the implementation carries `num_threads(...)` with
 * this, so the cap is a hard cap rather than a hint and one stray region
 * cannot go wide behind the caller's back.
 *
 * This is NOT `beam_model_effective_threads()`, which is public API and
 * answers the different question "what is configured". Conflating the two
 * would break a caller's introspection and the recipe for reproducing a
 * pinned benchmark.
 */
int beam_threads(void);

#endif /* BEAM_INTERNAL_PLAN_H */
