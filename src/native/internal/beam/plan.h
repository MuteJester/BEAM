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
    int      omp_max;    /**< omp_get_max_threads(): respects OMP_NUM_THREADS.*/
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
 *  2. Otherwise, `OMP_NUM_THREADS` set by the caller stands, as `omp_max`. It
 *     is the same intent expressed through OpenMP, and BEAM should not be the
 *     one library on the machine that disregards it.
 *
 *     `omp_max` and `cores` are separate fields because they are different
 *     numbers: `omp_get_num_procs()` ignores OMP_NUM_THREADS entirely. An
 *     earlier version of this policy returned `cores` here, which meant that
 *     OMP_NUM_THREADS=2 on a sixteen-core box planned SIXTEEN threads --
 *     overriding the caller's explicit setting with the very core count this
 *     work exists to stop asking for. The suite runner, which sets that
 *     variable, is what caught it.
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

/**
 * The widest team the policy will ever plan in auto mode: where this
 * workload's speedup flattened. Reported by the probe so that a caller can
 * see WHICH limit bound their call -- the machine or the library.
 */
int beam_plan_max_width(void);

/**
 * The count to size a memory LAYOUT from: the widest team `beam_plan_threads`
 * could return for this `in` if the machine were idle. It reads every field
 * except `available`.
 *
 * A layout is the stretch width of a per-length sweep, the number of
 * stretches, the per-thread workspace, and the figure quoted when one of them
 * does not fit. Derive those from the plan and they move with the machine's
 * load: two identical `log_mellin` calls a second apart quoted "about
 * 2.81 MB of scratch" and "about 2.82 MB", and at a low enough budget the
 * same drift decides whether the call is REFUSED. The plan is supposed to
 * follow the weather; a figure the caller is shown, and a refusal they must
 * act on, are not.
 *
 * Sizing for more threads than turn up is the safe direction and the reason
 * this works at all: a wider team divides the same budget into NARROWER
 * stretches, so a narrower team stays inside it with room to spare. Sizing
 * for fewer is how a budget check passes and the allocation behind it does
 * not -- which is why `test_the_layout_count_never_undercuts_the_plan` sweeps
 * the property rather than spot-checking it.
 *
 * With `cores` unknown (0) this returns the ceiling rather than 0: the plan
 * trusts `available` alone in that case, and no layout may undercut it.
 */
int beam_plan_layout_threads(const BeamPlanIn *in);

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

/**
 * Fill `in` from the machine, without making it current. The probe needs the
 * same inputs the real path uses, and duplicating the gathering is how the
 * two would drift apart.
 */
void beam_plan_inputs(BeamPlanIn *in);

/**
 * `beam_plan_enter` with the inputs read from the machine: the configured cap,
 * whether the caller set OMP_NUM_THREADS, the cores present and the free-core
 * estimate. This is what a public entry point calls, and the once-per-call
 * read that plan.h's contract requires happens here.
 *
 * The memory cap and the item count are deliberately NOT gathered here. They
 * are known only deeper in, where `beam_budget_concurrency` already applies
 * them on top of the plan, so passing 0 for both means "not binding yet"
 * rather than "unlimited".
 */
int beam_plan_enter_default(void);

/** Restore the plan that `beam_plan_enter` replaced. */
void beam_plan_leave(int token);

/**
 * Threads the next parallel region should use: the plan in force on this
 * thread if there is one, otherwise the cap from `beam_model_set_threads`,
 * otherwise the OpenMP runtime's own maximum. Always 1 without OpenMP.
 *
 * Every `omp parallel` in the implementation carries `num_threads(...)`,
 * naming either this directly (64 regions) or a count already resolved from
 * it and the memory budget -- `conc`, `nth`, `team` (29). So the cap is a
 * hard cap rather than a hint and one stray region cannot go wide behind the
 * caller's back. That is verified rather than asserted:
 * `check_lib.thread_count_clauses` fails the gate on a region that names no
 * count, and on one that names an expression outside the permitted set. It
 * was written because `occupancy/poisson_binomial/pgf.c` was exactly such a
 * stray region, and nothing noticed.
 *
 * This is NOT `beam_model_effective_threads()`, which is public API and
 * answers the different question "what is configured". Conflating the two
 * would break a caller's introspection and the recipe for reproducing a
 * pinned benchmark.
 */
int beam_threads(void);

/**
 * `beam_plan_layout_threads` with the inputs read from the machine, and never
 * below what `beam_threads()` is about to report.
 *
 * That floor is not decoration. `make test` builds the C library on its own,
 * where nothing enters a plan and `beam_threads()` falls back to the
 * configured cap -- which can be wider than the ceiling. A layout sized at
 * the ceiling would then be short of the team that actually runs.
 */
int beam_layout_threads(void);

#endif /* BEAM_INTERNAL_PLAN_H */
