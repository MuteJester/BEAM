/**
 * @file avail.h
 * @brief Observing the machine: how many cores are free right now.
 *
 * This is the half of the thread policy that reads the world. plan.h is the
 * other half and reads nothing; keeping them in separate files is what makes
 * the decision testable without a stopwatch and the observation testable
 * without a cgroup.
 *
 * Why a free-core estimate exists at all. Measured across three machines, a
 * fixed default is wrong in both directions: a sixteen-core box with fifteen
 * cores free wants eight threads, while a saturated forty-eight-core node
 * wants one, and asking for the core count there cost 3814x. No pure function
 * of the core count returns both, so the policy has to be told what is free.
 *
 * Why the runnable count and not the load average. Both live in
 * /proc/loadavg. Measured against a known number of CPU-bound competitors on
 * sixteen cores:
 *
 *     competitors   loadavg[0]   nr_running
 *               0         0.71            3
 *               4         0.97            7
 *               8         1.99           11
 *              12         3.70           16
 *
 * The one-minute average is wrong by -6 at eight competitors, because it is a
 * decaying average and no short observation lets it converge. Worse, it stays
 * wrong after the load goes away: half a second after twelve spinners exited
 * it still read 9.22, and ten seconds later 8.03, while the runnable count had
 * already returned to 2 and then 1. A per-call decision built on the average
 * would serialise every call for a minute after BEAM's own previous call
 * finished. The runnable count tracks truth with a constant offset and no lag.
 *
 * Its one weakness is that it is a single instantaneous sample, so a quiet
 * instant on a busy machine can read low and a transient can read high. The
 * error is bounded and lands on the safe side: it reports slightly MORE
 * runnable than there are competitors, so the estimate is conservative, and
 * conservative is the cheap direction -- planning too narrow costs about 2x,
 * planning too wide cost up to 3814x.
 *
 * Every reader returns -1 for "could not tell" rather than guessing, and -1
 * means "assume no contention". It is the one answer that cannot be wrong in
 * the expensive direction by more than the ceiling allows: clause 3 of the
 * policy is min(available, 8), so an unknown machine is planned at 8 and the
 * error is bounded by that, not by the core count. (An earlier version of
 * plan.c halved availability instead, and this paragraph used to credit the
 * halving with the conservative answer; the halving is gone and the ceiling
 * does that work now.)
 *
 * What the platforms without /proc do. They have no instantaneous runnable
 * count, and the load average they DO have is ruled out above. So they
 * difference two cumulative CPU-time readings instead -- see
 * beam_competitors_from_cpu_time and beam_competitors_from_samples -- which
 * measures competitors over a window rather than at an instant. It is second
 * best, and the second-best-ness is explicit: it lags, it reports unknown
 * until a thread has a window worth measuring, and every failure path lands
 * on the same -1 those platforms used to return on every call. It cannot be
 * worse than what it replaced; it can only start being right sooner.
 */
#ifndef BEAM_INTERNAL_AVAIL_H
#define BEAM_INTERNAL_AVAIL_H

#include <stdint.h>

/**
 * Runnable scheduling entities, from the fourth field of /proc/loadavg
 * ("running/total"). -1 when the text cannot be read.
 *
 * Takes the text rather than the path so that it can be tested against
 * captured input, including captures from a saturated cluster node.
 */
int beam_parse_nr_running(const char *loadavg_text);

/**
 * Whole cores a cgroup v2 `cpu.max` allows ("<quota> <period>", or "max").
 * -1 when there is no limit or the text cannot be read.
 *
 * A fractional quota FLOORS, with a minimum of one: 1.5 cores of quota cannot
 * run two threads without throttling, and the conservative direction is the
 * cheap one.
 */
int beam_parse_cpu_max(const char *cpu_max_text);

/** Cores the machine has, irrespective of what this process may use. */
int beam_cores_present(void);

/**
 * Cores this process is allowed to run on: the affinity mask, narrowed by a
 * cgroup v2 quota if one is in force. Never above `beam_cores_present()`.
 *
 * This part is exact and static, which is why it comes first. It covers
 * taskset, cpusets, SLURM allocations and container limits. It is NOT enough
 * on its own: a shared login node without cgroups honestly reports every
 * core, and one such node was 99.9% busy while `nproc` said 48.
 */
int beam_allowed_cores(void);

/**
 * `allowed` minus the competitors currently runnable, floored at one and never
 * above `allowed`. Pure, so the arithmetic is tested directly.
 *
 * One is subtracted for the caller, which is runnable by definition at the
 * moment it reads. A non-positive `nr_running` means "unknown", and yields
 * `allowed` unchanged.
 */
int beam_available_from(int allowed, int nr_running);

/**
 * Minimum window worth measuring, in nanoseconds. CPU time is accounted in
 * ticks of roughly 10 ms on both platforms that need the sampled estimate, so
 * a shorter window carries no information: the deltas are zero or one tick
 * and the quotient is noise.
 */
#define BEAM_AVAIL_MIN_WINDOW_NS 20000000u

/**
 * Competitors implied by two cumulative CPU-time samples, in the units
 * `beam_available_from` takes: runnable entities including the caller. -1 when
 * the window is too short to mean anything.
 *
 * This is how a machine with no /proc answers the question Linux answers with
 * the runnable count. The quantity is competitors, NOT total load, which is
 * why `own_ns` is subtracted: a window spanning BEAM's own eight-thread region
 * reads as eight cores busy, and counting that as contention would narrow the
 * next call BECAUSE OF THE PREVIOUS ONE -- the feedback loop this file rejects
 * the load average for. Linux escapes it for free, the runnable count being
 * instantaneous: by the time it reads, BEAM's workers have exited.
 *
 * Both reads are averages over the window rather than instants, so this signal
 * lags where the runnable count does not. The lag is bounded and lands on the
 * safe side: a machine that was busy and is now idle plans narrow, costing
 * about 2x, while the error in the other direction cost up to 3814x. A
 * fraction of a core therefore rounds UP, and a sample whose own time exceeds
 * the system total -- the two clocks are not read atomically -- reports no
 * competition rather than negative competition.
 *
 * Pure, so the arithmetic is tested directly; the platform readers that
 * produce the samples are the only part a test cannot pin to an exact answer.
 */
int beam_competitors_from_cpu_time(uint64_t sys_busy_ns, uint64_t own_ns,
                                   uint64_t wall_ns, int allowed);

/**
 * One thread's previous CPU-time sample. Exposed so that the sample-to-sample
 * step below can be driven from a test rather than from two clocks.
 */
typedef struct {
    uint64_t sys;      /**< Busy CPU nanoseconds across every CPU. */
    uint64_t own;      /**< This process's CPU nanoseconds. */
    uint64_t wall;     /**< Monotonic nanoseconds. */
    int have;          /**< Zero until the first reading is stored. */
} BeamAvailSample;

/**
 * The estimate from a new reading and the previous sample, updating the sample
 * in place. -1 when the pair cannot be interpreted yet.
 *
 * Pure apart from `*prev`, which is the whole point: the three rules that make
 * the sampled estimate work are here rather than in platform code, so a test
 * can pin them on any machine.
 *
 *   1. A thread's FIRST reading becomes the baseline and reports unknown.
 *      There is nothing to difference against.
 *   2. A window below `BEAM_AVAIL_MIN_WINDOW_NS` reports unknown and LEAVES
 *      THE BASELINE ALONE, so it keeps ageing. Replacing it on every call is
 *      the bug that would make the mechanism useless: a tight loop of short
 *      calls would restart the window each time and never reach twenty
 *      milliseconds, so the estimate would stay unknown on exactly the
 *      workload that needs it.
 *   3. A reading that moves backwards in any of the three series -- which the
 *      clocks promise not to do, and which a test must still pin -- resets the
 *      baseline and reports unknown rather than subtracting into an underflow.
 */
int beam_competitors_from_samples(BeamAvailSample *prev, uint64_t sys,
                                  uint64_t own, uint64_t wall, int allowed);

/**
 * The estimate the policy is given: free cores right now, at least one.
 *
 * `BEAM_AVAILABLE_CORES` pins it, for a benchmark leg, a CI run or a test that
 * needs a repeatable plan -- the same role `beam_model_set_threads` plays for
 * the cap. A pin that is not a positive integer is ignored rather than obeyed,
 * and a pin above the cores present is clamped: a typo must not be able to
 * serialise the library or to claim a thousand free cores.
 */
int beam_available_cores(void);

/**
 * The same estimate, also reporting whether it was MEASURED or assumed.
 * `*measured` becomes 1 when the machine answered and 0 when the estimate fell
 * back to "assume no contention". `measured` may be NULL.
 *
 * One read, both answers, which is the point: asking a second time would take
 * a second sample, and on the sampled platforms a sample consumes the window
 * it is measured over -- so a probe that re-read would report "not measured"
 * almost always, and would perturb the signal it claims to describe.
 *
 * This exists because an idle machine and a dead code path produce the same
 * number. On a quiet runner `available == allowed` whether the reading worked
 * or returned -1, so without this flag there is no way to tell a working
 * estimate from one that never ran -- and CI would pass either way.
 */
int beam_available_cores_ex(int *measured);

#endif /* BEAM_INTERNAL_AVAIL_H */
