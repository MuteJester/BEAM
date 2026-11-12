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
 * Every parser returns -1 for "could not tell" rather than guessing. -1 then
 * means "assume no contention", and the policy's own halving supplies the
 * conservative answer. That is also exactly what a machine with no /proc at
 * all gets, so the non-Linux fallback is not a special case anywhere.
 */
#ifndef BEAM_INTERNAL_AVAIL_H
#define BEAM_INTERNAL_AVAIL_H

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

/** The estimate the policy is given: free cores right now, at least one. */
int beam_available_cores(void);

#endif /* BEAM_INTERNAL_AVAIL_H */
