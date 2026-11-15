/**
 * @file avail.c
 * @brief How many cores are free right now.
 *
 * See avail.h for why this exists, why it reads the runnable count rather than
 * the load average, and why every parser returns -1 instead of guessing.
 *
 * The parsers take text, not paths. Only the two readers at the bottom touch
 * the filesystem, and they are the only part of this file a test cannot pin to
 * an exact answer.
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "beam/avail.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <sched.h>
#endif

/*
 * The sampled estimate, for the platforms with no /proc. See avail.h for why
 * the quantity is competitors rather than load, and why that costs a second
 * clock read.
 */
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#ifdef _MSC_VER
  #define TLS __declspec(thread)
#else
  #define TLS _Thread_local
#endif

/* ── Parsers ─────────────────────────────────────────────────── */

int beam_parse_nr_running(const char *loadavg_text) {
    if (!loadavg_text) return -1;
    /*
     * Walk to the fourth whitespace-separated field rather than scanning the
     * three averages, so a change in how they are formatted cannot shift which
     * field is read. The field is "running/total" and only the first half is
     * wanted.
     */
    const char *p = loadavg_text;
    for (int field = 0; field < 3; ++field) {
        while (*p == ' ' || *p == '\t') ++p;
        if (!*p || *p == '\n') return -1;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') ++p;
    }
    while (*p == ' ' || *p == '\t') ++p;
    if (*p < '0' || *p > '9') return -1;        /* also rejects a leading '-' */
    char *end = NULL;
    const long n = strtol(p, &end, 10);
    /* The slash must actually be there: a bare number is a different format. */
    if (!end || end == p || *end != '/') return -1;
    if (n < 0 || n > (1L << 24)) return -1;
    return (int)n;
}

int beam_parse_cpu_max(const char *cpu_max_text) {
    if (!cpu_max_text) return -1;
    const char *p = cpu_max_text;
    while (*p == ' ' || *p == '\t') ++p;
    /* "max" is cgroup v2's spelling for "no limit", and is not a cap. */
    if (strncmp(p, "max", 3) == 0) return -1;
    char *end = NULL;
    const long quota = strtol(p, &end, 10);
    if (!end || end == p || quota <= 0) return -1;
    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p < '0' || *p > '9') return -1;        /* a period must follow */
    const long period = strtol(p, &end, 10);
    if (!end || end == p || period <= 0) return -1;
    /*
     * Floor, with a minimum of one. 1.5 cores of quota cannot run two threads
     * without being throttled, and phase 0 measured the asymmetry: planning
     * too wide cost up to 3814x, planning too narrow costs about 2x.
     */
    const long cores = quota / period;
    if (cores < 1) return 1;
    if (cores > (1L << 20)) return -1;
    return (int)cores;
}

/* ── Composition ─────────────────────────────────────────────── */

int beam_available_from(int allowed, int nr_running) {
    if (allowed < 1) allowed = 1;
    /*
     * Non-positive means the runnable count could not be read. Assume no
     * contention: the policy halves whatever it is given, so this yields the
     * conservative constant that a machine with no /proc should get, with no
     * special case anywhere.
     */
    if (nr_running < 1) return allowed;
    const int busy = nr_running - 1;            /* minus the caller itself */
    const int avail = allowed - busy;
    return avail < 1 ? 1 : avail;
}

int beam_competitors_from_cpu_time(uint64_t sys_busy_ns, uint64_t own_ns,
                                   uint64_t wall_ns, int allowed) {
    if (allowed < 1) allowed = 1;
    if (wall_ns < BEAM_AVAIL_MIN_WINDOW_NS) return -1;
    /*
     * Our own time can exceed the system total across a non-atomic pair of
     * reads. Equal means the machine ran nothing but us.
     */
    if (sys_busy_ns <= own_ns) return 1;
    const uint64_t other = sys_busy_ns - own_ns;
    /* Ceiling: half a core of competition is one competitor, not none. */
    uint64_t cores = (other + wall_ns - 1u) / wall_ns;
    if (cores > (uint64_t)allowed) cores = (uint64_t)allowed;
    return (int)cores + 1;
}

int beam_competitors_from_samples(BeamAvailSample *prev, uint64_t sys,
                                  uint64_t own, uint64_t wall, int allowed) {
    if (!prev) return -1;

    if (!prev->have) {                              /* rule 1 */
        prev->sys = sys; prev->own = own; prev->wall = wall; prev->have = 1;
        return -1;
    }
    if (wall <= prev->wall || sys < prev->sys || own < prev->own) {  /* rule 3 */
        prev->sys = sys; prev->own = own; prev->wall = wall;
        return -1;
    }
    const uint64_t window = wall - prev->wall;
    if (window < BEAM_AVAIL_MIN_WINDOW_NS) return -1;               /* rule 2 */

    const int nr = beam_competitors_from_cpu_time(sys - prev->sys,
                                                  own - prev->own,
                                                  window, allowed);
    prev->sys = sys; prev->own = own; prev->wall = wall;
    return nr;
}

/* ── Reading the machine ─────────────────────────────────────── */

int beam_cores_present(void) {
#ifdef _OPENMP
    const int n = omp_get_num_procs();
    return n > 0 ? n : 1;
#else
    return 1;
#endif
}

/* Whole text of a small file, or an empty string. Never partial. */
static int read_small_file(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "re");
    if (!f) return 0;
    const size_t n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = '\0';
    return n > 0;
}

int beam_allowed_cores(void) {
    const int present = beam_cores_present();
    int allowed = present;

#if defined(__linux__)
    /*
     * The affinity mask is exact and static, and covers taskset, cpusets,
     * SLURM allocations and container pinning. CPU_COUNT over the mask this
     * process actually has, not the mask it could ask for.
     */
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0 && n < allowed) allowed = n;
    }

    /*
     * A cgroup v2 quota narrows it further, and is how a container's CPU limit
     * becomes visible: affinity alone would report the whole host.
     */
    char buf[64];
    if (read_small_file("/sys/fs/cgroup/cpu.max", buf, sizeof buf)) {
        const int q = beam_parse_cpu_max(buf);
        if (q > 0 && q < allowed) allowed = q;
    }
#endif

    if (allowed < 1) allowed = 1;
    if (allowed > present) allowed = present;   /* a misparse cannot widen */
    return allowed;
}

#if defined(__APPLE__) || defined(_WIN32)
/*
 * Busy CPU time across every CPU, our own process's CPU time, and a monotonic
 * wall clock -- all in nanoseconds, so the pure arithmetic above never sees a
 * platform's tick unit. Each returns 0 on failure, and a failed read means
 * "unknown", which is the existing safe fallback rather than a special case.
 */
static int read_system_busy_ns(uint64_t *out) {
#if defined(__APPLE__)
    host_cpu_load_info_data_t info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    /*
     * mach_host_self() hands out a send right, and this runs at every entry
     * point: without the deallocate it leaks one Mach port reference per BEAM
     * call, which no test would notice and a long-lived process would.
     */
    const host_t host = mach_host_self();
    const kern_return_t kr = host_statistics(host, HOST_CPU_LOAD_INFO,
                                             (host_info_t)&info, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (kr != KERN_SUCCESS) return 0;
    /* HOST_CPU_LOAD_INFO is already summed over the CPUs. Idle is excluded. */
    const uint64_t ticks = (uint64_t)info.cpu_ticks[CPU_STATE_USER]
                         + (uint64_t)info.cpu_ticks[CPU_STATE_SYSTEM]
                         + (uint64_t)info.cpu_ticks[CPU_STATE_NICE];
    const long hz = sysconf(_SC_CLK_TCK);
    if (hz <= 0) return 0;
    *out = ticks * (1000000000ull / (uint64_t)hz);
    return 1;
#else
    FILETIME idle, kernel, user;
    if (!GetSystemTimes(&idle, &kernel, &user)) return 0;
    ULARGE_INTEGER i, k, u;
    i.LowPart = idle.dwLowDateTime;     i.HighPart = idle.dwHighDateTime;
    k.LowPart = kernel.dwLowDateTime;   k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;     u.HighPart = user.dwHighDateTime;
    /* Kernel time INCLUDES idle, so busy is kernel + user - idle. */
    const uint64_t total = (k.QuadPart + u.QuadPart) * 100ull;  /* 100-ns units */
    const uint64_t spare = i.QuadPart * 100ull;
    *out = total > spare ? total - spare : 0ull;
    return 1;
#endif
}

static int read_own_cpu_ns(uint64_t *out) {
#if defined(__APPLE__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    *out = (uint64_t)ru.ru_utime.tv_sec * 1000000000ull
         + (uint64_t)ru.ru_utime.tv_usec * 1000ull
         + (uint64_t)ru.ru_stime.tv_sec * 1000000000ull
         + (uint64_t)ru.ru_stime.tv_usec * 1000ull;
    return 1;
#else
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited,
                         &kernel, &user)) return 0;
    ULARGE_INTEGER k, u;
    k.LowPart = kernel.dwLowDateTime;   k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;     u.HighPart = user.dwHighDateTime;
    *out = (k.QuadPart + u.QuadPart) * 100ull;
    return 1;
#endif
}

static int read_wall_ns(uint64_t *out) {
#if defined(__APPLE__)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    *out = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    return 1;
#else
    LARGE_INTEGER freq, now;
    if (!QueryPerformanceFrequency(&freq) || freq.QuadPart <= 0) return 0;
    if (!QueryPerformanceCounter(&now) || now.QuadPart < 0) return 0;
    const uint64_t q = (uint64_t)now.QuadPart, f = (uint64_t)freq.QuadPart;
    /*
     * Seconds and remainder separately. Counter times a billion overflows
     * int64 after roughly fifteen minutes of uptime at the usual 10 MHz --
     * longer than a CI runner lives, so the obvious one-liner passes here and
     * returns nonsense on a machine that has been up since Tuesday.
     */
    *out = (q / f) * 1000000000ull + ((q % f) * 1000000000ull) / f;
    return 1;
#endif
}

/*
 * The previous sample, per thread.
 *
 * Thread-local for the reason threads.c gives for the plan itself: two BEAM
 * calls on two threads sharing one baseline would measure each other's window
 * and neither would get a window it could interpret. A thread's first call
 * therefore reports unknown, which is what this platform did on every call
 * before.
 *
 * A window shorter than the accounting resolution does NOT replace the
 * baseline. Overwriting it on every call is the bug that makes this whole
 * mechanism useless: a tight loop of short calls would reset the start of the
 * window each time and never accumulate twenty milliseconds, so the estimate
 * would be unknown forever on exactly the workload that needs it. The
 * baseline is left to age instead.
 */
/*
 * The previous sample, per thread.
 *
 * Thread-local for the reason threads.c gives for the plan itself: two BEAM
 * calls on two threads sharing one baseline would measure each other's window
 * and neither would get a window it could interpret. A thread's first call
 * therefore reports unknown, which is what this platform answered on EVERY
 * call before, so nothing here can be worse than it was.
 */
static TLS BeamAvailSample tl_sample;

static int sampled_competitors(int allowed) {
    uint64_t sys = 0u, own = 0u, wall = 0u;
    if (!read_system_busy_ns(&sys)) return -1;
    if (!read_own_cpu_ns(&own)) return -1;
    if (!read_wall_ns(&wall)) return -1;
    return beam_competitors_from_samples(&tl_sample, sys, own, wall, allowed);
}
#endif  /* __APPLE__ || _WIN32 */

/*
 * An explicit pin, for reproducibility.
 *
 * A load-aware plan is not repeatable between runs, which is the accepted cost
 * of being right about a busy machine. This is how determinism is bought back
 * where it matters: a benchmark A/B leg, a CI run, a test. It is the same role
 * `beam_model_set_threads` plays for the cap, and it is read from the
 * environment so that it needs no code change to use.
 *
 * Nonsense is IGNORED rather than obeyed. A typo must not be able to serialise
 * the library or to claim a thousand free cores, so anything that is not a
 * positive integer falls through to measuring.
 */
static int available_pin(void) {
    const char *v = getenv("BEAM_AVAILABLE_CORES");
    if (!v || !*v) return -1;
    char *end = NULL;
    const long n = strtol(v, &end, 10);
    if (!end || end == v || *end != '\0') return -1;   /* trailing junk */
    if (n < 1 || n > (1L << 20)) return -1;
    return (int)n;
}

int beam_available_cores(void) {
    const int allowed = beam_allowed_cores();
    const int pin = available_pin();
    if (pin > 0) return pin < allowed ? pin : allowed;

    int nr = -1;
#if defined(__linux__)
    char buf[128];
    if (read_small_file("/proc/loadavg", buf, sizeof buf)) {
        nr = beam_parse_nr_running(buf);
    }
#elif defined(__APPLE__) || defined(_WIN32)
    /*
     * No /proc, so no instantaneous runnable count. The sampled estimate is
     * second best and says so: it lags, and it reports unknown until a thread
     * has a window worth measuring. Unknown is what this platform answered on
     * every call before, so nothing here can be worse than it was.
     */
    nr = sampled_competitors(allowed);
#endif
    return beam_available_from(allowed, nr);
}
