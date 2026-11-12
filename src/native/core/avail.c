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

int beam_available_cores(void) {
    int nr = -1;
#if defined(__linux__)
    char buf[128];
    if (read_small_file("/proc/loadavg", buf, sizeof buf)) {
        nr = beam_parse_nr_running(buf);
    }
#endif
    return beam_available_from(beam_allowed_cores(), nr);
}
