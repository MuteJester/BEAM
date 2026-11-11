/**
 * @file beam_test.c
 * @brief BEAM's C test harness: the counters, the clock and the record.
 *
 * Allocates NOTHING. tests/native/test_expected_counts_accounting.c links
 * with --wrap=malloc,calloc,realloc,free and sums what the allocator actually
 * handed out during one analytic, to check the figure that analytic reports
 * against it. A harness that allocated would be counted into that sum, and
 * the one test whose whole purpose is exact accounting would be the one test
 * this file broke. So the records are a fixed array and a message is COPIED
 * into a fixed field of one.
 *
 * Copied, and not kept as the caller's pointer, because fifteen suites build
 * their failure text with snprintf into one static buffer: storing the
 * pointer made every record show whatever was written there last, since the
 * JSON is emitted after the run. The printed line was right and the
 * machine-readable record was wrong, which is the half a reader is least
 * able to check and the half tools/qa/run reports from.
 */
#define _DEFAULT_SOURCE          /* getloadavg, sysconf */

#include "beam_test.h"

#undef printf                    /* the harness prints through out() below */

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_TESTS 512
#define MAX_SELECTED 64

typedef enum { OUT_NONE = 0, OUT_PASSED, OUT_FAILED, OUT_SKIPPED } Outcome;

#define MAX_MESSAGE 200          /* longer messages are cut, never allocated */

typedef struct {
    const char *name;        /* the RUN_TEST literal: the caller owns it */
    char message[MAX_MESSAGE];
    int has_message;
    double seconds;
    int pass_before;
    int fail_before;
    Outcome outcome;
} Record;

int pass_count = 0;
int fail_count = 0;
int beam_test_list_only = 0;

static Record records[MAX_TESTS];
static int record_count = 0;
static int current = -1;
/* Whether anything has been written on the current test's line. The harness
 * terminates every line itself, so a body that reports nothing at all --
 * a bare fail_count++, or an early return -- cannot run into the next
 * test's name. */
static int wrote_verdict = 0;

static const char *selected[MAX_SELECTED];
static char matched[MAX_SELECTED];
static int selected_count = 0;

static const char *json_path = NULL;
static const char *suite_name = "tests";
static struct timespec test_started, run_started;

static double since(const struct timespec *t0) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - t0->tv_sec)
         + (double)(now.tv_nsec - t0->tv_nsec) / 1e9;
}

/* Everything the harness prints, silenced while listing. */
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
static int out(const char *fmt, ...) {
    if (beam_test_list_only) return 0;
    va_list ap;
    va_start(ap, fmt);
    const int n = vprintf(fmt, ap);
    va_end(ap);
    return n;
}

int beam_test_printf(const char *fmt, ...) {
    if (beam_test_list_only) return 0;
    va_list ap;
    va_start(ap, fmt);
    const int n = vprintf(fmt, ap);
    va_end(ap);
    return n;
}

int beam_test_wanted(const char *name) {
    if (selected_count == 0) return 1;
    int wanted = 0;
    for (int i = 0; i < selected_count; i++) {
        if (strcmp(name, selected[i]) == 0) {
            matched[i] = 1;
            wanted = 1;
        }
    }
    return wanted;
}

void beam_test_begin(const char *name) {
    if (record_count == MAX_TESTS) {
        fprintf(stderr, "\n%s: more than %d tests in one binary; raise "
                        "MAX_TESTS in tests/harness/beam_test.c\n",
                suite_name, MAX_TESTS);
        fflush(NULL);
        exit(2);
    }
    current = record_count++;
    records[current].name = name;
    records[current].message[0] = '\0';
    records[current].has_message = 0;
    records[current].seconds = 0.0;
    records[current].outcome = OUT_NONE;
    records[current].pass_before = pass_count;
    records[current].fail_before = fail_count;
    wrote_verdict = 0;
    out("  %-55s ", name);
    clock_gettime(CLOCK_MONOTONIC, &test_started);
}

void beam_test_pass(void) {
    out("PASS");
    wrote_verdict = 1;
    pass_count++;
}

/* Copy, truncating rather than allocating. Never overwrites a message the
 * record already holds: the FIRST failure is the one that explains a test. */
static void keep_message(Record *r, const char *message) {
    if (r->has_message || !message) return;
    size_t n = strlen(message);
    if (n >= sizeof r->message) n = sizeof r->message - 1;
    memcpy(r->message, message, n);
    r->message[n] = '\0';
    r->has_message = 1;
}

void beam_test_fail(const char *message) {
    out("FAIL: %s ", message);
    wrote_verdict = 1;
    fail_count++;
    if (current >= 0) {
        /* A skip earlier in the same body explains the sub-case it declined,
         * not the failure that came after it: test_row_cursor is that shape.
         * The failure replaces it. */
        if (records[current].outcome == OUT_SKIPPED)
            records[current].has_message = 0;
        keep_message(&records[current], message);
    }
}

void beam_test_skip_here(const char *reason) {
    out("SKIP (%s) ", reason);
    wrote_verdict = 1;
    if (current >= 0) {
        records[current].outcome = OUT_SKIPPED;
        keep_message(&records[current], reason);
    }
}

void beam_test_end(void) {
    if (current < 0) return;
    Record *r = &records[current];
    r->seconds = since(&test_started);
    /*
     * Decided from what the test DID, not from what it said. Eleven files
     * increment fail_count directly in 74 places rather than going through
     * ASSERT_MSG, and four read it to decide whether to pass; a harness that
     * trusted only its own entry points would record those as no verdict.
     */
    if (fail_count > r->fail_before)      r->outcome = OUT_FAILED;
    else if (pass_count > r->pass_before) r->outcome = OUT_PASSED;
    else if (r->outcome != OUT_SKIPPED)   r->outcome = OUT_NONE;
    /* One line per test, ended here. */
    if (!wrote_verdict) {
        if (r->outcome == OUT_FAILED)
            out("FAILED (a failure was counted, with no message)");
        else
            out("NO VERDICT (neither passed nor failed)");
    }
    out("\n");
    current = -1;
}

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s [--list] [--run NAME]... [--json PATH]\n", prog);
}

void beam_test_init(int argc, char **argv) {
    if (argc > 0 && argv[0]) {
        const char *slash = strrchr(argv[0], '/');
        suite_name = slash ? slash + 1 : argv[0];
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--list") == 0) {
            beam_test_list_only = 1;
        } else if (strcmp(argv[i], "--run") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: --run needs a test name\n", argv[0]);
                usage(argv[0]);
                exit(2);
            }
            if (selected_count == MAX_SELECTED) {
                fprintf(stderr, "%s: more than %d --run names\n",
                        argv[0], MAX_SELECTED);
                exit(2);
            }
            selected[selected_count++] = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: --json needs a path\n", argv[0]);
                usage(argv[0]);
                exit(2);
            }
            json_path = argv[++i];
        } else {
            fprintf(stderr, "%s: unrecognised argument: %s\n", argv[0], argv[i]);
            usage(argv[0]);
            exit(2);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &run_started);
}

static const char *outcome_name(Outcome o) {
    switch (o) {
        case OUT_PASSED:  return "passed";
        case OUT_FAILED:  return "failed";
        case OUT_SKIPPED: return "skipped";
        default:          return "no verdict";
    }
}

static int count_of(Outcome o) {
    int n = 0;
    for (int i = 0; i < record_count; i++) if (records[i].outcome == o) n++;
    return n;
}

static void write_escaped(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        switch (*s) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f);  break;
            case '\t': fputs("\\t", f);  break;
            case '\r': fputs("\\r", f);  break;
            default:
                if ((unsigned char)*s < 0x20) fprintf(f, "\\u%04x", *s);
                else fputc(*s, f);
        }
    }
    fputc('"', f);
}

static int thread_cap(void) {
    /*
     * The cap as the caller set it. BEAM never edits global OpenMP state
     * (src/native/core/threads.c:24-26), so the environment variable is the
     * cap; omp_get_max_threads() is what it resolved to.
     */
    const char *env = getenv("OMP_NUM_THREADS");
    if (env && *env) return atoi(env);
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void beam_test_report_json(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "%s: could not write %s\n", suite_name, path);
        return;
    }
    double load[3] = {-1.0, -1.0, -1.0};
    (void)getloadavg(load, 3);
    const long cores = sysconf(_SC_NPROCESSORS_ONLN);

    fputs("{\n \"suite\": ", f);
    write_escaped(f, suite_name);
    fprintf(f, ",\n \"total\": %d,\n \"passed\": %d,\n \"failed\": %d,\n"
               " \"skipped\": %d,\n \"no_verdict\": %d,\n \"seconds\": %.3f,\n",
            record_count, count_of(OUT_PASSED), count_of(OUT_FAILED),
            count_of(OUT_SKIPPED), count_of(OUT_NONE), since(&run_started));
    /*
     * A duration with no record of the conditions is not a measurement: this
     * suite took 3.4 minutes on an idle box and 106 minutes at load 40, and
     * comparing two such numbers credits a change with someone else's quiet
     * afternoon.
     */
    fprintf(f, " \"conditions\": {\n  \"load_average\": [%.2f, %.2f, %.2f],\n"
               "  \"thread_cap\": %d,\n  \"core_count\": %ld\n },\n",
            load[0], load[1], load[2], thread_cap(), cores > 0 ? cores : 0);
    fputs(" \"tests\": [\n", f);
    for (int i = 0; i < record_count; i++) {
        fputs("  {\"name\": ", f);
        write_escaped(f, records[i].name);
        fputs(", \"outcome\": ", f);
        write_escaped(f, outcome_name(records[i].outcome));
        fprintf(f, ", \"seconds\": %.6f, \"message\": ", records[i].seconds);
        if (records[i].has_message) write_escaped(f, records[i].message);
        else fputs("null", f);
        fputs(i + 1 < record_count ? "},\n" : "}\n", f);
    }
    fputs(" ]\n}\n", f);
    fclose(f);
}

int beam_test_finish(void) {
    if (beam_test_list_only) return 0;

    /*
     * A --run name this binary does not have would otherwise run nothing,
     * report "0 passed, 0 failed" and exit 0 -- a selection typo that reads
     * exactly like a suite that passed.
     */
    int unmatched = 0;
    for (int i = 0; i < selected_count; i++) {
        if (!matched[i]) {
            fprintf(stderr, "NO SUCH TEST: %s\n", selected[i]);
            unmatched = 1;
        }
    }

    const int failed = count_of(OUT_FAILED);
    const int skipped = count_of(OUT_SKIPPED);
    const int none = count_of(OUT_NONE);
    out("\n  %s: %d passed, %d failed, %d skipped",
        suite_name, count_of(OUT_PASSED), failed, skipped);
    if (none) out(", %d WITH NO VERDICT", none);
    out("  in %.2f s\n", since(&run_started));
    if (none)
        out("  A test that reaches no verdict verified nothing and said so to "
            "no one. Treated as a failure.\n");

    if (json_path) beam_test_report_json(json_path);
    fflush(NULL);
    if (unmatched) return 2;
    return (failed || none) ? 1 : 0;
}
