/*
 * beam_bench -- the native benchmark harness for BEAM's C library.
 *
 * Per benchmark, in order:
 *   1. COLD    every fixture's rebuildable caches are dropped, then one call,
 *              memory tracked. Reported separately: it is what a first call
 *              on a fresh model costs, live-cache construction included.
 *   2. WARMUP  further calls until --warmup-time has passed (at least
 *              --warmup calls).
 *   3. CALIBRATE  calls faster than --target-window are batched (reps > 1)
 *              so one timing window is long enough that clock overhead is
 *              noise. Single-shot benchmarks are never batched.
 *   4. MEMORY  one tracked call (folded into the first sample when reps = 1).
 *   5. SAMPLES at least --min-samples, then more until --time-budget is
 *              spent or --samples are taken. Every value is time PER CALL.
 *
 * Output is JSON lines (--jsonl): a meta record, corpus and fixture records,
 * then a "begin" and a "result" record per benchmark, then "end". Each line
 * is flushed when written, so a crash leaves the finished results and the
 * name of the benchmark that crashed. Statistics are computed downstream, by
 * benchmarks/python, from the raw per-call samples recorded here.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "bench_framework.h"
#include "bench_json.h"

#define HARNESS_SCHEMA 2
#define BENCH_MAX_SAMPLES 1000

/* Linker-synthesised bounds of the registration section. */
extern const BeamBenchmark *const __start_beam_bench[];
extern const BeamBenchmark *const __stop_beam_bench[];

/* ── Failure reporting used by the framework macros ─────────────── */

void bench_fail(BenchContext *ctx, int code, const char *fmt, ...) {
    if (ctx->failed) return;   /* keep the first, which is the cause */
    ctx->failed = true;
    ctx->fail_code = code;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->fail_msg, sizeof ctx->fail_msg, fmt, ap);
    va_end(ap);
}

void bench_skip(BenchContext *ctx, const char *fmt, ...) {
    ctx->skipped = true;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->skip_msg, sizeof ctx->skip_msg, fmt, ap);
    va_end(ap);
}

void *bench_scratch(BenchContext *ctx, size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (bytes > ctx->scratch_size) {
        free(ctx->scratch);
        ctx->scratch = malloc(bytes);
        ctx->scratch_size = ctx->scratch ? bytes : 0;
        if (!ctx->scratch) {
            bench_fail(ctx, -1, "out of memory allocating %zu bytes of scratch", bytes);
            return NULL;
        }
    }
    /* Zeroed every time, so a call never inherits a previous benchmark's
     * output and its pages are already faulted in before the window opens. */
    memset(ctx->scratch, 0, bytes);
    return ctx->scratch;
}

/* ── Options ──────────────────────────────────────────────────── */

typedef struct Options {
    BenchWorkload w;
    int         order_set, max_length_set, length_set;
    int         threads;
    uint32_t    max_samples, min_samples, min_warmup;
    double      time_budget_s, warmup_time_s, target_window_s, slow_call_s;
    const char *suites, *filter, *exclude, *jsonl_path;
    int         list, list_json;
} Options;

static void usage(const char *prog) {
    printf("Usage: %s [options]\n\n"
           "Selection\n"
           "  --list | --list-json      list registered benchmarks and exit\n"
           "  --suite a,b               only these suites (domains)\n"
           "  --filter s1,s2            only benchmarks whose domain or name contains one of these\n"
           "  --exclude d::n,...        skip these benchmarks\n"
           "Workload\n"
           "  --alphabet aa|nt          (default aa)\n"
           "  --order K                 Markov order (default 4 for aa, 6 for nt)\n"
           "  --length L                per-length argument; also max_length unless given (default 27 aa, 81 nt)\n"
           "  --max-length M            model max_length (default = --length)\n"
           "  --corpus N                synthetic corpus size, or cap on --corpus-file (default 20000)\n"
           "  --corpus-file PATH        train fixtures on real sequences, one per line\n"
           "  --seed S                  synthetic corpus seed (default 1)\n"
           "  --threads N               BEAM thread cap, 0 = library default (default 0)\n"
           "Sampling\n"
           "  --samples N               at most N samples per benchmark (default 20)\n"
           "  --min-samples N           at least N samples (default 3)\n"
           "  --time-budget S           stop sampling after S seconds once min-samples is met (default 2)\n"
           "  --warmup N                at least N warmup calls after the cold call (default 0)\n"
           "  --warmup-time S           warm up for at least S seconds (default 0.2)\n"
           "  --target-window S         batch fast calls until one window lasts S seconds (default 0.002)\n"
           "  --slow-call S             a cold call longer than S seconds gets one sample (default 30)\n"
           "Output\n"
           "  --jsonl PATH              write JSON-lines results to PATH\n",
           prog);
}

static int parse_options(int argc, char **argv, Options *o) {
    memset(o, 0, sizeof *o);
    o->w.alphabet = BEAM_ALPHABET_AA;
    o->w.corpus_size = 20000u;
    o->w.seed = 1u;
    o->max_samples = 20u;
    o->min_samples = 3u;
    o->time_budget_s = 2.0;
    o->warmup_time_s = 0.2;
    o->target_window_s = 0.002;
    o->slow_call_s = 30.0;
    unsigned long order = 0, length = 0, max_length = 0;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define TAKE()                                                                               \
    do {                                                                                     \
        if (!v) { fprintf(stderr, "option %s needs a value\n", a); return -1; }             \
        ++i;                                                                                 \
    } while (0)
        if (!strcmp(a, "--list")) o->list = 1;
        else if (!strcmp(a, "--list-json")) o->list_json = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); exit(0); }
        else if (!strcmp(a, "--suite")) { TAKE(); o->suites = v; }
        else if (!strcmp(a, "--filter")) { TAKE(); o->filter = v; }
        else if (!strcmp(a, "--exclude")) { TAKE(); o->exclude = v; }
        else if (!strcmp(a, "--jsonl")) { TAKE(); o->jsonl_path = v; }
        else if (!strcmp(a, "--alphabet")) {
            TAKE();
            if (!strcmp(v, "aa")) o->w.alphabet = BEAM_ALPHABET_AA;
            else if (!strcmp(v, "nt")) o->w.alphabet = BEAM_ALPHABET_NT;
            else { fprintf(stderr, "--alphabet must be aa or nt\n"); return -1; }
        }
        else if (!strcmp(a, "--order")) { TAKE(); order = strtoul(v, NULL, 10); o->order_set = 1; }
        else if (!strcmp(a, "--length")) { TAKE(); length = strtoul(v, NULL, 10); o->length_set = 1; }
        else if (!strcmp(a, "--max-length")) { TAKE(); max_length = strtoul(v, NULL, 10); o->max_length_set = 1; }
        else if (!strcmp(a, "--corpus")) { TAKE(); o->w.corpus_size = (uint32_t)strtoul(v, NULL, 10); }
        else if (!strcmp(a, "--corpus-file")) { TAKE(); o->w.corpus_file = v; }
        else if (!strcmp(a, "--seed")) { TAKE(); o->w.seed = strtoull(v, NULL, 10); }
        else if (!strcmp(a, "--threads")) { TAKE(); o->threads = atoi(v); }
        else if (!strcmp(a, "--samples") || !strcmp(a, "--iterations")) { TAKE(); o->max_samples = (uint32_t)strtoul(v, NULL, 10); }
        else if (!strcmp(a, "--min-samples")) { TAKE(); o->min_samples = (uint32_t)strtoul(v, NULL, 10); }
        else if (!strcmp(a, "--time-budget")) { TAKE(); o->time_budget_s = atof(v); }
        else if (!strcmp(a, "--warmup")) { TAKE(); o->min_warmup = (uint32_t)strtoul(v, NULL, 10); }
        else if (!strcmp(a, "--warmup-time")) { TAKE(); o->warmup_time_s = atof(v); }
        else if (!strcmp(a, "--target-window")) { TAKE(); o->target_window_s = atof(v); }
        else if (!strcmp(a, "--slow-call")) { TAKE(); o->slow_call_s = atof(v); }
        else { fprintf(stderr, "unknown option %s (see --help)\n", a); return -1; }
#undef TAKE
    }

    const bool nt = o->w.alphabet == BEAM_ALPHABET_NT;
    if (!o->order_set) order = nt ? 6u : 4u;
    if (!o->length_set) length = o->max_length_set ? max_length : (nt ? 81u : 27u);
    if (!o->max_length_set) max_length = length;
    if (order < 1 || order > BEAM_MAX_ORDER) {
        fprintf(stderr, "--order must be 1..%d\n", (int)BEAM_MAX_ORDER);
        return -1;
    }
    if (max_length < 1 || max_length > 255) { fprintf(stderr, "--max-length must be 1..255\n"); return -1; }
    if (length < 1 || length > max_length) {
        fprintf(stderr, "--length (%lu) must be 1..max_length (%lu)\n", length, max_length);
        return -1;
    }
    if (o->max_samples < 1) o->max_samples = 1;
    if (o->max_samples > BENCH_MAX_SAMPLES) o->max_samples = BENCH_MAX_SAMPLES;
    if (o->min_samples < 1) o->min_samples = 1;
    if (o->min_samples > o->max_samples) o->min_samples = o->max_samples;
    o->w.order = (uint8_t)order;
    o->w.length = (uint32_t)length;
    o->w.max_length = (uint8_t)max_length;
    return 0;
}

/* Does `name` appear as an item of the comma-separated `list`?
 * With `substring`, an item need only be contained in `name`. */
static bool list_has(const char *list, const char *name, bool substring) {
    if (!list) return false;
    const char *p = list;
    while (*p) {
        const char *q = strchr(p, ',');
        const size_t len = q ? (size_t)(q - p) : strlen(p);
        if (len > 0) {
            if (substring) {
                char item[128];
                const size_t n = len < sizeof item - 1 ? len : sizeof item - 1;
                memcpy(item, p, n);
                item[n] = '\0';
                if (strstr(name, item)) return true;
            } else if (strlen(name) == len && !strncmp(name, p, len)) {
                return true;
            }
        }
        if (!q) break;
        p = q + 1;
    }
    return false;
}

static bool selected(const Options *o, const BeamBenchmark *b) {
    char key[160];
    snprintf(key, sizeof key, "%s::%s", b->domain, b->name);
    if (o->suites && !list_has(o->suites, b->domain, false)) return false;
    if (o->filter && !list_has(o->filter, key, true)) return false;
    if (o->exclude && list_has(o->exclude, key, false)) return false;
    return true;
}

static int compare_bench(const void *a, const void *b) {
    const BeamBenchmark *x = *(const BeamBenchmark *const *)a;
    const BeamBenchmark *y = *(const BeamBenchmark *const *)b;
    const int d = strcmp(x->domain, y->domain);
    if (d) return d;
    const int f = strcmp(x->file, y->file);
    if (f) return f;
    return x->line - y->line;
}

/* ── Running one call ─────────────────────────────────────────── */

static void run_call(const BeamBenchmark *b, BenchContext *ctx, BenchPhase phase, uint32_t reps, bool track) {
    ctx->phase = phase;
    ctx->reps = reps;
    ctx->track_memory = track;
    ctx->failed = false;
    ctx->fail_code = 0;
    ctx->fail_msg[0] = '\0';
    ctx->skipped = false;
    ctx->skip_msg[0] = '\0';
    ctx->blocks_entered = 0;
    ctx->window_ns = 0;
    memset(&ctx->mem, 0, sizeof ctx->mem);
    b->func(ctx);
    if (!ctx->failed && !ctx->skipped && ctx->blocks_entered == 0)
        bench_fail(ctx, -1, "benchmark returned before its timed block");
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
static int cmp_dbl(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* What one empty timing window costs: the floor under every measurement. */
static double measure_timer_overhead(void) {
    BenchContext ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.reps = 0;
    enum { N = 2001 };
    static uint64_t v[N];
    for (int i = 0; i < N; ++i) {
        ctx.blocks_entered = 0;
        BEAM_TIMED_BLOCK(&ctx) {}
        v[i] = ctx.window_ns;
    }
    qsort(v, N, sizeof v[0], cmp_u64);
    return (double)v[N / 2];
}

static const char *fmt_ns(double ns, char *buf, size_t n) {
    if (ns < 1e3) snprintf(buf, n, "%.1f ns", ns);
    else if (ns < 1e6) snprintf(buf, n, "%.2f us", ns * 1e-3);
    else if (ns < 1e9) snprintf(buf, n, "%.2f ms", ns * 1e-6);
    else snprintf(buf, n, "%.2f s", ns * 1e-9);
    return buf;
}

static const char *fmt_bytes(double b, char *buf, size_t n) {
    const double a = b < 0 ? -b : b;
    if (a < 1024.0) snprintf(buf, n, "%.0f B", b);
    else if (a < 1048576.0) snprintf(buf, n, "%.1f KiB", b / 1024.0);
    else if (a < 1073741824.0) snprintf(buf, n, "%.1f MiB", b / 1048576.0);
    else snprintf(buf, n, "%.2f GiB", b / 1073741824.0);
    return buf;
}

/*
 * CPU time used by OTHER processes while a benchmark ran, in CPUs: total busy
 * ticks of the machine (/proc/stat) minus this process's own (/proc/self/stat,
 * all threads), over the wall time. The load average cannot answer this,
 * because the harness's own OpenMP threads dominate it. Multi-threaded kernels
 * slow down badly when anything else takes a core: a 10 us call measured 6 ms
 * while a compiler ran alongside.
 */
typedef struct CpuSnap { unsigned long long busy, self; uint64_t wall_ns; bool ok; } CpuSnap;

static CpuSnap cpu_snapshot(void) {
    CpuSnap c = {0, 0, bench_now_ns(), false};
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return c;
    unsigned long long v[8] = {0};
    const int n = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                         &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    fclose(f);
    if (n < 8) return c;
    c.busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];   /* user nice system irq softirq steal */
    f = fopen("/proc/self/stat", "r");
    if (!f) return c;
    char buf[2048];
    const size_t len = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[len] = '\0';
    const char *p = strrchr(buf, ')');            /* comm may contain spaces */
    unsigned long long ut = 0, st = 0;
    if (!p || sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &ut, &st) != 2) return c;
    c.self = ut + st;
    c.ok = true;
    return c;
}

/* Average CPUs used by other processes (`foreign`) and by this one (`self`). */
static bool cpu_usage(const CpuSnap *a, const CpuSnap *b, double *foreign, double *self) {
    if (!a->ok || !b->ok || b->wall_ns <= a->wall_ns) return false;
    const double ticks = (double)sysconf(_SC_CLK_TCK);
    const double secs = (double)(b->wall_ns - a->wall_ns) * 1e-9;
    const double mine = (double)(b->self - a->self);
    const double other = (double)(b->busy - a->busy) - mine;
    *self = mine / ticks / secs;
    *foreign = other > 0.0 ? other / ticks / secs : 0.0;
    return true;
}

static void emit_mem(FILE *f, int *first, const char *prefix, const BenchMemStats *m) {
    char k[64];
    snprintf(k, sizeof k, "%speak_bytes", prefix);      BJ_U64(f, first, k, m->peak_bytes);
    snprintf(k, sizeof k, "%sretained_bytes", prefix);  BJ_I64(f, first, k, m->retained_bytes);
    snprintf(k, sizeof k, "%salloc_bytes", prefix);     BJ_U64(f, first, k, m->alloc_bytes);
    snprintf(k, sizeof k, "%salloc_calls", prefix);     BJ_U64(f, first, k, m->alloc_calls);
}

typedef struct Outcome {
    const char   *status;
    double        samples[BENCH_MAX_SAMPLES];
    uint32_t      n;
    uint32_t      reps;
    uint64_t      cold_ns;
    BenchMemStats cold_mem, mem;
    bool          have_mem;
    uint32_t      warmup_calls;
    uint64_t      wall_ns;
    double        foreign_cpus;   /* < 0 if unknown */
    double        self_cpus;
} Outcome;

static void run_benchmark(const Options *o, const BeamBenchmark *b, BenchContext *ctx, Outcome *out) {
    memset(out, 0, sizeof *out);
    const uint64_t wall0 = bench_now_ns();
    const bool single = (b->flags & BENCH_SINGLE_SHOT) != 0;
    ctx->items_per_op = 0.0;
    ctx->item_unit = NULL;

    out->foreign_cpus = -1.0;

    /* 1. Cold. */
    bench_fixtures_drop_caches();
    run_call(b, ctx, BENCH_PHASE_COLD, 1, true);
    if (ctx->failed || ctx->skipped) {
        out->status = ctx->failed ? "error" : "skipped";
        out->wall_ns = bench_now_ns() - wall0;
        return;
    }
    out->cold_ns = ctx->window_ns;
    out->cold_mem = ctx->mem;

    /* 2. Warmup. At least one warm call whenever the cold call took under a
     * second, because a cold call is often slow only for building caches
     * (path_count: 0.7 s cold, 16 ms warm) and must not set the calibration. */
    uint64_t last = out->cold_ns;
    const uint64_t warm_budget = (uint64_t)(o->warmup_time_s * 1e9);
    const uint64_t w0 = bench_now_ns();
    for (uint32_t c = 0; c < 100000u; ++c) {
        const bool need = c < o->min_warmup || (c == 0 && out->cold_ns < 1000000000ULL) ||
                          (bench_now_ns() - w0 < warm_budget && out->cold_ns < warm_budget);
        if (!need) break;
        run_call(b, ctx, BENCH_PHASE_WARMUP, 1, false);
        if (ctx->failed) { out->status = "error"; out->wall_ns = bench_now_ns() - wall0; return; }
        last = ctx->window_ns;
        out->warmup_calls++;
    }

    /* 3. Calibrate. */
    uint32_t reps = 1;
    if (!single) {
        const double target = o->target_window_s * 1e9;
        const double per = last > 0 ? (double)last : 1.0;
        if (per < target) {
            const double r = target / per;
            reps = r > 1e7 ? 10000000u : (uint32_t)(r + 1.0);
        }
    }
    out->reps = reps;

    /* 4. Memory, separately only when samples are batched. */
    if (reps > 1) {
        run_call(b, ctx, BENCH_PHASE_MEMORY, 1, true);
        if (ctx->failed) { out->status = "error"; out->wall_ns = bench_now_ns() - wall0; return; }
        out->mem = ctx->mem;
        out->have_mem = true;
    }

    /* 5. Samples. A call slower than --slow-call gets one, reported as a
     * single sample: three samples of a five-minute call would cost a quarter
     * of an hour, and replicated processes (--ab) are what settle a change. */
    const uint64_t budget = (uint64_t)(o->time_budget_s * 1e9);
    const bool slow = (double)out->cold_ns > o->slow_call_s * 1e9;
    const uint32_t min_samples = slow ? 1u : o->min_samples;
    const uint32_t max_samples = slow ? 1u : o->max_samples;
    const uint64_t s0 = bench_now_ns();
    for (uint32_t i = 0; i < max_samples; ++i) {
        const bool track = !out->have_mem && i == 0;
        run_call(b, ctx, BENCH_PHASE_SAMPLE, reps, track);
        if (ctx->failed) { out->status = "error"; out->wall_ns = bench_now_ns() - wall0; return; }
        if (track) { out->mem = ctx->mem; out->have_mem = true; }
        out->samples[out->n++] = (double)ctx->window_ns / (double)reps;
        if (out->n >= min_samples && bench_now_ns() - s0 >= budget) break;
    }
    out->status = "ok";
    out->wall_ns = bench_now_ns() - wall0;
}

static void emit_result(FILE *f, const BeamBenchmark *b, const BenchContext *ctx, const Outcome *out) {
    int first = 1;
    fputc('{', f);
    BJ_STR(f, &first, "type", "result");
    BJ_STR(f, &first, "domain", b->domain);
    BJ_STR(f, &first, "name", b->name);
    BJ_STR(f, &first, "description", b->description);
    BJ_STR(f, &first, "file", b->file);
    BJ_I64(f, &first, "line", b->line);
    BJ_BOOL(f, &first, "single_shot", (b->flags & BENCH_SINGLE_SHOT) != 0);
    BJ_STR(f, &first, "status", out->status);
    if (ctx->failed) {
        BJ_I64(f, &first, "error_code", ctx->fail_code);
        BJ_STR(f, &first, "error_message", ctx->fail_msg);
    }
    if (ctx->skipped) BJ_STR(f, &first, "skip_reason", ctx->skip_msg);
    BJ_U64(f, &first, "reps", out->reps);
    BJ_U64(f, &first, "warmup_calls", out->warmup_calls);
    BJ_U64(f, &first, "cold_ns", out->cold_ns);
    emit_mem(f, &first, "cold_", &out->cold_mem);
    if (out->have_mem) emit_mem(f, &first, "", &out->mem);
    if (ctx->items_per_op > 0.0) {
        BJ_NUM(f, &first, "items_per_op", ctx->items_per_op);
        BJ_STR(f, &first, "item_unit", ctx->item_unit ? ctx->item_unit : "items");
    }
    BJ_U64(f, &first, "wall_ns", out->wall_ns);
    if (out->foreign_cpus >= 0.0) {
        BJ_NUM(f, &first, "foreign_cpus", out->foreign_cpus);
        BJ_NUM(f, &first, "self_cpus", out->self_cpus);
    }
    bench_json_key(f, "samples_ns", &first);
    fputc('[', f);
    for (uint32_t i = 0; i < out->n; ++i) {
        if (i) fputc(',', f);
        fprintf(f, "%.6g", out->samples[i]);
    }
    fputc(']', f);
    fputs("}\n", f);
    fflush(f);
}

static void print_row(const BeamBenchmark *b, const BenchContext *ctx, const Outcome *out) {
    char m[32], c[32], p[32];
    if (strcmp(out->status, "ok") != 0) {
        printf("%-11s %-34s %12s  %s\n", b->domain, b->name,
               ctx->failed ? "ERROR" : "SKIPPED", ctx->failed ? ctx->fail_msg : ctx->skip_msg);
        return;
    }
    double s[BENCH_MAX_SAMPLES];
    memcpy(s, out->samples, out->n * sizeof(double));
    qsort(s, out->n, sizeof(double), cmp_dbl);
    const double med = out->n % 2 ? s[out->n / 2] : 0.5 * (s[out->n / 2 - 1] + s[out->n / 2]);
    printf("%-11s %-34s %12s %12s %12s  n=%-3u reps=%u\n", b->domain, b->name,
           fmt_ns(med, m, sizeof m), fmt_ns((double)out->cold_ns, c, sizeof c),
           fmt_bytes((double)out->mem.peak_bytes, p, sizeof p), out->n, out->reps);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    Options o;
    if (parse_options(argc, argv, &o) != 0) return 2;

    const size_t total = (size_t)(__stop_beam_bench - __start_beam_bench);
    const BeamBenchmark **list = (const BeamBenchmark **)malloc((total ? total : 1) * sizeof *list);
    if (!list) return 1;
    for (size_t i = 0; i < total; ++i) list[i] = __start_beam_bench[i];
    qsort(list, total, sizeof *list, compare_bench);

    if (o.list || o.list_json) {
        for (size_t i = 0; i < total; ++i) {
            const BeamBenchmark *b = list[i];
            if (o.list_json) {
                int first = 1;
                fputc('{', stdout);
                BJ_STR(stdout, &first, "domain", b->domain);
                BJ_STR(stdout, &first, "name", b->name);
                BJ_STR(stdout, &first, "description", b->description);
                BJ_STR(stdout, &first, "file", b->file);
                BJ_I64(stdout, &first, "line", b->line);
                BJ_BOOL(stdout, &first, "single_shot", (b->flags & BENCH_SINGLE_SHOT) != 0);
                fputs("}\n", stdout);
            } else {
                printf("%-11s %-34s %s%s\n", b->domain, b->name, b->description,
                       (b->flags & BENCH_SINGLE_SHOT) ? " [single-shot]" : "");
            }
        }
        free(list);
        return 0;
    }

    FILE *jf = NULL;
    if (o.jsonl_path) {
        jf = fopen(o.jsonl_path, "a");
        if (!jf) { fprintf(stderr, "cannot open %s for writing\n", o.jsonl_path); return 1; }
    }

    beam_model_set_threads(o.threads);
    const double overhead = measure_timer_overhead();

    if (jf) {
        int first = 1;
        fputc('{', jf);
        BJ_STR(jf, &first, "type", "meta");
        BJ_I64(jf, &first, "schema", HARNESS_SCHEMA);
        BJ_STR(jf, &first, "compiler", __VERSION__);
        BJ_STR(jf, &first, "harness_built", __DATE__ " " __TIME__);
        BJ_I64(jf, &first, "pid", getpid());
        BJ_I64(jf, &first, "started_unix", (long long)time(NULL));
        BJ_STR(jf, &first, "alphabet", o.w.alphabet == BEAM_ALPHABET_NT ? "nt" : "aa");
        BJ_U64(jf, &first, "order", o.w.order);
        BJ_U64(jf, &first, "max_length", o.w.max_length);
        BJ_U64(jf, &first, "length", o.w.length);
        BJ_U64(jf, &first, "corpus_size", o.w.corpus_size);
        BJ_STR(jf, &first, "corpus_file", o.w.corpus_file ? o.w.corpus_file : "");
        BJ_U64(jf, &first, "seed", o.w.seed);
        BJ_I64(jf, &first, "threads_requested", o.threads);
        BJ_I64(jf, &first, "threads_effective", beam_model_effective_threads());
        BJ_I64(jf, &first, "logical_cpus", sysconf(_SC_NPROCESSORS_ONLN));
        BJ_U64(jf, &first, "memory_budget_bytes", beam_model_memory_budget());
        BJ_NUM(jf, &first, "timer_overhead_ns", overhead);
        BJ_U64(jf, &first, "max_samples", o.max_samples);
        BJ_U64(jf, &first, "min_samples", o.min_samples);
        BJ_NUM(jf, &first, "time_budget_s", o.time_budget_s);
        BJ_NUM(jf, &first, "warmup_time_s", o.warmup_time_s);
        BJ_U64(jf, &first, "min_warmup", o.min_warmup);
        BJ_NUM(jf, &first, "target_window_s", o.target_window_s);
        BJ_NUM(jf, &first, "slow_call_s", o.slow_call_s);
        fputs("}\n", jf);
        fflush(jf);
    }

    printf("beam_bench | %s K=%u max_length=%u L=%u | corpus %s | threads %d (effective %d) | timer floor %.0f ns\n",
           o.w.alphabet == BEAM_ALPHABET_NT ? "nt" : "aa", o.w.order, o.w.max_length, o.w.length,
           o.w.corpus_file ? o.w.corpus_file : "synthetic", o.threads, beam_model_effective_threads(), overhead);

    if (!bench_fixtures_init(&o.w, jf)) {
        fprintf(stderr, "corpus: %s\n", bench_fixture_error());
        if (jf) fclose(jf);
        free(list);
        return 1;
    }
    /* Build the primary model now, so a workload that cannot be built fails
     * before any benchmark runs rather than as ninety identical errors. */
    if (!bench_fixture(BENCH_VARIANT_PRIMARY)) {
        fprintf(stderr, "fixture: %s\n", bench_fixture_error());
        bench_fixtures_teardown();
        if (jf) fclose(jf);
        free(list);
        return 1;
    }

    printf("%-11s %-34s %12s %12s %12s\n", "SUITE", "BENCHMARK", "MEDIAN/CALL", "COLD CALL", "PEAK HEAP");

    BenchContext ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.alphabet = o.w.alphabet;
    ctx.order = o.w.order;
    ctx.max_length = o.w.max_length;
    ctx.length = o.w.length;

    static Outcome out;
    size_t ran = 0;
    for (size_t i = 0; i < total; ++i) {
        const BeamBenchmark *b = list[i];
        if (!selected(&o, b)) continue;
        if (jf) {
            int first = 1;
            fputc('{', jf);
            BJ_STR(jf, &first, "type", "begin");
            BJ_STR(jf, &first, "domain", b->domain);
            BJ_STR(jf, &first, "name", b->name);
            fputs("}\n", jf);
            fflush(jf);
        }
        const CpuSnap c0 = cpu_snapshot();
        run_benchmark(&o, b, &ctx, &out);
        const CpuSnap c1 = cpu_snapshot();
        if (!cpu_usage(&c0, &c1, &out.foreign_cpus, &out.self_cpus)) out.foreign_cpus = -1.0;
        print_row(b, &ctx, &out);
        if (jf) emit_result(jf, b, &ctx, &out);
        ran++;
    }

    if (jf) {
        int first = 1;
        fputc('{', jf);
        BJ_STR(jf, &first, "type", "end");
        BJ_U64(jf, &first, "executed", ran);
        BJ_U64(jf, &first, "process_peak_rss_bytes", bench_mem_process_peak_rss());
        fputs("}\n", jf);
        fclose(jf);
    }
    free(ctx.scratch);
    bench_fixtures_teardown();
    free(list);
    printf("%zu benchmarks run\n", ran);
    return 0;
}
