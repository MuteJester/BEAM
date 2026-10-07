#ifndef BENCH_FRAMEWORK_H
#define BENCH_FRAMEWORK_H

/*
 * How a benchmark is written:
 *
 *     BEAM_BENCHMARK(math, entropy, "Exact Shannon entropy of the model") {
 *         BENCH_MODEL(ctx, m);                       // the workload's model
 *         double h = 0.0;
 *         BEAM_TIMED_BLOCK(ctx) {
 *             BENCH_CHECK(ctx, beam_model_entropy(m, &h));
 *         }
 *         BENCH_KEEP(h);
 *     }
 *
 * The function body is SETUP + exactly one timed block + TEARDOWN, and is
 * called once per sample. The harness decides how many times the block body
 * runs inside one timing window (`ctx->reps`): calls too fast for the clock
 * are batched until a window lasts long enough, and every reported time is
 * PER CALL of the block body. A benchmark whose body mutates state so that a
 * second run would do different work (observe, compact, subtract, ...) is
 * declared BENCH_SINGLE_SHOT and is never batched: each sample re-runs setup.
 *
 * Correctness is checked, not assumed. The harness clears BEAM's thread-local
 * error before the window and reads it after, and BENCH_CHECK records any
 * non-OK return. A benchmark that failed is reported as ERROR with BEAM's own
 * message, and its time is never ranked -- the first version of this harness
 * timed error paths and reported them as fast functions.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "beam/common.h"
#include "beam/errors.h"
#include "beam/model.h"
#include "bench_clock.h"
#include "bench_fixtures.h"
#include "bench_memory.h"

/* Never batch: setup must be redone before every call of the block body. */
#define BENCH_SINGLE_SHOT 0x1u

typedef enum BenchPhase {
    BENCH_PHASE_COLD = 0,   /* first call, fixtures' caches dropped */
    BENCH_PHASE_WARMUP,
    BENCH_PHASE_MEMORY,     /* one tracked call, when samples are batched */
    BENCH_PHASE_SAMPLE
} BenchPhase;

typedef struct BenchContext {
    /* Workload (read only for benchmarks). */
    BeamAlphabet alphabet;
    uint8_t      order;
    uint8_t      max_length;
    uint32_t     length;

    /* Set by the harness for each call. */
    BenchPhase   phase;
    uint32_t     reps;          /* block-body runs per timing window */
    bool         track_memory;

    /* Set by the benchmark. */
    double       items_per_op;  /* e.g. 100 sequences per sample_100 call */
    const char  *item_unit;

    /* Outcome of the last call. */
    bool         failed;
    int          fail_code;
    char         fail_msg[384];
    bool         skipped;
    char         skip_msg[256];
    uint32_t     blocks_entered;
    uint64_t     window_ns;
    BenchMemStats mem;

    /* Scratch owned by the harness, grown on request outside the window. */
    void        *scratch;
    size_t       scratch_size;
} BenchContext;

typedef struct BeamBenchmark {
    const char *domain;
    const char *name;
    const char *description;
    uint32_t    flags;
    void      (*func)(BenchContext *ctx);
    const char *file;
    int         line;
} BeamBenchmark;

/*
 * Registration: a POINTER to each descriptor goes into the "beam_bench"
 * section, and the linker brackets the section with __start_/__stop_. Pointers
 * rather than the descriptors themselves, because the compiler may pad large
 * objects in a section to 32 bytes, which would break array-stride iteration
 * the moment the descriptor stopped being exactly 32 bytes.
 */
#define BEAM_BENCHMARK_EX(d, n, desc, fl)                                                    \
    static void bench_fn_##d##_##n(BenchContext *ctx);                                       \
    static const BeamBenchmark bench_desc_##d##_##n = {                                      \
        #d, #n, desc, (fl), bench_fn_##d##_##n, __FILE__, __LINE__};                         \
    __attribute__((used, section("beam_bench"))) static const BeamBenchmark *const           \
        bench_ptr_##d##_##n = &bench_desc_##d##_##n;                                         \
    static void bench_fn_##d##_##n(BenchContext *ctx)

#define BEAM_BENCHMARK(d, n, desc) BEAM_BENCHMARK_EX(d, n, desc, 0u)

/* ── Timed window ──────────────────────────────────────────────── */

void bench_fail(BenchContext *ctx, int code, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
void bench_skip(BenchContext *ctx, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

static inline int bench_window_open(BenchContext *ctx) {
    if (ctx->blocks_entered++ > 0) {
        bench_fail(ctx, -1, "benchmark entered BEAM_TIMED_BLOCK more than once");
        return 0;
    }
    if (ctx->failed || ctx->skipped) return 0;
    beam_clear_error();
    if (ctx->track_memory) bench_mem_begin();
    ctx->window_ns = bench_now_ns();
    return 1;
}

static inline int bench_window_close(BenchContext *ctx) {
    const uint64_t t = bench_now_ns();
    ctx->window_ns = t - ctx->window_ns;
    if (ctx->track_memory) bench_mem_end(&ctx->mem);
    const BeamError e = beam_last_error();
    if (e != BEAM_OK && !ctx->failed)
        bench_fail(ctx, (int)e, "%s", beam_error_message());
    return 0;
}

#define BEAM_TIMED_BLOCK(ctx)                                                                \
    for (int _bw = bench_window_open(ctx); _bw; _bw = bench_window_close(ctx))               \
        for (uint32_t _br = 0; _br < (ctx)->reps; ++_br)

/* ── Helpers for benchmark bodies ─────────────────────────────── */

/* Record a failure if a BeamError-returning call did not succeed. */
#define BENCH_CHECK(ctx, call)                                                               \
    do {                                                                                     \
        const BeamError _bc = (call);                                                        \
        if (_bc != BEAM_OK && !(ctx)->failed)                                                \
            bench_fail((ctx), (int)_bc, "%s returned %d: %s", #call, (int)_bc,              \
                       beam_error_message());                                                \
    } while (0)

/* The workload's model, or fail the benchmark and return. */
#define BENCH_MODEL_VARIANT(ctx, var, variant)                                               \
    BeamModel *var = bench_fixture(variant);                                                 \
    if (!var) {                                                                              \
        bench_fail((ctx), -1, "fixture unavailable: %s", bench_fixture_error());             \
        return;                                                                              \
    }
#define BENCH_MODEL(ctx, var) BENCH_MODEL_VARIANT(ctx, var, BENCH_VARIANT_PRIMARY)

/* Fail and return if a setup condition does not hold. */
#define BENCH_REQUIRE(ctx, cond, ...)                                                        \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            bench_fail((ctx), -1, __VA_ARGS__);                                              \
            return;                                                                          \
        }                                                                                    \
    } while (0)

/* Skip (not a failure) when the workload does not apply, e.g. length > max. */
#define BENCH_SKIP_IF(ctx, cond, ...)                                                        \
    do {                                                                                     \
        if (cond) {                                                                          \
            bench_skip((ctx), __VA_ARGS__);                                                  \
            return;                                                                          \
        }                                                                                    \
    } while (0)

/* Keep a result alive so the call producing it cannot be optimised away. */
#define BENCH_KEEP(x)                                                                        \
    do {                                                                                     \
        __typeof__(x) _bk = (x);                                                             \
        __asm__ volatile("" : : "r"(&_bk) : "memory");                                       \
    } while (0)

/* At least `bytes` of zeroed scratch, valid until the next call. Grown
 * outside the timed window; returns NULL (benchmark failed) on OOM. */
void *bench_scratch(BenchContext *ctx, size_t bytes);
#define BENCH_SCRATCH(ctx, type, var, count)                                                 \
    type *var = (type *)bench_scratch((ctx), (size_t)(count) * sizeof(type));                \
    if (!var) return;

/* Declare what one call processes, for throughput in natural units. */
static inline void bench_set_items(BenchContext *ctx, double n, const char *unit) {
    ctx->items_per_op = n;
    ctx->item_unit = unit;
}

#endif /* BENCH_FRAMEWORK_H */
