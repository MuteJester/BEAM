/* Observe the WIDTH of every OpenMP team BEAM actually creates.
 *
 * Every check BEAM had before this one was static. `check_lib.thread_count_clauses`
 * proves all 93 parallel regions NAME a thread count, and the GIL scan proves
 * every GIL-releasing region RESOLVES a plan. Neither can see what the runtime
 * granted. That blind spot is not hypothetical: `beam_threads()` returned the
 * configured cap to a worker thread for as long as the plan existed, so an
 * 8-thread plan opened eight teams of sixteen under OMP_MAX_ACTIVE_LEVELS=2 --
 * and `thread counts` was OK in the same gate run, because the clause was
 * present and correct and only its VALUE on a worker was wrong.
 *
 * This observes the shipped artifact rather than a special build. BEAM creates
 * teams through exactly two libgomp entry points (`nm --undefined-only` on the
 * extension: GOMP_parallel and the fused GOMP_parallel_loop_nonmonotonic_dynamic;
 * the other eight GOMP imports are work-sharing, atomics and critical sections,
 * which create no team). Interposing those two via LD_PRELOAD measures the real
 * `_cbeam*.so` the gate hashes, with no recompilation -- a check that needed its
 * own build would be checking a different binary.
 * `test_teamwidth.py::test_the_observer_wraps_every_team_creating_entry_point`
 * fails if a third ever appears, because the observer would then be blind and
 * still report zero violations.
 *
 * REQUESTED vs ACTUAL, and why both. The `nthreads` argument is what BEAM's
 * `num_threads(...)` clause evaluated to -- the plan's answer, which until now
 * was only ever checked by reading the source. The team that runs can be
 * narrower (the runtime is allowed to give less) but never wider, so the two
 * numbers bracket the question: `requested` audits BEAM's arithmetic, `actual`
 * audits the runtime's grant. Recording only one of them would leave either the
 * plan or the runtime unexamined.
 *
 * Build:  make teamwidth
 * Use:    LD_PRELOAD=build/teamwidth.so .venv/bin/python ...
 *         then read the counters through ctypes.CDLL(None) -- an LD_PRELOAD
 *         library is in the global symbol scope, so its exports resolve there.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

/* Widths, split by whether the region ITSELF is nested. The split is the whole
 * point: the defect that motivated this file was invisible in the outer team's
 * width (a correct 8) and only showed in the inner one (16). */
static atomic_uint_fast64_t w_outer    = 0;  /* widest depth-1 team          */
static atomic_uint_fast64_t w_nested   = 0;  /* widest team at depth >= 2    */
static atomic_uint_fast64_t w_any      = 0;  /* widest team at any depth     */
static atomic_uint_fast64_t r_any      = 0;  /* largest `nthreads` requested */
static atomic_uint_fast64_t n_unnamed  = 0;  /* entries with nthreads == 0   */
static atomic_uint_fast64_t n_entries  = 0;  /* team creations observed      */
static atomic_uint_fast64_t d_max      = 0;  /* deepest level reached        */

typedef int  (*int_fn_t)(void);
typedef void (*fn_t)(void *);
typedef void (*plain_t)(fn_t, void *, unsigned, unsigned);
typedef void (*loop_t)(fn_t, void *, unsigned, long, long, long, long, unsigned);

static plain_t  real_plain;
static loop_t   real_loop;
static int_fn_t omp_level;
static int_fn_t omp_width;

/*
 * RTLD_NEXT cannot find libgomp here: CPython dlopens extension modules with
 * RTLD_LOCAL, so libgomp lands in a local scope RTLD_NEXT never searches and
 * reports "undefined symbol". dlopen on the already-loaded library returns a
 * handle to that same mapping, and dlsym through the handle works.
 */
static void *gomp(void)
{
    static void *h;
    if (!h) {
        h = dlopen("libgomp.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!h)
            h = dlopen("libgomp.so.1", RTLD_NOW);
        if (!h) {
            fprintf(stderr, "teamwidth: dlopen libgomp.so.1 failed: %s\n",
                    dlerror());
            _exit(96);
        }
    }
    return h;
}

static void *need(const char *sym, int code)
{
    void *p = dlsym(gomp(), sym);
    if (!p) {
        fprintf(stderr, "teamwidth: dlsym %s failed\n", sym);
        _exit(code);
    }
    return p;
}

static void resolve(void)
{
    if (!real_plain) real_plain = (plain_t)need("GOMP_parallel", 97);
    if (!real_loop)
        real_loop = (loop_t)need("GOMP_parallel_loop_nonmonotonic_dynamic", 98);
    /*
     * Both of these must come from libgomp, not from a thread-local counter of
     * our own. A worker thread entering a region from inside its team sees its
     * own TLS depth as 0 and would be charged as outermost -- the error that
     * made an earlier shim report 755% of wall at 16 threads.
     */
    if (!omp_level) omp_level = (int_fn_t)need("omp_get_level", 99);
    if (!omp_width) omp_width = (int_fn_t)need("omp_get_num_threads", 95);
}

static void bump_max(atomic_uint_fast64_t *slot, uint_fast64_t v)
{
    uint_fast64_t prev = atomic_load(slot);
    while (v > prev && !atomic_compare_exchange_weak(slot, &prev, v))
        ;
}

/*
 * Runs as the team's body, once per thread. Every member of a team sees the
 * same `omp_get_num_threads()`, so the atomic max collapses them to one
 * observation per region and no count is needed.
 */
struct tramp { fn_t fn; void *data; };

static void observe(void *p)
{
    struct tramp *t = (struct tramp *)p;
    /*
     * Inside the region, so level 1 means this region is the outermost and
     * level >= 2 means it is nested. An INACTIVE nested region -- the default,
     * where OMP_MAX_ACTIVE_LEVELS is 1 -- still raises the level but has a
     * width of 1, so it is recorded truthfully as a team of one and cannot
     * look like a violation.
     */
    const uint_fast64_t lvl   = (uint_fast64_t)omp_level();
    const uint_fast64_t width = (uint_fast64_t)omp_width();

    bump_max(&w_any, width);
    bump_max(lvl >= 2 ? &w_nested : &w_outer, width);
    bump_max(&d_max, lvl);

    t->fn(t->data);
}

/*
 * `t` lives on this frame, which is safe because GOMP_parallel is synchronous:
 * it creates the team, runs the body on every member, joins, and only then
 * returns. The trampoline cannot outlive the call that set it up.
 */
static void record_request(unsigned nthreads)
{
    atomic_fetch_add(&n_entries, 1);
    if (nthreads == 0)
        atomic_fetch_add(&n_unnamed, 1);
    else
        bump_max(&r_any, (uint_fast64_t)nthreads);
}

void GOMP_parallel(fn_t fn, void *data, unsigned nthreads, unsigned flags)
{
    resolve();
    record_request(nthreads);
    struct tramp t = { fn, data };
    real_plain(observe, &t, nthreads, flags);
}

void GOMP_parallel_loop_nonmonotonic_dynamic(fn_t fn, void *data,
                                             unsigned nthreads, long start,
                                             long end, long incr, long chunk,
                                             unsigned flags)
{
    resolve();
    record_request(nthreads);
    struct tramp t = { fn, data };
    real_loop(observe, &t, nthreads, start, end, incr, chunk, flags);
}

uint64_t beam_tw_outer(void)     { return atomic_load(&w_outer); }
uint64_t beam_tw_nested(void)    { return atomic_load(&w_nested); }
uint64_t beam_tw_widest(void)    { return atomic_load(&w_any); }
uint64_t beam_tw_requested(void) { return atomic_load(&r_any); }
uint64_t beam_tw_unnamed(void)   { return atomic_load(&n_unnamed); }
uint64_t beam_tw_entries(void)   { return atomic_load(&n_entries); }
uint64_t beam_tw_depth(void)     { return atomic_load(&d_max); }

/*
 * Reset before the call under test. Without this the counters would carry
 * whatever `import BEAM` and the fixture build did, and the widest team in the
 * process is not evidence about one API call.
 */
void beam_tw_reset(void)
{
    atomic_store(&w_outer, 0);
    atomic_store(&w_nested, 0);
    atomic_store(&w_any, 0);
    atomic_store(&r_any, 0);
    atomic_store(&n_unnamed, 0);
    atomic_store(&n_entries, 0);
    atomic_store(&d_max, 0);
}
