/**
 * @file runtime.c
 * @brief Process-wide thread and memory budget controls.
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "beam/internal.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif


/* ── Allocator policy ────────────────────────────────────────── */

/*
 * Stop glibc ratcheting its mmap threshold, which is what makes a finished
 * model cost many times its own size in resident memory.
 *
 * The threshold starts at 128 KiB and is DYNAMIC: freeing an mmap-ed block
 * raises it to that block's size, up to 32 MiB. After the first large payload
 * is freed, every later one is served from the heap instead -- and the heap
 * can only be returned to the OS from its top, so a single live allocation
 * above a freed region pins all of it. Nothing is leaked; it is simply never
 * given back.
 *
 * Measured on this machine, checksums identical throughout:
 *
 *                                      default    pinned
 *   per-repertoire model, 13.8 MB
 *     resident after build              450 MB     27 MB
 *     resident after one analytic       678 MB     38 MB
 *     resident after destroying it      348 MB    4.7 MB
 *   foundation model, 806.7 MB
 *     resident after load             1,345 MB    774 MB
 *     peak during load                1,593 MB  1,352 MB
 *
 * A fourteen-megabyte model holding 678 MB is not a trade anyone chose. The
 * cost is about 9% on the Hill sweep, consistently, and nothing measurable on
 * scoring or on building.
 *
 * THIS IS PROCESS-WIDE, and the thread cap above deliberately is not -- it is
 * named in every `num_threads` clause precisely so BEAM never edits global
 * state the rest of the process shares. This breaks that rule, knowingly,
 * because the alternative is a library whose memory cost is dominated by an
 * allocator heuristic its callers have no reason to know about. Two escape
 * hatches keep it from being imposed:
 *
 *   - `MALLOC_MMAP_THRESHOLD_` set in the environment means the caller has
 *     already chosen, and glibc has already applied it. Left alone.
 *   - `BEAM_MMAP_THRESHOLD` sets a different value, or disables this entirely
 *     when it is 0.
 *
 * The value restores glibc's own STATIC default. It is the dynamic growth
 * that is being switched off, not the default that is being second-guessed.
 */
#define BEAM_MMAP_THRESHOLD_DEFAULT (128u * 1024u)

static int g_beam_mmap_threshold = 0;   /* what was applied, 0 if nothing */

int beam_applied_mmap_threshold(void) { return g_beam_mmap_threshold; }

static void beam_apply_allocator_policy(void) {
#if defined(__GLIBC__)
    if (getenv("MALLOC_MMAP_THRESHOLD_")) return;      /* caller already chose */
    unsigned long want = BEAM_MMAP_THRESHOLD_DEFAULT;
    const char *s = getenv("BEAM_MMAP_THRESHOLD");
    if (s && *s) {
        char *end = NULL;
        const unsigned long n = strtoul(s, &end, 10);
        if (end && *end == '\0') want = n;            /* ignore anything else */
    }
    if (want == 0u) return;                            /* explicitly disabled */
    if (want > (unsigned long)INT_MAX) want = (unsigned long)INT_MAX;
    if (mallopt(M_MMAP_THRESHOLD, (int)want) == 1)
        g_beam_mmap_threshold = (int)want;
#endif
}

/*
 * A constructor rather than a call from `beam_model_new`: the ratchet is
 * raised by the first large free anywhere in the process, which may well be
 * before any model exists. runtime.c is pulled in by `beam_threads`, which
 * every parallel region names, so the object is never dropped by the linker.
 */
#if defined(__GNUC__)
__attribute__((constructor))
#endif
static void beam_runtime_init(void) { beam_apply_allocator_policy(); }


/* ── Parallelism ─────────────────────────────────────────────── */

/*
 * Zero means "whatever the runtime would do", which is what a library should
 * default to: the caller's OMP_NUM_THREADS, or one thread per core. Anything
 * positive is a hard cap, applied by naming it in every `num_threads` clause
 * rather than by calling `omp_set_num_threads`, so BEAM never edits global
 * runtime state that the rest of the process shares.
 *
 * Plain `int` rather than an atomic: this is a knob set once at start-up, and
 * the header says so. Making it atomic would suggest a guarantee about
 * changing it mid-computation that the surrounding code does not offer.
 */
static int g_beam_threads = 0;

void beam_model_set_threads(int n) {
    g_beam_threads = n > 0 ? n : 0;
}

int beam_model_threads(void) {
    return g_beam_threads;
}

int beam_threads(void) {
#ifdef _OPENMP
    if (g_beam_threads > 0) return g_beam_threads;
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int beam_model_effective_threads(void) {
    return beam_threads();
}

/*
 * Scratch the per-length dynamic programs may hold at once.
 *
 * Only the spectrum needs this: its per-state histogram grids are 1.6 GB at
 * order 4 with 512 bins, so running one length per core would ask for
 * twenty-five gigabytes and find the swap rather than the speed-up. Every
 * other per-length program needs tens of megabytes and is capped by the
 * thread count alone.
 *
 * Four gigabytes by default: enough for full parallelism at order 2 or 3,
 * and for two or three concurrent lengths at order 4, on a machine that can
 * hold the model at all.
 *
 * It is the most scratch one operation may hold, which caps concurrency AND
 * refuses a single program too large to fit. It used to say the opposite --
 * "a cap on CONCURRENCY, never on what a single length is allowed to
 * allocate, one length always runs, however large" -- while `positional.c`
 * and the query layer refused above it from the same setter. The refusing
 * half was kept: a single sweep reaches 6.6 GB on a nucleotide model at the
 * ceiling order, and a budget that cannot say no to it is decoration.
 * See src/native/internal/beam/budget.h.
 */
static uint64_t g_beam_budget = 4ull << 30;

void beam_model_set_memory_budget(uint64_t bytes) {
    g_beam_budget = bytes ? bytes : (4ull << 30);
}

uint64_t beam_model_memory_budget(void) {
    return g_beam_budget;
}
