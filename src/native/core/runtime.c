/**
 * @file runtime.c
 * @brief Process-wide thread and memory budget controls.
 */
#include <stdio.h>

#include "beam/internal.h"


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
