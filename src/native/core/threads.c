/**
 * @file threads.c
 * @brief Process-wide parallelism control.
 *
 * Memory-side process controls -- the allocator policy and the scratch
 * budget -- live under src/native/memory/ rather than here, so that the
 * one file that edits global allocator state is the one place to look for
 * it. This file deliberately edits none: the thread cap is applied by
 * naming it in every `num_threads` clause.
 */
#include <stdio.h>

#include "beam/internal.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif


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
