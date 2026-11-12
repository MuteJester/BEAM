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
#include <stdlib.h>      /* getenv */

#include "beam/internal.h"
#include "beam/avail.h"
#include "beam/plan.h"

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

/*
 * The plan in force on THIS thread, or 0 when none is. The policy never
 * returns 0 -- it floors at one -- so 0 is an unambiguous "no plan active",
 * which is what keeps every build before the entry points are wired inert.
 *
 * Thread-local, and plan.h says why at length: a process-global plan would be
 * visible to OpenMP workers, but two BEAM calls on two threads would then
 * overwrite each other's plan mid-call, which is the buffer-versus-team
 * mismatch this mechanism exists to prevent.
 */
#ifdef _MSC_VER
  #define TLS __declspec(thread)
#else
  #define TLS _Thread_local
#endif

static TLS int tl_plan = 0;

/* What `beam_model_set_threads` configured, with no plan involved. */
static int threads_cap(void) {
#ifdef _OPENMP
    if (g_beam_threads > 0) return g_beam_threads;
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int beam_plan_enter(const BeamPlanIn *in) {
    const int prev = tl_plan;
    tl_plan = beam_plan_threads(in);
    return prev;
}

void beam_plan_leave(int token) {
    tl_plan = token;
}

void beam_plan_inputs(BeamPlanIn *in) {
    if (!in) return;
    in->user_cap = g_beam_threads;
    /*
     * OMP_NUM_THREADS set by the caller is the same intent as an explicit cap,
     * expressed through OpenMP, and the policy stands aside for it. Read here
     * rather than inside the policy, which reads nothing.
     */
    in->omp_env = getenv("OMP_NUM_THREADS") != NULL;
    in->mem_cap = 0u;                /* applied deeper, on top of the plan */
    in->items = 0u;                  /* not known at an entry point */
    in->cores = beam_cores_present();
#ifdef _OPENMP
    in->omp_max = omp_get_max_threads();    /* respects OMP_NUM_THREADS */
#else
    in->omp_max = 1;
#endif
    in->available = beam_available_cores();
}

int beam_plan_enter_default(void) {
    BeamPlanIn in;
    beam_plan_inputs(&in);
    return beam_plan_enter(&in);
}

int beam_threads(void) {
#ifdef _OPENMP
    if (tl_plan > 0) return tl_plan;
    return threads_cap();
#else
    return 1;
#endif
}

/*
 * Public API, and deliberately NOT the plan. This answers "what is
 * configured"; `beam_threads()` answers "what will run". They were the same
 * function until the policy arrived, and conflating them again would break a
 * caller's introspection and the recipe for reproducing a pinned benchmark --
 * which is how the paper's performance table is reproducible at all.
 */
int beam_model_effective_threads(void) {
    return threads_cap();
}
