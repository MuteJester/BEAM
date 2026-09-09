/**
 * @file allocator.c
 * @brief Process-wide allocator configuration.
 *
 * One of the files under src/native/memory/, which together own every
 * memory-management decision the library makes: how much scratch an
 * operation may hold, how the allocator is configured, what a model reports
 * about its own size, and what the per-length cache is allowed to keep.
 */
#include "beam/internal.h"
#include "beam/memory.h"

#include <stdlib.h>
#include <limits.h>

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
