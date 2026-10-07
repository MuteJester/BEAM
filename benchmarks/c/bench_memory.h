#ifndef BENCH_MEMORY_H
#define BENCH_MEMORY_H

/*
 * Exact heap accounting for one timed window, by wrapping the allocator at
 * link time (-Wl,--wrap=malloc,calloc,realloc,free).
 *
 * Every block carries a 16-byte header recording its size and the EPOCH it
 * was allocated in. A window opens a new epoch, so a free counts against the
 * window only when the block was also allocated inside it. Without that, a
 * window that frees memory allocated earlier (a model built in setup, a cache
 * from the previous call) would drive the live count negative and understate
 * the peak -- which is what the first version of this harness reported.
 *
 *   peak_bytes      high-water mark of bytes allocated in the window and
 *                   still live, i.e. the scratch the call needed at once
 *   retained_bytes  allocated in the window and not freed by its end
 *                   (results handed back, caches built)
 *   alloc_bytes     total bytes requested in the window
 *   alloc_calls     malloc + calloc + realloc calls in the window
 *
 * malloc, calloc, realloc, aligned_alloc and free are wrapped (BEAM uses no
 * other allocator). Only calls from objects linked into the harness are seen,
 * which includes all of libbeam.a. libgomp and libc internals are not seen.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct BenchMemStats {
    uint64_t peak_bytes;
    int64_t  retained_bytes;
    uint64_t alloc_bytes;
    uint64_t alloc_calls;
    uint64_t free_calls;
} BenchMemStats;

/* Open a window: counters reset, a new epoch begins. */
void bench_mem_begin(void);

/* Close the window and report what happened inside it. */
void bench_mem_end(BenchMemStats *out);

/* Peak resident set of the whole process so far (VmHWM), in bytes; 0 if unknown. */
uint64_t bench_mem_process_peak_rss(void);

#endif /* BENCH_MEMORY_H */
