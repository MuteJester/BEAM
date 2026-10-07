#include "bench_memory.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

extern void *__real_malloc(size_t size);
extern void *__real_calloc(size_t nmemb, size_t size);
extern void *__real_realloc(void *ptr, size_t size);
extern void *__real_aligned_alloc(size_t alignment, size_t size);
extern void  __real_free(void *ptr);

/*
 * Every wrapped block is [padding][header][payload], the header always the
 * 16 bytes just before the payload so free() knows where to look:
 *
 *   tag  = 56-bit magic << 8 | log2(offset from the raw block to the payload)
 *   meta = size << 24 | epoch (24 bits; 0 = allocated outside every window)
 *
 * The offset is 16 for malloc/calloc/realloc and the alignment for
 * aligned_alloc (which BEAM uses for the Mellin line and real work buffers;
 * missing it would understate exactly the analytics that allocate most).
 * Sizes up to 2^40 bytes are representable.
 */
typedef struct BenchAllocHeader {
    uint64_t tag;
    uint64_t meta;
} BenchAllocHeader;

_Static_assert(sizeof(BenchAllocHeader) == 16, "header must keep 16-byte alignment");

#define MAGIC56     0x4245414D4D454DULL   /* "BEAMMEM" */
#define EPOCH_BITS  24u
#define EPOCH_MASK  ((1u << EPOCH_BITS) - 1u)
#define MAX_SIZE    ((1ULL << 40) - 1u)

static _Atomic uint32_t g_epoch = 0;       /* epoch of the open window, 0 if none */
static _Atomic uint32_t g_last_epoch = 0;
static _Atomic int64_t  g_live = 0;
static _Atomic uint64_t g_peak = 0;
static _Atomic uint64_t g_alloc_bytes = 0;
static _Atomic uint64_t g_alloc_calls = 0;
static _Atomic uint64_t g_free_calls = 0;

static inline bool is_ours(const BenchAllocHeader *h) { return (h->tag >> 8) == MAGIC56; }
static inline size_t hdr_size(const BenchAllocHeader *h) { return (size_t)(h->meta >> EPOCH_BITS); }
static inline uint32_t hdr_epoch(const BenchAllocHeader *h) { return (uint32_t)(h->meta & EPOCH_MASK); }
static inline void *hdr_raw(BenchAllocHeader *h) {
    return (char *)h + sizeof(BenchAllocHeader) - ((size_t)1 << (h->tag & 0xFFu));
}

static inline void note_peak(uint64_t cur) {
    uint64_t prev = atomic_load_explicit(&g_peak, memory_order_relaxed);
    while (cur > prev &&
           !atomic_compare_exchange_weak_explicit(&g_peak, &prev, cur,
                                                  memory_order_relaxed, memory_order_relaxed)) {
    }
}

static inline void account_alloc(size_t size, uint32_t epoch) {
    if (epoch == 0) return;
    atomic_fetch_add_explicit(&g_alloc_bytes, size, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_alloc_calls, 1, memory_order_relaxed);
    const int64_t cur = atomic_fetch_add_explicit(&g_live, (int64_t)size, memory_order_relaxed) + (int64_t)size;
    if (cur > 0) note_peak((uint64_t)cur);
}

/* A free counts only against the window its block was allocated in. */
static inline void account_free(size_t size, uint32_t epoch) {
    if (epoch == 0 || epoch != atomic_load_explicit(&g_epoch, memory_order_relaxed)) return;
    atomic_fetch_add_explicit(&g_free_calls, 1, memory_order_relaxed);
    atomic_fetch_sub_explicit(&g_live, (int64_t)size, memory_order_relaxed);
}

static void *finish(void *raw, size_t offset_log2, size_t size) {
    char *payload = (char *)raw + ((size_t)1 << offset_log2);
    BenchAllocHeader *h = (BenchAllocHeader *)(payload - sizeof(BenchAllocHeader));
    const uint32_t epoch = atomic_load_explicit(&g_epoch, memory_order_relaxed);
    h->tag = (MAGIC56 << 8) | offset_log2;
    h->meta = ((uint64_t)size << EPOCH_BITS) | epoch;
    account_alloc(size, epoch);
    return payload;
}

void *__wrap_malloc(size_t size) {
    if (size > MAX_SIZE) return NULL;
    void *raw = __real_malloc(sizeof(BenchAllocHeader) + size);
    return raw ? finish(raw, 4u, size) : NULL;
}

void *__wrap_calloc(size_t nmemb, size_t size) {
    if (size && nmemb > SIZE_MAX / size) return NULL;
    const size_t bytes = nmemb * size;
    if (bytes > MAX_SIZE) return NULL;
    /* Real calloc, so large zeroed blocks keep their lazily-faulted pages. */
    void *raw = __real_calloc(1, sizeof(BenchAllocHeader) + bytes);
    return raw ? finish(raw, 4u, bytes) : NULL;
}

void *__wrap_aligned_alloc(size_t alignment, size_t size) {
    if (alignment < sizeof(BenchAllocHeader)) alignment = sizeof(BenchAllocHeader);
    if ((alignment & (alignment - 1u)) != 0 || size > MAX_SIZE) return NULL;
    size_t log2 = 0;
    while (((size_t)1 << log2) < alignment) ++log2;
    /* aligned_alloc wants a multiple of the alignment. */
    size_t total = alignment + size;
    total = (total + alignment - 1u) & ~(alignment - 1u);
    void *raw = __real_aligned_alloc(alignment, total);
    return raw ? finish(raw, log2, size) : NULL;
}

void __wrap_free(void *ptr) {
    if (!ptr) return;
    BenchAllocHeader *h = (BenchAllocHeader *)((char *)ptr - sizeof(BenchAllocHeader));
    if (!is_ours(h)) {
        __real_free(ptr);   /* from an allocator we do not wrap */
        return;
    }
    account_free(hdr_size(h), hdr_epoch(h));
    void *raw = hdr_raw(h);
    h->tag = 0;
    __real_free(raw);
}

void *__wrap_realloc(void *ptr, size_t size) {
    if (!ptr) return __wrap_malloc(size);
    if (size == 0) {
        __wrap_free(ptr);
        return NULL;
    }
    if (size > MAX_SIZE) return NULL;
    BenchAllocHeader *h = (BenchAllocHeader *)((char *)ptr - sizeof(BenchAllocHeader));
    if (!is_ours(h)) return __real_realloc(ptr, size);

    const size_t old = hdr_size(h);
    const uint32_t old_epoch = hdr_epoch(h);
    if ((h->tag & 0xFFu) == 4u) {
        /* The real realloc, so a block that can grow in place still does and
         * the timing is not distorted by a copy. */
        BenchAllocHeader *nh = (BenchAllocHeader *)__real_realloc(h, sizeof(BenchAllocHeader) + size);
        if (!nh) return NULL;
        account_free(old, old_epoch);
        const uint32_t epoch = atomic_load_explicit(&g_epoch, memory_order_relaxed);
        nh->meta = ((uint64_t)size << EPOCH_BITS) | epoch;
        account_alloc(size, epoch);
        return (char *)nh + sizeof(BenchAllocHeader);
    }
    /* An over-aligned block: realloc does not preserve alignment anyway. */
    void *fresh = __wrap_malloc(size);
    if (!fresh) return NULL;
    memcpy(fresh, ptr, old < size ? old : size);
    __wrap_free(ptr);
    return fresh;
}

void bench_mem_begin(void) {
    atomic_store(&g_live, 0);
    atomic_store(&g_peak, 0);
    atomic_store(&g_alloc_bytes, 0);
    atomic_store(&g_alloc_calls, 0);
    atomic_store(&g_free_calls, 0);
    uint32_t e = (atomic_fetch_add(&g_last_epoch, 1) + 1) & EPOCH_MASK;
    if (e == 0) e = (atomic_fetch_add(&g_last_epoch, 1) + 1) & EPOCH_MASK;   /* 0 is reserved */
    atomic_store(&g_epoch, e);
}

void bench_mem_end(BenchMemStats *out) {
    atomic_store(&g_epoch, 0);
    if (!out) return;
    out->peak_bytes = atomic_load(&g_peak);
    out->retained_bytes = atomic_load(&g_live);
    out->alloc_bytes = atomic_load(&g_alloc_bytes);
    out->alloc_calls = atomic_load(&g_alloc_calls);
    out->free_calls = atomic_load(&g_free_calls);
}

uint64_t bench_mem_process_peak_rss(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    unsigned long kb = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmHWM: %lu kB", &kb) == 1) break;
    }
    fclose(f);
    return (uint64_t)kb * 1024u;
}
