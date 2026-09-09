/**
 * @file live_cache.c
 * @brief What the per-length live cache keeps, and when it lets go.
 *
 * One of the files under src/native/memory/. It owns the cache's storage,
 * its bound and its lifetime; dp/liveness.c owns building a live set and
 * deciding when one is wanted. The two met in a single function before, with
 * the DP's construction and the cache's locking interleaved, which is why the
 * cache's cost was invisible to anyone reading either concern alone.
 *
 * NOT counted by `beam_model_memory_bytes`, which reports the model's tables.
 * `beam_model_live_cache_bytes` reports this, and on a per-repertoire model
 * it is the larger of the two: 15.1 MB of cache against a 13.8 MB model.
 */
#include <stdlib.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/memory.h"

/**
 * One live set per length, held on the model until something invalidates it.
 *
 * WHY THIS IS NOT AN OPTIMISATION YOU CAN SKIP. Once a dynamic program
 * iterates the live index instead of all 194,481 states, the structure build
 * becomes the largest single term left in it -- 78% of a converted DP on a
 * fifteen-thousand-sequence donor and 52% on the 927M foundation. Eight
 * analytics each rebuilding it is 3.6x and 2.1x of pure waste, and at the top
 * end it is worse than that: converting the sweeps WITHOUT sharing the
 * structure measures as a 1.4x regression against the unconverted code on the
 * foundation and 1.5x on an open-world model, because the build is repaid
 * over one sweep rather than eight.
 *
 * WHAT IT COSTS. Index lists, edge masks and a one-bit-per-state liveness
 * map, summed over every live length: 7.9 MB for a fifteen-thousand-sequence
 * donor, 388 MB for the foundation, 422 MB for a 45%-live open-world model.
 * The bound below refuses anything past a quarter of the process budget and
 * falls back to building per call, so a model whose live structure does not
 * fit stays correct and merely stops being fast.
 *
 * THREAD SAFETY. Entries are immutable once published, so concurrent readers
 * need no lock; only the publish is serialised, and it happens at most once
 * per length. Two threads racing on the same length both build, one publishes
 * and the other frees its own copy and uses the winner's. Mutating a model
 * while an analytic runs on it is unsupported here exactly as it is
 * everywhere else in the library.
 */
struct BeamLiveCache {
    BeamLive **slot;     /* [n_slots], NULL until a length is first asked for */
    uint32_t   n_slots;
    uint64_t   bytes;    /* what the published entries hold, for the bound    */
};

void beam_live_cache_drop(BeamModel *m) {
    if (!m || !m->live_cache) return;
    BeamLiveCache *c = m->live_cache;
    /* Unhook first, so a caller that somehow re-enters sees no cache rather
     * than half of one. */
    m->live_cache = NULL;
    for (uint32_t i = 0; i < c->n_slots; ++i) {
        if (!c->slot[i]) continue;
        beam_live_free(c->slot[i]);
        free(c->slot[i]);
    }
    free(c->slot);
    free(c);
}

uint64_t beam_live_cache_bytes(const BeamModel *m) {
    if (!m || !m->live_cache) return 0u;
    return m->live_cache->bytes;
}

/**
 * Ceiling on what the cache may hold, in bytes.
 *
 * A quarter of the process memory budget. The budget's other job is capping
 * how many per-length dynamic programs run at once, and the cache is a
 * PERSISTENT claim against the same pool, so it takes a minority share rather
 * than the whole of it. At the 4 GB default that is 1 GB, which holds the
 * foundation's 388 MB and the open-world model's 422 MB with room to spare;
 * lowering the budget lowers this with it.
 */
static uint64_t live_cache_cap(void) {
    return beam_model_memory_budget() / 4u;
}

const BeamLive *beam_live_cache_lookup(const BeamModel *m, uint32_t len) {
    BeamModel *mm = (BeamModel *)m;      /* the cache is state a const model keeps */
    const BeamLive *hit = NULL;
#ifdef _OPENMP
    #pragma omp critical(beam_live_cache)
#endif
    {
        if (mm->live_cache && mm->live_cache->slot[len])
            hit = mm->live_cache->slot[len];
    }
    return hit;
}

const BeamLive *beam_live_cache_publish(const BeamModel *m, uint32_t len,
                                        const BeamLive *built, uint64_t bytes,
                                        bool *took_ownership) {
    BeamModel *mm = (BeamModel *)m;
    const BeamLive *hit = NULL, *moved = NULL;
    *took_ownership = false;
#ifdef _OPENMP
    #pragma omp critical(beam_live_cache)
#endif
    {
        if (!mm->live_cache) {
            BeamLiveCache *c =
                (BeamLiveCache *)calloc(1, sizeof(BeamLiveCache));
            if (c) {
                c->slot = (BeamLive **)calloc(m->n_len_slots, sizeof(BeamLive *));
                c->n_slots = m->n_len_slots;
                if (c->slot) mm->live_cache = c;
                else free(c);
            }
        }
        if (mm->live_cache) {
            if (mm->live_cache->slot[len]) {
                hit = mm->live_cache->slot[len];      /* another thread won */
            } else if (mm->live_cache->bytes + bytes <= live_cache_cap()) {
                BeamLive *e = (BeamLive *)malloc(sizeof(BeamLive));
                if (e) {
                    *e = *built;
                    mm->live_cache->slot[len] = e;
                    mm->live_cache->bytes += bytes;
                    moved = e;
                }
            }
        }
    }
    if (moved) { *took_ownership = true; return moved; }
    return hit;                       /* NULL means: keep and free your own */
}

/* Published lengths, for the test hooks that assert the cache is being used. */
uint32_t beam_live_cache_entries(const BeamModel *m) {
    if (!m || !m->live_cache) return 0u;
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->live_cache->n_slots; ++i)
        if (m->live_cache->slot[i]) n++;
    return n;
}
