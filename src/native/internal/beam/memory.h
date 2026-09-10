/**
 * @file memory.h
 * @brief The library's memory-management surface, in one place.
 *
 * Internal to src/native/. Not a public header and not installed.
 *
 * Everything that decides how much memory BEAM uses, reports, or gives back
 * is implemented under src/native/memory/ and declared here:
 *
 *   allocator.c    how the process allocator is configured, and why
 *   budget.c       the scratch one operation may hold at once
 *   account.c      the running total one operation has taken against it
 *   accounting.c   what a model reports about its own size
 *   live_cache.c   what the per-length cache keeps, and when it lets go
 *
 * They were previously spread across core/runtime.c, model/introspection.c
 * and dp/liveness.c, each sharing a file with an unrelated responsibility.
 * The allocator policy in particular is the one place the library edits
 * global process state, and it should not take reading a file about thread
 * counts to find it.
 */
#ifndef BEAM_INTERNAL_MEMORY_H
#define BEAM_INTERNAL_MEMORY_H

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/live_frontier.h"   /* BeamLive, named below */

/* ── account.c ───────────────────────────────────────────────── */

/**
 * What a single operation currently holds against the memory budget.
 *
 * The budget is "the most scratch ONE OPERATION may hold at once", and an
 * operation is usually several allocations that are live together: a
 * forward-backward workspace, a set of accumulators, the answer they emit.
 * Checking each against the whole budget independently is not observing it --
 * four allocations that each pass at 90% of the figure hold three and a half
 * times it between them. One account, opened by the entry point, and each
 * allocation asks for what is left.
 *
 * ZERO-INITIALISE and pass by pointer. There is deliberately no constructor
 * and no null-account convenience: an optional account is what
 * `math/positional.c` had first, where the reader tested for null and the
 * writer did not, so the one caller that passed nothing dereferenced it. An
 * account that is always there has no half of the contract to get wrong.
 *
 * `peak` is the high-water mark, which is what an operation should REPORT: a
 * caller sizing a job needs what was actually held, and for anything whose
 * footprint depends on its answer rather than on its inputs that figure
 * cannot be predicted, only measured.
 *
 * `err` carries a refusal out of a function with no room to return one --
 * `double *` with NULL for failure, say. Without it a budget refusal and a
 * genuine allocation failure reach the caller as the same bare code and the
 * message naming the figure and the way out is lost. The account records
 * WHICH; the wording stays with the caller, which is the only place that
 * knows what it was allocating.
 */
typedef struct {
    uint64_t  held;     /**< bytes this operation currently holds   */
    uint64_t  peak;     /**< the most it held at once, for reporting */
    BeamError err;      /**< BEAM_OK, or the refusal that stopped it */
} BeamMemAccount;

/** How much of the budget is still unclaimed. Zero, never negative. */
uint64_t beam_account_room(const BeamMemAccount *a);

/**
 * Take `bytes` if they fit, and say whether they did. Sets no error: the
 * caller words its own, because only the caller knows what it was for.
 *
 * REFUSE BEFORE ALLOCATING, as budget.h requires. Charge the figure you are
 * about to hand `malloc`, and allocate only if this returns true.
 */
bool beam_account_take(BeamMemAccount *a, uint64_t bytes);

/** Give `bytes` back. Never more than was taken. */
void beam_account_release(BeamMemAccount *a, uint64_t bytes);


/* ── allocator.c ─────────────────────────────────────────────── */

/**
 * The mmap threshold BEAM pinned at start-up, or 0 if it pinned none --
 * because the caller had already set `MALLOC_MMAP_THRESHOLD_`, because
 * `BEAM_MMAP_THRESHOLD` is 0, or because this is not glibc.
 */
int beam_applied_mmap_threshold(void);

/**
 * Apply the allocator policy, once per process.
 *
 * A constructor in allocator.c calls this, and so does model construction.
 * Both are needed: the constructor runs earliest, but a static archive drops
 * an object nothing references, and this file's symbols are referenced by
 * nobody else. Idempotent.
 */
void beam_memory_init(void);

/* ── live_cache.c ────────────────────────────────────────────── */

/**
 * The cached live set for `len`, or NULL. Takes the cache lock.
 */
const BeamLive *beam_live_cache_lookup(const BeamModel *m, uint32_t len);

/**
 * Offer a freshly built live set to the cache, under one lock.
 *
 * Returns the entry the caller should use. `*took_ownership` says whether the
 * cache adopted `built` -- in which case the caller must not free it -- and
 * is false both when another thread published first and when the cache
 * declined for want of budget. Those two are distinguished by the return
 * value: an entry that is not `built` came from the winner, and NULL means
 * the caller keeps and frees its own copy.
 *
 * One function rather than a lookup and an insert, because the check and the
 * insert must happen under the same lock or two threads publish the same
 * length and one of the entries is leaked.
 */
const BeamLive *beam_live_cache_publish(const BeamModel *m, uint32_t len,
                                        const BeamLive *built, uint64_t bytes,
                                        bool *took_ownership);

/** Release every cached live set. Safe on a model that never built one. */
void beam_live_cache_drop(BeamModel *m);

/** What the published entries hold. Zero when there is no cache. */
uint64_t beam_live_cache_bytes(const BeamModel *m);

/** How many lengths are published, for the tests that assert cache use. */
uint32_t beam_live_cache_entries(const BeamModel *m);

/**
 * Free one live set's arrays. Implemented in dp/liveness.c, which builds
 * them; declared here because the cache is what ultimately frees them.
 */
void beam_live_free(BeamLive *L);

#endif /* BEAM_INTERNAL_MEMORY_H */
