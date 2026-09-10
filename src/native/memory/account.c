/**
 * @file account.c
 * @brief What one operation has taken from the budget so far.
 *
 * One of the files under src/native/memory/, which together own every
 * memory-management decision the library makes: how much scratch an
 * operation may hold, how the allocator is configured, what a model reports
 * about its own size, and what the per-length cache is allowed to keep.
 *
 * `budget.c` holds the FIGURE. This holds the RUNNING TOTAL against it, which
 * is the half that was missing: `beam_model_memory_budget()` says what one
 * operation may hold at once, and an operation that checks each of its
 * allocations against that figure separately has not observed it. Four
 * allocations that each pass at 90% of the budget hold three and a half times
 * it between them.
 *
 * See internal/beam/memory.h for the contract and for why the refusal is
 * carried on the account rather than only returned.
 */
#include "beam/internal.h"
#include "beam/memory.h"

uint64_t beam_account_room(const BeamMemAccount *a) {
    const uint64_t budget = beam_model_memory_budget();
    return budget > a->held ? budget - a->held : 0u;
}

bool beam_account_take(BeamMemAccount *a, uint64_t bytes) {
    if (bytes > beam_account_room(a)) return false;
    a->held += bytes;
    if (a->held > a->peak) a->peak = a->held;
    return true;
}

void beam_account_release(BeamMemAccount *a, uint64_t bytes) {
    a->held -= bytes;
}
