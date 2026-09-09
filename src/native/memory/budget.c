/**
 * @file budget.c
 * @brief The scratch one operation may hold at once.
 *
 * One of the files under src/native/memory/, which together own every
 * memory-management decision the library makes: how much scratch an
 * operation may hold, how the allocator is configured, what a model reports
 * about its own size, and what the per-length cache is allowed to keep.
 */
#include "beam/internal.h"
#include "beam/memory.h"

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
