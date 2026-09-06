/**
 * @file budget.h
 * @brief One meaning for the memory budget, enforced two ways.
 *
 * Internal to src/native/. Not a public header and not installed.
 *
 * The budget is the most scratch the library will hold at once for one
 * operation. Two consequences follow from that single sentence, and every
 * per-length dynamic program in the library needs both:
 *
 *   - a program whose working set exceeds the budget is REFUSED, before it
 *     allocates, by `beam_budget_refuse`;
 *   - lengths that run concurrently divide it, so `beam_budget_concurrency`
 *     returns how many may run at once rather than letting the thread count
 *     multiply the footprint.
 *
 * Before this header the two halves disagreed. `runtime.c` and `model.h` both
 * promised the budget was "a cap on CONCURRENCY, never on what a single length
 * is allowed to allocate -- one length always runs, however large", while
 * `positional.c` and the whole query layer refused outright above it. Both
 * behaviours were reachable from the same setter, and a call that worked at
 * 4 GB failed at 1 MB in exactly the paths the documentation said it could
 * not. Refusing is the behaviour that was kept, because the alternative
 * protects nothing: a single sweep is 6.6 GB on a nucleotide model at the
 * alphabet's ceiling order, and letting it through unconditionally is how a
 * machine dies.
 *
 * REFUSE BEFORE ALLOCATING, never after. `malloc` and `calloc` on Linux grant
 * an absurd request lazily and fail on first write, by which time the OOM
 * killer is choosing its victim from the whole machine rather than from this
 * process. A guard that runs after the allocation has already lost.
 *
 * TWO LAYERS, deliberately. A by-length driver refuses up front, on the
 * calling thread, because `beam_set_error` is thread-local: a refusal raised
 * inside an OpenMP worker reaches the driver as a bare code and the message
 * naming the figure and the way out is lost. The sweep then guards its own
 * allocation as well, which is redundant on every path that has a driver and
 * is the only protection on a path that does not.
 *
 * A consequence worth stating, because it looks like a hole: removing a
 * sweep's own guard does not fail the suite where a driver refuses first. That
 * is the redundancy working, not a missing test.
 *
 * The guard's figure must come from the SAME expression as the allocation it
 * guards. A budget check computed from its own copy of the formula is not a
 * check, and the difference shows up only on the machine that runs out of
 * memory. Each caller therefore computes `need` once, next to the allocation,
 * and passes that one value here.
 */
#ifndef BEAM_INTERNAL_BUDGET_H
#define BEAM_INTERNAL_BUDGET_H

#include <stdint.h>

#include "beam/model.h"
#include "beam/error_macros.h"

/** True when a working set of `need` bytes fits the budget in force. */
static inline bool beam_budget_fits(uint64_t need) {
    return need <= beam_model_memory_budget();
}

/**
 * A byte count and the unit to print it in.
 *
 * Fixing the unit at gigabytes made both halves of a refusal read "0.00 GB"
 * whenever the budget had been set small, which is the case a user is most
 * likely to be looking at deliberately: the message existed to report a
 * figure and reported nothing. Each side picks its own unit, so a sweep
 * needing 260 kB against a 2 kB budget says so.
 */
static inline double beam_budget_scaled(uint64_t bytes, const char **unit) {
    if (bytes >= (1ull << 30)) { *unit = "GB"; return (double)bytes / 1e9; }
    if (bytes >= (1ull << 20)) { *unit = "MB"; return (double)bytes / 1e6; }
    if (bytes >= (1ull << 10)) { *unit = "kB"; return (double)bytes / 1e3; }
    *unit = "bytes";
    return (double)bytes;
}

/**
 * The standard refusal, naming what was asked for and the way out.
 *
 * `what` is a noun phrase for the program that will not fit, in the caller's
 * own words -- "a tilted forward sweep", "the surprisal spectrum grid" -- so
 * the message says which analytic refused rather than only that one did.
 */
static inline BeamError beam_budget_refuse(const char *what, uint32_t length,
                                           uint64_t need) {
    const char *nu, *bu;
    const double nv = beam_budget_scaled(need, &nu);
    const double bv = beam_budget_scaled(beam_model_memory_budget(), &bu);
    return BEAM_FAIL(BEAM_ERR_BUDGET,
                     "%s at length %u needs about %.2f %s of scratch and the "
                     "memory budget is %.2f %s; raise it with "
                     "BEAM.set_memory_budget, or ask at a lower order",
                     what, (unsigned)length, nv, nu, bv, bu);
}

/**
 * The same refusal for a program that is not indexed by a sequence length.
 *
 * The publicness PGF is sized by the number of atoms and groups rather than by
 * a length, and inventing a length for the message would be a worse lie than
 * leaving it out.
 */
static inline BeamError beam_budget_refuse_bytes(const char *what,
                                                 uint64_t need) {
    const char *nu, *bu;
    const double nv = beam_budget_scaled(need, &nu);
    const double bv = beam_budget_scaled(beam_model_memory_budget(), &bu);
    return BEAM_FAIL(BEAM_ERR_BUDGET,
                     "%s needs about %.2f %s of scratch and the memory budget "
                     "is %.2f %s; raise it with BEAM.set_memory_budget",
                     what, nv, nu, bv, bu);
}

/** `beam_budget_fits` and `beam_budget_refuse` in the shape callers want. */
#define BEAM_BUDGET_OR_REFUSE(what, length, need)                          \
    do {                                                                   \
        const uint64_t beam_need_ = (need);                                \
        if (!beam_budget_fits(beam_need_))                                 \
            return beam_budget_refuse((what), (length), beam_need_);       \
    } while (0)

/**
 * How many per-length programs of `per_item` bytes may run at once.
 *
 * Never more than `n_items` and never more than the thread cap, so this only
 * ever lowers the concurrency the scheduler would otherwise use. Returns 0
 * when a SINGLE item does not fit, which the caller must turn into a refusal
 * rather than silently running it anyway -- 0 is not "run none", it is "this
 * does not fit at all".
 *
 * `per_item` of 0 means the caller could not measure one, and is treated as
 * fitting: refusing on an unmeasured size would refuse everything.
 */
static inline uint32_t beam_budget_concurrency(uint64_t per_item,
                                               uint32_t n_items) {
    if (per_item == 0u) return n_items ? n_items : 1u;
    const uint64_t conc = beam_model_memory_budget() / per_item;
    if (conc == 0u) return 0u;                    /* one does not even fit */
    uint32_t c = conc > n_items ? n_items : (uint32_t)conc;
    const int th = beam_model_effective_threads();
    if (th > 0 && c > (uint32_t)th) c = (uint32_t)th;
    return c < 1u ? 1u : c;
}

#endif /* BEAM_INTERNAL_BUDGET_H */
