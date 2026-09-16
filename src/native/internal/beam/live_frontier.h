/**
 * @file live_frontier.h
 * @brief Forward/live frontiers and the live-structure test hooks.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_LIVE_FRONTIER_H
#define BEAM_INTERNAL_LIVE_FRONTIER_H

#include <string.h>

#include "beam/model.h"
#include "beam/model_state.h"

/* ── Forward/live frontiers ─────────────────────────────── */

/**
 * Per-length live structure of the trellis, cached on the model.
 *
 * Opaque here on purpose: memory/live_cache.c owns the layout, the bound and
 * the lifetime, and nothing outside it may reach into a cached entry. Its
 * interface is in memory.h; what belongs HERE is the one rule every caller
 * of this file has to obey, which is absolute:
 *
 *   **Any change to the counts or to the finalized tables invalidates it.**
 *
 * The cache is a projection of the finalized tables, so a mutation that
 * leaves it in place would answer later questions about the model as it used
 * to be. Every entry point that touches counts or re-finalizes calls
 * `beam_live_cache_drop` FIRST -- `observe`, `add`, `subtract`,
 * `posterior`, `compact` and `finalize` -- and `destroy` frees it. Copies are
 * built by adding into a fresh model, so they start with no cache and cannot
 * inherit a stale one.
 *
 * It is never serialised. A saved model stores counts, so loading re-finalizes
 * and the reader builds its own.
 *
 * `beam_live_cache_drop` releases every cached live set and resets the
 * accounting. Safe on a model that never built one, and safe to call twice.
 * NOT safe to call while another thread is inside an analytic on the same
 * model, which is the same rule the rest of the library follows for mutation.
 */
/* Declared in memory.h, with the cache it belongs to. Named here because
 * this is the file whose callers must obey the rule above. */

/* ── Live-structure test hooks ───────────────────────────────── */

/*
 * `beam_live_selfcheck` returns 0 when every claim the frontier makes
 * about one length holds, and otherwise the first one that does not.
 *
 * These are not diagnostics for callers; they exist because the invariants
 * they check are INVISIBLE in the published answers. A wrong residue bit on a
 * state that happens to carry no mass, or an index list that is a permutation
 * rather than an ascending run, changes nothing today and changes a Hill
 * number tomorrow.
 */
#define BEAM_LIVE_SELFCHECK_BUILD        1  /* the frontier would not build */
#define BEAM_LIVE_SELFCHECK_ROOT         2  /* layer 0 is not just the root */
#define BEAM_LIVE_SELFCHECK_REACH_ORDER  3  /* `reached` not strictly ascending */
#define BEAM_LIVE_SELFCHECK_LIVE_ORDER   4  /* `live` not strictly ascending */
#define BEAM_LIVE_SELFCHECK_MASK         5  /* mask and `live` disagree */
#define BEAM_LIVE_SELFCHECK_SUBSET       6  /* a live state was never reached */
#define BEAM_LIVE_SELFCHECK_BITS         7  /* an edge bit is not the weight */

int beam_live_selfcheck(const BeamModel *m, uint32_t len);

/** Bytes the live cache currently holds; 0 when it holds nothing. */

/** Lengths the live cache currently holds. */

/* One bit per state. `static inline` because live_bit_get is a bit test in
 * the inner loop of the path-count and topology sweeps, called once per
 * (state, layer); an external call there would be pure overhead. */
static inline void live_bit_set(uint64_t *b, uint32_t i) {
    b[i >> 6] |= 1ull << (i & 63u);
}

static inline uint32_t live_bit_get(const uint64_t *b, uint32_t i) {
    return (uint32_t)((b[i >> 6] >> (i & 63u)) & 1ull);
}

/**
 * Acquire the live/reached frontier for one length, from the cache when it
 * is there. Release with `beam_live_release`.
 */
typedef struct {
    uint64_t **mask;       /* [len+1][BITWORDS(S)+1]  live, one bit per state */
    uint32_t **live;       /* [len+1][n_live[i]]      ascending             */
    uint32_t  *n_live;
    uint32_t **reached;    /* [len+1][n_reached[i]]   ascending             */
    uint32_t  *n_reached;
    uint32_t **edge_bits;  /* [len][n_reached[i]]     parallel to reached   */
    uint32_t   len;
    uint32_t   n_states;   /* so a cached set can size its own bitsets      */
} BeamLive;

typedef struct {
    const BeamLive *L;
    BeamLive        scratch;   /* holds the set when it was not cached */
    bool            cached;
} LiveHandle;

BeamError beam_live_acquire(const BeamModel *m, uint32_t len,
                            LiveHandle *h);
void beam_live_release(LiveHandle *h);

/** Producible lengths, longest first. Returns how many were written. */
uint32_t beam_live_lengths_desc(const BeamModel *m, uint32_t *out);

/**
 * Byte mask of the states reached at layer `i`, for `beam_layer_weights_live`.
 *
 * The frontier stores membership as a bitset plus an ascending index list;
 * the layer-weight builder wants one byte per state. Rebuilding it per layer
 * costs a memset over the state space, which is far less than building
 * weights for states no path reaches.
 *
 * With `out_max = NULL`, only callers that never read an unmarked row may use
 * this mask. With `out_max != NULL`, `beam_layer_weights_live` still computes
 * the exact global maximum while materialising the marked rows.
 */
static inline void beam_reached_mask(const BeamLive *L, uint32_t i,
                                     uint32_t n_states, uint8_t *alive) {
    memset(alive, 0, n_states);
    const uint32_t  n = L->n_reached[i];
    const uint32_t *v = L->reached[i];
    for (uint32_t k = 0; k < n; ++k) alive[v[k]] = 1;
}

#endif /* BEAM_INTERNAL_LIVE_FRONTIER_H */
