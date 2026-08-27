/**
 * @file model_state.h
 * @brief Parallelism, fixed directions, and shared constants; The BeamModel_ layout; Element counts of each table.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_MODEL_STATE_H
#define BEAM_INTERNAL_MODEL_STATE_H

#include "beam/model.h"
#include "beam/sparse_rows.h"


/* ── Parallelism ─────────────────────────────────────────────── */

/**
 * Threads the next parallel region should use.
 *
 * Every `omp parallel` in the implementation carries `num_threads(...)` with
 * this, so `beam_model_set_threads` is a hard cap rather than a hint and
 * one stray region cannot go wide behind the caller's back.
 */
int beam_threads(void);

/**
 * Chunks a parallel SUM is split into, fixed and independent of the thread
 * count.
 *
 * A sum is the one reduction whose answer depends on the order it is taken
 * in, so `reduction(+:)` would make a result depend on how many cores
 * happened to be free. Splitting the range into a fixed number of chunks,
 * summing each serially and then adding the chunk totals in index order gives
 * one answer for every thread count — the property the tests pin — at the
 * cost of differing from a single serial sweep in the last bit or two.
 */
#define BEAM_SUM_CHUNKS 256u

#define BEAM_DIRS   2u
#define BEAM_FWD    0u   /* N-anchored: indexes by distance from the front */
#define BEAM_BWD    1u   /* C-anchored: indexes by distance from the back  */

/** Floor applied before taking a log, matching the reference implementation. */
#define BEAM_PROB_FLOOR 1e-300

typedef struct BeamLiveCache BeamLiveCache;

struct BeamModel_ {
    BeamModelConfig cfg;

    /* Derived dimensions, all fixed at construction. */
    uint32_t order;          /* the WIDE expert's context order               */
    /**
     * Context order of the DAG state, `max(order, 3)`.
     *
     * Not the same as `order`, and conflating them is a real bug rather than a
     * tidiness issue: the mid expert is always order 3, so a state holding
     * fewer than 3 residues cannot supply its context and the order-3 index
     * would silently read a neighbouring cell instead. Everything about the
     * trellis — node count, window slide, which residue the backward expert
     * settles, the lag before it can settle anything — is governed by this,
     * while `order` governs only the wide table's own width.
     */
    uint32_t state_order;
    uint32_t n_states;       /* A^state_order, nodes per DAG layer            */
    uint32_t state_shift;    /* A^(state_order-1), slides the state window    */
    uint32_t n_ctx_wide;     /* A^order, the wide table's own context count   */
    uint32_t n_ctx_mid;      /* A^3                                           */
    uint32_t n_ctx_len;      /* A^2                                           */
    uint32_t n_depth_wide;   /* depth_cap + 1                                 */
    uint32_t max_len;        /* W                                             */
    uint32_t n_len_slots;    /* W + 1, so index [L] is valid for L = max_len  */

    /* Raw abundance counts. NULL after beam_model_compact(). */
    double *cnt_wide[BEAM_DIRS];
    double *cnt_mid[BEAM_DIRS];
    double *cnt_len[BEAM_DIRS];
    double *cnt_length;      /* [n_len_slots] observed length distribution    */

    /* Finalized natural-log probability tables. NULL until finalize. */
    double *lp_wide[BEAM_DIRS];
    double *lp_mid[BEAM_DIRS];
    double *lp_len[BEAM_DIRS];

    /* Sparse counterparts. Populated only when cfg.sparse is true. */
    BeamSparseRows sp_cnt_wide[BEAM_DIRS];
    BeamSparseRows sp_cnt_mid[BEAM_DIRS];
    BeamSparseRows sp_cnt_len[BEAM_DIRS];
    BeamSparseRows sp_lp_wide[BEAM_DIRS];
    BeamSparseRows sp_lp_mid[BEAM_DIRS];
    BeamSparseRows sp_lp_len[BEAM_DIRS];

    double *log_pi;          /* [n_len_slots] ln pi(L)                        */
    double *log_z;           /* [n_len_slots] ln Z(L)                         */

    double   total_weight;
    uint64_t n_observed;
    uint64_t n_skipped;
    bool     finalized;
    bool     compacted;

    /**
     * True when no cell of any finalized table is -inf.
     *
     * Hierarchical backoff leaves every cell positive unless a kappa is zero,
     * so this holds on every model the defaults produce. It is measured, not
     * assumed: `beam_to_log_inplace` reports whether it had to floor anything.
     *
     * What it buys is the live-state trim (`trim` in analytics.c). With no
     * dead cell there is no dead edge, so liveness is a property of the
     * trellis rather than of the weights and the whole pass collapses to a
     * reachability sweep — no edge weights, no table reads. That halves
     * `path_count`, `topology` and the spectrum, each of which otherwise
     * builds every layer's weights twice.
     */
    bool     tables_all_finite;

    /**
     * Per-length live trellis structure, built on demand by `analytics.c`.
     *
     * Mutable through a `const BeamModel *`, because building it is what
     * the read-only dynamic programs do and it changes no answer they give.
     * Bounded by a quarter of `beam_model_memory_budget()`; past that the
     * live sets are built per call and discarded, which is what the code did
     * before the cache existed.
     */
    BeamLiveCache *live_cache;
};

/* ── Element counts of each table ────────────────────────────── */

static inline size_t beam_n_wide(const BeamModel *m) {
    return (size_t)m->n_depth_wide * m->n_ctx_wide * BEAM_RESIDUES;
}
static inline size_t beam_n_mid(const BeamModel *m) {
    return (size_t)m->max_len * m->n_ctx_mid * BEAM_RESIDUES;
}
static inline size_t beam_n_len(const BeamModel *m) {
    return (size_t)m->n_len_slots * m->max_len * m->n_ctx_len * BEAM_RESIDUES;
}

#endif /* BEAM_INTERNAL_MODEL_STATE_H */
