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
#include "beam/alphabet.h"
#include "beam/sparse_rows.h"
#include "beam/frozen_counts.h"


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
 * The mmap threshold BEAM applied at start-up, or 0 if it applied none --
 * because the caller had already set `MALLOC_MMAP_THRESHOLD_`, because
 * `BEAM_MMAP_THRESHOLD` is 0, or because this is not glibc. See
 * `memory/allocator.c` for what it buys and what it costs.
 */

/*
 * Here rather than in `error_macros.h`, which is where it was: it is the one
 * refusal macro that dereferences a model, so it belongs with the struct it
 * reads rather than dragging that struct into every file that wants
 * `BEAM_ENULL`.
 */
#define BEAM_EBADLEN(mdl, L)                                              \
    BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,                                      \
             "%s: length %u is not one this model generates (lengths 1..%u "  \
             "with a finite length prior are)", __func__, (unsigned)(L),      \
             (unsigned)(mdl)->max_len)

#define BEAM_DIRS   2u
#define BEAM_FWD    0u   /* N-anchored: indexes by distance from the front */
#define BEAM_BWD    1u   /* C-anchored: indexes by distance from the back  */

/** Floor applied before taking a log, matching the reference implementation. */
#define BEAM_PROB_FLOOR 1e-300

typedef struct BeamLiveCache BeamLiveCache;

struct BeamModel_ {
    BeamModelConfig cfg;

    /**
     * The alphabet, resolved once at construction.
     *
     * `alpha` is the descriptor; the three counts beside it are the only
     * parts of it an inner loop needs, held as `uint32_t` so a kernel reads
     * one word rather than chasing a pointer and widening a `uint8_t`. They
     * are exactly `alpha->residues`, `alpha->radix` and `alpha->none`.
     */
    const BeamAlphabetInfo *alpha;
    uint32_t n_residues;     /* R: emitted symbols, the stride of every row  */
    uint32_t n_alphabet;     /* A: R + 1, the context-packing radix          */
    uint32_t none_idx;       /* the context sentinel, == n_residues          */

    /* Derived dimensions, all fixed at construction. */
    uint32_t order;          /* k, the context order of both tables           */
    /**
     * Context order of the DAG state. Identical to `order`.
     *
     * The field stays because the trellis reads more clearly for it: `order`
     * is a statement about the tables and `state_order` about the graph, and
     * every window slide, settle lag and node count below is a property of
     * the graph. They were different while a second expert of fixed order 3
     * existed, because a state holding fewer than 3 residues could not supply
     * that expert's context. With one table per direction there is nothing to
     * be wider than, so a state holds exactly the `k` residues its context
     * needs.
     */
    uint32_t state_order;
    uint32_t n_states;       /* A^state_order, nodes per DAG layer            */
    uint32_t state_shift;    /* A^(state_order-1), slides the state window    */
    uint32_t n_ctx;          /* A^order, the context count of both tables     */
    uint32_t max_len;        /* W, and the depth axis: depth is in [0, W)     */
    uint32_t n_len_slots;    /* W + 1, so index [L] is valid for L = max_len  */

    /*
     * Provenance: the three facts a finished model cannot recompute about
     * itself. See internal/beam/provenance.h for why the clocks differ.
     *
     * `observe_nanos` aggregates across every call, so a model built by
     * streaming a hundred chunks reports what the whole build cost rather
     * than what the last chunk did. `finalize_nanos` does the same across
     * re-finalizes, which is the honest reading: re-finalizing is work the
     * model has had done to it.
     */
    int64_t  created_unix;
    int64_t  observe_nanos;
    int64_t  finalize_nanos;

    /* Raw abundance counts. NULL after beam_model_compact(). */
    double *cnt[BEAM_DIRS];
    double *cnt_length;      /* [n_len_slots] observed length distribution    */

    /* Finalized natural-log probability tables. NULL until finalize. */
    double *lp[BEAM_DIRS];

    /* Sparse counterparts. Populated only when cfg.sparse is true. */
    BeamSparseRows sp_cnt[BEAM_DIRS];
    BeamSparseRows sp_lp[BEAM_DIRS];

    /**
     * Counts after observation stopped, or empty structs when they are still
     * mutable. Exactly one of these and `sp_cnt_*` holds the counts at any
     * moment; `counts_frozen` says which.
     *
     * Finalize freezes, and every operation that needs to write counts again
     * -- observe, add, subtract, posterior, re-finalize -- thaws first. Save
     * reads the frozen form directly rather than thawing, because walking
     * the sorted rows is both cheaper than a thaw and cheaper than the dense
     * probe it replaces.
     */
    BeamFrozenCounts frz[BEAM_DIRS];
    bool     counts_frozen;

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
     * What it buys is the live-state trim (`trim_frontier` in `dp/liveness.c`). With no
     * dead cell there is no dead edge, so liveness is a property of the
     * trellis rather than of the weights and the whole pass collapses to a
     * reachability sweep — no edge weights, no table reads. That halves
     * `path_count`, `topology` and the spectrum, each of which otherwise
     * builds every layer's weights twice.
     */
    bool     tables_all_finite;

    /**
     * Per-length live trellis structure, built on demand by `dp/liveness.c`.
     *
     * Mutable through a `const BeamModel *`, because building it is what
     * the read-only dynamic programs do and it changes no answer they give.
     * Bounded by a quarter of `beam_model_memory_budget()`; past that the
     * live sets are built per call and discarded, which is what the code did
     * before the cache existed.
     */
    BeamLiveCache *live_cache;
};

/* ── Element count of one direction's table ──────────────────── */

/*
 * One row per (depth, context), one cell per residue. The depth axis is
 * `max_len` because a forward depth is at most `L-1` and a backward depth is
 * at most `L-1`, and `L <= max_len`.
 */
static inline size_t beam_n_table(const BeamModel *m) {
    return (size_t)m->max_len * m->n_ctx * m->n_residues;
}

#endif /* BEAM_INTERNAL_MODEL_STATE_H */
