/**
 * @file internal.h
 * @brief Shared layout and helpers for the BEAM implementation.
 *
 * Not part of the public API. Do not include from outside lib/edge_model/.
 *
 * ## Table layout, and why it is what it is
 *
 * Every table has the following LOGICAL indexing, packed
 * most-distant-context-first and matching the reference implementation exactly
 * so dense and sparse models can be compared cell by cell. The open-world
 * backend is a flat `double` array. The zero-backoff strict backend keeps the
 * same row index in a dense uint32 directory and stores only occupied
 * 20-double rows. For a context of order `k` at depth `d`, scoring residue `x`:
 *
 *     wide : [ min(d, depth_cap) ][ c_k ]...[ c_1 ][ x ]
 *     mid  : [ d                 ][ c_3 ][ c_2 ][ c_1 ][ x ]
 *     len  : [ L ][ d            ][ c_2 ][ c_1 ][ x ]
 *
 * where `c_1` is the context residue NEAREST the scored one and `c_k` the
 * furthest. Context slots that fall outside the sequence hold
 * `BEAM_NONE`.
 *
 * ONE table breaks that rule: the finalized C-anchored WIDE table is packed
 * nearest-first, `[ min(d, depth_cap) ][ c_1 ]...[ c_k ][ x ]`. That is a
 * performance decision, it is confined to the finalized table (the counts and
 * the file format keep the common packing), and the reasoning is set out in
 * full at `beam_ctx_backward_wide` below. Anything reading
 * `lp_wide[BEAM_BWD]` must index it through one of the two helpers there.
 *
 * ## The two directions read opposite ways, and the packing follows
 *
 * For the forward (N-anchored) direction at position `p`, `c_j = s[p-j]`, so
 * the context is exactly the DAG state read oldest-first and a flat context
 * index is a suffix of the state index. For the backward (C-anchored)
 * direction, `c_j = s[p+j]`, so the digits come out of the state in the
 * REVERSE order. `beam_ctx_backward_from_state` is the only place that
 * asymmetry lives; getting it backwards is the single most likely way to
 * produce a model that looks plausible and is wrong, so it is isolated and
 * unit-tested directly by `test_backward_context_from_state_matches_the_sequence`
 * in tests/c_unit/test_model.c — against hand-computed indices, against
 * the sequence form of the same context, and against the forward reading of
 * the same state, which it must NOT equal. Reading the state forwards leaves
 * the partition function normalising perfectly, so nothing else catches it.
 */
#ifndef BEAM_MODEL_INTERNAL_H
#define BEAM_MODEL_INTERNAL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/model.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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

/* ── Sparse 20-value row storage ────────────────────────────── */

#define BEAM_ROW_MISSING UINT32_MAX

/**
 * Dense row directory with sparse 20-double payloads.
 *
 * Count tables are partitioned by the depth task that owns their rows. That
 * lets the existing parallel observation loop grow each payload without a
 * lock: one task is the sole writer of one partition. Finalized tables have a
 * single payload whose rows are emitted in dense-index order, preserving the
 * sequential access pattern used by the dynamic programs.
 */
typedef struct {
    uint32_t  *index;       /* dense row -> slot within its owning partition */
    double   **values;      /* [n_parts][capacity[p] * 20]                   */
    uint32_t **row_ids;     /* slot -> dense row, for every table.  Finalized
                             * tables fill ascending so theirs is sorted;
                             * mutable count tables fill in observation order
                             * and theirs is not. */
    uint32_t  *size;        /* occupied rows per partition                   */
    uint32_t  *capacity;    /* allocated rows per partition                  */
    uint32_t   n_dense_rows;
    uint32_t   n_parts;
    bool       mutable_rows;
} BeamSparseRows;

BeamError beam_sparse_init(BeamSparseRows *t, uint32_t n_dense_rows,
                              uint32_t n_parts, bool mutable_rows);
void beam_sparse_destroy(BeamSparseRows *t);
const uint32_t *beam_sparse_rows(const BeamSparseRows *t, uint32_t part,
                                     uint32_t *n);
const double *beam_sparse_row(const BeamSparseRows *t, uint32_t row,
                                  uint32_t part);
double *beam_sparse_row_mut(BeamSparseRows *t, uint32_t row,
                                uint32_t part, bool create);
bool beam_sparse_erase(BeamSparseRows *t, uint32_t row, uint32_t part);
void beam_sparse_shrink(BeamSparseRows *t);
uint64_t beam_sparse_memory_bytes(const BeamSparseRows *t);
uint64_t beam_sparse_n_rows(const BeamSparseRows *t);

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
     * assumed: `to_log_inplace` reports whether it had to floor anything.
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

/* ── Forward/live frontiers ─────────────────────────────── */

/**
 * Per-length live structure of the trellis, cached on the model.
 *
 * Opaque here on purpose: `analytics.c` owns both the layout and the
 * construction, and nothing outside it may reach into a cached entry. What
 * the rest of the implementation needs to know is the lifetime rule, which is
 * absolute:
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
void beam_live_cache_drop(BeamModel *m);

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
uint64_t beam_live_cache_bytes(const BeamModel *m);

/** Lengths the live cache currently holds. */
uint32_t beam_live_cache_entries(const BeamModel *m);

/* ── Diagnosable failures ────────────────────────────────────── */

/*
 * Every BEAM entry point returns an error CODE, and a bare code reaches Python
 * as "invalid argument" with nothing to act on. These attach the message that
 * `beam_error_message()` reports, so a caller learns which argument, which
 * length, or which table ran out.
 *
 * `__func__` names the function that refused, which for a static helper is the
 * helper: that is the more useful half of the answer, since the public entry
 * point is already obvious from the call site.
 */
#define BEAM_ENULL(what)                                                  \
    BEAM_FAIL(BEAM_ERR_NULL_ARG, "%s: %s must not be NULL", __func__, (what))

#define BEAM_ENOTBUILT()                                                  \
    BEAM_FAIL(BEAM_ERR_NOT_BUILT,                                               \
             "%s: the model is not finalized; observe sequences and call "    \
             "finalize before scoring or analysing it", __func__)

#define BEAM_ECOMPACTED(what)                                             \
    BEAM_FAIL(BEAM_ERR_NOT_BUILT,                                               \
             "%s: the model is compacted, so %s", __func__, (what))

#define BEAM_EALLOC(what)                                                 \
    BEAM_FAIL(BEAM_ERR_ALLOC, "%s: out of memory allocating %s", __func__, (what))

#define BEAM_EBADLEN(mdl, L)                                              \
    BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,                                      \
             "%s: length %u is not one this model generates (lengths 1..%u "  \
             "with a finite length prior are)", __func__, (unsigned)(L),      \
             (unsigned)(mdl)->max_len)

#define BEAM_ERANGE(...)                                                  \
    BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE, __VA_ARGS__)

#define BEAM_ENOPATHS(what)                                               \
    BEAM_FAIL(BEAM_ERR_NO_LIVE_PATHS, "%s: %s", __func__, (what))

/**
 * CRC-32C, hardware-accelerated where the running CPU offers it.
 *
 * Identical in value to `beam_crc32c_update`, which it falls back to; the only
 * difference is that a gigabyte of counts checksums in a fifth of the time.
 * Exposed here rather than kept static in io.c so the equivalence can be
 * tested directly.
 */
uint32_t beam_model_crc32c(uint32_t crc, const void *data, size_t len);

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

/* ── Alphabet ────────────────────────────────────────────────── */

/** ASCII → residue index, or BEAM_NONE for anything not an amino acid. */
extern const uint8_t beam_char_to_idx[256];

/** Residue index → ASCII. Index BEAM_NONE maps to '.'. */
extern const char beam_idx_to_char[BEAM_ALPHABET];

/**
 * Encode a NUL-terminated sequence into residue indices.
 * Returns false if the sequence is empty, longer than `max_len`, or contains
 * a character outside the 20 amino acids.
 */
bool beam_encode(const BeamModel *m, const char *seq,
                     uint8_t *out, uint32_t *out_len);

/* ── Context extraction ──────────────────────────────────────── */

/**
 * Forward context of order `k` for the residue at `pos`, packed
 * most-distant-first: `((c_k * A + c_(k-1)) * A + ...) * A + c_1` with
 * `c_j = s[pos - j]`, sentinel-padded below zero.
 */
static inline uint32_t beam_ctx_forward(const uint8_t *s, int32_t pos,
                                            uint32_t k) {
    uint32_t idx = 0;
    for (int32_t j = (int32_t)k; j >= 1; --j) {
        int32_t q = pos - j;
        uint32_t c = (q >= 0) ? s[q] : BEAM_NONE;
        idx = idx * BEAM_ALPHABET + c;
    }
    return idx;
}

/**
 * Backward context of order `k` for the residue at `pos`, packed
 * most-distant-first with `c_j = s[pos + j]`, sentinel-padded past `len`.
 */
static inline uint32_t beam_ctx_backward(const uint8_t *s, int32_t pos,
                                             uint32_t len, uint32_t k) {
    uint32_t idx = 0;
    for (int32_t j = (int32_t)k; j >= 1; --j) {
        int32_t q = pos + j;
        uint32_t c = (q < (int32_t)len) ? s[q] : BEAM_NONE;
        idx = idx * BEAM_ALPHABET + c;
    }
    return idx;
}

/**
 * Backward context of order `k` read out of a DAG state, for the residue the
 * state's OLDEST slot holds, given that `x` is the residue being emitted now.
 *
 * The state packs `(s[i-K], ..., s[i-1])` oldest-most-significant. The residue
 * being backward-scored is `s[i-K]`, and its right-context runs
 * `c_1 = s[i-K+1] ... c_(K-1) = s[i-1], c_K = x`. Because the backward table
 * is packed most-distant-first, the digits come out of the state in reverse:
 * `c_K = x` is most significant and `c_1` (the state's SECOND-oldest slot) is
 * least. This is the mirror image of the forward case, where a context index
 * is just a suffix of the state index.
 *
 * `k` may be less than the model order, in which case only the `k` residues
 * nearest the scored one are used, which for the backward direction means
 * dropping `x` first.
 */
static inline uint32_t beam_ctx_backward_from_state(uint32_t state,
                                                        uint32_t x,
                                                        uint32_t order,
                                                        uint32_t k) {
    /* Slots of the state, oldest first: slot[0] = s[i-K] is the scored one.
     *
     * `order` is validated to 2..BEAM_MAX_ORDER at construction, so
     * slot[order] below is the last valid element and never past it. gcc
     * cannot see that from this signature and warns about a 1-byte write into
     * a region of size 0; clamping here is free at runtime and lets the
     * compiler prove the bound instead of being told to ignore it. */
    uint8_t slot[BEAM_MAX_ORDER + 1];
    if (order > BEAM_MAX_ORDER) order = BEAM_MAX_ORDER;
    uint32_t q = state;
    for (int32_t t = (int32_t)order - 1; t >= 0; --t) {
        slot[t] = (uint8_t)(q % BEAM_ALPHABET);
        q /= BEAM_ALPHABET;
    }
    slot[order] = (uint8_t)x;      /* c_K, the furthest right-context residue */

    /* c_j = slot[j]; pack c_k .. c_1 most-distant-first. */
    uint32_t idx = 0;
    for (uint32_t j = k; j >= 1; --j) {
        idx = idx * BEAM_ALPHABET + slot[j];
    }
    return idx;
}

/* ── The backward WIDE table is packed nearest-first ─────────── */

/*
 * One table, and one table only, departs from the most-distant-first packing
 * described at the top of this file: the C-anchored wide table.
 *
 * The reason is memory, and it is the difference between an implementation
 * that is bandwidth-bound and one that is latency-bound. In the trellis the
 * backward context of the residue a state settles is `(s[i-K+1] .. s[i-1], x)`
 * — the state's low digits, in order, followed by the residue being emitted.
 * Packed nearest-first that is exactly `(state % state_shift) * A + x`, so
 * sweeping a layer reads the table front to back. Packed most-distant-first
 * the same sweep reads `x * A^(K-1) + reverse(state % state_shift)`, which
 * scatters 3.9 million lookups over half a gigabyte and spends its whole
 * budget on cache misses. At order 4 that one difference is most of the cost
 * of every dynamic program in the library.
 *
 * Nothing outside the finalized backward-wide probability table changes: the
 * COUNTS keep the common layout, so the smoothing hierarchy, `add`, and
 * `subtract` are untouched, and only the final write relocates each row. The
 * two helpers below are the only places the packing is spelled out; both must
 * agree, and
 * `test_backward_wide_context_matches_the_sequence` checks that they do.
 */

/** Reverse `order` base-A digits: most-distant-first index -> nearest-first. */
static inline uint32_t beam_ctx_reverse(uint32_t ctx, uint32_t order) {
    uint32_t out = 0;
    for (uint32_t j = 0; j < order; ++j) {
        out = out * BEAM_ALPHABET + (ctx % BEAM_ALPHABET);
        ctx /= BEAM_ALPHABET;
    }
    return out;
}

/**
 * Backward wide context read off a sequence, packed NEAREST-first:
 * `((c_1 * A + c_2) * A + ...) * A + c_order` with `c_j = s[pos + j]`.
 */
static inline uint32_t beam_ctx_backward_wide(const uint8_t *s, int32_t pos,
                                                  uint32_t len, uint32_t order) {
    uint32_t idx = 0;
    for (uint32_t j = 1; j <= order; ++j) {
        const int32_t q = pos + (int32_t)j;
        idx = idx * BEAM_ALPHABET
            + ((q < (int32_t)len) ? s[q] : BEAM_NONE);
    }
    return idx;
}

/**
 * The same context read out of a DAG state, for the residue in the state's
 * oldest slot, given that `x` is being emitted now.
 *
 * Nearest-first packing makes this arithmetic rather than a digit shuffle:
 * `t = state % state_shift` already holds `(c_1 .. c_(K-1))` in order, so a
 * full-width context is `t * A + x` and a narrower one drops trailing digits.
 */
static inline uint32_t beam_ctx_backward_wide_from_state(uint32_t state,
                                                             uint32_t x,
                                                             uint32_t state_shift,
                                                             uint32_t state_order,
                                                             uint32_t order) {
    const uint32_t t = state % state_shift;      /* c_1 .. c_(K-1), in order */
    if (order >= state_order) return t * BEAM_ALPHABET + x;
    uint32_t drop = state_order - 1u - order;    /* digits past c_order      */
    uint32_t idx = t;
    while (drop--) idx /= BEAM_ALPHABET;
    return idx;
}

/** The residue a DAG state's oldest slot holds. */
static inline uint32_t beam_state_oldest(uint32_t state,
                                             uint32_t state_shift) {
    return state / state_shift;
}

/* ── Table addressing ────────────────────────────────────────── */

static inline size_t beam_ix_wide(const BeamModel *m, uint32_t depth,
                                      uint32_t ctx, uint32_t x) {
    uint32_t d = depth < m->cfg.depth_cap ? depth : m->cfg.depth_cap;
    return ((size_t)d * m->n_ctx_wide + ctx) * BEAM_RESIDUES + x;
}
static inline size_t beam_ix_mid(const BeamModel *m, uint32_t depth,
                                     uint32_t ctx, uint32_t x) {
    return ((size_t)depth * m->n_ctx_mid + ctx) * BEAM_RESIDUES + x;
}
static inline size_t beam_ix_len(const BeamModel *m, uint32_t len,
                                     uint32_t depth, uint32_t ctx, uint32_t x) {
    return (((size_t)len * m->max_len + depth) * m->n_ctx_len + ctx)
           * BEAM_RESIDUES + x;
}

/* ── Dense/sparse row access ────────────────────────────────── */

static inline uint32_t beam_depth_part(const BeamModel *m,
                                           uint32_t depth) {
    return depth < m->cfg.depth_cap ? depth : m->cfg.depth_cap;
}

static inline uint32_t beam_wide_part_for_row(const BeamModel *m,
                                                   uint32_t row) {
    return row / m->n_ctx_wide;
}

static inline uint32_t beam_mid_part_for_row(const BeamModel *m,
                                                  uint32_t row) {
    return beam_depth_part(m, row / m->n_ctx_mid);
}

static inline uint32_t beam_len_part_for_row(const BeamModel *m,
                                                  uint32_t row) {
    const uint32_t depth = (row / m->n_ctx_len) % m->max_len;
    return beam_depth_part(m, depth);
}

static inline const double *beam_cnt_wide_row(const BeamModel *m,
                                                   uint32_t dir,
                                                   uint32_t row) {
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_cnt_wide[dir], row,
                              beam_wide_part_for_row(m, row))
        : m->cnt_wide[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_cnt_mid_row(const BeamModel *m,
                                                  uint32_t dir,
                                                  uint32_t row) {
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_cnt_mid[dir], row,
                              beam_mid_part_for_row(m, row))
        : m->cnt_mid[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_cnt_len_row(const BeamModel *m,
                                                  uint32_t dir,
                                                  uint32_t row) {
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_cnt_len[dir], row,
                              beam_len_part_for_row(m, row))
        : m->cnt_len[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline double *beam_cnt_wide_row_mut(BeamModel *m, uint32_t dir,
                                                 uint32_t row, bool create) {
    return m->cfg.sparse
        ? beam_sparse_row_mut(&m->sp_cnt_wide[dir], row,
                                  beam_wide_part_for_row(m, row), create)
        : m->cnt_wide[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline double *beam_cnt_mid_row_mut(BeamModel *m, uint32_t dir,
                                                uint32_t row, bool create) {
    return m->cfg.sparse
        ? beam_sparse_row_mut(&m->sp_cnt_mid[dir], row,
                                  beam_mid_part_for_row(m, row), create)
        : m->cnt_mid[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline double *beam_cnt_len_row_mut(BeamModel *m, uint32_t dir,
                                                uint32_t row, bool create) {
    return m->cfg.sparse
        ? beam_sparse_row_mut(&m->sp_cnt_len[dir], row,
                                  beam_len_part_for_row(m, row), create)
        : m->cnt_len[dir] + (size_t)row * BEAM_RESIDUES;
}

typedef enum {
    BEAM_TABLE_WIDE = 0,
    BEAM_TABLE_MID  = 1,
    BEAM_TABLE_LEN  = 2
} BeamTableKind;

static inline uint32_t beam_count_n_rows(const BeamModel *m,
                                              BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE)
        return m->n_depth_wide * m->n_ctx_wide;
    if (kind == BEAM_TABLE_MID)
        return m->max_len * m->n_ctx_mid;
    return m->n_len_slots * m->max_len * m->n_ctx_len;
}

static inline uint32_t beam_count_part_for_row(const BeamModel *m,
                                                    BeamTableKind kind,
                                                    uint32_t row) {
    if (kind == BEAM_TABLE_WIDE) return beam_wide_part_for_row(m, row);
    if (kind == BEAM_TABLE_MID) return beam_mid_part_for_row(m, row);
    return beam_len_part_for_row(m, row);
}

static inline BeamSparseRows *beam_sparse_count_table(
        BeamModel *m, uint32_t dir, BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->sp_cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return &m->sp_cnt_mid[dir];
    return &m->sp_cnt_len[dir];
}

static inline const BeamSparseRows *beam_sparse_count_table_const(
        const BeamModel *m, uint32_t dir, BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->sp_cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return &m->sp_cnt_mid[dir];
    return &m->sp_cnt_len[dir];
}

static inline double *beam_dense_count_table(BeamModel *m,
                                                  uint32_t dir,
                                                  BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return m->cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return m->cnt_mid[dir];
    return m->cnt_len[dir];
}

static inline const double *beam_dense_count_table_const(
        const BeamModel *m, uint32_t dir, BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return m->cnt_wide[dir];
    if (kind == BEAM_TABLE_MID) return m->cnt_mid[dir];
    return m->cnt_len[dir];
}

static inline const double *beam_count_row(const BeamModel *m,
                                                uint32_t dir,
                                                BeamTableKind kind,
                                                uint32_t row) {
    if (!m->cfg.sparse)
        return beam_dense_count_table_const(m, dir, kind)
             + (size_t)row * BEAM_RESIDUES;
    return beam_sparse_row(beam_sparse_count_table_const(m, dir, kind),
                               row, beam_count_part_for_row(m, kind, row));
}

static inline double *beam_count_row_mut(BeamModel *m, uint32_t dir,
                                              BeamTableKind kind,
                                              uint32_t row, bool create) {
    if (!m->cfg.sparse)
        return beam_dense_count_table(m, dir, kind)
             + (size_t)row * BEAM_RESIDUES;
    return beam_sparse_row_mut(beam_sparse_count_table(m, dir, kind),
                                   row,
                                   beam_count_part_for_row(m, kind, row),
                                   create);
}

static inline bool beam_count_erase(BeamModel *m, uint32_t dir,
                                         BeamTableKind kind, uint32_t row) {
    if (!m->cfg.sparse) return false;
    return beam_sparse_erase(beam_sparse_count_table(m, dir, kind),
                                 row,
                                 beam_count_part_for_row(m, kind, row));
}

static inline const double *beam_lp_wide_row(const BeamModel *m,
                                                  uint32_t dir,
                                                  uint32_t depth,
                                                  uint32_t ctx) {
    const uint32_t d = beam_depth_part(m, depth);
    const uint32_t row = d * m->n_ctx_wide + ctx;
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_lp_wide[dir], row, 0u)
        : m->lp_wide[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_lp_mid_row(const BeamModel *m,
                                                 uint32_t dir,
                                                 uint32_t depth,
                                                 uint32_t ctx) {
    const uint32_t row = depth * m->n_ctx_mid + ctx;
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_lp_mid[dir], row, 0u)
        : m->lp_mid[dir] + (size_t)row * BEAM_RESIDUES;
}

static inline const double *beam_lp_len_row(const BeamModel *m,
                                                 uint32_t dir, uint32_t len,
                                                 uint32_t depth,
                                                 uint32_t ctx) {
    const uint32_t row = (len * m->max_len + depth) * m->n_ctx_len + ctx;
    return m->cfg.sparse
        ? beam_sparse_row(&m->sp_lp_len[dir], row, 0u)
        : m->lp_len[dir] + (size_t)row * BEAM_RESIDUES;
}

/* ── Cross-file helpers ──────────────────────────────────────── */

/**
 * Mixed log-weight of one residue in one direction, before the direction
 * weight is applied. `ctx_wide`, `ctx_mid` and `ctx_len` are pre-packed
 * context indices of order `order`, 3 and 2 respectively.
 */
double beam_mix(const BeamModel *m, uint32_t dir, uint32_t len,
                    uint32_t depth, uint32_t ctx_wide, uint32_t ctx_mid,
                    uint32_t ctx_len, uint32_t x);

/**
 * Edge weights of one DAG layer: `out[state * 20 + x]`, the full weight of
 * emitting `x` from `state` at position `i` of a length-`len` sequence.
 * Includes the forward term for the residue being emitted and the backward
 * term for the residue emitted `order` steps ago, which only now has its
 * complete right-hand context.
 *
 * `out` must hold `n_states * 20` doubles.
 *
 * `out_max`, if given, receives the largest weight in the layer. Every sweep
 * needs it to rescale, and computing it here costs nothing while the values
 * are still in register: computing it afterwards means a second pass over 31
 * MB per layer, which measured at a fifth of the sweep. A maximum is
 * order-independent, so the fused value is bit-identical to the separate one.
 * `-INFINITY` means the layer has no live edge at all.
 */
/** The layer's maximum edge weight, computed without building the layer. */
double beam_layer_max(const BeamModel *m, uint32_t i, uint32_t len);

void beam_layer_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            double *out, double *out_max);

/**
 * As above, but skipping states `alive` marks dead and filling their rows with
 * -inf. See the definition in score.c for when this is safe: only where the
 * caller provably never reads a dead row, which excludes every caller that
 * asks for `out_max`.
 */
void beam_layer_weights_live(const BeamModel *m, uint32_t i, uint32_t len,
                                 double *out, double *out_max,
                                 const uint8_t *alive);

/**
 * Weights of the 20 edges leaving ONE state, identical to the corresponding
 * row of `beam_layer_weights`.
 *
 * For a consumer that visits states in an unpredictable order — the
 * best-first search in topk.c is the only one — building the whole layer to
 * read one row of it is 3.9 million wasted edges per visit.
 */
void beam_state_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            uint32_t state, double *out);

/**
 * Sink weights: `out[state]`, settling the backward terms of the final
 * `min(order, len)` residues, which never get a full right-hand context
 * during the walk. `out` must hold `n_states` doubles.
 */
void beam_tail_weights(const BeamModel *m, uint32_t len, double *out);

/** Compute `log_z[len]`. Returns BEAM_OK or an allocation error. */
BeamError beam_compute_log_z(BeamModel *m, uint32_t len, double *out);

/**
 * Backward values of the trellis: `beta[i][state]`, the log of the total
 * weight of every completion from `state` at layer `i`. `beta` must be an
 * array of `len + 1` pointers, each to `n_states` doubles.
 */
BeamError beam_betas(const BeamModel *m, uint32_t len, double **beta);

/**
 * Backward values of the `q`-TILTED trellis, i.e. with every edge weight
 * scaled by `q` before exponentiating. `q = 1` reproduces `beam_betas`.
 *
 * Needed wherever a posterior under a tilted law is required, such as
 * attributing surprisal to depths at `q != 1`.
 */
BeamError beam_betas_tilted(const BeamModel *m, uint32_t len, double q,
                               double **beta);

/**
 * Lengths the model actually generates.
 *
 * `finalize` zeroes the prior of any length whose trellis has no surviving
 * path, so a finite prior implies a finite partition function and this single
 * test is sufficient. Before that renormalisation the two could disagree, and
 * a length with prior mass but no support would poison every mean it entered.
 */
static inline bool beam_length_live(const BeamModel *m, uint32_t len) {
    return len >= 1 && len <= m->max_len && isfinite(m->log_pi[len]);
}

#endif /* BEAM_MODEL_INTERNAL_H */
