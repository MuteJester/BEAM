/**
 * @file indexing.h
 * @brief Context extraction, reversal, and table addressing.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_INDEXING_H
#define BEAM_INTERNAL_INDEXING_H

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/alphabet.h"

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

#endif /* BEAM_INTERNAL_INDEXING_H */
