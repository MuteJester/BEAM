/**
 * @file score.c
 * @brief Sequence probability, DAG edge weights, and the partition function.
 *
 * Two ways of computing the same number live here, and that is deliberate.
 *
 *   - `beam_model_log_pseq` walks the sequence directly, one forward and
 *     one backward lookup per residue. O(L), no trellis. This is what callers
 *     use.
 *   - `beam_layer_weights` / `beam_tail_weights` build the DAG's edge
 *     weights, which the partition function, the spectrum, the path counts and
 *     the sampler all consume.
 *
 * They must agree exactly, and `test_layer_weights_reproduce_the_scorer` in
 * tests/c_unit/test_model.c asserts it by walking the trellis along a
 * known sequence and summing the edge weights. If they ever diverge, the
 * analytics are describing a different model from the one being scored, which
 * is the failure mode this file is arranged to make impossible to miss.
 */
#include "internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Mixed per-residue weight ────────────────────────────────── */

static inline double mix_values(const BeamModel *m, double a, double b,
                                double c) {
    /* Any dead expert kills the edge: a geometric product with a zero factor
     * is zero, and -inf * a positive coefficient must not become NaN. */
    if (a == -INFINITY || b == -INFINITY || c == -INFINITY) return -INFINITY;
    return m->cfg.coef_wide * a + m->cfg.coef_mid * b + m->cfg.coef_len * c;
}

static inline double mix_dense(const BeamModel *m, uint32_t dir,
                               uint32_t len, uint32_t depth,
                               uint32_t ctx_wide, uint32_t ctx_mid,
                               uint32_t ctx_len, uint32_t x) {
    const double a = m->lp_wide[dir][beam_ix_wide(m, depth, ctx_wide, x)];
    const double b = m->lp_mid[dir][beam_ix_mid(m, depth, ctx_mid, x)];
    const double c = m->lp_len[dir][beam_ix_len(m, len, depth, ctx_len, x)];
    return mix_values(m, a, b, c);
}

static inline double mix_sparse(const BeamModel *m, uint32_t dir,
                                uint32_t len, uint32_t depth,
                                uint32_t ctx_wide, uint32_t ctx_mid,
                                uint32_t ctx_len, uint32_t x) {
    const double *a = beam_lp_wide_row(m, dir, depth, ctx_wide);
    const double *b = beam_lp_mid_row(m, dir, depth, ctx_mid);
    const double *c = beam_lp_len_row(m, dir, len, depth, ctx_len);
    if (!a || !b || !c) return -INFINITY;
    return mix_values(m, a[x], b[x], c[x]);
}

double beam_mix(const BeamModel *m, uint32_t dir, uint32_t len,
                    uint32_t depth, uint32_t ctx_wide, uint32_t ctx_mid,
                    uint32_t ctx_len, uint32_t x) {
    return m->cfg.sparse
         ? mix_sparse(m, dir, len, depth, ctx_wide, ctx_mid, ctx_len, x)
         : mix_dense(m, dir, len, depth, ctx_wide, ctx_mid, ctx_len, x);
}

/* ── Direct sequence scoring ─────────────────────────────────── */

/**
 * Sum of both directions' contributions over every residue, before the length
 * prior and the partition function. This is the quantity the DAG's edge
 * weights plus its tail weight also sum to.
 */
static double raw_log_weight(const BeamModel *m, const uint8_t *s,
                             uint32_t len) {
    double total = 0.0;
    for (uint32_t p = 0; p < len; ++p) {
        const uint32_t x = s[p];

        const double f = beam_mix(
            m, BEAM_FWD, len, p,
            beam_ctx_forward(s, (int32_t)p, m->order),
            beam_ctx_forward(s, (int32_t)p, 3u),
            beam_ctx_forward(s, (int32_t)p, 2u), x);
        if (f == -INFINITY) return -INFINITY;

        const uint32_t r = len - 1u - p;
        const double b = beam_mix(
            m, BEAM_BWD, len, r,
            beam_ctx_backward_wide(s, (int32_t)p, len, m->order),
            beam_ctx_backward(s, (int32_t)p, len, 3u),
            beam_ctx_backward(s, (int32_t)p, len, 2u), x);
        if (b == -INFINITY) return -INFINITY;

        total += m->cfg.weight_forward * f + m->cfg.weight_backward * b;
    }
    return total;
}

double beam_model_log_pseq(const BeamModel *m, const char *seq) {
    if (!m || !m->finalized || !seq) return -INFINITY;
    uint8_t buf[BEAM_MAX_LENGTH];
    uint32_t len = 0;
    if (!beam_encode(m, seq, buf, &len)) return -INFINITY;
    if (!isfinite(m->log_pi[len]) || !isfinite(m->log_z[len])) return -INFINITY;

    const double raw = raw_log_weight(m, buf, len);
    if (raw == -INFINITY) return -INFINITY;
    return raw + m->log_pi[len] - m->log_z[len];
}

BeamError beam_model_log_pseq_batch(const BeamModel *m,
                                       const char *const *seqs,
                                       uint32_t n, double *out) {
    if (!m || !seqs || !out)
        return BEAM_ENULL(!m ? "model" : (!seqs ? "sequences" : "output buffer"));
    if (!m->finalized) return BEAM_ENOTBUILT();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        out[i] = beam_model_log_pseq(m, seqs[i]);
    }
    return BEAM_OK;
}

BeamError beam_model_explain(const BeamModel *m, const char *seq,
                                double *out_forward, double *out_backward,
                                uint32_t *out_len) {
    if (!m || !seq) return BEAM_ENULL(!m ? "model" : "sequence");
    if (!m->finalized) return BEAM_ENOTBUILT();
    uint8_t buf[BEAM_MAX_LENGTH];
    uint32_t len = 0;
    if (!beam_encode(m, seq, buf, &len))
        return BEAM_FAIL(BEAM_ERR_INVALID_SEQUENCE,
                        "%s: the sequence is empty, longer than max_length "
                        "(%u), or contains a character outside the 20 amino "
                        "acids", __func__, (unsigned)m->max_len);

    for (uint32_t p = 0; p < len; ++p) {
        const uint32_t x = buf[p];
        if (out_forward) {
            out_forward[p] = beam_mix(
                m, BEAM_FWD, len, p,
                beam_ctx_forward(buf, (int32_t)p, m->order),
                beam_ctx_forward(buf, (int32_t)p, 3u),
                beam_ctx_forward(buf, (int32_t)p, 2u), x);
        }
        if (out_backward) {
            const uint32_t r = len - 1u - p;
            out_backward[p] = beam_mix(
                m, BEAM_BWD, len, r,
                beam_ctx_backward_wide(buf, (int32_t)p, len, m->order),
                beam_ctx_backward(buf, (int32_t)p, len, 3u),
                beam_ctx_backward(buf, (int32_t)p, len, 2u), x);
        }
    }
    if (out_len) *out_len = len;
    return BEAM_OK;
}

/* ── DAG edge weights ────────────────────────────────────────── */

/**
 * Weights of every edge leaving layer `i` of the length-`len` trellis.
 *
 * A state packs the last `order` residues emitted, oldest-most-significant.
 * Emitting `x` from state `s` settles two scores:
 *
 *   - the FORWARD score of `x` itself, at depth `i`, whose context is exactly
 *     the state (that is why the forward context index and the state index
 *     coincide);
 *   - the BACKWARD score of the residue `order` steps back, which is the
 *     state's oldest slot, and which only now has its complete right-hand
 *     context. Its depth from the C terminus is `len - 1 - (i - order)`.
 *
 * Before `i` reaches `order` there is no residue old enough to settle, and the
 * last `order` residues never get one during the walk; `beam_tail_weights`
 * pays those at the sink.
 */
/**
 * One state's row, shared by the whole-layer sweep and the single-row entry
 * point so the two cannot drift apart.
 */
static inline void state_weights_dense(const BeamModel *m, uint32_t i,
                                       uint32_t len, uint32_t s,
                                       bool settle_back, uint32_t r,
                                       double *row) {
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t K  = m->state_order;
    const uint32_t n2 = m->n_ctx_len;
    const uint32_t n3 = m->n_ctx_mid;
    const uint32_t nw = m->n_ctx_wide;

    /* Forward: the context IS the state. A state whose newest slots hold
     * sentinels is only reachable in the first few layers, and the tables
     * carry sentinel-padded rows for exactly that case. Every forward context
     * is a suffix of the state, because the state is packed oldest-first and
     * a shorter context is simply the nearer residues. */
    for (uint32_t x = 0; x < R; ++x) {
        row[x] = m->cfg.weight_forward
               * mix_dense(m, BEAM_FWD, len, i,
                           s % nw, s % n3, s % n2, x);
    }
    if (!settle_back) return;

    const uint32_t scored = beam_state_oldest(s, m->state_shift);
    /* A sentinel in that slot means there is no residue to score.
     * Unreachable for i >= K, but cheap to be explicit about. */
    if (scored >= R) return;
    for (uint32_t x = 0; x < R; ++x) {
        if (row[x] == -INFINITY) continue;
        const double b = mix_dense(
            m, BEAM_BWD, len, r,
            beam_ctx_backward_wide_from_state(s, x, m->state_shift, K,
                                                  m->order),
            beam_ctx_backward_from_state(s, x, K, 3u),
            beam_ctx_backward_from_state(s, x, K, 2u), scored);
        row[x] = (b == -INFINITY) ? -INFINITY
                                  : row[x] + m->cfg.weight_backward * b;
    }
}

/** Sparse counterpart with the three forward directory lookups hoisted. */
static inline void state_weights_sparse(const BeamModel *m, uint32_t i,
                                        uint32_t len, uint32_t s,
                                        bool settle_back, uint32_t r,
                                        double *row) {
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t K  = m->state_order;
    const uint32_t n2 = m->n_ctx_len;
    const uint32_t n3 = m->n_ctx_mid;
    const uint32_t nw = m->n_ctx_wide;

    const double *fw = beam_lp_wide_row(m, BEAM_FWD, i, s % nw);
    const double *fm = beam_lp_mid_row(m, BEAM_FWD, i, s % n3);
    const double *fl = beam_lp_len_row(m, BEAM_FWD, len, i, s % n2);
    if (!fw || !fm || !fl) {
        for (uint32_t x = 0; x < R; ++x) row[x] = -INFINITY;
    } else {
        for (uint32_t x = 0; x < R; ++x)
            row[x] = m->cfg.weight_forward
                   * mix_values(m, fw[x], fm[x], fl[x]);
    }
    if (!settle_back) return;

    const uint32_t scored = beam_state_oldest(s, m->state_shift);
    if (scored >= R) return;

    /*
     * Forty of every sixty lookups in this loop were loop-invariant.
     *
     * The settled residue sits at p = i - K. Its order-3 and order-2 backward
     * contexts are s[p+1..p+3] and s[p+1..p+2], every one of them INSIDE the
     * DAG state; the residue x being emitted is s[p+4], outside it. So the mid
     * and length rows do not depend on x and were being refetched twenty times
     * with byte-identical arguments. Only the wide expert sees x, and because
     * the finalized C-anchored table is packed nearest-first, x lands in the
     * last context digit -- so its twenty rows are twenty CONSECUTIVE dense
     * indices starting at the x = 0 context.
     *
     * Verified exhaustively over all 194,481 states rather than argued: the
     * wide context is exactly cw(0) + x every time, and the mid and len
     * contexts are byte-identical across x. On the 927M foundation this loop
     * was 1.49e9 of 1.75e9 row lookups, and hoisting is worth 1.75x there.
     *
     * The invariance needs state_order >= 3 so the order-3 context fits inside
     * the state. That is enforced once at construction rather than tested
     * here -- this function runs tens of millions of times per analytic and a
     * branch that can never be taken still costs.
     */
    /*
     * Which of the three backward lookups depend on the residue x being
     * emitted is a function of the CONFIGURATION, not a constant.
     *
     * The settled residue sits at p = i - K, and its order-J backward context
     * is s[p+1 .. p+J] = s[i-K+1 .. i-K+J]. The emitted residue is s[i]. So x
     * is inside that context exactly when J >= K. With K = max(order, 3):
     *
     *   order 4  K=4:  wide J=4 >= K -> x is the last digit, cw(x) = cw(0)+x
     *                  mid  J=3 <  K -> INVARIANT, hoistable
     *                  len  J=2 <  K -> INVARIANT, hoistable
     *   order 3  K=3:  wide J=3 >= K -> x in context
     *                  mid  J=3 >= K -> x IN CONTEXT, NOT hoistable
     *   order 2  K=3:  wide J=2 <  K -> x NOT in the wide context at all,
     *                                   so cw(x) = cw(0) and "+x" is WRONG
     *
     * Only order 4 has both properties the hoist needs, and it is the default
     * and the only configuration the exhaustive 194,481-state verification
     * covered. Hoisting unconditionally made every partition function -inf at
     * order 2 -- the model could generate nothing -- which is what this gate
     * exists to prevent. Other orders take the original path; they are rare
     * and correctness is not negotiable for a speedup.
     */
    if (K > 3u && m->order == K) {
        const uint32_t cm = beam_ctx_backward_from_state(s, 0u, K, 3u);
        const uint32_t cl = beam_ctx_backward_from_state(s, 0u, K, 2u);
        const double *bm = beam_lp_mid_row(m, BEAM_BWD, r, cm);
        const double *bl = beam_lp_len_row(m, BEAM_BWD, len, r, cl);
        if (!bm || !bl) {
            for (uint32_t x = 0; x < R; ++x) row[x] = -INFINITY;
            return;
        }
        const uint32_t cw0 = beam_ctx_backward_wide_from_state(
            s, 0u, m->state_shift, K, m->order);
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY) continue;
            const double *bw = beam_lp_wide_row(m, BEAM_BWD, r, cw0 + x);
            const double b = !bw ? -INFINITY
                           : mix_values(m, bw[scored], bm[scored], bl[scored]);
            row[x] = (b == -INFINITY) ? -INFINITY
                                      : row[x] + m->cfg.weight_backward * b;
        }
        return;
    }

    for (uint32_t x = 0; x < R; ++x) {
        if (row[x] == -INFINITY) continue;
        const uint32_t cw = beam_ctx_backward_wide_from_state(
            s, x, m->state_shift, K, m->order);
        const uint32_t cm = beam_ctx_backward_from_state(s, x, K, 3u);
        const uint32_t cl = beam_ctx_backward_from_state(s, x, K, 2u);
        const double *bw = beam_lp_wide_row(m, BEAM_BWD, r, cw);
        const double *bm = beam_lp_mid_row(m, BEAM_BWD, r, cm);
        const double *bl = beam_lp_len_row(m, BEAM_BWD, len, r, cl);
        const double b = (!bw || !bm || !bl)
                       ? -INFINITY
                       : mix_values(m, bw[scored], bm[scored], bl[scored]);
        row[x] = (b == -INFINITY) ? -INFINITY
                                  : row[x] + m->cfg.weight_backward * b;
    }
}

void beam_state_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            uint32_t state, double *out) {
    const uint32_t K = m->state_order;
    const bool settle_back = (i >= K);
    const uint32_t r = settle_back ? (len - 1u - (i - K)) : 0u;
    if (m->cfg.sparse)
        state_weights_sparse(m, i, len, state, settle_back, r, out);
    else
        state_weights_dense(m, i, len, state, settle_back, r, out);
}

/**
 * The layer's maximum edge weight, without building the layer.
 *
 * Every caller that asks `beam_layer_weights` for `out_max` uses it as a
 * numerical rescale -- `compute_log_z`, `tilted_log_z`, `jet_sweep`,
 * `accumulator_max_at`. That is why those sweeps could not be restricted to
 * the live states: the maximum is taken over ALL states, and a state that
 * carries it need not be able to reach a sink. Measured on the 927M
 * foundation, 22 of 375 layers have exactly that, so a live-only maximum
 * changes the rescale and moves the last bits of every score.
 *
 * Restricting to states with an OCCUPIED forward wide row is different, and
 * is exact rather than lucky: `state_weights_sparse` returns -inf for a state
 * missing any forward row, and -inf can never carry a maximum. So the two
 * maxima agree by construction. Verified anyway at 116, 272 and 375 layers on
 * a synthetic model, a real donor and the foundation -- zero differences.
 *
 * About 1% of states carry forward rows on a per-repertoire model, so this
 * replaces a sweep over 194,481 states with one over a few thousand.
 *
 * Dense models fall through to the full scan: every cell is populated there,
 * so there is nothing to skip and the row map does not exist.
 */
double beam_layer_max(const BeamModel *m, uint32_t i, uint32_t len) {
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t K  = m->state_order;
    const uint32_t S  = m->n_states;
    const uint32_t nw = m->n_ctx_wide;
    const bool settle_back = (i >= K);
    const uint32_t r = settle_back ? (len - 1u - (i - K)) : 0u;

    if (!m->cfg.sparse) {
        double *w = (double *)malloc((size_t)S * R * sizeof(double));
        if (!w) return NAN;
        double hi = -INFINITY;
        beam_layer_weights(m, i, len, w, &hi);
        free(w);
        return hi;
    }

    uint32_t n_rows = 0;
    const uint32_t *rows =
        beam_sparse_rows(&m->sp_lp_wide[BEAM_FWD], 0u, &n_rows);
    if (!rows) return -INFINITY;

    /* Rows carry the depth bucket in their high digits; only this layer's
     * bucket can contribute. The map is ascending, so the bucket is a
     * contiguous run, but a scan is already proportional to the occupied
     * rows and costs nothing worth a binary search. */
    const uint32_t want = beam_depth_part(m, i);
    const uint32_t stride = S / nw;      /* 1 when order == state_order */

    double hi = -INFINITY;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:hi) \
            num_threads(beam_threads()) if(n_rows > 4096)
#endif
    for (int64_t k = 0; k < (int64_t)n_rows; ++k) {
        const uint32_t row = rows[k];
        if (row / nw != want) continue;
        const uint32_t ctx = row % nw;
        double buf[BEAM_RESIDUES];
        /* One wide context can back several states when the state carries
         * more residues than the wide expert's order. */
        for (uint32_t rep = 0; rep < stride; ++rep) {
            const uint32_t st = rep * nw + ctx;
            if (st >= S) break;
            /* These states came FROM the occupied row map, so the forward
             * wide row is present by construction; no probe needed here. */
            state_weights_sparse(m, i, len, st, settle_back, r, buf);
            for (uint32_t x = 0; x < R; ++x) if (buf[x] > hi) hi = buf[x];
        }
    }
    return hi;
}

void beam_layer_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            double *out, double *out_max) {
    beam_layer_weights_live(m, i, len, out, out_max, NULL);
}

/**
 * As `beam_layer_weights`, but skipping states `alive` marks dead.
 *
 * A dead state has no path to a sink, so no dynamic program can route through
 * it and none of them read its row. Computing it anyway is the dominant cost
 * of every per-repertoire analytic: on a 15,000-sequence donor 0.12% of the
 * trellis is live, so 99.88% of the six-lookup-per-edge work is thrown away.
 * `path_count` alone spent 17 of its 32 seconds there.
 *
 * Dead rows are left UNTOUCHED. Filling them with -inf was measured and is
 * slower than the work it saves: the buffer is 20 doubles per state, 31 MB at
 * order 4, so writing every dead row rewrites it once per layer. The caller
 * contract below is therefore not advisory -- a caller that reads a dead row
 * gets stale data from a previous layer, not a safe sentinel.
 *
 * PASS `alive` ONLY WHERE DEAD ROWS ARE PROVABLY UNREAD. The forward sweeps
 * qualify: their accumulator is non-zero only for states that were live
 * targets on the previous layer, so `cur[src] != 0` implies `alive[i][src]`.
 * Callers that ask for `out_max` do NOT qualify -- the fused maximum scans
 * every row, and excluding dead states would change the surprisal grid and
 * the tilted-sweep rescale, which is a change in results, not in speed.
 */
void beam_layer_weights_live(const BeamModel *m, uint32_t i, uint32_t len,
                                 double *out, double *out_max,
                                 const uint8_t *alive) {
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t A  = BEAM_ALPHABET;
    const uint32_t K  = m->state_order;
    const uint32_t SK = m->state_shift;

    const bool settle_back = (i >= K);
    const uint32_t r = settle_back ? (len - 1u - (i - K)) : 0u;
    double hi = -INFINITY;

    if (m->cfg.sparse) {
        const uint32_t nw_fwd = m->n_ctx_wide;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:hi) \
                num_threads(beam_threads())
#endif
        for (int64_t t64 = 0; t64 < (int64_t)SK; ++t64) {
            const uint32_t t = (uint32_t)t64;
            for (uint32_t oldest = 0; oldest < A; ++oldest) {
                const uint32_t s = oldest * SK + t;
                double *row = out + (size_t)s * R;
                /* Left UNTOUCHED, not filled with -inf. `out` is 20 doubles
                 * per state -- 31 MB at order 4 -- so writing every dead row
                 * rewrites the whole buffer each layer and costs more memory
                 * traffic than the lookups it saves. Measured: filling gave
                 * 1.24x, skipping gives far more. The contract above is what
                 * makes this safe: dead rows are never read. */
                if (alive && !alive[s]) continue;
                /*
                 * Probe the forward wide row first. A state missing it scores
                 * exactly twenty -inf and nothing else, so deciding that with
                 * ONE lookup saves two further probes and the whole backward
                 * settle loop. On a per-repertoire model 98.6% of states have
                 * no such row, which is why this is worth 3.4x there -- and
                 * it writes the identical bytes, so the outputs do not move.
                 */
                if (!beam_lp_wide_row(m, BEAM_FWD, i, s % nw_fwd)) {
                    for (uint32_t x = 0; x < R; ++x) row[x] = -INFINITY;
                    continue;
                }
                state_weights_sparse(m, i, len, s, settle_back, r, row);
                for (uint32_t x = 0; x < R; ++x)
                    if (row[x] > hi) hi = row[x];
            }
        }
        if (out_max) *out_max = hi;
        return;
    }

    /*
     * Iterate by the state's LOW digits, not by the state.
     *
     * Every state writes only its own row, so any order is correct and the
     * results are bit-identical whichever is chosen; what changes by two to
     * three times is the time it takes. Splitting the state as
     * `s = oldest * state_shift + t` and putting `t` outermost lines all
     * three big streams up:
     *
     *   - the backward wide table is indexed `t * A + x` (see
     *     edge_model_internal.h), so one `t` reads one contiguous 3 kB block
     *     and consecutive `t` read consecutive blocks: a single sequential
     *     scan of the half-gigabyte table instead of 3.9 million scattered
     *     lookups;
     *   - the forward wide table and `out` are both indexed by the state, so
     *     each of the 21 `oldest` values walks its own slab forwards. That is
     *     21 sequential streams, which the hardware prefetcher handles, not
     *     21 random walks;
     *   - the mid and length tables are indexed by the low digits alone, so
     *     the rows a given `t` needs are read once and reused across all 21
     *     `oldest` values instead of being re-fetched for each.
     *
     * This is the hot path of every dynamic program in the library: six table
     * lookups per edge, 20 edges per state, called once per layer per length.
     */
    /* A maximum is order-independent, so the fused value is bit-identical to
     * the separate pass and does not move with the thread count. The dead
     * layer reports -inf because OpenMP initialises a floating-point `max`
     * reduction to the least representable number, which for an IEEE double
     * is -inf; a private copy no iteration touches therefore cannot outrank a
     * genuinely dead layer. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:hi) \
            num_threads(beam_threads())
#endif
    for (int64_t t64 = 0; t64 < (int64_t)SK; ++t64) {
        const uint32_t t = (uint32_t)t64;
        for (uint32_t oldest = 0; oldest < A; ++oldest) {
            const uint32_t s = oldest * SK + t;
            double *row = out + (size_t)s * R;
            if (alive && !alive[s]) continue;   /* see the sparse branch */
            state_weights_dense(m, i, len, s, settle_back, r, row);
            for (uint32_t x = 0; x < R; ++x) if (row[x] > hi) hi = row[x];
        }
    }
    if (out_max) *out_max = hi;
}

/**
 * Sink weights, settling the backward scores of the final `min(order, len)`
 * residues.
 *
 * At the sink the state holds `(s[len-K], ..., s[len-1])`. The residue at
 * distance `r` from the C terminus sits in slot `K-1-r`, and its right-hand
 * context runs off the end of the sequence after `r` residues, so the
 * remaining context slots are sentinels.
 */
void beam_tail_weights(const BeamModel *m, uint32_t len, double *out) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t K = m->state_order;
    const uint32_t n_tail = (K < len) ? K : len;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t s64 = 0; s64 < (int64_t)m->n_states; ++s64) {
        const uint32_t s = (uint32_t)s64;
        /* Unpack the state, oldest first. */
        uint8_t slot[BEAM_MAX_ORDER + 1];
        uint32_t q = s;
        for (int32_t t = (int32_t)K - 1; t >= 0; --t) {
            slot[t] = (uint8_t)(q % A);
            q /= A;
        }

        double total = 0.0;
        for (uint32_t r = 0; r < n_tail && total != -INFINITY; ++r) {
            const uint32_t pos = K - 1u - r;          /* slot of the scored residue */
            const uint32_t scored = slot[pos];
            if (scored >= R) { total = -INFINITY; break; }

            /* c_j = the residue j places to the right, or a sentinel past the
             * end of the sequence. The mid and length tables are packed
             * most-distant-first, so they take the digits descending; the
             * wide table is packed nearest-first, so it takes them ascending.
             * See edge_model_internal.h for why the wide one differs. */
            uint32_t cm = 0, cl = 0;
            for (uint32_t j = K; j >= 1; --j) {
                uint32_t c = (j <= r) ? slot[pos + j] : BEAM_NONE;
                if (j <= 3u) cm = cm * A + c;
                if (j <= 2u) cl = cl * A + c;
            }
            uint32_t cw = 0;
            for (uint32_t j = 1; j <= m->order; ++j) {
                cw = cw * A + ((j <= r) ? slot[pos + j] : BEAM_NONE);
            }
            const double b = beam_mix(m, BEAM_BWD, len, r,
                                          cw, cm, cl, scored);
            if (b == -INFINITY) { total = -INFINITY; break; }
            total += m->cfg.weight_backward * b;
        }
        out[s] = total;
    }
}

/* ── Partition function ──────────────────────────────────────── */

/**
 * Forward sweep over the trellis, in linear space with per-layer rescaling.
 *
 * Log space would cost a `log` per target state and an `exp` per predecessor;
 * rescaling costs one `exp` per edge and one `log` per layer, which is an
 * order of magnitude cheaper on a 194k-state trellis. Underflow within a layer
 * is harmless: a path 700 nats below the layer's best contributes less than
 * 1e-300 of the total.
 */
BeamError beam_compute_log_z(BeamModel *m, uint32_t len, double *out) {
    const uint32_t A  = BEAM_ALPHABET;
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)malloc((size_t)S * sizeof(double));
    double *next = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !cur || !next || !tail) {
        free(w); free(cur); free(next); free(tail);
        return BEAM_EALLOC("the partition-function trellis buffers");
    }

    /* The walk starts in the all-sentinel state. */
    for (uint32_t s = 0; s < S; ++s) cur[s] = 0.0;
    cur[S - 1] = 1.0;
    double log_scale = 0.0;
    /* How many states the sweep can currently reach, carried from the layer
     * that produced `cur` rather than recounted. Counting it costs one pass
     * over S per layer, which is the very thing the count is used to avoid. */
    uint32_t reach = 1u;
    /*
     * Decided once, from the model's own occupancy, rather than per layer.
     * The per-state build wins only when the sweep reaches a small slice of
     * the trellis, and that tracks how full the tables are: a per-repertoire
     * model fills 0.2% of its cells, the 927M foundation fills 42.7%. Below
     * the cut the thin path is 4x; above it the batched build's sequential
     * scan wins and even COUNTING the reached states costs more than it
     * saves -- measured at 0.85x on a 200,000-sequence model when the count
     * ran unconditionally.
     */
    bool try_thin = false;
    if (m->cfg.sparse) {
        /* Occupancy read off the sparse tables' own row counts, which is a
         * handful of loads. `beam_model_n_nonzero_cells` would answer the
         * same question by counting all 147 million cells -- once per length,
         * which is worse than anything it could save. */
        uint32_t nw = 0, nm = 0, nl = 0;
        (void)beam_sparse_rows(&m->sp_lp_wide[BEAM_FWD], 0u, &nw);
        (void)beam_sparse_rows(&m->sp_lp_mid[BEAM_FWD],  0u, &nm);
        (void)beam_sparse_rows(&m->sp_lp_len[BEAM_FWD],  0u, &nl);
        const uint64_t occupied = (uint64_t)nw + nm + nl;
        const uint64_t total = (uint64_t)m->n_depth_wide * m->n_ctx_wide
                             + (uint64_t)m->max_len * m->n_ctx_mid
                             + (uint64_t)m->n_len_slots * m->max_len * m->n_ctx_len;
        try_thin = occupied * 20u < total;
    }

    for (uint32_t i = 0; i < len; ++i) {
        /* Rescale the layer's weights by their maximum so the exponentials
         * stay in range regardless of how deep the tables go. The maximum
         * comes out of the layer build itself; taking a second pass over the
         * 31 MB of weights to find it cost a fifth of the sweep. */
        /*
         * The maximum comes from `beam_layer_max`, which reaches it
         * through the occupied forward rows instead of building all 194,481
         * states. Identical by construction -- a state with no forward row
         * scores -inf and cannot carry a maximum -- and verified layer by
         * layer against the full scan on the foundation.
         *
         * The weights themselves are then built only for states the sweep can
         * reach: `cur[src] != 0` is exactly the set the accumulation below
         * reads, so every other row would be computed and never looked at.
         * On a per-repertoire model that is about 1% of the trellis.
         *
         * This sweep is the partition function, run once per live length at
         * the end of finalize, and it was the 1.6s floor on every build -- and
         * on every load, since the file stores counts and loading refinalizes.
         */
        double wmax;
        if (try_thin && (uint64_t)reach * 8u < (uint64_t)S) {
            /* Sparse and thinly reached: build only the rows the accumulation
             * reads. `cur[src] != 0` is exactly that set. */
            wmax = beam_layer_max(m, i, len);
            if (wmax != -INFINITY)
                for (uint32_t s = 0; s < S; ++s)
                    if (cur[s] != 0.0)
                        beam_state_weights(m, i, len, s, w + (size_t)s * R);
        } else {
            /* Densely reached, or a dense model: the batched build wins. Its
             * loop order turns the wide table into one sequential scan rather
             * than millions of scattered lookups -- worth 2-3x -- and that
             * beats skipping rows once most of them are wanted. Measured:
             * per-state was 2.4x FASTER at 100 sequences and 1.2x SLOWER at
             * 200,000, which is what the threshold is for. */
            wmax = -INFINITY;
            beam_layer_weights(m, i, len, w, &wmax);
        }
        if (wmax == -INFINITY) {           /* no live edge: no length-L path */
            free(w); free(cur); free(next); free(tail);
            *out = -INFINITY;
            return BEAM_OK;
        }

        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        double nmax = 0.0;
        int64_t nreach = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                reduction(+:nreach) num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < (int64_t)S; ++t) {
            const uint32_t x = (uint32_t)t % A;
            if (x >= R) { next[t] = 0.0; continue; }
            const uint32_t rest = (uint32_t)t / A;
            double acc = 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                const double v = cur[src];
                if (v == 0.0) continue;
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                acc += v * exp(wv - wmax);
            }
            next[t] = acc;
            if (try_thin && acc != 0.0) nreach++;
            if (acc > nmax) nmax = acc;
        }
        if (try_thin) reach = (uint32_t)nreach;
        log_scale += wmax;

        if (!(nmax > 0.0)) {
            free(w); free(cur); free(next); free(tail);
            *out = -INFINITY;
            return BEAM_OK;
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) next[s] /= nmax;
        log_scale += log(nmax);

        double *tmp = cur; cur = next; next = tmp;
    }

    beam_tail_weights(m, len, tail);
    double tmax = -INFINITY;
    for (uint32_t s = 0; s < S; ++s)
        if (cur[s] > 0.0 && tail[s] > tmax) tmax = tail[s];
    if (tmax == -INFINITY) {
        free(w); free(cur); free(next); free(tail);
        *out = -INFINITY;
        return BEAM_OK;
    }
    double acc = 0.0;
    for (uint32_t s = 0; s < S; ++s) {
        if (cur[s] == 0.0 || tail[s] == -INFINITY) continue;
        acc += cur[s] * exp(tail[s] - tmax);
    }

    *out = log_scale + tmax + log(acc);
    free(w); free(cur); free(next); free(tail);
    return BEAM_OK;
}

/* ── Backward values ─────────────────────────────────────────── */

/**
 * `beta[i][s]` is the natural log of the total weight of every completion of a
 * length-`len` path from state `s` at layer `i`, including the sink term.
 *
 * Kept in log space rather than rescaled: the sampler indexes it randomly
 * rather than sweeping it, so a per-layer scale factor would have to be
 * carried alongside and applied at every draw.
 */
BeamError beam_betas(const BeamModel *m, uint32_t len, double **beta) {
    return beam_betas_tilted(m, len, 1.0, beta);
}

BeamError beam_betas_tilted(const BeamModel *m, uint32_t len, double q,
                               double **beta) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    if (!w) return BEAM_EALLOC("one layer of edge weights");

    beam_tail_weights(m, len, beta[len]);
    if (q != 1.0)
        for (uint32_t s = 0; s < S; ++s)
            if (beta[len][s] != -INFINITY) beta[len][s] *= q;

    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, len, w, NULL);
        const double *nb = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
            const uint32_t s = (uint32_t)s64;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = q * row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            if (mx == -INFINITY) { beta[i][s] = -INFINITY; continue; }
            double acc = 0.0;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                acc += exp(q * row[x] + nb[base + x] - mx);
            }
            beta[i][s] = mx + log(acc);
        }
    }
    free(w);
    return BEAM_OK;
}
