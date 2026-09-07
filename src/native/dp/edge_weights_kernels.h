/**
 * @file edge_weights_kernels.h
 * @brief Edge weights over the trellis, compiled once per alphabet.
 *
 * DELIBERATELY WITHOUT AN INCLUDE GUARD. edge_weights.c includes this three
 * times -- with the amino-acid dimensions, the nucleotide ones, and the
 * model's own -- and each inclusion defines a differently suffixed set of
 * file-static functions. The includer defines BEAM_K_SUF, BEAM_K_A, BEAM_K_R.
 *
 * ## Why compiled rather than dispatched
 *
 * This is the only one of the fifteen files the alphabet migration measurably
 * slowed, and it is the file every dynamic program reaches its inner loop
 * through. Pinning its radix back to a literal recovers the whole regression:
 * path_count 1.00x, entropy 1.01x, spectrum 1.01x against the amino-acid-only
 * build. Everywhere else a runtime radix costs nothing.
 *
 * Four ways of handing the constant back WITHOUT duplicating the code were
 * measured, and all four recovered nothing: specialising the state split;
 * dispatching the two leaf kernels at their call sites; dispatching the
 * exported functions into always_inline bodies; and extracting each parallel
 * loop's body into a specialised slice. Rebuilt without -fopenmp the third of
 * those recovers everything, which is the tell -- GCC lowers a parallel region
 * into an outlined function early and passes its variables through a data
 * struct, so a radix that is a literal at the call site is a runtime load
 * inside the loop.
 *
 * Duplicating the text is therefore not laziness in place of a clever fix; it
 * is the only thing measured to work. The cost is roughly three times the
 * object code for this one file, and the general copy means an alphabet that
 * is neither of the two stays correct.
 *
 * THAT GENERAL COPY IS UNREACHABLE TODAY, AND THEREFORE UNTESTED. The
 * dispatch in edge_weights.c picks `_aa` or `_nt` for every alphabet that
 * exists, so nothing exercises `_gen` and nothing checks that it agrees with
 * them bit for bit. That is a known state rather than an oversight: it is
 * insurance whose premium is being unverified until a third alphabet makes it
 * live. Whoever adds that alphabet should compare `_gen` against its
 * specialised copy before trusting either.
 */

/* Forward declarations: layer_max calls layer_weights, which is defined
 * below it, and without these the call would introduce a non-static
 * declaration that the definition then contradicts. */
static void BEAM_KFN(layer_weights)(const BeamModel *m, uint32_t i,
                                    uint32_t len, double *out,
                                    double *out_max);
static void BEAM_KFN(layer_weights_live)(const BeamModel *m, uint32_t i,
                                         uint32_t len, double *out,
                                         double *out_max,
                                         const uint8_t *alive);

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
 * last `order` residues never get one during the walk; `BEAM_KFN(tail_weights)`
 * pays those at the sink.
 */
/**
 * One state's row, shared by the whole-layer sweep and the single-row entry
 * point so the two cannot drift apart.
 */
static inline void BEAM_KFN(state_weights_dense)(const BeamModel *m, uint32_t i,
                                       uint32_t len, uint32_t s,
                                       bool settle_back, uint32_t r,
                                       double *row) {
    const uint32_t R  = BEAM_K_R;
    const uint32_t A  = BEAM_K_A;
    const uint32_t K  = m->state_order;
    const uint32_t n2 = m->n_ctx_len;
    const uint32_t n3 = m->n_ctx_mid;
    const uint32_t nw = m->n_ctx_wide;

    /* Forward: the context IS the state. A state whose newest slots hold
     * sentinels is only reachable in the first few layers, and the tables
     * carry sentinel-padded rows for exactly that case. Every forward context
     * is a suffix of the state, because the state is packed oldest-first and
     * a shorter context is simply the nearer residues. */
    /*
     * The forward half is guarded the way the backward half already is. A dead
     * cell mixes to -inf, and `weight_forward = 0` then makes `0 * -inf` a
     * NaN rather than the -inf it must be. Nothing rejected that weight --
     * `config.c` validates only `>= 0` -- and the NaNs were latent, because
     * every sweep iterates reached states and skipped them. On a
     * 200-sequence closed-world model they were not latent: `finalize`
     * reported "no length has a finite partition function, so the model can
     * generate nothing" about a perfectly generative model. Found by review;
     * the asymmetry with the backward half below was the tell.
     */
    for (uint32_t x = 0; x < R; ++x) {
        const double f = mix_dense(m, BEAM_FWD, len, i,
                                   s % nw, s % n3, s % n2, x);
        row[x] = (f == -INFINITY) ? -INFINITY : m->cfg.weight_forward * f;
    }
    if (!settle_back) return;

    const uint32_t scored = beam_state_oldest(s, m->state_shift);
    /* A sentinel in that slot means there is no residue to score.
     * Unreachable for i >= K, but cheap to be explicit about. */
    if (scored >= R) return;

    /*
     * The same hoist the sparse path performs below, and for the same reason.
     * See the long comment there for why it is sound and why the gate is what
     * it is: at order 4 the mid and length backward contexts sit entirely
     * inside the DAG state, so `x` cannot reach them, and the wide context is
     * `cw(0) + x` exactly.
     *
     * The dense path was left out when the sparse one was done, and here it
     * costs more than a lookup: each of these helpers runs a base-21 digit
     * loop with an integer division, twenty times per state, to recompute
     * three values that do not move. Worth 2.60x single-threaded and 1.69x on
     * sixteen threads over a full layer, with a bit-identical result -- the
     * prototype in `.dev/loop_study/` checks the whole 194,481 x 20 output
     * with memcmp and reports a maximum difference of 0 ULP.
     *
     * The gate is not optional. Run ungated, this rewrites 99.76% of the
     * output at orders 2 and 3, because `n_ctx_mid` is 21^3 while the state
     * shift is only 21^2 there, so the mid context does move with `x`.
     */
    if (K > 3u && m->order == K) {
        const uint32_t cw0 = beam_ctx_backward_wide_from_state(
            s, 0u, m->state_shift, K, m->order, A);
        const uint32_t cm = beam_ctx_backward_from_state(s, 0u, K, 3u, A);
        const uint32_t cl = beam_ctx_backward_from_state(s, 0u, K, 2u, A);
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY) continue;
            const double b = mix_dense(m, BEAM_BWD, len, r,
                                       cw0 + x, cm, cl, scored);
            row[x] = (b == -INFINITY) ? -INFINITY
                                      : row[x] + m->cfg.weight_backward * b;
        }
        return;
    }

    for (uint32_t x = 0; x < R; ++x) {
        if (row[x] == -INFINITY) continue;
        const double b = mix_dense(
            m, BEAM_BWD, len, r,
            beam_ctx_backward_wide_from_state(s, x, m->state_shift, K,
                                                  m->order, A),
            beam_ctx_backward_from_state(s, x, K, 3u, A),
            beam_ctx_backward_from_state(s, x, K, 2u, A), scored);
        row[x] = (b == -INFINITY) ? -INFINITY
                                  : row[x] + m->cfg.weight_backward * b;
    }
}

/** Sparse counterpart with the three forward directory lookups hoisted. */
static inline void BEAM_KFN(state_weights_sparse)(const BeamModel *m, uint32_t i,
                                        uint32_t len, uint32_t s,
                                        bool settle_back, uint32_t r,
                                        double *row) {
    const uint32_t R  = BEAM_K_R;
    const uint32_t A  = BEAM_K_A;
    const uint32_t K  = m->state_order;
    const uint32_t n2 = m->n_ctx_len;
    const uint32_t n3 = m->n_ctx_mid;
    const uint32_t nw = m->n_ctx_wide;

    const BeamLogRow fw = beam_lp_wide_row(m, BEAM_FWD, i, s % nw);
    const BeamLogRow fm = beam_lp_mid_row(m, BEAM_FWD, i, s % n3);
    const BeamLogRow fl = beam_lp_len_row(m, BEAM_FWD, len, i, s % n2);
    if (!fw.present || !fm.present || !fl.present) {
        for (uint32_t x = 0; x < R; ++x) row[x] = -INFINITY;
    } else {
        /*
         * Guarded exactly as the dense kernel above is, and for the same
         * reason: `mix_values` returns -INFINITY when any expert cell is
         * dead, and `weight_forward = 0` makes that `0 * -inf`, which is NaN.
         *
         * The dense half was fixed and this one was not, because the
         * regression test that caught it builds from `beam_model_config_default`,
         * where `sparse` is false. The Python layer selects sparse for the
         * default closed world, so the shipped path was the unguarded one:
         * measured at 115 NaN edge weights of 3,519,180 on a ten-sequence
         * order-3 strict model. The sparse BACKWARD half below was already
         * guarded, and that asymmetry is what gave this away.
         */
        /* Decoded once each rather than ranked per residue: this loop reads
         * every cell of all three rows, and a packed row's indices are in
         * residue order, so one walk beats twenty popcounts. */
        double fwv[BEAM_MAX_RESIDUES], fmv[BEAM_MAX_RESIDUES],
               flv[BEAM_MAX_RESIDUES];
        beam_log_row_into(&fw, R, fwv);
        beam_log_row_into(&fm, R, fmv);
        beam_log_row_into(&fl, R, flv);
        for (uint32_t x = 0; x < R; ++x) {
            const double f = mix_values(m, fwv[x], fmv[x], flv[x]);
            row[x] = (f == -INFINITY) ? -INFINITY : m->cfg.weight_forward * f;
        }
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
        const uint32_t cm = beam_ctx_backward_from_state(s, 0u, K, 3u, A);
        const uint32_t cl = beam_ctx_backward_from_state(s, 0u, K, 2u, A);
        const BeamLogRow bm = beam_lp_mid_row(m, BEAM_BWD, r, cm);
        const BeamLogRow bl = beam_lp_len_row(m, BEAM_BWD, len, r, cl);
        if (!bm.present || !bl.present) {
            for (uint32_t x = 0; x < R; ++x) row[x] = -INFINITY;
            return;
        }
        const uint32_t cw0 = beam_ctx_backward_wide_from_state(
            s, 0u, m->state_shift, K, m->order, A);
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY) continue;
            const BeamLogRow bw = beam_lp_wide_row(m, BEAM_BWD, r, cw0 + x);
            const double b = !bw.present ? -INFINITY
                           : mix_values(m, beam_log_cell(&bw, scored),
                                        beam_log_cell(&bm, scored),
                                        beam_log_cell(&bl, scored));
            row[x] = (b == -INFINITY) ? -INFINITY
                                      : row[x] + m->cfg.weight_backward * b;
        }
        return;
    }

    for (uint32_t x = 0; x < R; ++x) {
        if (row[x] == -INFINITY) continue;
        const uint32_t cw = beam_ctx_backward_wide_from_state(
            s, x, m->state_shift, K, m->order, A);
        const uint32_t cm = beam_ctx_backward_from_state(s, x, K, 3u, A);
        const uint32_t cl = beam_ctx_backward_from_state(s, x, K, 2u, A);
        const BeamLogRow bw = beam_lp_wide_row(m, BEAM_BWD, r, cw);
        const BeamLogRow bm = beam_lp_mid_row(m, BEAM_BWD, r, cm);
        const BeamLogRow bl = beam_lp_len_row(m, BEAM_BWD, len, r, cl);
        const double b = (!bw.present || !bm.present || !bl.present)
                       ? -INFINITY
                       : mix_values(m, beam_log_cell(&bw, scored),
                                    beam_log_cell(&bm, scored),
                                    beam_log_cell(&bl, scored));
        row[x] = (b == -INFINITY) ? -INFINITY
                                  : row[x] + m->cfg.weight_backward * b;
    }
}

static void BEAM_KFN(state_weights)(const BeamModel *m, uint32_t i, uint32_t len,
                            uint32_t state, double *out) {
    const uint32_t K = m->state_order;
    const bool settle_back = (i >= K);
    const uint32_t r = settle_back ? (len - 1u - (i - K)) : 0u;
    if (m->cfg.sparse)
        BEAM_KFN(state_weights_sparse)(m, i, len, state, settle_back, r, out);
    else
        BEAM_KFN(state_weights_dense)(m, i, len, state, settle_back, r, out);
}

/**
 * The layer's maximum edge weight, without building the layer.
 *
 * Every caller that asks `BEAM_KFN(layer_weights)` for `out_max` uses it as a
 * numerical rescale -- `compute_log_z`, `tilted_log_z`, `jet_sweep`,
 * `accumulator_max_at`. That is why those sweeps could not be restricted to
 * the live states: the maximum is taken over ALL states, and a state that
 * carries it need not be able to reach a sink. Measured on the 927M
 * foundation, 22 of 375 layers have exactly that, so a live-only maximum
 * changes the rescale and moves the last bits of every score.
 *
 * Restricting to states with an OCCUPIED forward wide row is different, and
 * is exact rather than lucky: `BEAM_KFN(state_weights_sparse)` returns -inf for a state
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
static double BEAM_KFN(layer_max)(const BeamModel *m, uint32_t i, uint32_t len) {
    const uint32_t R  = BEAM_K_R;
    const uint32_t K  = m->state_order;
    const uint32_t S  = m->n_states;
    const uint32_t nw = m->n_ctx_wide;
    const bool settle_back = (i >= K);
    const uint32_t r = settle_back ? (len - 1u - (i - K)) : 0u;

    if (!m->cfg.sparse) {
        double *w = (double *)malloc((size_t)S * R * sizeof(double));
        if (!w) return NAN;
        double hi = -INFINITY;
        BEAM_KFN(layer_weights)(m, i, len, w, &hi);
        free(w);
        return hi;
    }

    /*
     * Rows carry the depth bucket in their high digits, so this layer's rows
     * are exactly the directory slice [want * nw, (want + 1) * nw). Walking
     * the slice reaches the same rows in the same ascending order as filtering
     * the whole slot -> row map did, and visits n_ctx_wide entries instead of
     * every occupied row in the table -- 194,481 rather than 1,846,623 on the
     * foundation, at any depth.
     *
     * That map was this function's only reader anywhere in the library, so
     * reading the directory instead is also what lets a finalized table stop
     * carrying one. The values are pinned in tests/native/test_layer_max.c
     * against the DEFINITION -- every state, no shortcut -- rather than
     * against whichever row source came before.
     */
    const BeamSparseRows *fw = &m->sp_lp_wide[BEAM_FWD];
    const uint32_t want = beam_depth_part(m, i);
    const uint32_t base = want * nw;
    if ((!fw->index && !fw->rank)
        || (uint64_t)base + nw > (uint64_t)fw->n_dense_rows)
        return -INFINITY;

    const uint32_t stride = S / nw;      /* 1 when order == state_order */

    double hi = -INFINITY;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:hi) \
            num_threads(beam_threads()) if(nw > 4096)
#endif
    for (int64_t c = 0; c < (int64_t)nw; ++c) {
        if (!beam_sparse_present(fw, base + (uint32_t)c)) continue;
        const uint32_t ctx = (uint32_t)c;
        double buf[BEAM_MAX_RESIDUES];
        /* One wide context can back several states when the state carries
         * more residues than the wide expert's order. */
        for (uint32_t rep = 0; rep < stride; ++rep) {
            const uint32_t st = rep * nw + ctx;
            if (st >= S) break;
            /* The directory just said this row is present, so the sparse
             * scorer needs no probe of its own here. */
            BEAM_KFN(state_weights_sparse)(m, i, len, st, settle_back, r, buf);
            for (uint32_t x = 0; x < R; ++x) if (buf[x] > hi) hi = buf[x];
        }
    }
    return hi;
}

static void BEAM_KFN(layer_weights)(const BeamModel *m, uint32_t i, uint32_t len,
                            double *out, double *out_max) {
    BEAM_KFN(layer_weights_live)(m, i, len, out, out_max, NULL);
}

/**
 * As `BEAM_KFN(layer_weights)`, but skipping states `alive` marks dead.
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
static void BEAM_KFN(layer_weights_live)(const BeamModel *m, uint32_t i, uint32_t len,
                                 double *out, double *out_max,
                                 const uint8_t *alive) {
    const uint32_t R  = BEAM_K_R;
    const uint32_t A  = BEAM_K_A;
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
                if (!beam_lp_wide_row(m, BEAM_FWD, i, s % nw_fwd).present) {
                    for (uint32_t x = 0; x < R; ++x) row[x] = -INFINITY;
                    continue;
                }
                BEAM_KFN(state_weights_sparse)(m, i, len, s, settle_back, r, row);
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
            BEAM_KFN(state_weights_dense)(m, i, len, s, settle_back, r, row);
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
static void BEAM_KFN(tail_weights)(const BeamModel *m, uint32_t len, double *out) {
    const uint32_t A = BEAM_K_A;
    const uint32_t R = BEAM_K_R;
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
                uint32_t c = (j <= r) ? slot[pos + j] : m->none_idx;
                if (j <= 3u) cm = cm * A + c;
                if (j <= 2u) cl = cl * A + c;
            }
            uint32_t cw = 0;
            for (uint32_t j = 1; j <= m->order; ++j) {
                cw = cw * A + ((j <= r) ? slot[pos + j] : m->none_idx);
            }
            const double b = beam_mix(m, BEAM_BWD, len, r,
                                          cw, cm, cl, scored);
            if (b == -INFINITY) { total = -INFINITY; break; }
            total += m->cfg.weight_backward * b;
        }
        out[s] = total;
    }
}
