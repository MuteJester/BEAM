/**
 * @file expected_counts.c
 * @brief Exact expected usage of every table cell, under the model itself.
 *
 * ## The quantity
 *
 * The same per-layer edge posterior the marginals use,
 *
 *     gamma_i(s, x) = alpha_i(s) * w_i(s, x) * beta_{i+1}(s -> x) / Z(L)
 *
 * summed not over states but into the table cell each edge would have
 * incremented during training. That is the E step of an EM iteration written
 * out exactly: the model's expected sufficient statistics under its own
 * distribution, with no sampling anywhere in the derivation. `math/fb_sweep.c`
 * produces gamma; `math/positional.c` answers the other question about it.
 *
 * ## Why this file is about memory as much as about arithmetic
 *
 * The marginals write into a buffer the caller supplied, of a size the caller
 * knew. This writes into six hash maps whose size is a property of the ANSWER
 * -- how many `(depth, context)` pairs the trellis actually reaches -- and on
 * the foundation model that was 896 MB against an 806 MB model, peaking at
 * 1,566 MB while the answer was copied out. Nothing bounded it and nothing
 * reported it.
 *
 * It cannot be bounded the way `beam_fb_alloc` bounds itself, by multiplying
 * out a shape known in advance. The only shape-derived bound here is the dense
 * table -- 210.0 million cells at order 4 -- which is the address space the
 * sparse accumulator exists in order not to allocate, and a guard built on it
 * refuses every call. So the account is charged AS the maps grow, where the
 * figure being asked for is exactly known, and the peak is returned in
 * `BeamExpectedCounts::working_set_bytes` because a figure that cannot be
 * predicted has to be measured.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"
#include "beam/fb_sweep.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* ── Expected cell usage ─────────────────────────────────────── */

/**
 * A growable map from a table's dense row index to an accumulator row.
 *
 * The alternative was an array the shape of the table itself, and it does not
 * fit: an order-4 model has 210.0 million cells, so two dense accumulators
 * would be seven gigabytes to hold an answer whose support is typically fifty
 * thousand rows. Keying only the rows an edge actually reaches makes the cost
 * proportional to the answer instead of to the address space.
 *
 * Open addressing with a power-of-two capacity, and no deletion, because
 * nothing here ever removes a row.
 */
typedef struct {
    uint32_t *key;      /* dense row index, or UINT32_MAX for an empty slot */
    uint32_t *at;       /* key slot -> row slot in `vals` and `ids`         */
    uint32_t  mask;     /* capacity - 1                                     */
    uint32_t  n;        /* occupied rows                                    */
    uint32_t  cap;      /* rows `vals` and `ids` can hold                   */
    uint32_t  stride;   /* doubles per row: the alphabet's residue count    */
    double   *vals;     /* [cap * stride]                                   */
    uint32_t *ids;      /* row slot -> dense row index                      */
    BeamMemAccount *pool;     /* the budget this map and its five siblings share  */
} RowMap;

/**
 * Take `bytes` from the shared account for a map, or refuse and record why.
 *
 * GROWTH-TIME rather than up front, which is the part worth explaining. Every
 * other program in this file bounds itself before it starts, from a shape it
 * knows: `fb_alloc` multiplies out `(len + 1) * n_states` and refuses. There
 * is no such expression here. The maps are keyed on the DP's `(depth, ctx)`
 * space, and the only bound derivable from the model's shape is the dense
 * table -- 148 million cells at order 4, which is the address space the map
 * exists in order NOT to allocate. A guard built on it would refuse every
 * call. What is available instead is exact: both growth paths double in
 * place, so each knows the figure it is about to ask for at the moment it
 * asks. That figure is charged here, and the answer is a measurement rather
 * than a prediction.
 *
 * REFUSE BEFORE ALLOCATING, as budget.h requires: callers charge the figure
 * they are about to hand `malloc`, and allocate only if this returns true.
 *
 * The refusal is recorded on the account rather than only returned, because
 * `rowmap_row` returns `double *` and has only NULL to fail with. Without it
 * a budget refusal and a genuine allocation failure would reach the caller as
 * the same bare `EALLOC`, and the message naming the figure and the way out
 * would be lost.
 */
static bool account_charge(BeamMemAccount *p, uint64_t bytes) {
    if (!beam_account_take(p, bytes)) {
        /* The figure is the WHOLE call's -- the backward field shares this
         * account -- so the refusal names the working set and not the
         * accumulators, which are only the part that asked last. */
        p->err = beam_budget_refuse_bytes("the expected-count working set",
                                          p->held + bytes);
        return false;
    }
    return true;
}

/** The key directory's bytes at capacity `slots`: two parallel uint32 arrays. */
static uint64_t rowmap_dir_bytes(uint64_t slots) {
    return slots * 2u * sizeof(uint32_t);
}

/** One row's bytes in `vals` and `ids` together. */
static uint64_t rowmap_row_bytes(uint32_t stride) {
    return (uint64_t)stride * sizeof(double) + sizeof(uint32_t);
}

/** What a map holds right now. Zero for one that was never initialised. */
static uint64_t rowmap_bytes(const RowMap *t) {
    if (!t->key) return 0;
    return rowmap_dir_bytes((uint64_t)t->mask + 1u)
         + (uint64_t)t->cap * rowmap_row_bytes(t->stride);
}

static void rowmap_free(RowMap *t) {
    if (t->pool) beam_account_release(t->pool, rowmap_bytes(t));
    free(t->key); free(t->at); free(t->vals); free(t->ids);
    memset(t, 0, sizeof(*t));
}

static bool rowmap_init(RowMap *t, BeamMemAccount *pool, uint32_t stride) {
    memset(t, 0, sizeof(*t));
    const uint32_t k0 = 1024u, r0 = 512u;
    const uint64_t want = rowmap_dir_bytes(k0)
                        + (uint64_t)r0 * rowmap_row_bytes(stride);
    if (!account_charge(pool, want)) return false;
    t->key = (uint32_t *)malloc((size_t)k0 * sizeof(uint32_t));
    t->at  = (uint32_t *)malloc((size_t)k0 * sizeof(uint32_t));
    t->vals = (double *)calloc((size_t)r0 * stride, sizeof(double));
    t->ids  = (uint32_t *)malloc((size_t)r0 * sizeof(uint32_t));
    if (!t->key || !t->at || !t->vals || !t->ids) {
        free(t->key); free(t->at); free(t->vals); free(t->ids);
        memset(t, 0, sizeof(*t));
        beam_account_release(pool, want);
        return false;
    }
    for (uint32_t i = 0; i < k0; ++i) t->key[i] = UINT32_MAX;
    t->mask = k0 - 1u;
    t->cap = r0;
    t->stride = stride;
    t->pool = pool;
    return true;
}

/* Knuth's multiplicative hash. Row indices are dense and highly structured --
 * consecutive contexts within a depth -- so the low bits alone collide badly
 * and the high bits are what carry the mixing. */
static inline uint32_t rowmap_hash(uint32_t row) {
    return row * 2654435761u;
}

static bool rowmap_rehash(RowMap *t) {
    const uint32_t ncap = (t->mask + 1u) * 2u;
    /* Both directories are live at once -- the new one is filled by walking
     * the old -- so the new size is charged on top of the old, and the old is
     * released only once it has actually been freed. */
    const uint64_t want = rowmap_dir_bytes(ncap);
    if (!account_charge(t->pool, want)) return false;
    uint32_t *key = (uint32_t *)malloc((size_t)ncap * sizeof(uint32_t));
    uint32_t *at  = (uint32_t *)malloc((size_t)ncap * sizeof(uint32_t));
    if (!key || !at) {
        free(key); free(at);
        beam_account_release(t->pool, want);
        return false;
    }
    for (uint32_t i = 0; i < ncap; ++i) key[i] = UINT32_MAX;
    const uint32_t nmask = ncap - 1u;
    for (uint32_t slot = 0; slot < t->n; ++slot) {
        const uint32_t row = t->ids[slot];
        uint32_t h = rowmap_hash(row) & nmask;
        while (key[h] != UINT32_MAX) h = (h + 1u) & nmask;
        key[h] = row;
        at[h] = slot;
    }
    free(t->key); free(t->at);
    beam_account_release(t->pool, rowmap_dir_bytes((uint64_t)t->mask + 1u));
    t->key = key; t->at = at; t->mask = nmask;
    return true;
}

/** The accumulator row for `row`, created zeroed if new. NULL only on OOM. */
static double *rowmap_row(RowMap *t, uint32_t row) {
    uint32_t h = rowmap_hash(row) & t->mask;
    while (t->key[h] != UINT32_MAX) {
        if (t->key[h] == row) return t->vals + (size_t)t->at[h] * t->stride;
        h = (h + 1u) & t->mask;
    }
    if (t->n == t->cap) {
        const uint32_t ncap = t->cap * 2u;
        /* Charged like the rehash: `realloc` may move rather than extend, in
         * which case both copies are live across the call, so the new size
         * goes on top of the old and the old comes off once the pair has
         * settled at `ncap`. */
        const uint64_t old  = (uint64_t)t->cap * rowmap_row_bytes(t->stride);
        const uint64_t want = (uint64_t)ncap * rowmap_row_bytes(t->stride);
        if (!account_charge(t->pool, want)) return NULL;
        /* `ids` first, the four-byte-per-row array, so that a failure to grow
         * the hundred-and-sixty-byte one leaves the smaller of the two
         * oversized rather than the larger. Either way `cap` is unchanged, so
         * the pool's release under-counts by that overhang -- which is not
         * observable: a NULL from here aborts the whole call, and the pool is
         * a stack local that dies with it. */
        uint32_t *ids = (uint32_t *)realloc(t->ids,
                                            (size_t)ncap * sizeof(uint32_t));
        if (!ids) { beam_account_release(t->pool, want); return NULL; }
        t->ids = ids;
        double *v = (double *)realloc(t->vals,
                                      (size_t)ncap * t->stride * sizeof(double));
        if (!v) { beam_account_release(t->pool, want); return NULL; }
        t->vals = v;
        memset(t->vals + (size_t)t->cap * t->stride, 0,
               (size_t)t->cap * t->stride * sizeof(double));
        t->cap = ncap;
        beam_account_release(t->pool, old);
    }
    const uint32_t slot = t->n++;
    t->ids[slot] = row;
    t->key[h] = row;
    t->at[h] = slot;
    /* Load factor kept at or below one half; past that the linear probe
     * degrades and the rehash is cheaper than the probes it saves. */
    if (t->n * 2u > t->mask + 1u && !rowmap_rehash(t)) return NULL;
    return t->vals + (size_t)slot * t->stride;
}

static int cmp_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/**
 * Move a map's contents into a block, rows ascending.
 *
 * Sorted because a hash map's insertion order is an artefact of the hash and
 * of which layer happened to reach a row first, and an output whose row order
 * depends on that is not reproducible even though its values are.
 */
static BeamError rowmap_emit(RowMap *t, BeamCountBlock *b) {
    b->n_rows = 0; b->rows = NULL; b->counts = NULL;
    if (t->n == 0) return BEAM_OK;
    /*
     * The block is the second copy of the map, and here the figure IS known
     * up front: `n` rows exactly. Charged and never released, because unlike
     * the maps it is the answer and outlives this function.
     *
     * This charge CANNOT be the one that refuses, and that is worth writing
     * down rather than leaving for someone to rediscover as a hole. A block
     * is `n` rows; the map it came from holds `cap >= n` rows plus a key
     * directory, so `block <= map` always. The caller emits and frees one map
     * at a time, so what is held while block `k` is charged is every map from
     * `k` on plus every block before it -- and term by term that is bounded by
     * the six maps together, which were all live a moment earlier. The peak is
     * behind us by the time we get here.
     *
     * It stays because that argument is a property of the emit ordering
     * above, not of this function. Emit both before releasing either -- which
     * is what this code did until it was measured at 1,566 MB on the
     * foundation model -- and this becomes the check that binds.
     */
    const uint64_t want = (uint64_t)t->n * rowmap_row_bytes(t->stride);
    if (!account_charge(t->pool, want)) return t->pool->err;
    uint32_t *rows = (uint32_t *)malloc((size_t)t->n * sizeof(uint32_t));
    double *counts = (double *)malloc((size_t)t->n * t->stride * sizeof(double));
    if (!rows || !counts) {
        free(rows); free(counts);
        beam_account_release(t->pool, want);
        return BEAM_EALLOC("an expected-count block");
    }
    memcpy(rows, t->ids, (size_t)t->n * sizeof(uint32_t));
    qsort(rows, t->n, sizeof(uint32_t), cmp_u32);
    for (uint32_t i = 0; i < t->n; ++i) {
        uint32_t h = rowmap_hash(rows[i]) & t->mask;
        while (t->key[h] != rows[i]) h = (h + 1u) & t->mask;
        memcpy(counts + (size_t)i * t->stride,
               t->vals + (size_t)t->at[h] * t->stride,
               (size_t)t->stride * sizeof(double));
    }
    b->n_rows = t->n;
    b->rows = rows;
    b->counts = counts;
    return BEAM_OK;
}

/* ── Row addressing ──────────────────────────────────────────── */

/*
 * It comes from `beam/indexing.h`, and that matters more here than anywhere
 * else those functions are used.
 *
 * These have to be the same rows `observe_one_position` writes, or the expected
 * counts are not comparable with the observed ones and the whole quantity means
 * nothing. This file used to spell them out again, arguing that deriving them a
 * second way would be a second thing to keep in step -- which is right, and is
 * an argument against keeping a copy here rather than for it. There is one
 * definition now, and `model/training.c` calls it too.
 */

/**
 * The two maps, indexed [anchor] exactly as BeamExpectedCounts is, together
 * with the budget they share.
 *
 * One struct rather than an array plus a pool passed beside it: the budget
 * and the maps it governs have to stay in step through eight call sites, and
 * two parameters that must agree are two chances for them not to.
 */
typedef struct {
    RowMap  map[2];
    BeamMemAccount pool;
} Accum;

/** Add `g` to one cell of one anchor's table. */
static bool bump(Accum *ac, uint32_t dir, uint32_t row, uint32_t x, double g) {
    double *w = rowmap_row(&ac->map[dir], row);
    if (!w) return false;
    w[x] += g;
    return true;
}

/**
 * Accumulate one length's edge posterior into the six maps, scaled by `weight`.
 *
 * Which cells an edge touches is not a choice made here: it is read off
 * `observe_one_position` in training.c, so that the expected counts land in
 * exactly the cells an observed sequence would have incremented, and the two
 * are directly comparable. Concretely, emitting `x` from state `s` at layer
 * `i` of a length-`L` walk settles
 *
 *   - the N-anchored score of `x` itself, at depth `i`, whose context is a
 *     suffix of the state index;
 *   - the C-anchored score of the residue the state's oldest slot holds, which
 *     sits at position `i - K` and is only now complete on its right. Its
 *     depth from the C terminus is `L - 1 - (i - K)`.
 *
 * and the sink settles the C-anchored scores of the final `min(K, L)`
 * residues, which never get a full right-hand context during the walk.
 * Together those cover every position once in each direction, which is why
 * every block sums to `L`.
 *
 * ONE asymmetry has to be got right and is easy to get wrong. The finalized
 * C-anchored WIDE table is packed nearest-first for cache reasons, but the
 * COUNT table it is derived from keeps the common most-distant-first packing.
 * These are expected COUNTS, so they use `beam_ctx_backward_counts_from_state`, the
 * most-distant-first reading, and not the `_wide_` variant the edge-weight
 * kernels use. Getting that backwards produces a plausible table addressed by
 * a reversed context, and nothing else in the library would catch it.
 *
 * The accumulation is serial. The edge weights are still built in parallel,
 * which is where the time goes; what is serialised is a handful of array
 * increments per edge, and doing them in one fixed order is what makes a cell
 * that many layers reach come out the same on every run.
 */
static BeamError expected_one_length(const BeamModel *m, uint32_t len,
                                     double weight, Accum *ac) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t K = m->state_order;
    const uint32_t nc = m->n_ctx;

    BeamFBWork k;
    BeamError err = beam_fb_alloc(m, len, &k, &ac->pool);
    if (err != BEAM_OK) return err;

    err = beam_fb_betas(m, len, NULL, k.beta, k.w);
    if (err != BEAM_OK) { beam_fb_free(&k); return err; }

    {
    const double logz = k.beta[0][S - 1u];
    if (!isfinite(logz)) {           /* a live length always has one path */
        beam_fb_free(&k);
        return BEAM_ENOPATHS("the trellis carries no path at this length");
    }

    for (uint32_t s = 0; s < S; ++s) k.cur[s] = -INFINITY;
    k.cur[S - 1u] = 0.0;                            /* ln 1 */
    bool oom = false;
    bool lost = false;

    for (uint32_t i = 0; i < len && !oom; ++i) {
        beam_layer_weights(m, i, len, k.w, NULL);
        const bool settle_back = (i >= K);
        const uint32_t r = settle_back ? (len - 1u - (i - K)) : 0u;
        const double *bn = k.beta[i + 1];

        for (uint32_t s = 0; s < S && !oom; ++s) {
            if (k.cur[s] == -INFINITY) continue;
            const double la = k.cur[s] - logz;
            const double *row = k.w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;

            /* Forward contexts do not depend on the residue emitted, so they
             * are hoisted; the state IS the forward context. */
            const uint32_t fw = beam_row(m, i, s % nc);
            const uint32_t scored = beam_state_oldest(s, SK);
            const bool back = settle_back && scored < R;

            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY) continue;
                const double nb = bn[base + x];
                if (nb == -INFINITY) continue;
                const double g = weight * exp(la + row[x] + nb);
                if (g == 0.0) continue;
                if (!bump(ac, BEAM_FWD, fw, x, g)) { oom = true; break; }
                if (!back) continue;
                const uint32_t bw = beam_row(
                    m, r, beam_ctx_backward_counts_from_state(s, x, K, m->order, A));
                if (!bump(ac, BEAM_BWD, bw, scored, g)) { oom = true; break; }
            }
        }
        if (oom) break;
        if (!beam_fb_advance(m, k.w, k.cur, k.next, 0u, R)) { lost = true; break; }
        double *tmp = k.cur; k.cur = k.next; k.next = tmp;
    }

    /*
     * The sink. Its posterior is `alpha_L(s) * tail(s) / Z`, and the contexts
     * are read off the final state exactly as `tail_weights` reads them: the
     * residue at distance `r` from the C terminus sits in slot `K-1-r`, and
     * its right-hand context runs off the end of the sequence after `r`
     * residues, so the remaining slots are sentinels.
     */
    if (!oom && !lost) {
        beam_tail_weights(m, len, k.tail);
        const uint32_t n_tail = (K < len) ? K : len;
        for (uint32_t s = 0; s < S && !oom; ++s) {
            if (k.cur[s] == -INFINITY || k.tail[s] == -INFINITY) continue;
            const double g = weight * exp(k.cur[s] + k.tail[s] - logz);
            if (g == 0.0) continue;
            uint8_t slot[BEAM_MAX_ORDER + 1];
            uint32_t q = s;
            for (int32_t t = (int32_t)K - 1; t >= 0; --t) {
                slot[t] = (uint8_t)(q % A);
                q /= A;
            }
            for (uint32_t rr = 0; rr < n_tail; ++rr) {
                const uint32_t pos = K - 1u - rr;
                const uint32_t scored = slot[pos];
                if (scored >= R) break;      /* not a valid ending */
                uint32_t cw = 0;
                for (uint32_t j = m->order; j >= 1u; --j)
                    cw = cw * A + ((j <= rr) ? slot[pos + j] : m->none_idx);
                if (!bump(ac, BEAM_BWD, beam_row(m, rr, cw), scored, g)) {
                    oom = true; break;
                }
            }
        }
    }

    beam_fb_free(&k);
    /* `rowmap_row` has only NULL to fail with, so which failure it was is
     * carried on the pool: a budget refusal names the figure and the way out,
     * and a genuine allocation failure does not pretend to. */
    if (oom)
        return ac->pool.err != BEAM_OK
             ? ac->pool.err : BEAM_EALLOC("an expected-count accumulator row");
    if (lost)
        return BEAM_ENOPATHS("the forward sweep lost the trellis's mass to "
                            "underflow, so the expected counts would not sum "
                            "to the length");
    }
    return BEAM_OK;
}

BeamError beam_model_expected_counts(const BeamModel *m, uint32_t length,
                                     BeamExpectedCounts *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output struct");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (length != 0u && !beam_length_live(m, length))
        return BEAM_EBADLEN(m, length);

    memset(out, 0, sizeof(*out));
    out->n_residues = m->n_residues;

    /* Zeroed FIRST, so the cleanup below is safe on the maps that were never
     * reached: `rowmap_free` frees four pointers, and on an uninitialised map
     * those are whatever was on the stack. */
    Accum ac;
    memset(&ac, 0, sizeof(ac));
    for (uint32_t d = 0; d < 2u; ++d) {
        if (!rowmap_init(&ac.map[d], &ac.pool, m->n_residues)) {
            const BeamError why = ac.pool.err;
            for (uint32_t dd = 0; dd < 2u; ++dd) rowmap_free(&ac.map[dd]);
            return why != BEAM_OK
                 ? why : BEAM_EALLOC("the expected-count accumulators");
        }
    }

    BeamError err = BEAM_OK;
    double total = 0.0;
    if (length != 0u) {
        err = expected_one_length(m, length, 1.0, &ac);
        total = (double)length;
    } else {
        /* Ascending length, so a cell many lengths reach accumulates its
         * contributions in one fixed order. Each length is weighted by its own
         * share of the mixture, which makes the pooled answer the expected
         * usage per sequence drawn from the model rather than per sequence of
         * some particular length. */
        for (uint32_t L = 1; L <= m->max_len && err == BEAM_OK; ++L) {
            if (!beam_length_live(m, L)) continue;
            const double pi = exp(m->log_pi[L]);
            if (pi == 0.0) continue;
            err = expected_one_length(m, L, pi, &ac);
            total += pi * (double)L;
        }
    }

    /*
     * Emit and free ONE MAP AT A TIME, not both then both.
     *
     * A block is a second copy of its map, so emitting both before
     * releasing any means the peak is every map plus every block at once --
     * measured at 1,566 MB on the foundation model, where releasing each map
     * as its block lands peaks at 1,185 MB for the same answer. The largest
     * map is 377 MB and is the first one emitted; holding it through five
     * more emissions buys nothing.
     *
     * `rowmap_free` runs unconditionally, including on the maps an earlier
     * error skipped, so every map is still freed exactly once.
     */
    for (uint32_t d = 0; d < 2u; ++d) {
        if (err == BEAM_OK)
            err = rowmap_emit(&ac.map[d], &out->block[d]);
        rowmap_free(&ac.map[d]);
    }

    if (err != BEAM_OK) { beam_model_expected_counts_free(out); return err; }
    out->expected_length = total;
    out->working_set_bytes = ac.pool.peak;
    return BEAM_OK;
}

void beam_count_block_free(BeamCountBlock *b) {
    if (!b) return;
    free(b->rows);
    free(b->counts);
    memset(b, 0, sizeof(*b));
}

void beam_model_expected_counts_free(BeamExpectedCounts *ec) {
    if (!ec) return;
    for (uint32_t d = 0; d < 2u; ++d) beam_count_block_free(&ec->block[d]);
    memset(ec, 0, sizeof(*ec));
}
