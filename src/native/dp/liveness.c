/**
 * @file liveness.c
 * @brief Forward/live frontiers, the per-length live cache, and its test hooks.
 */
#include "beam/internal.h"
#include "beam/memory.h"


/* ── Frontier trim ───────────────────────────────────────────── */

/**
 * Live states of one length, discovered by walking rather than by scanning.
 *
 * `trim` above answers the same question in `S * R` work per layer, which is
 * 3.9M at order 4 no matter how little of the trellis is reachable. Under the
 * hierarchical prior every reachable state is live, so the walk's value is in
 * the early layers, where few states are reachable at all.
 *
 * TWO SETS, NOT ONE, and the distinction is load-bearing:
 *
 *   `reached`  forward-reachable from the root. What the tilted and jet
 *              sweeps need: they carry a per-layer rescale computed from the
 *              states PRESENT at that layer, and a prefix that is reachable
 *              but cannot reach a sink still contributed to that scale.
 *              Restricting them to `live` would drop those contributions and
 *              move the last bits of D2 and the entropy.
 *
 *   `live`     reachable AND able to reach a sink. What path counting,
 *              topology, the spectra and top-k need, since those are
 *              properties of complete paths.
 *
 * Both index lists are ASCENDING, which keeps every accumulation in the order
 * the dense loops used and so keeps the arithmetic bit-identical.
 *
 * THE THIRD THING, and the reason nothing downstream reads a weight:
 *
 *   `edge_bits[i][k]` holds bit `x` iff the edge leaving `reached[i][k]` on
 *   residue `x` is finite. Pass 1 has to build those weights anyway to know
 *   where the walk can go; keeping the twenty bits it distilled from them
 *   costs four bytes per reached state and saves pass 1's work being redone
 *   by pass 2, by `path_count` and by `topology`, none of which ever look at
 *   a weight VALUE -- every one of them only ever asks `w[x] != -INFINITY`.
 *   Measured on the 927M foundation: the frontier falls from 70.3 s to 28.4 s
 *   and the path-count sweep from 36.1 s to 0.97 s, for 124 MB of masks
 *   against the 3.6 GB a materialised weight graph would have cost.
 *
 * `mask` is ONE BIT per state, not one byte. At order 4 a byte per state per
 * layer is 56 MB of cache for a fifteen-thousand-sequence donor and 136 MB
 * for the foundation, which would make liveness the largest term in the R3
 * cache by an order of magnitude and would make the donor's cache larger than
 * the foundation's index lists.
 */

/* ── One bit per state ───────────────────────────────────────── */

/*
 * The bitsets carry ONE SPARE WORD past `BITWORDS(S)`. `bit_window20` reads
 * the word after the one holding `base` whenever the twenty bits straddle a
 * word boundary, and paying for eight bytes is cheaper than branching on
 * whether `base` is near the end of the trellis.
 */
#define BEAM_LIVE_BITWORDS(S) ((((size_t)(S)) + 63u) / 64u)


/**
 * The twenty membership bits for states `base .. base + 19`, which is exactly
 * the residue fan-out of one source state: `target = base + x`.
 */
static inline uint32_t live_bit_window20(const uint64_t *b, uint32_t base) {
    const uint32_t w = base >> 6, s = base & 63u;
    uint64_t v = b[w] >> s;
    if (s > 44u) v |= b[w + 1u] << (64u - s);   /* s <= 63, so never a UB shift */
    return (uint32_t)(v & 0xFFFFFull);
}

void beam_live_free(BeamLive *L) {
    if (!L) return;
    if (L->mask)      { for (uint32_t i = 0; i <= L->len; ++i) free(L->mask[i]);      free(L->mask); }
    /* `live[i]` is either its own allocation or exactly `reached[i]` -- see
     * the two build sites below -- so pointer equality is the exact test for
     * "not mine to free", and needs no flag kept in step by hand. `reached`
     * is still live here because it is freed on the next line, not this one. */
    if (L->live) {
        for (uint32_t i = 0; i <= L->len; ++i)
            if (!L->reached || L->live[i] != L->reached[i]) free(L->live[i]);
        free(L->live);
    }
    if (L->reached)   { for (uint32_t i = 0; i <= L->len; ++i) free(L->reached[i]);   free(L->reached); }
    if (L->edge_bits) { for (uint32_t i = 0; i <= L->len; ++i) free(L->edge_bits[i]); free(L->edge_bits); }
    free(L->n_live); free(L->n_reached);
    memset(L, 0, sizeof *L);
}

/** Heap bytes one live set holds, for the cache's budget accounting. */
static uint64_t live_bytes(const BeamLive *L) {
    const uint64_t words = (uint64_t)BEAM_LIVE_BITWORDS(L->n_states) + 1u;
    uint64_t b = sizeof(BeamLive)
               + (uint64_t)(L->len + 1u) * (4u * sizeof(void *) + 2u * sizeof(uint32_t));
    for (uint32_t i = 0; i <= L->len; ++i) {
        b += words * sizeof(uint64_t);
        if (!L->live || !L->reached || L->live[i] != L->reached[i])
            b += (uint64_t)L->n_live[i] * sizeof(uint32_t);
        b += (uint64_t)L->n_reached[i] * sizeof(uint32_t);
        if (L->edge_bits && L->edge_bits[i])
            b += (uint64_t)L->n_reached[i] * sizeof(uint32_t);
    }
    return b;
}

static int cmp_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/**
 * Order the ENTRIES of an ascending index list so the low digits vary
 * slowest: `in[out[k]] % sk` non-decreasing in `k`.
 *
 * A counting sort over the `state_shift` buckets, which is 9,261 at order 4 --
 * cheaper than a comparison sort and stable, though stability does not matter
 * here since the destination is only a fetch order.
 *
 * It emits POSITIONS rather than states so the caller can write what it
 * learns about `in[j]` straight back into a list that is parallel to `in` and
 * therefore still ascending. `out` must hold `n` entries and `cnt` must hold
 * `sk + 1`; the input is left untouched, because the caller's index list has
 * to stay ascending for the accumulations that read it.
 */
static void sort_low_digit(const uint32_t *in, uint32_t n, uint32_t sk,
                           uint32_t *cnt, uint32_t *out) {
    if (!n) return;
    memset(cnt, 0, ((size_t)sk + 1u) * sizeof(uint32_t));
    for (uint32_t k = 0; k < n; ++k) cnt[in[k] % sk + 1u]++;
    for (uint32_t b = 0; b < sk; ++b) cnt[b + 1u] += cnt[b];
    for (uint32_t k = 0; k < n; ++k) out[cnt[in[k] % sk]++] = k;
}

static BeamError trim_frontier(const BeamModel *m, uint32_t len, BeamLive *L) {
    const uint32_t A  = m->n_alphabet;
    const uint32_t R  = m->n_residues;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;
    const size_t   BW = BEAM_LIVE_BITWORDS(S) + 1u;

    memset(L, 0, sizeof *L);
    L->len       = len;
    L->n_states  = S;
    L->mask      = (uint64_t **)calloc(len + 1u, sizeof(uint64_t *));
    L->live      = (uint32_t **)calloc(len + 1u, sizeof(uint32_t *));
    L->reached   = (uint32_t **)calloc(len + 1u, sizeof(uint32_t *));
    L->edge_bits = (uint32_t **)calloc(len + 1u, sizeof(uint32_t *));
    L->n_live    = (uint32_t *)calloc(len + 1u, sizeof(uint32_t));
    L->n_reached = (uint32_t *)calloc(len + 1u, sizeof(uint32_t));
    uint8_t  *seen = (uint8_t *)calloc(S, 1);
    uint32_t *buf  = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t *order_buf = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t *bucket = (uint32_t *)malloc(((size_t)SK + 1u) * sizeof(uint32_t));
    double   *row  = (double *)malloc((size_t)R * sizeof(double));
#define BEAM_FRONTIER_FAIL()                                                   \
    do { free(seen); free(buf); free(row); free(order_buf); free(bucket);     \
         beam_live_free(L);                                                        \
         return BEAM_EALLOC("the frontier-trim buffers"); } while (0)
    if (!L->mask || !L->live || !L->reached || !L->edge_bits || !L->n_live
        || !L->n_reached || !seen || !buf || !row || !order_buf || !bucket)
        BEAM_FRONTIER_FAIL();
    for (uint32_t i = 0; i <= len; ++i) {
        L->mask[i] = (uint64_t *)calloc(BW, sizeof(uint64_t));
        if (!L->mask[i]) BEAM_FRONTIER_FAIL();
    }

    /* ── pass 1: forward reachability, and the edge bits it distils ── */
    L->reached[0] = (uint32_t *)malloc(sizeof(uint32_t));
    if (!L->reached[0]) BEAM_FRONTIER_FAIL();
    L->reached[0][0] = S - 1u;          /* the all-sentinel root */
    L->n_reached[0]  = 1u;

    for (uint32_t i = 0; i < len; ++i) {
        const uint32_t nr = L->n_reached[i];
        uint32_t n = 0;
        if (nr) {
            L->edge_bits[i] = (uint32_t *)malloc((size_t)nr * sizeof(uint32_t));
            if (!L->edge_bits[i]) BEAM_FRONTIER_FAIL();
        }
        uint32_t *eb = L->edge_bits[i];
        /*
         * VISIT ORDER, not sparsity, is what decides this loop's cost.
         *
         * `beam_layer_weights` iterates by the state's LOW digits, and its
         * comment explains why: the backward wide table is indexed `t*A + x`,
         * so low-digit-major turns a half-gigabyte table into one sequential
         * scan instead of millions of scattered lookups. Worth two to three
         * times.
         *
         * Walking `reached` in ascending state order threw that away. It did
         * not show at 0.12% liveness, where skipping 99.9% of the trellis
         * swamps everything; it showed at 40%, where the frontier LOST to the
         * dense build it replaced and `hill_number(0)` became the slowest
         * analytic on the 927M foundation -- 59.9s against 29.3s for an
         * unconverted one. Sorting the frontier into low-digit order restores
         * the access pattern and is worth 1.17x-2.01x on the same states.
         *
         * The index lists this function emits stay ASCENDING: consumers
         * accumulate in that order and changing it would move the arithmetic.
         * Only the order in which weights are FETCHED changes here. That is
         * why the sort hands back POSITIONS: the bits land in `eb[j]`, which
         * is parallel to the ascending `reached[i]`, whatever order they were
         * computed in.
         */
        sort_low_digit(L->reached[i], nr, SK, bucket, order_buf);
        for (uint32_t k = 0; k < nr; ++k) {
            const uint32_t j = order_buf[k];
            beam_state_weights(m, i, len, L->reached[i][j], row);
            uint32_t b = 0;
            for (uint32_t x = 0; x < R; ++x)
                if (row[x] != -INFINITY) b |= 1u << x;
            eb[j] = b;
        }
        /* Expansion reads the bits, so it costs no table lookups at all and
         * is free to run in ASCENDING source order. Which target set comes
         * out does not depend on the order -- it is a set, and it is sorted
         * below either way. */
        for (uint32_t j = 0; j < nr; ++j) {
            const uint32_t base = (L->reached[i][j] % SK) * A;
            uint32_t b = eb[j];
            while (b) {
                const uint32_t x = (uint32_t)__builtin_ctz(b);
                b &= b - 1u;
                const uint32_t to = base + x;
                if (seen[to]) continue;
                seen[to] = 1u;
                buf[n++] = to;
            }
        }
        for (uint32_t k = 0; k < n; ++k) seen[buf[k]] = 0u;   /* touched only */
        qsort(buf, n, sizeof(uint32_t), cmp_u32);
        L->n_reached[i + 1] = n;
        if (n) {
            L->reached[i + 1] = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
            if (!L->reached[i + 1]) BEAM_FRONTIER_FAIL();
            memcpy(L->reached[i + 1], buf, (size_t)n * sizeof(uint32_t));
        }
    }

    /* ── pass 2: backward liveness, over reached states only ── */
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!tail) BEAM_FRONTIER_FAIL();
    beam_tail_weights(m, len, tail);   /* O(S), once per length, not per layer */
    {
        uint32_t n = 0;
        for (uint32_t k = 0; k < L->n_reached[len]; ++k) {
            const uint32_t st = L->reached[len][k];
            if (tail[st] != -INFINITY) { live_bit_set(L->mask[len], st); buf[n++] = st; }
        }
        L->n_live[len] = n;
        if (n == L->n_reached[len]) {
            /* Every reached state survived. `live` is a subset of `reached`
             * and both are strictly ascending, so equal counts prove the two
             * lists are equal -- alias rather than copy. Consumers only ever
             * read through the pointer, so they cannot tell the difference;
             * 126 of the foundation's 400 layers land here, 5.42 MiB. */
            L->live[len] = L->reached[len];
        } else if (n) {
            L->live[len] = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
            if (!L->live[len]) { free(tail); BEAM_FRONTIER_FAIL(); }
            memcpy(L->live[len], buf, (size_t)n * sizeof(uint32_t));
        }
    }
    free(tail);

    /*
     * No weights here, and no fetch order to get right: pass 1 already
     * reduced every edge to the one bit this test needs, and the twenty
     * targets of a source are twenty CONSECUTIVE states, so the whole
     * `for x: w[x] finite && next_layer_live[base + x]` loop collapses to one
     * masked window fetch and an AND. This is where the second half of R2's
     * saving comes from: pass 2 used to recompute everything pass 1 had
     * thrown away, in the slower of the two possible orders.
     */
    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        const uint64_t *nb = L->mask[i + 1];
        const uint32_t *rd = L->reached[i];
        const uint32_t *eb = L->edge_bits[i];
        const uint32_t nr = L->n_reached[i];
        uint32_t n = 0;
        /* `reached[i]` IS ascending and `live` is assembled straight from it,
         * so the list stays in the order consumers accumulate in. */
        for (uint32_t j = 0; j < nr; ++j) {
            const uint32_t st = rd[j];
            if (eb[j] & live_bit_window20(nb, (st % SK) * A)) {
                live_bit_set(L->mask[i], st);
                buf[n++] = st;
            }
        }
        L->n_live[i] = n;
        if (n == nr) {
            L->live[i] = L->reached[i];        /* equal lists; see above */
        } else if (n) {
            L->live[i] = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
            if (!L->live[i]) BEAM_FRONTIER_FAIL();
            memcpy(L->live[i], buf, (size_t)n * sizeof(uint32_t));
        }
    }
#undef BEAM_FRONTIER_FAIL

    free(seen); free(buf); free(row); free(order_buf); free(bucket);
    return BEAM_OK;
}


/* ── Acquiring a live set ────────────────────────────────────── */

/**
 * A live set for `len`, from the cache when there is one and freshly built
 * when there is not.
 *
 * The caller MUST pair this with `beam_live_release`, and must not hold the
 * returned pointer across anything that could mutate the model.
 *
 * The cache's storage, bound and locking are in memory/live_cache.c. What
 * stays here is the decision to build and the ownership of what was built:
 * `beam_live_cache_publish` either adopts the set -- after which this handle
 * must not free it -- or hands back the entry another thread published, or
 * declines for want of budget and leaves the caller to use and free its own.
 */
void beam_live_release(LiveHandle *h) {
    if (!h) return;
    if (!h->cached) beam_live_free(&h->scratch);
    h->L = NULL;
}

BeamError beam_live_acquire(const BeamModel *m, uint32_t len,
                             LiveHandle *h) {
    memset(h, 0, sizeof *h);

    if (len >= m->n_len_slots) {          /* callers check, but do not trust it */
        return trim_frontier(m, len, &h->scratch);
    }

    const BeamLive *hit = beam_live_cache_lookup(m, len);
    if (hit) { h->L = hit; h->cached = true; return BEAM_OK; }

    BeamError err = trim_frontier(m, len, &h->scratch);
    if (err != BEAM_OK) return err;

    bool adopted = false;
    const BeamLive *entry = beam_live_cache_publish(
        m, len, &h->scratch, live_bytes(&h->scratch), &adopted);

    if (adopted) {
        /* The set now belongs to the cache; the handle must not free it. */
        memset(&h->scratch, 0, sizeof h->scratch);
        h->L = entry;
        h->cached = true;
    } else if (entry) {
        beam_live_free(&h->scratch);      /* another thread published first */
        h->L = entry;
        h->cached = true;
    } else {
        h->L = &h->scratch;               /* over budget: build and discard */
        h->cached = false;
    }
    return BEAM_OK;
}


/* ── Test hooks ──────────────────────────────────────────────── */

/**
 * Re-derive everything `trim_frontier` claims and report the first claim that
 * does not hold. See `edge_model_internal.h` for the codes.
 *
 * The expensive one is code 5: for EVERY reached state of EVERY layer it
 * rebuilds the weights the frontier distilled its mask from and checks the
 * twenty bits one at a time. That is the single new invariant R2 introduces,
 * and it is the one that cannot be inferred from a matching answer -- a wrong
 * bit on a state nothing else reaches would leave every published number
 * unchanged until the day a model reached it.
 */
int beam_live_selfcheck(const BeamModel *m, uint32_t len) {
    if (!m || !m->finalized) return BEAM_LIVE_SELFCHECK_BUILD;
    const uint32_t R  = m->n_residues;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;

    BeamLive L;
    if (trim_frontier(m, len, &L) != BEAM_OK)
        return BEAM_LIVE_SELFCHECK_BUILD;

    int rc = 0;
    double *row = (double *)malloc((size_t)R * sizeof(double));
    if (!row) { beam_live_free(&L); return BEAM_LIVE_SELFCHECK_BUILD; }

    if (L.n_reached[0] != 1u || L.reached[0][0] != S - 1u)
        rc = BEAM_LIVE_SELFCHECK_ROOT;

    for (uint32_t i = 0; !rc && i <= len; ++i) {
        /* Both index lists are STRICTLY ascending. Not a style rule: the
         * sweeps accumulate in this order and a permutation of it would move
         * the last bits of every floating-point total. */
        for (uint32_t k = 1; k < L.n_reached[i]; ++k)
            if (L.reached[i][k] <= L.reached[i][k - 1u])
                { rc = BEAM_LIVE_SELFCHECK_REACH_ORDER; break; }
        for (uint32_t k = 1; !rc && k < L.n_live[i]; ++k)
            if (L.live[i][k] <= L.live[i][k - 1u])
                { rc = BEAM_LIVE_SELFCHECK_LIVE_ORDER; break; }
        if (rc) break;

        /* `live` is exactly the set `mask` marks, and it is a subset of
         * `reached`: a state nothing reaches cannot be on a complete path. */
        uint32_t marked = 0;
        for (uint32_t s = 0; s < S; ++s) if (live_bit_get(L.mask[i], s)) marked++;
        if (marked != L.n_live[i]) { rc = BEAM_LIVE_SELFCHECK_MASK; break; }
        for (uint32_t k = 0; k < L.n_live[i]; ++k)
            if (!live_bit_get(L.mask[i], L.live[i][k]))
                { rc = BEAM_LIVE_SELFCHECK_MASK; break; }
        if (rc) break;
        {
            uint32_t a = 0;
            for (uint32_t k = 0; k < L.n_live[i]; ++k) {
                while (a < L.n_reached[i] && L.reached[i][a] < L.live[i][k]) a++;
                if (a >= L.n_reached[i] || L.reached[i][a] != L.live[i][k])
                    { rc = BEAM_LIVE_SELFCHECK_SUBSET; break; }
            }
        }
        if (rc || i == len) continue;

        /* THE new invariant: bit x of edge_bits[i][k] iff the edge leaving
         * reached[i][k] on residue x is finite. */
        for (uint32_t k = 0; k < L.n_reached[i]; ++k) {
            beam_state_weights(m, i, len, L.reached[i][k], row);
            uint32_t want = 0;
            for (uint32_t x = 0; x < R; ++x)
                if (row[x] != -INFINITY) want |= 1u << x;
            if (L.edge_bits[i][k] != want)
                { rc = BEAM_LIVE_SELFCHECK_BITS; break; }
        }
        (void)SK;
    }

    free(row);
    beam_live_free(&L);
    return rc;
}

