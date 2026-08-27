/**
 * @file analytics.c
 * @brief Exact dynamic programs over the BEAM DAG: support size, surprisal
 *        spectrum, entropy, Hill numbers.
 *
 * Nothing here samples. Every quantity is a sweep over the same layered DAG
 * the scorer uses, which is the whole reason the model is built as a DAG in
 * the first place.
 *
 * Reference: `.private/edge/src/spectrum.py`.
 */
#include "beam/internal.h"

/* ── Live-state trimming ─────────────────────────────────────── */

/**
 * `alive[i][s]` is true iff a live path runs from state `s` at layer `i` to a
 * live sink.
 *
 * On the models this library builds, that is every state: hierarchical backoff
 * leaves every cell positive, so no transition is dead. The pass is kept
 * anyway, because the alternative is to ASSUME saturation rather than measure
 * it — and because the DPs downstream are written against a trimmed trellis,
 * so a smoother that ever did leave a cell at zero would be handled correctly
 * instead of silently counting sequences the model cannot produce.
 *
 * `alive` must be `len + 1` arrays of `n_states` bytes.
 */
static BeamError trim(const BeamModel *m, uint32_t len, uint8_t **alive) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *t = (double *)malloc((size_t)S * sizeof(double));
    if (!t) return BEAM_EALLOC("the live-state trimming buffers");
    beam_tail_weights(m, len, t);
    for (uint32_t s = 0; s < S; ++s) alive[len][s] = (t[s] != -INFINITY);
    free(t);

    /*
     * When no table cell is -inf, no BEAM is -inf either, so which states are
     * live stops depending on the weights at all and becomes pure
     * reachability. That is the common case — hierarchical backoff leaves
     * every cell positive — and skipping the weights here is what stops
     * `path_count`, `topology` and the spectrum from building every layer
     * twice. The flag is measured at finalize, not assumed; the general path
     * below still runs when it is false.
     */
    if (m->tables_all_finite) {
        for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
            const uint8_t *nb = alive[i + 1];
            uint8_t *cu = alive[i];
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
            for (int64_t s = 0; s < (int64_t)S; ++s) {
                const uint32_t base = ((uint32_t)s % SK) * A;
                uint8_t ok = 0;
                for (uint32_t x = 0; x < R && !ok; ++x) ok = nb[base + x];
                cu[s] = ok;
            }
        }
        return BEAM_OK;
    }

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    if (!w) return BEAM_EALLOC("the live-state trimming buffers");
    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, len, w, NULL);
        const uint8_t *nb = alive[i + 1];
        uint8_t *cu = alive[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) {
            const double *row = w + (size_t)s * R;
            const uint32_t base = ((uint32_t)s % SK) * A;
            uint8_t ok = 0;
            for (uint32_t x = 0; x < R && !ok; ++x)
                ok = (row[x] != -INFINITY) && nb[base + x];
            cu[s] = ok;
        }
    }
    free(w);
    return BEAM_OK;
}

/* ── Frontier trim ───────────────────────────────────────────── */

/**
 * Live states of one length, discovered by walking rather than by scanning.
 *
 * `trim` above answers the same question in `S * R` work per layer, which is
 * 3.9M at order 4 no matter how little of the trellis is reachable. On a
 * per-repertoire strict model the widest layer holds a few thousand live
 * states out of 194,481, so almost all of that work is spent proving that
 * states nothing reaches are also states nothing can leave.
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

/* ── One bit per state ───────────────────────────────────────── */

/*
 * The bitsets carry ONE SPARE WORD past `BITWORDS(S)`. `bit_window20` reads
 * the word after the one holding `base` whenever the twenty bits straddle a
 * word boundary, and paying for eight bytes is cheaper than branching on
 * whether `base` is near the end of the trellis.
 */
#define BEAM_LIVE_BITWORDS(S) ((((size_t)(S)) + 63u) / 64u)

static inline void live_bit_set(uint64_t *b, uint32_t i) {
    b[i >> 6] |= 1ull << (i & 63u);
}

static inline uint32_t live_bit_get(const uint64_t *b, uint32_t i) {
    return (uint32_t)((b[i >> 6] >> (i & 63u)) & 1ull);
}

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

static void live_free(BeamLive *L) {
    if (!L) return;
    if (L->mask)      { for (uint32_t i = 0; i <= L->len; ++i) free(L->mask[i]);      free(L->mask); }
    if (L->live)      { for (uint32_t i = 0; i <= L->len; ++i) free(L->live[i]);      free(L->live); }
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
    const uint32_t A  = BEAM_ALPHABET;
    const uint32_t R  = BEAM_RESIDUES;
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
         live_free(L);                                                        \
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
        if (n) {
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
        if (n) {
            L->live[i] = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
            if (!L->live[i]) BEAM_FRONTIER_FAIL();
            memcpy(L->live[i], buf, (size_t)n * sizeof(uint32_t));
        }
    }
#undef BEAM_FRONTIER_FAIL

    free(seen); free(buf); free(row); free(order_buf); free(bucket);
    return BEAM_OK;
}


/* ── The per-length live cache ───────────────────────────────── */

/**
 * One live set per length, held on the model until something invalidates it.
 *
 * WHY THIS IS NOT AN OPTIMISATION YOU CAN SKIP. Once a dynamic program
 * iterates the live index instead of all 194,481 states, the structure build
 * becomes the largest single term left in it -- 78% of a converted DP on a
 * fifteen-thousand-sequence donor and 52% on the 927M foundation. Eight
 * analytics each rebuilding it is 3.6x and 2.1x of pure waste, and at the top
 * end it is worse than that: converting the sweeps WITHOUT sharing the
 * structure measures as a 1.4x regression against the unconverted code on the
 * foundation and 1.5x on an open-world model, because the build is repaid
 * over one sweep rather than eight.
 *
 * WHAT IT COSTS. Index lists, edge masks and a one-bit-per-state liveness
 * map, summed over every live length: 7.9 MB for a fifteen-thousand-sequence
 * donor, 388 MB for the foundation, 422 MB for a 45%-live open-world model.
 * The bound below refuses anything past a quarter of the process budget and
 * falls back to building per call, so a model whose live structure does not
 * fit stays correct and merely stops being fast.
 *
 * THREAD SAFETY. Entries are immutable once published, so concurrent readers
 * need no lock; only the publish is serialised, and it happens at most once
 * per length. Two threads racing on the same length both build, one publishes
 * and the other frees its own copy and uses the winner's. Mutating a model
 * while an analytic runs on it is unsupported here exactly as it is
 * everywhere else in the library.
 */
struct BeamLiveCache {
    BeamLive **slot;     /* [n_slots], NULL until a length is first asked for */
    uint32_t   n_slots;
    uint64_t   bytes;    /* what the published entries hold, for the bound    */
};

void beam_live_cache_drop(BeamModel *m) {
    if (!m || !m->live_cache) return;
    BeamLiveCache *c = m->live_cache;
    /* Unhook first, so a caller that somehow re-enters sees no cache rather
     * than half of one. */
    m->live_cache = NULL;
    for (uint32_t i = 0; i < c->n_slots; ++i) {
        if (!c->slot[i]) continue;
        live_free(c->slot[i]);
        free(c->slot[i]);
    }
    free(c->slot);
    free(c);
}

/**
 * Ceiling on what the cache may hold, in bytes.
 *
 * A quarter of the process memory budget. The budget's other job is capping
 * how many per-length dynamic programs run at once, and the cache is a
 * PERSISTENT claim against the same pool, so it takes a minority share rather
 * than the whole of it. At the 4 GB default that is 1 GB, which holds the
 * foundation's 388 MB and the open-world model's 422 MB with room to spare;
 * lowering the budget lowers this with it.
 */
static uint64_t live_cache_cap(void) {
    return beam_model_memory_budget() / 4u;
}

/**
 * A live set for `len`, from the cache when there is one and freshly built
 * when there is not.
 *
 * The caller MUST pair this with `live_release`, and must not hold the
 * returned pointer across anything that could mutate the model.
 */
typedef struct {
    const BeamLive *L;
    BeamLive        scratch;   /* holds the set when it was not cached */
    bool            cached;
} LiveHandle;

static void live_release(LiveHandle *h) {
    if (!h) return;
    if (!h->cached) live_free(&h->scratch);
    h->L = NULL;
}

static BeamError live_acquire(const BeamModel *m, uint32_t len,
                             LiveHandle *h) {
    /* The cache is model state that read-only analytics maintain; the models
     * they are handed are const, and this is the one place that admits it. */
    BeamModel *mm = (BeamModel *)m;
    memset(h, 0, sizeof *h);

    if (len >= m->n_len_slots) {          /* callers check, but do not trust it */
        return trim_frontier(m, len, &h->scratch);
    }

    const BeamLive *hit = NULL;
#ifdef _OPENMP
    #pragma omp critical(beam_live_cache)
#endif
    {
        if (mm->live_cache && mm->live_cache->slot[len])
            hit = mm->live_cache->slot[len];
    }
    if (hit) { h->L = hit; h->cached = true; return BEAM_OK; }

    BeamError err = trim_frontier(m, len, &h->scratch);
    if (err != BEAM_OK) return err;

    const uint64_t want = live_bytes(&h->scratch);
    BeamLive *moved = NULL;
#ifdef _OPENMP
    #pragma omp critical(beam_live_cache)
#endif
    {
        if (!mm->live_cache) {
            BeamLiveCache *c =
                (BeamLiveCache *)calloc(1, sizeof(BeamLiveCache));
            if (c) {
                c->slot = (BeamLive **)calloc(m->n_len_slots, sizeof(BeamLive *));
                c->n_slots = m->n_len_slots;
                if (c->slot) mm->live_cache = c;
                else free(c);
            }
        }
        if (mm->live_cache) {
            if (mm->live_cache->slot[len]) {
                hit = mm->live_cache->slot[len];      /* another thread won */
            } else if (mm->live_cache->bytes + want <= live_cache_cap()) {
                BeamLive *e = (BeamLive *)malloc(sizeof(BeamLive));
                if (e) {
                    *e = h->scratch;
                    mm->live_cache->slot[len] = e;
                    mm->live_cache->bytes += want;
                    moved = e;
                }
            }
        }
    }
    if (moved) {
        /* The set now belongs to the cache; the handle must not free it. */
        memset(&h->scratch, 0, sizeof h->scratch);
        h->L = moved;
        h->cached = true;
    } else if (hit) {
        live_free(&h->scratch);
        h->L = hit;
        h->cached = true;
    } else {
        h->L = &h->scratch;                /* over budget: build and discard */
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
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;

    BeamLive L;
    if (trim_frontier(m, len, &L) != BEAM_OK)
        return BEAM_LIVE_SELFCHECK_BUILD;

    int rc = 0;
    double *row = (double *)malloc((size_t)R * sizeof(double));
    if (!row) { live_free(&L); return BEAM_LIVE_SELFCHECK_BUILD; }

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
    live_free(&L);
    return rc;
}

uint64_t beam_live_cache_bytes(const BeamModel *m) {
    if (!m || !m->live_cache) return 0;
    return m->live_cache->bytes;
}

uint32_t beam_live_cache_entries(const BeamModel *m) {
    if (!m || !m->live_cache) return 0;
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->live_cache->n_slots; ++i)
        if (m->live_cache->slot[i]) n++;
    return n;
}

static uint8_t **alloc_alive(uint32_t len, uint32_t n_states) {
    uint8_t **a = (uint8_t **)calloc(len + 1u, sizeof(uint8_t *));
    if (!a) return NULL;
    for (uint32_t i = 0; i <= len; ++i) {
        a[i] = (uint8_t *)calloc(n_states, 1);
        if (!a[i]) {
            for (uint32_t j = 0; j <= i; ++j) free(a[j]);
            free(a);
            return NULL;
        }
    }
    return a;
}

static void free_alive(uint8_t **a, uint32_t len) {
    if (!a) return;
    for (uint32_t i = 0; i <= len; ++i) free(a[i]);
    free(a);
}

/* ── Support size ────────────────────────────────────────────── */

/**
 * Support size at one length.
 *
 * Returns 0.0 for three different reasons — an unusable model, a length the
 * model does not generate, and an allocation failure — which a `double` cannot
 * distinguish. Each sets the thread-local error context, so a caller that
 * needs to tell them apart reads `beam_last_error()` after a zero:
 * `BEAM_OK` means the length genuinely has no paths.
 */

double beam_model_path_count(const BeamModel *m, uint32_t length) {
    if (!m || !m->finalized) {
        if (!m) beam_set_error(BEAM_ERR_NULL_ARG,
                              "path_count: model must not be NULL");
        else    (void)BEAM_ENOTBUILT();
        return 0.0;
    }
    beam_clear_error();
    if (!beam_length_live(m, length)) { (void)BEAM_EBADLEN(m, length); return 0.0; }

    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    LiveHandle h;
    if (live_acquire(m, length, &h) != BEAM_OK) return 0.0;
    const BeamLive *L = h.L;

    /*
     * No weights, anywhere in this sweep.
     *
     * Path counting never asks what an edge WEIGHS, only whether it exists,
     * and the frontier already reduced that to one bit per (state, residue)
     * while it was discovering reachability. `bits[s]` is the twenty-bit
     * residue mask of state `s` at the current layer, scattered out of the
     * frontier's `reached`-parallel list once per layer. On the 927M
     * foundation this took the sweep from 36.1 s to 0.97 s; the arithmetic is
     * untouched, because `w[x] == -INFINITY` and `bit x clear` are the same
     * predicate over the same weights.
     */
    uint32_t *bits = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    double *cur  = (double *)calloc(S, sizeof(double));
    /* calloc, then cleared only where it was last written: see the loop.
     * A full memset of S doubles per layer is 1.5 MB at order 4, which over
     * every layer of every length moves more memory than the sweep it serves. */
    double *next = (double *)calloc(S, sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!bits || !cur || !next || !tail) {
        free(bits); free(cur); free(next); free(tail);
        live_release(&h);
        (void)BEAM_EALLOC("the path-counting sweep buffers");
        return 0.0;
    }

    cur[S - 1] = live_bit_get(L->mask[0], S - 1) ? 1.0 : 0.0;
    for (uint32_t i = 0; i < length; ++i) {
        /* Scatter this layer's edge masks so the sweep can index them by
         * state. Only `reached` entries are written, and only a state that
         * carries mass is ever read -- mass implies live implies reached --
         * so the stale entries left from earlier layers are unreachable. */
        {
            const uint32_t *rd = L->reached[i];
            const uint32_t *eb = L->edge_bits[i];
            const uint32_t nr = L->n_reached[i];
            for (uint32_t k = 0; k < nr; ++k) bits[rd[k]] = eb[k];
        }
        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        /* `next` is the buffer that held layer i-1 before the last swap, so the
         * only entries that can be stale are the ones live at i-1. Clearing
         * exactly those keeps the invariant "everything not written this layer
         * is zero" at O(live) rather than O(S). */
        if (i >= 1) {
            const uint32_t np = L->n_live[i - 1];
            const uint32_t *pidx = L->live[i - 1];
            for (uint32_t k = 0; k < np; ++k) next[pidx[k]] = 0.0;
        }
        const uint32_t nlive = L->n_live[i + 1];
        const uint32_t *tidx = L->live[i + 1];
#ifdef _OPENMP
        /* `num_threads` is not optional, and `if` is not an optimisation.
         * Without the first this ignores beam_set_threads() and takes every
         * core the machine has. Without the second a team is created and torn
         * down once per layer even when the layer holds a single live state:
         * on a per-repertoire model that overhead WAS the whole runtime --
         * 0.600s of a 0.628s path_count, for about fifty microseconds of
         * arithmetic. The threshold is deliberately generous; below it the
         * serial loop is faster than the fork/join. */
        #pragma omp parallel for schedule(static) \
                num_threads(beam_threads()) if(nlive > 4096)
#endif
        for (int64_t k = 0; k < (int64_t)nlive; ++k) {
            const uint32_t t = tidx[k];
            const uint32_t x = t % A;
            if (x >= R) continue;
            const uint32_t rest = t / A;
            double acc = 0.0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                /* cur[src] != 0 implies src was live at this layer, so
                 * bits[src] was written above and holds its residue mask. */
                if (cur[src] == 0.0) continue;
                if (!((bits[src] >> x) & 1u)) continue;
                acc += cur[src];
            }
            next[t] = acc;
        }
        double *tmp = cur; cur = next; next = tmp;
    }
    beam_tail_weights(m, length, tail);
    double total = 0.0;
    for (uint32_t s = 0; s < S; ++s)
        if (tail[s] != -INFINITY) total += cur[s];

    free(bits); free(cur); free(next); free(tail);
    live_release(&h);
    return total;
}

/**
 * Live lengths, longest first, written into `out`. Returns how many there are.
 *
 * Longest first because the per-length dynamic programs are the unit of
 * parallel work and their cost is proportional to the length: handing out the
 * 25-layer job last leaves fifteen idle cores waiting for it, and handing it
 * out first hides it behind everything else. `out` must hold `n_len_slots`.
 */
static uint32_t live_lengths_desc(const BeamModel *m, uint32_t *out) {
    uint32_t n = 0;
    for (uint32_t L = m->n_len_slots; L-- > 1; )
        if (beam_length_live(m, L)) out[n++] = L;
    return n;
}

double beam_model_path_count_total(const BeamModel *m) {
    if (!m || !m->finalized) return 0.0;
    const uint32_t n = m->n_len_slots;
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *per = (double *)calloc(n, sizeof(double));
    if (!work || !per) {
        free(work); free(per);
        (void)BEAM_EALLOC("the per-length path counts");
        return 0.0;
    }
    const uint32_t nw = live_lengths_desc(m, work);
    /* The lengths are independent dynamic programs and this is where the
     * parallelism belongs: one whole DP per thread beats sixteen threads
     * inside one DP's layer, which has to synchronise once per layer. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k)
        per[work[k]] = beam_model_path_count(m, work[k]);
    /* Summed in ascending length, exactly as the serial loop did. */
    double t = 0.0;
    for (uint32_t L = 1; L < n; ++L) t += per[L];
    free(work); free(per);
    return t;
}

/* ── Topology ────────────────────────────────────────────────── */

BeamError beam_model_topology(const BeamModel *m, uint32_t length,
                                 double *out_nodes, double *out_edges,
                                 uint32_t *out_layer_nodes) {
    if (!m) return BEAM_ENULL("model");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);

    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    LiveHandle h;
    BeamError err = live_acquire(m, length, &h);
    if (err != BEAM_OK) return err;
    const BeamLive *L = h.L;

    uint32_t *bits = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    if (!bits) {
        live_release(&h);
        return BEAM_EALLOC("the topology sweep buffers");
    }

    /*
     * A node counts only if it is BOTH forward-reachable and able to reach a
     * live sink, which is what makes this the size of the graph a walk can
     * actually traverse rather than the size of the state space -- and that
     * intersection is precisely what the frontier's `live` list already is.
     *
     * The dense form of this counted `cur[s] > 0.0 && alive[i][s]` over all
     * 194,481 states per layer, where `cur > 0` marked "reachable along a
     * path of live states". The two agree state for state: a state that is
     * forward-reachable and backward-live has a witness path on which every
     * state is itself forward-reachable (a prefix of the witness) and
     * backward-live (it reaches the state, which reaches a sink), so the
     * dense sweep marks it; and `cur > 0` cannot hold for anything the
     * frontier failed to reach. `nodes_by_layer` is therefore unchanged, and
     * that is worth checking against a known model rather than believing --
     * `topology(13)` on the reference 14,615-sequence donor gives
     * 1,1,7,30,155,552,1281,2022,2534,2222,1423,631,237,88.
     */
    double nodes = 0.0;
    for (uint32_t i = 0; i <= length; ++i) {
        if (out_layer_nodes) out_layer_nodes[i] = L->n_live[i];
        nodes += (double)L->n_live[i];
    }

    /*
     * Edges, counted by target for the same reason path counting is: the 21
     * sources of a target are the target's private business, so no two
     * threads write the same accumulator. Both totals are counts, so summing
     * them in any order is exact and the reduction is safe.
     *
     * Restricting the targets to `live[i + 1]` drops nothing: a target with
     * at least one qualifying source is forward-reached by construction, and
     * the dense loop already required it to be backward-live, so the pairs
     * counted are the same pairs.
     */
    int64_t edges = 0;
    for (uint32_t i = 0; i < length; ++i) {
        {
            const uint32_t *rd = L->reached[i];
            const uint32_t *eb = L->edge_bits[i];
            const uint32_t nr = L->n_reached[i];
            for (uint32_t k = 0; k < nr; ++k) bits[rd[k]] = eb[k];
        }
        const uint64_t *mcur = L->mask[i];
        const uint32_t nlive = L->n_live[i + 1];
        const uint32_t *tidx = L->live[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:edges) \
                num_threads(beam_threads()) if(nlive > 4096)
#endif
        for (int64_t k = 0; k < (int64_t)nlive; ++k) {
            const uint32_t t = tidx[k];
            const uint32_t x = t % A;
            if (x >= R) continue;
            const uint32_t rest = t / A;
            int64_t hits = 0;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                /* Liveness first: only a live state is guaranteed to have had
                 * its mask scattered into `bits` this layer. */
                if (!live_bit_get(mcur, src)) continue;
                if (!((bits[src] >> x) & 1u)) continue;
                hits++;
            }
            edges += hits;
        }
    }

    if (out_nodes) *out_nodes = nodes;
    if (out_edges) *out_edges = (double)edges;
    free(bits);
    live_release(&h);
    return BEAM_OK;
}

/* ── Accumulator range ───────────────────────────────────────── */

/**
 * Max-plus sweep giving the largest value the spectrum DP must be able to
 * represent, which is the larger of two things that are NOT the same:
 *
 *   - the running accumulator, which grows monotonically along the path;
 *   - the final surprisal, which is the accumulator plus the per-length
 *     constant `ln Z(L) - ln pi(L) - tail`.
 *
 * Neither dominates in general, so the larger of the two is returned. On these
 * models the surprisal is the binding one — the sink term `-tail` is positive
 * and the per-length constant is a fraction of a nat either way — so bounding
 * only the accumulator would size a grid too small. A strongly negative
 * constant flips it, and a grid sized to the surprisal alone would then index
 * past its own top. Both are covered here.
 */
static double accumulator_max_at(const BeamModel *m, uint32_t len) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)malloc((size_t)S * sizeof(double));
    double *next = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !cur || !next || !tail) {
        free(w); free(cur); free(next); free(tail);
        (void)BEAM_EALLOC("the max-plus accumulator sweep buffers");
        return 0.0;
    }
    double hi = 0.0;
    for (uint32_t s = 0; s < S; ++s) cur[s] = -INFINITY;
    cur[S - 1] = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_layer_weights(m, i, len, w, NULL);
        for (uint32_t s = 0; s < S; ++s) next[s] = -INFINITY;
        for (uint32_t s = 0; s < S; ++s) {
            if (cur[s] == -INFINITY) continue;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY) continue;
                const double v = cur[s] - row[x];      /* surprisal accumulates */
                if (v > next[base + x]) next[base + x] = v;
            }
        }
        double *tmp = cur; cur = next; next = tmp;
    }
    beam_tail_weights(m, len, tail);
    {
        const double off = m->log_z[len] - m->log_pi[len];
        for (uint32_t s = 0; s < S; ++s) {
            if (cur[s] == -INFINITY) continue;
            if (cur[s] > hi) hi = cur[s];
            if (tail[s] == -INFINITY) continue;
            const double surprisal = cur[s] - tail[s] + off;
            if (surprisal > hi) hi = surprisal;
        }
    }
    free(w); free(cur); free(next); free(tail);
    return hi;
}

/**
 * Largest value a spectrum grid has to represent.
 *
 * Returns 0.0 both for a model that generates nothing and for an allocation
 * failure. Both set the thread-local error context, so `beam_last_error()`
 * separates them after a zero.
 */
double beam_model_accumulator_max(const BeamModel *m) {
    if (!m || !m->finalized) {
        if (!m) beam_set_error(BEAM_ERR_NULL_ARG,
                              "accumulator_max: model must not be NULL");
        else    (void)BEAM_ENOTBUILT();
        return 0.0;
    }
    beam_clear_error();
    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    if (!work) { (void)BEAM_EALLOC("the per-length work list"); return 0.0; }
    const uint32_t nw = live_lengths_desc(m, work);
    double hi = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:hi) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        const double v = accumulator_max_at(m, work[k]);
        if (v > hi) hi = v;
    }
    free(work);
    return hi;
}

/* ── Surprisal spectrum ──────────────────────────────────────── */

/**
 * Occupied bin range of one node's histogram, as a half-open interval.
 *
 * Carrying this is what makes the spectrum DP affordable. Early layers occupy
 * a handful of bins out of a thousand, and without bounds every edge would
 * sweep the whole grid: the production FlashBack transport
 * (`lib/flashback/graph_analytics.c:pseq_grid_transport`) keeps the same
 * bookkeeping for the same reason.
 */
typedef struct { uint32_t lo, hi; } BinRange;

/**
 * Add `src` shifted up by `u` bins into `dst`, splitting linearly between the
 * two adjacent bins so total mass is conserved and the mean is unbiased.
 * Source bins that fall outside the grid contribute nothing; that must hold at
 * BOTH ends, because the sink shift can be negative.
 *
 * `sr` bounds the source's occupied bins; `dr` is widened to cover whatever
 * this call writes.
 */
static void transport(const double *src, double *dst, uint32_t bins,
                      double u, double factor, BinRange sr, BinRange *dr) {
    const int64_t k = (int64_t)floor(u);
    const double  f = u - (double)k;
    const double  g = 1.0 - f;
    for (uint32_t b = sr.lo; b < sr.hi; ++b) {
        const double v = src[b];
        if (v == 0.0) continue;
        const int64_t lo = (int64_t)b + k;
        if (lo >= 0 && lo < (int64_t)bins) {
            dst[lo] += v * g * factor;
            if ((uint32_t)lo < dr->lo) dr->lo = (uint32_t)lo;
            if ((uint32_t)lo + 1u > dr->hi) dr->hi = (uint32_t)lo + 1u;
        }
        const int64_t hi = lo + 1;
        if (hi >= 0 && hi < (int64_t)bins) {
            dst[hi] += v * f * factor;
            if ((uint32_t)hi < dr->lo) dr->lo = (uint32_t)hi;
            if ((uint32_t)hi + 1u > dr->hi) dr->hi = (uint32_t)hi + 1u;
        }
    }
}

/**
 * Add `src` shifted up by `u` bins into `dst` for TWO measures at once,
 * splitting linearly between the two adjacent bins exactly as `transport`
 * does for one.
 *
 * The point is not to save a pass over the source bins — it is that the shift
 * `u`, and therefore the integer bin `k` and the split fractions `f` and `g`,
 * are computed ONCE and used for both. Two separate transports would compute
 * them twice from the same inputs and get the same answer today, but nothing
 * would hold them together, and the ratio of the two measures per bin is only
 * meaningful if the mass was split identically. That ratio is the exact
 * mass-weighted mean probability of the sequences in the bin, which is what
 * the discovery-curve arithmetic is built on.
 */
static void transport2(const double *src_a, const double *src_b,
                       double *dst_a, double *dst_b, uint32_t bins,
                       double u, double factor_a, double factor_b,
                       BinRange sr, BinRange *dr) {
    const int64_t k = (int64_t)floor(u);
    const double  f = u - (double)k;
    const double  g = 1.0 - f;
    for (uint32_t b = sr.lo; b < sr.hi; ++b) {
        const double va = src_a[b], vb = src_b[b];
        if (va == 0.0 && vb == 0.0) continue;
        const int64_t lo = (int64_t)b + k;
        if (lo >= 0 && lo < (int64_t)bins) {
            dst_a[lo] += va * g * factor_a;
            dst_b[lo] += vb * g * factor_b;
            if ((uint32_t)lo < dr->lo) dr->lo = (uint32_t)lo;
            if ((uint32_t)lo + 1u > dr->hi) dr->hi = (uint32_t)lo + 1u;
        }
        const int64_t hi = lo + 1;
        if (hi >= 0 && hi < (int64_t)bins) {
            dst_a[hi] += va * f * factor_a;
            dst_b[hi] += vb * f * factor_b;
            if ((uint32_t)hi < dr->lo) dr->lo = (uint32_t)hi;
            if ((uint32_t)hi + 1u > dr->hi) dr->hi = (uint32_t)hi + 1u;
        }
    }
}

/**
 * One length's surprisal spectrum, carrying either one measure or both.
 *
 * `out_b` NULL means one measure, governed by `counting`. Non-NULL means both
 * are carried through the same transport: `out_a` receives the counting
 * measure and `out_b` the generated one, and `counting` is ignored. Both
 * accumulate into whatever `out_a` / `out_b` already hold, so pooling over
 * lengths is a matter of calling this repeatedly.
 */
static BeamError spectrum_one(const BeamModel *m, bool counting,
                             uint32_t len, uint32_t bins, double ds,
                             double *out_a, double *out_b) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t nm = out_b ? 2u : 1u;      /* measures carried per state */

    uint8_t **alive = alloc_alive(len, S);
    if (!alive) return BEAM_EALLOC("the live-state map");
    BeamError err = trim(m, len, alive);
    if (err != BEAM_OK) { free_alive(alive, len); return err; }

    const size_t hist = (size_t)nm * bins;    /* doubles per state */
    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    double *cur  = (double *)calloc((size_t)S * hist, sizeof(double));
    double *next = (double *)calloc((size_t)S * hist, sizeof(double));
    BinRange *rc = (BinRange *)malloc((size_t)S * sizeof(BinRange));
    BinRange *rn = (BinRange *)malloc((size_t)S * sizeof(BinRange));
    if (!w || !tail || !cur || !next || !rc || !rn) {
        free(w); free(tail); free(cur); free(next); free(rc); free(rn);
        free_alive(alive, len);
        return BEAM_EALLOC("the per-state surprisal histograms");
    }

    for (uint32_t s = 0; s < S; ++s) { rc[s].lo = bins; rc[s].hi = 0; }
    for (uint32_t mi = 0; mi < nm; ++mi)
        cur[(size_t)(S - 1) * hist + (size_t)mi * bins + 0] = 1.0;
    rc[S - 1].lo = 0; rc[S - 1].hi = 1;

    for (uint32_t i = 0; i < len; ++i) {
        /* Pass the mask this function already computed. The transport loop
         * below tests `av_cur[s]` / `av_next[tt]` before reading a row, so the
         * "dead rows are never read" contract holds and skipping them is
         * bit-identical -- it was simply never wired up. Worth 1.36x on a
         * per-repertoire spectrum. */
        beam_layer_weights_live(m, i, len, w, NULL, alive[i]);
        /*
         * Accumulate by TARGET rather than by source, for the same reason the
         * forward sweeps do: 21 sources land on every target, so a
         * source-major scatter races. Gathering makes each target histogram
         * one thread's private property.
         *
         * It is also bit-identical rather than merely equivalent. The sources
         * of target `rest * A + x` are `a * state_shift + rest` for ascending
         * `a`, which is ascending in the source state too, so the gather adds
         * them in exactly the order the scatter did.
         *
         * This is the single most expensive loop in the library — a whole
         * histogram is moved per edge, not a number — and it was the one loop
         * of its kind still running on one core.
         */
        const uint8_t *av_cur = alive[i], *av_next = alive[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(beam_threads())
#endif
        for (int64_t tt = 0; tt < (int64_t)S; ++tt) {
            rn[tt].lo = bins; rn[tt].hi = 0;
            const uint32_t x = (uint32_t)tt % A;
            if (x >= R || !av_next[tt]) continue;
            const uint32_t rest = (uint32_t)tt / A;
            double *dst = next + (size_t)tt * hist;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t s = a * SK + rest;
                if (rc[s].lo >= rc[s].hi) continue;    /* no mass here */
                if (!av_cur[s]) continue;
                const double wv = w[(size_t)s * R + x];
                if (wv == -INFINITY) continue;
                const double *src = cur + (size_t)s * hist;
                if (nm == 1u) {
                    transport(src, dst, bins, -wv / ds,
                              counting ? 1.0 : exp(wv), rc[s], &rn[tt]);
                } else {
                    transport2(src, src + bins, dst, dst + bins, bins,
                               -wv / ds, 1.0, exp(wv), rc[s], &rn[tt]);
                }
            }
        }
        /* Clear only what was written, so the sweep costs occupied bins
         * rather than the whole grid. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) {
            if (rc[s].lo >= rc[s].hi) continue;
            for (uint32_t mi = 0; mi < nm; ++mi)
                memset(cur + (size_t)s * hist + (size_t)mi * bins + rc[s].lo, 0,
                       (size_t)(rc[s].hi - rc[s].lo) * sizeof(double));
        }
        { double *t = cur; cur = next; next = t; }
        { BinRange *t = rc; rc = rn; rn = t; }
    }

    /* Sink: the backward tail, plus the per-length constant that turns the
     * accumulated weight into a surprisal. */
    beam_tail_weights(m, len, tail);
    {
        const double off = m->log_z[len] - m->log_pi[len];
        BinRange sink = { bins, 0 };
        for (uint32_t s = 0; s < S; ++s) {
            if (tail[s] == -INFINITY || rc[s].lo >= rc[s].hi) continue;
            const double *src = cur + (size_t)s * hist;
            const double u = (-tail[s] + off) / ds;
            if (nm == 1u) {
                transport(src, out_a, bins, u,
                          counting ? 1.0 : exp(tail[s] - off), rc[s], &sink);
            } else {
                transport2(src, src + bins, out_a, out_b, bins, u,
                           1.0, exp(tail[s] - off), rc[s], &sink);
            }
        }
    }

    free(w); free(tail); free(cur); free(next); free(rc); free(rn);
    free_alive(alive, len);
    return BEAM_OK;
}

/**
 * Grid spacing for a spectrum, sized automatically when `spacing` is not
 * positive. Shared by every spectrum entry point so they cannot disagree
 * about what grid they are on — which for the paired and per-length forms is
 * not a tidiness point but the property that makes their outputs comparable.
 */
static BeamError spectrum_spacing(const BeamModel *m, uint32_t bins,
                                 double spacing, double *out_ds) {
    if (bins < 2)
        return BEAM_ERANGE("%s: a spectrum needs at least 2 bins, got %u",
                               __func__, (unsigned)bins);
    if (spacing > 0.0) { *out_ds = spacing; return BEAM_OK; }

    const double hi = beam_model_accumulator_max(m);
    if (!(hi > 0.0))
        return BEAM_ENOPATHS("the accumulator bound is zero, so the "
                                 "model generates nothing to bin");
    /*
     * Headroom above the exact bound, and it has to scale with length.
     *
     * Linear splitting is a smoothing operation: each layer can push a sliver
     * of mass one bin past where the exact arithmetic would put it, so after L
     * layers plus the sink the occupied range can run up to L+1 bins beyond
     * `hi`. Reserving a fixed handful of bins is enough for short sequences
     * and silently truncates long ones.
     */
    const uint32_t reserve = m->max_len + 9u;
    if (bins <= reserve)
        return BEAM_ERANGE("%s: %u bins is not enough to size a grid "
                               "automatically at max_length %u; %u are "
                               "reserved for splitting headroom, or pass an "
                               "explicit spacing", __func__, (unsigned)bins,
                               (unsigned)m->max_len, (unsigned)reserve);
    *out_ds = hi * 1.02 / (double)(bins - reserve);
    return BEAM_OK;
}

BeamError beam_model_spectrum(const BeamModel *m, bool counting,
                                 uint32_t length, uint32_t bins, double spacing,
                                 double *weights, double *out_spacing) {
    if (!m || !weights) return BEAM_ENULL(!m ? "model" : "weights buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    double ds = 0.0;
    BeamError e = spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(weights, 0, (size_t)bins * sizeof(double));
    if (length != 0) {
        if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
        return spectrum_one(m, counting, length, bins, ds, weights, NULL);
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        e = spectrum_one(m, counting, L, bins, ds, weights, NULL);
        if (e != BEAM_OK) return e;
    }
    return BEAM_OK;
}

BeamError beam_model_spectrum_pair(const BeamModel *m, uint32_t length,
                                      uint32_t bins, double spacing,
                                      double *counting_weights,
                                      double *generated_weights,
                                      double *out_spacing) {
    if (!m || !counting_weights || !generated_weights)
        return BEAM_ENULL(!m ? "model"
                                 : (!counting_weights ? "counting weights"
                                                      : "generated weights"));
    if (!m->finalized) return BEAM_ENOTBUILT();

    double ds = 0.0;
    BeamError e = spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(counting_weights,  0, (size_t)bins * sizeof(double));
    memset(generated_weights, 0, (size_t)bins * sizeof(double));
    if (length != 0) {
        if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
        return spectrum_one(m, false, length, bins, ds,
                            counting_weights, generated_weights);
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        e = spectrum_one(m, false, L, bins, ds,
                         counting_weights, generated_weights);
        if (e != BEAM_OK) return e;
    }
    return BEAM_OK;
}

/**
 * Scratch one length of the spectrum DP needs, in bytes.
 *
 * Dominated by the two per-state histogram grids, which at order 4 with 512
 * bins are 800 MB each. That is why the per-length form caps its concurrency
 * by memory rather than by core count: sixteen lengths at once would ask for
 * twenty-five gigabytes.
 */
static uint64_t spectrum_bytes(const BeamModel *m, uint32_t bins,
                               uint32_t nm) {
    const uint64_t S = m->n_states;
    return 2ull * S * nm * bins * sizeof(double)      /* cur + next   */
         + S * BEAM_RESIDUES * sizeof(double)     /* edge weights */
         + S * (2ull * sizeof(BinRange) + 2ull * sizeof(double))
         + (uint64_t)(m->max_len + 1u) * S;           /* live-state map */
}

BeamError beam_model_spectra_by_length(const BeamModel *m, bool counting,
                                          uint32_t max_length, uint32_t bins,
                                          double spacing, double *weights,
                                          double *out_spacing) {
    if (!m || !weights) return BEAM_ENULL(!m ? "model" : "weights buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (max_length > m->max_len)
        return BEAM_ERANGE("%s: length %u is beyond this model's "
                               "max_length of %u", __func__,
                               (unsigned)max_length, (unsigned)m->max_len);
    if (max_length == 0) max_length = m->max_len;

    double ds = 0.0;
    BeamError e = spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(weights, 0, (size_t)(max_length + 1u) * bins * sizeof(double));

    uint32_t *work = (uint32_t *)malloc((size_t)(max_length + 1u) * sizeof(uint32_t));
    if (!work) return BEAM_EALLOC("the per-length work list");
    uint32_t nw = 0;
    for (uint32_t L = max_length + 1u; L-- > 1; )
        if (beam_length_live(m, L)) work[nw++] = L;
    if (nw == 0) { free(work); return BEAM_OK; }

    /*
     * Which axis to parallelise, decided by MEMORY.
     *
     * Each length needs its own pair of per-state histogram grids, and at
     * order 4 with 512 bins that is 1.6 GB apiece: sixteen at once would ask
     * for twenty-five gigabytes and find the swap instead of the speed-up.
     *
     * When the budget cannot hold one length per thread, the lengths run one
     * at a time and each gets every thread INSIDE its own sweeps, which are
     * parallel in their own right. That is not a compromise, it is the faster
     * of the two: OpenMP does not nest by default, so an outer loop of two
     * threads would put the whole job on two cores. Running the lengths
     * concurrently is worth it only when every thread can have one.
     *
     * Either way the answers are identical — the rows are written
     * independently — so this decides only speed and peak memory.
     */
    const uint64_t need = spectrum_bytes(m, bins, 1u);
    const int th = beam_threads();
    uint32_t conc = (uint32_t)(need ? beam_model_memory_budget() / need
                                    : (uint64_t)nw);
    if (conc > nw) conc = nw;
    if (th > 0 && conc > (uint32_t)th) conc = (uint32_t)th;
    if (th > 1 && conc < (uint32_t)th && conc < nw) conc = 1u;
    if (conc < 1u) conc = 1u;

    int failed = BEAM_OK;
    if (conc <= 1u) {
        /* No enclosing parallel region at all: one with a single thread is
         * still an ACTIVE region, and would serialise every sweep inside it. */
        for (uint32_t k = 0; k < nw; ++k) {
            const BeamError err = spectrum_one(m, counting, work[k], bins, ds,
                                              weights + (size_t)work[k] * bins,
                                              NULL);
            if (err != BEAM_OK && (int)err > failed) failed = (int)err;
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
                num_threads((int)conc)
#endif
        for (int64_t k = 0; k < (int64_t)nw; ++k) {
            const uint32_t L = work[k];
            const BeamError err = spectrum_one(m, counting, L, bins, ds,
                                              weights + (size_t)L * bins, NULL);
            if (err != BEAM_OK) { if ((int)err > failed) failed = (int)err; }
        }
    }
    free(work);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                        "spectra_by_length: a per-length sweep failed");
    return BEAM_OK;
}

/* ── Entropy and Hill numbers ────────────────────────────────── */

/**
 * Tilted partition function `sum over length-L paths of exp(alpha * raw)`.
 *
 * `alpha = 0` counts paths, `alpha = 1` is the ordinary partition function,
 * and general `alpha` gives the power sums the Hill numbers are built from.
 */
static BeamError tilted_log_z(const BeamModel *m, uint32_t len, double alpha,
                             double *out) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w    = (double *)malloc((size_t)S * R * sizeof(double));
    double *cur  = (double *)calloc(S, sizeof(double));
    double *next = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !cur || !next || !tail) {
        free(w); free(cur); free(next); free(tail);
        return BEAM_EALLOC("the tilted forward-sweep buffers");
    }
    cur[S - 1] = 1.0;
    double log_scale = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        double wmax = -INFINITY;
        beam_layer_weights(m, i, len, w, &wmax);
        if (wmax == -INFINITY) goto dead;
        const double amax = alpha * wmax;

        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        double nmax = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
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
                acc += v * exp(alpha * wv - amax);
            }
            next[t] = acc;
            if (acc > nmax) nmax = acc;
        }
        log_scale += amax;
        if (!(nmax > 0.0)) goto dead;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) next[s] /= nmax;
        log_scale += log(nmax);
        double *tmp = cur; cur = next; next = tmp;
    }

    beam_tail_weights(m, len, tail);
    {
        double tmax = -INFINITY;
        for (uint32_t s = 0; s < S; ++s)
            if (cur[s] > 0.0 && tail[s] != -INFINITY && alpha * tail[s] > tmax)
                tmax = alpha * tail[s];
        if (tmax == -INFINITY) goto dead;
        double acc = 0.0;
        for (uint32_t s = 0; s < S; ++s) {
            if (cur[s] == 0.0 || tail[s] == -INFINITY) continue;
            acc += cur[s] * exp(alpha * tail[s] - tmax);
        }
        *out = log_scale + tmax + log(acc);
    }
    free(w); free(cur); free(next); free(tail);
    return BEAM_OK;
dead:
    free(w); free(cur); free(next); free(tail);
    *out = -INFINITY;
    return BEAM_OK;
}

/**
 * Expected accumulated edge weight under the length-`L` conditional, by the
 * expectation semiring: carry both the path mass and the mass-weighted sum of
 * accumulated weight, and divide at the sink. Exact, one forward sweep.
 */
static BeamError expected_raw(const BeamModel *m, uint32_t len, double *out) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w  = (double *)malloc((size_t)S * R * sizeof(double));
    double *a  = (double *)calloc(S, sizeof(double));
    double *b  = (double *)calloc(S, sizeof(double));
    double *na = (double *)malloc((size_t)S * sizeof(double));
    double *nb = (double *)malloc((size_t)S * sizeof(double));
    double *tail = (double *)malloc((size_t)S * sizeof(double));
    if (!w || !a || !b || !na || !nb || !tail) {
        free(w); free(a); free(b); free(na); free(nb); free(tail);
        return BEAM_EALLOC("the expectation-semiring buffers");
    }
    a[S - 1] = 1.0;

    for (uint32_t i = 0; i < len; ++i) {
        double wmax = -INFINITY;
        beam_layer_weights(m, i, len, w, &wmax);
        if (wmax == -INFINITY) { *out = 0.0; goto out; }

        /* Accumulate by TARGET rather than by source. The scatter
         * `next[(s % SK) * A + x] += ...` has 21 sources per target, so
         * parallelising over sources would race; inverting it makes each
         * target the private property of one thread. */
        double nmax = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < (int64_t)S; ++t) {
            const uint32_t x = (uint32_t)t % A;
            if (x >= R) { na[t] = 0.0; nb[t] = 0.0; continue; }
            const uint32_t rest = (uint32_t)t / A;
            double sa = 0.0, sb = 0.0;
            for (uint32_t k = 0; k < A; ++k) {
                const uint32_t src = k * SK + rest;
                if (a[src] == 0.0 && b[src] == 0.0) continue;
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                const double e = exp(wv - wmax);
                sa += a[src] * e;
                sb += (b[src] + a[src] * wv) * e;
            }
            na[t] = sa; nb[t] = sb;
            if (sa > nmax) nmax = sa;
        }
        /* Rescale both accumulators by the same factor: the ratio is what we
         * want, so the scale cancels and never has to be tracked. */
        if (!(nmax > 0.0)) { *out = 0.0; goto out; }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < (int64_t)S; ++s) { na[s] /= nmax; nb[s] /= nmax; }
        { double *t; t = a; a = na; na = t; t = b; b = nb; nb = t; }
    }

    beam_tail_weights(m, len, tail);
    {
        double tmax = -INFINITY;
        for (uint32_t s = 0; s < S; ++s)
            if (a[s] > 0.0 && tail[s] != -INFINITY && tail[s] > tmax) tmax = tail[s];
        if (tmax == -INFINITY) { *out = 0.0; goto out; }
        double A_ = 0.0, B_ = 0.0;
        for (uint32_t s = 0; s < S; ++s) {
            if (tail[s] == -INFINITY || (a[s] == 0.0 && b[s] == 0.0)) continue;
            const double e = exp(tail[s] - tmax);
            A_ += a[s] * e;
            B_ += (b[s] + a[s] * tail[s]) * e;
        }
        *out = (A_ > 0.0) ? B_ / A_ : 0.0;
    }
out:
    free(w); free(a); free(b); free(na); free(nb); free(tail);
    return BEAM_OK;
}

BeamError beam_model_entropy(const BeamModel *m, double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    const uint32_t n = m->n_len_slots;
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    double *per = (double *)calloc(n, sizeof(double));
    if (!work || !per) {
        free(work); free(per);
        return BEAM_EALLOC("the per-length expected weights");
    }
    uint32_t nw = 0;
    for (uint32_t L = n; L-- > 1; )
        if (beam_length_live(m, L) && exp(m->log_pi[L]) > 0.0) work[nw++] = L;

    /* `BEAM_OK` is zero and every failure code is positive, so a max-reduction
     * both avoids a data race on the flag and picks the same code however the
     * iterations are distributed. */
    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double e = 0.0;
        const BeamError err = expected_raw(m, work[k], &e);
        if (err != BEAM_OK) { if ((int)err > failed) failed = (int)err; }
        else per[work[k]] = e;
    }
    if (failed != BEAM_OK) {
        free(work); free(per);
        return BEAM_FAIL((BeamError)failed,
                        "entropy: a per-length sweep could not be computed");
    }
    /* Accumulated in ascending length, exactly as the serial loop did. */
    double h = 0.0;
    for (uint32_t L = 1; L < n; ++L) {
        if (!beam_length_live(m, L)) continue;
        const double pi = exp(m->log_pi[L]);
        if (!(pi > 0.0)) continue;
        h -= pi * (m->log_pi[L] - m->log_z[L] + per[L]);
    }
    free(work); free(per);
    *out = h;
    return BEAM_OK;
}

BeamError beam_model_hill_number(const BeamModel *m, double alpha,
                                    double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();

    if (fabs(alpha - 1.0) < 1e-12) {
        double h = 0.0;
        BeamError e = beam_model_entropy(m, &h);
        if (e != BEAM_OK) return e;
        *out = exp(h);
        return BEAM_OK;
    }
    if (alpha == 0.0) {
        *out = beam_model_path_count_total(m);
        return BEAM_OK;
    }

    /* sum_s P(s)^alpha = sum_L (pi(L)/Z(L))^alpha * Z_alpha(L), summed in the
     * log domain so a long tail of tiny lengths cannot underflow the total. */
    double mx = -INFINITY;
    double *terms = (double *)malloc(m->n_len_slots * sizeof(double));
    if (!terms) return BEAM_EALLOC("the per-length Hill terms");
    for (uint32_t L = 0; L < m->n_len_slots; ++L) terms[L] = -INFINITY;

    uint32_t *work = (uint32_t *)malloc((size_t)m->n_len_slots * sizeof(uint32_t));
    double *za = (double *)malloc((size_t)m->n_len_slots * sizeof(double));
    if (!work || !za) {
        free(work); free(za); free(terms);
        return BEAM_EALLOC("the per-length tilted partition functions");
    }
    const uint32_t nw = live_lengths_desc(m, work);
    int failed = BEAM_OK;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t k = 0; k < (int64_t)nw; ++k) {
        double v = 0.0;
        const BeamError e = tilted_log_z(m, work[k], alpha, &v);
        if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
        else za[work[k]] = v;
    }
    if (failed != BEAM_OK) {
        free(work); free(za); free(terms);
        return BEAM_FAIL((BeamError)failed,
                        "hill_number: a per-length tilted sweep failed");
    }
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        if (za[L] == -INFINITY) continue;
        terms[L] = alpha * (m->log_pi[L] - m->log_z[L]) + za[L];
        if (terms[L] > mx) mx = terms[L];
    }
    free(work); free(za);
    if (mx == -INFINITY) { free(terms); *out = 0.0; return BEAM_OK; }
    double acc = 0.0;
    for (uint32_t L = 0; L < m->n_len_slots; ++L)
        if (terms[L] != -INFINITY) acc += exp(terms[L] - mx);
    free(terms);

    const double log_sum = mx + log(acc);
    *out = exp(log_sum / (1.0 - alpha));
    return BEAM_OK;
}
