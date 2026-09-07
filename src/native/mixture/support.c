/**
 * @file support.c
 * @brief `|union of the members' supports|`, by a bitmask forward sweep.
 *
 * The q = 0 corner of the mixture. `pbar(s) > 0` exactly when SOME SINGLE
 * member spells every edge of `s`, so the support of the mixture is the union
 * of the members' supports -- and that is a different set from the support of
 * a model trained on the pooled counts, which may splice an edge learned from
 * one member onto an edge learned from another. `D0(union) <= D0(merged)`
 * always. See the header for the measured size of that gap.
 *
 * ## Why a bitmask and not inclusion-exclusion
 *
 * |union| = sum over nonempty subsets T of (-1)^(|T|+1) |intersection over T|,
 * and every one of those intersections is a path count the product sweep next
 * door already computes. It would be a dozen lines. It is also useless: the
 * terms alternate in sign, each is astronomically large -- a support of 1e10
 * is ordinary here -- and the union is frequently many orders of magnitude
 * below the leading term. Double precision has 16 digits and the cancellation
 * eats all of them.
 *
 * So the sweep carries, per trellis state, the set of members still consistent
 * with the path taken to reach it, and counts paths per (state, mask) pair.
 * Every sequence arrives at exactly one such pair -- the members that spell it
 * are a property of the sequence, not of the route -- so nothing is counted
 * twice and nothing has to be subtracted back off. The sweep only ever adds
 * positive numbers.
 *
 * ## What actually limits M
 *
 * The mask is a `uint64_t`, so 64 members is structural. It is almost never
 * what binds. The cost is the number of DISTINCT masks alive per state, which
 * grows with how much the members disagree, and a cohort of unrelated donors
 * reaches the cap far below 64. `out_peak` reports the largest population any
 * layer held so a caller can see where it stands rather than guess; exceeding
 * the cap is BEAM_ERR_BUDGET, never a silently degraded answer.
 */
#include "beam/internal.h"
#include "beam/mixture.h"


/* ── The (state, mask) table ─────────────────────────────────── */

/*
 * Open addressing with linear probing, and a mask of 0 marking an empty slot.
 * A live mask is never 0 -- a path with no member left consistent is dropped
 * at the transition that emptied it -- so no separate occupancy array is
 * needed and a cleared slot is a zeroed word.
 *
 * `slots` is always twice `cap`, so the load factor cannot pass 0.5 and the
 * probe stays short.
 *
 * The table STARTS SMALL AND DOUBLES, up to a `limit` fixed from the memory
 * budget before the sweep begins. Sizing it at the limit from the start was
 * the first version and it was wrong in the way that matters: a two-member
 * mixture over a 9,000-state trellis would reserve gigabytes of address space
 * it could never populate, and under the address-space ceiling everything here
 * runs beneath, reserving is failing. Growth keeps the footprint proportional
 * to the answer while the budget still bounds the worst case.
 *
 * Growth is bounded rather than free, so a caller still learns that the
 * members disagree too much for an exact union instead of watching the machine
 * swap.
 */
typedef struct {
    uint32_t *state;
    uint64_t *mask;      /* 0 marks empty                                   */
    double   *count;
    uint64_t *used;      /* slot indices in insertion order, for clearing   */
    uint64_t  n_used;
    uint64_t  slots;     /* power of two, always 2 * cap                    */
    uint64_t  cap;       /* population the current allocation holds         */
    uint64_t  limit;     /* population ceiling; growth stops here           */
} MaskTable;

static inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void mt_free(MaskTable *t) {
    free(t->state); free(t->mask); free(t->count); free(t->used);
    t->state = NULL; t->mask = NULL; t->count = NULL; t->used = NULL;
}

/** The smallest power of two at least `v`. The probe mask needs one. */
static inline uint64_t next_pow2(uint64_t v) {
    uint64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static bool mt_alloc(MaskTable *t, uint64_t cap, uint64_t limit) {
    memset(t, 0, sizeof(*t));
    t->cap   = cap;
    t->limit = limit;
    /* `mt_insert` masks with `slots - 1`, so this must be a power of two --
     * NOT simply `2 * cap`, which a caller-supplied ceiling need not make one.
     * At least twice `cap`, so the load factor cannot pass 0.5 and an empty
     * slot always exists for the probe to stop at. */
    t->slots = next_pow2(2ull * cap);
    t->state = (uint32_t *)malloc(t->slots * sizeof(uint32_t));
    t->mask  = (uint64_t *)calloc(t->slots, sizeof(uint64_t));
    t->count = (double   *)malloc(t->slots * sizeof(double));
    t->used  = (uint64_t *)malloc(cap * sizeof(uint64_t));
    if (!t->state || !t->mask || !t->count || !t->used) { mt_free(t); return false; }
    return true;
}

/** Accumulate into (s, mk). False only when the allocation is full. */
static inline bool mt_insert(MaskTable *t, uint32_t s, uint64_t mk, double c) {
    uint64_t i = mix64(((uint64_t)s * 0x100000001B3ull) ^ mk) & (t->slots - 1u);
    for (;;) {
        if (t->mask[i] == 0ull) break;
        if (t->mask[i] == mk && t->state[i] == s) { t->count[i] += c; return true; }
        i = (i + 1u) & (t->slots - 1u);
    }
    if (t->n_used >= t->cap) return false;
    t->mask[i]  = mk;
    t->state[i] = s;
    t->count[i] = c;
    t->used[t->n_used++] = i;
    return true;
}

/** Double the allocation and rehash. False at the ceiling, or out of memory. */
static bool mt_grow(MaskTable *t) {
    if (t->cap >= t->limit) return false;
    uint64_t ncap = t->cap * 2ull;
    if (ncap > t->limit) ncap = t->limit;
    MaskTable n;
    if (!mt_alloc(&n, ncap, t->limit)) return false;
    /* Every (state, mask) in the old table is distinct, so no reinsertion can
     * merge two entries, and `ncap > n_used` means none can fail. */
    for (uint64_t j = 0; j < t->n_used; ++j) {
        const uint64_t slot = t->used[j];
        (void)mt_insert(&n, t->state[slot], t->mask[slot], t->count[slot]);
    }
    mt_free(t);
    *t = n;
    return true;
}

/** Accumulate into (s, mk). False when the population ceiling is reached. */
static inline bool mt_add(MaskTable *t, uint32_t s, uint64_t mk, double c) {
    if (mt_insert(t, s, mk, c)) return true;
    if (!mt_grow(t)) return false;
    return mt_insert(t, s, mk, c);
}

/** Clear only the slots that were written, not the whole table. */
static inline void mt_clear(MaskTable *t) {
    for (uint64_t j = 0; j < t->n_used; ++j) t->mask[t->used[j]] = 0ull;
    t->n_used = 0;
}


/* ── Sizing ──────────────────────────────────────────────────── */

/**
 * Bytes the sweep holds that do NOT depend on the mask population.
 *
 * The edge-liveness bitset is the same size as one layer of edge weights, one
 * `uint64_t` per (state, residue) against one `double`. That is what makes M
 * free in the layer: whether one member or sixty-four can spell an edge costs
 * the same word.
 */
static inline uint64_t union_base_bytes(uint32_t S, uint32_t R) {
    return (uint64_t)S * R * sizeof(uint64_t)   /* per-edge member bitset   */
         + (uint64_t)S * R * sizeof(double)     /* one layer of weights     */
         + (uint64_t)S * sizeof(uint64_t)       /* per-state sink bitset    */
         + (uint64_t)S * sizeof(double)         /* one sink weight vector   */
         + 2ull * S                             /* two byte masks           */
         + (uint64_t)S * sizeof(uint32_t);      /* the mass-carrying states */
}

/** Bytes ONE table holds at a population ceiling of `cap`. */
static inline uint64_t union_table_bytes(uint64_t cap) {
    return next_pow2(2ull * cap)
             * (sizeof(uint32_t) + sizeof(uint64_t) + sizeof(double))
         + cap * sizeof(uint64_t);
}

/**
 * Smallest population ceiling worth DERIVING from the budget.
 *
 * A sweep that cannot hold a few thousand pairs cannot answer anything useful,
 * and letting it start only to refuse three layers in would waste the setup
 * and report the failure at a length rather than at the budget. Refused up
 * front instead. Applies only to a ceiling this file works out for itself; a
 * caller who names one gets exactly that.
 */
#define BEAM_UNION_MIN_CAP 4096ull

/**
 * Where a derived ceiling begins allocating, before any doubling.
 *
 * Small on purpose. Most mixtures are small -- a pair of toy models peaks at
 * 64 live pairs -- and reserving the whole derived ceiling for them wastes
 * address space that the sweep runs under a hard cap on. Growth is geometric,
 * so reaching a hundred thousand pairs from here costs about twice the final
 * table in total rehashing, against a sweep that is orders of magnitude more
 * work.
 *
 * It is also what makes the growth path REACHABLE from an ordinary test. At
 * 4096 nothing in the suite ever doubled, and a rehash nobody runs is a rehash
 * nobody has checked.
 */
#define BEAM_UNION_START_CAP 64ull


/* ── The sweep ───────────────────────────────────────────────── */

BeamError beam_mixture_support_union(const BeamModel *const *models,
                                     uint32_t n_models, uint32_t len,
                                     uint64_t cap, double *out,
                                     uint64_t *out_peak) {
    if (!out) return BEAM_ENULL("output buffer");
    const BeamError shape = beam_mixture_check_shape(models, n_models);
    if (shape != BEAM_OK) return shape;

    const BeamModel *m0 = models[0];
    if (len < 1u || len > m0->max_len) return BEAM_EBADLEN(m0, len);

    const uint32_t A  = m0->n_alphabet;
    const uint32_t R  = m0->n_residues;
    const uint32_t S  = m0->n_states;
    const uint32_t SK = m0->state_shift;

    /* The members that produce this length at all. A member whose length prior
     * is zero here contributes no sequence of this length however much of the
     * trellis it can spell, so it never enters a mask. */
    uint64_t seed = 0ull;
    for (uint32_t f = 0; f < n_models; ++f)
        if (beam_length_live(models[f], len)) seed |= 1ull << f;
    if (seed == 0ull) { *out = 0.0; if (out_peak) *out_peak = 0; return BEAM_OK; }

    /*
     * The ceiling, in pairs. `cap == 0` means "as much as the budget allows",
     * derived here rather than guessed at a fixed number so that the one knob
     * the library exposes governs this sweep the way it governs every other.
     *
     * Derived by doubling from the floor while the result still fits, so the
     * figure the budget is charged is the figure the table can actually reach.
     * The first version divided the remaining budget by the bytes per pair and
     * charged the quotient, which `next_pow2` inside `union_table_bytes` then
     * rounded straight past the budget -- and a two-member toy mixture was
     * refused on a 4 GB budget.
     */
    const uint64_t base = union_base_bytes(S, R);
    const bool explicit_cap = (cap != 0ull);
    if (cap == 0ull) {
        const uint64_t budget = beam_model_memory_budget();
        const uint64_t room = budget > base ? budget - base : 0ull;
        if (2ull * union_table_bytes(BEAM_UNION_MIN_CAP) > room)
            return beam_budget_refuse("a mixture support-union sweep", len,
                base + 2ull * union_table_bytes(BEAM_UNION_MIN_CAP));
        cap = BEAM_UNION_MIN_CAP;
        while (2ull * union_table_bytes(cap * 2ull) <= room) cap *= 2ull;
    } else {
        /* An explicit ceiling is honoured EXACTLY. It is a deliberate
         * instruction -- usually "prove to me that this refuses" -- and
         * quietly raising it to the floor below would make the refusal path
         * unreachable and untestable. The floor applies only to a ceiling this
         * function derived for itself. */
        BEAM_BUDGET_OR_REFUSE("a mixture support-union sweep", len,
                              base + 2ull * union_table_bytes(cap));
    }

    LiveHandle h[BEAM_MIXTURE_MAX_MEMBERS];
    const BeamLive *L[BEAM_MIXTURE_MAX_MEMBERS];
    uint32_t got = 0;
    for (uint32_t f = 0; f < n_models; ++f) {
        if (!(seed & (1ull << f))) { L[f] = NULL; continue; }
        if (beam_live_acquire(models[f], len, &h[got]) != BEAM_OK) {
            for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
            return BEAM_EALLOC("the forward frontier");
        }
        L[f] = h[got].L;
        ++got;
    }

    uint64_t *lm  = (uint64_t *)malloc((size_t)S * R * sizeof(uint64_t));
    double   *w   = (double   *)malloc((size_t)S * R * sizeof(double));
    uint64_t *tm  = (uint64_t *)calloc(S, sizeof(uint64_t));
    double   *tl  = (double   *)malloc((size_t)S * sizeof(double));
    uint8_t  *mm  = (uint8_t  *)calloc(S, 1);
    uint8_t  *fm  = (uint8_t  *)malloc(S);
    uint32_t *ms  = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    MaskTable cur = {0}, nxt = {0};
    /*
     * A ceiling this file derived starts small and doubles; a ceiling the
     * caller named is allocated outright. The second is what the budget was
     * charged for a few lines above, so allocating it makes the charge and the
     * footprint the same number -- and it gives a test a way to run the same
     * sweep with the growth path and without it, which is the only check the
     * rehash gets.
     */
    const uint64_t cap0 = explicit_cap
                        ? cap
                        : (cap < BEAM_UNION_START_CAP ? cap
                                                      : BEAM_UNION_START_CAP);

    if (!lm || !w || !tm || !tl || !mm || !fm || !ms
        || !mt_alloc(&cur, cap0, cap) || !mt_alloc(&nxt, cap0, cap)) {
        free(lm); free(w); free(tm); free(tl); free(mm); free(fm); free(ms);
        mt_free(&cur); mt_free(&nxt);
        for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
        return BEAM_EALLOC("the mixture support-union buffers");
    }

    BeamError rc = BEAM_OK;
    uint64_t peak = 0;
    double total = 0.0;
    uint32_t overflow_at = 0;

    (void)mt_add(&cur, S - 1u, seed, 1.0);      /* the root, all members live */

    for (uint32_t i = 0; i < len; ++i) {
        /*
         * The states this layer actually carries mass at, deduplicated across
         * the masks sharing each one. Only their rows of the edge bitset are
         * cleared and rebuilt, which is what keeps this cheaper than a memset
         * of the whole 31 MB layer once per layer per length.
         */
        uint32_t n_ms = 0;
        for (uint64_t j = 0; j < cur.n_used; ++j) {
            const uint32_t s = cur.state[cur.used[j]];
            if (!mm[s]) { mm[s] = 1u; ms[n_ms++] = s; }
        }
        if (n_ms == 0u) break;
        for (uint32_t j = 0; j < n_ms; ++j)
            memset(lm + (size_t)ms[j] * R, 0, (size_t)R * sizeof(uint64_t));

        /* One bit per member per edge. Built member by member into a single
         * weight layer, so the footprint does not grow with M. A member is
         * only ever asked about states it reaches: elsewhere its bit must stay
         * clear, and `beam_layer_weights_live` leaves rows it skips UNTOUCHED
         * rather than filling them with -INFINITY, so reading one would be
         * reading a previous layer. */
        for (uint32_t f = 0; f < n_models; ++f) {
            if (!L[f]) continue;
            memset(fm, 0, S);
            const uint32_t  nr = L[f]->n_reached[i];
            const uint32_t *rv = L[f]->reached[i];
            uint32_t any = 0;
            for (uint32_t j = 0; j < nr; ++j)
                if (mm[rv[j]]) { fm[rv[j]] = 1u; ++any; }
            if (!any) continue;
            beam_layer_weights_live(models[f], i, len, w, NULL, fm);
            const uint64_t bit = 1ull << f;
            for (uint32_t j = 0; j < nr; ++j) {
                const uint32_t s = rv[j];
                if (!fm[s]) continue;
                const double   *row = w  + (size_t)s * R;
                uint64_t       *dst = lm + (size_t)s * R;
                for (uint32_t x = 0; x < R; ++x)
                    if (row[x] != -INFINITY) dst[x] |= bit;
            }
        }

        /* Scatter. The mask narrows monotonically along the path: a member
         * that cannot spell this edge is gone from every extension of it, and
         * a path with no member left is not a sequence any member spells, so
         * it is dropped here rather than carried to the sink and filtered. */
        for (uint64_t j = 0; j < cur.n_used; ++j) {
            const uint64_t slot = cur.used[j];
            const uint32_t s    = cur.state[slot];
            const uint64_t mk   = cur.mask[slot];
            const double   c    = cur.count[slot];
            const uint64_t *row = lm + (size_t)s * R;
            const uint32_t rest = s % SK;
            for (uint32_t x = 0; x < R; ++x) {
                const uint64_t nb = mk & row[x];
                if (nb == 0ull) continue;
                if (!mt_add(&nxt, rest * A + x, nb, c)) {
                    overflow_at = i;
                    rc = BEAM_ERR_BUDGET;
                    goto done;
                }
            }
        }

        for (uint32_t j = 0; j < n_ms; ++j) mm[ms[j]] = 0u;
        if (nxt.n_used > peak) peak = nxt.n_used;
        mt_clear(&cur);
        { MaskTable t = cur; cur = nxt; nxt = t; }
    }

    if (cur.n_used > peak) peak = cur.n_used;

    /* The sink settles the backward terms of the last residues. A member that
     * cannot pay them cannot spell the sequence, so the tail narrows the mask
     * exactly as an edge does. */
    for (uint32_t f = 0; f < n_models; ++f) {
        if (!L[f]) continue;
        beam_tail_weights(models[f], len, tl);
        const uint64_t bit = 1ull << f;
        for (uint64_t j = 0; j < cur.n_used; ++j) {
            const uint32_t s = cur.state[cur.used[j]];
            if (tl[s] != -INFINITY) tm[s] |= bit;
        }
    }
    for (uint64_t j = 0; j < cur.n_used; ++j) {
        const uint64_t slot = cur.used[j];
        if ((cur.mask[slot] & tm[cur.state[slot]]) != 0ull)
            total += cur.count[slot];
    }

done:
    free(lm); free(w); free(tm); free(tl); free(mm); free(fm); free(ms);
    mt_free(&cur); mt_free(&nxt);
    for (uint32_t g = 0; g < got; ++g) beam_live_release(&h[g]);
    if (out_peak) *out_peak = peak;
    if (rc == BEAM_ERR_BUDGET) {
        const char *bu;
        const double bv = beam_budget_scaled(base + 2ull * union_table_bytes(cap),
                                             &bu);
        return BEAM_FAIL(BEAM_ERR_BUDGET,
                         "the support union at length %u outgrew its ceiling "
                         "of %llu live (state, member-set) pairs at layer %u, "
                         "holding about %.2f %s; %u members disagree too much "
                         "for an exact union at this budget -- raise it with "
                         "BEAM.set_memory_budget, or take the bounds instead",
                         (unsigned)len, (unsigned long long)cap,
                         (unsigned)overflow_at, bv, bu, (unsigned)n_models);
    }
    *out = total;
    return BEAM_OK;
}


BeamError beam_mixture_support_union_by_length(
        const BeamModel *const *models, uint32_t n_models, uint64_t cap,
        double *out_by_length, double *out_total, uint64_t *out_peak) {
    const BeamError shape = beam_mixture_check_shape(models, n_models);
    if (shape != BEAM_OK) return shape;
    if (!out_total) return BEAM_ENULL("output buffer");

    const BeamModel *m0 = models[0];
    const uint32_t n = m0->n_len_slots;
    double *val = (double *)calloc(n, sizeof(double));
    uint32_t *work = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    uint64_t *pk = (uint64_t *)calloc(n, sizeof(uint64_t));
    if (!val || !work || !pk) {
        free(val); free(work); free(pk);
        return BEAM_EALLOC("the per-length support union");
    }

    /* Longest first: the deepest trellis carries the most masks and is the
     * length most likely to hit the ceiling, so it is the one to find out
     * about first as well as the one to schedule first. */
    uint32_t nw = 0;
    for (uint32_t Lx = n; Lx-- > 1; ) {
        for (uint32_t f = 0; f < n_models; ++f)
            if (beam_length_live(models[f], Lx)) { work[nw++] = Lx; break; }
    }

    /*
     * Serial. The union sweep is a hash table per layer with no safe way to
     * scatter into it from many threads, and running several LENGTHS at once
     * would multiply the table footprint by the thread count -- which is the
     * one thing the budget exists to stop. The inner product parallelises over
     * lengths precisely because its footprint is bounded and known; this one's
     * is the quantity under test.
     */
    BeamError rc = BEAM_OK;
    for (uint32_t j = 0; j < nw && rc == BEAM_OK; ++j)
        rc = beam_mixture_support_union(models, n_models, work[j], cap,
                                        &val[work[j]], &pk[work[j]]);
    if (rc != BEAM_OK) { free(val); free(work); free(pk); return rc; }

    double total = 0.0;
    uint64_t peak = 0;
    for (uint32_t Lx = 1; Lx < n; ++Lx) {
        total += val[Lx];
        if (pk[Lx] > peak) peak = pk[Lx];
    }
    if (out_by_length)
        for (uint32_t Lx = 0; Lx < n; ++Lx) out_by_length[Lx] = val[Lx];
    *out_total = total;
    if (out_peak) *out_peak = peak;
    free(val); free(work); free(pk);
    return BEAM_OK;
}
