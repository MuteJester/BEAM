/**
 * @file top_k.c
 * @brief The k most probable sequences, exactly.
 *
 * Best-first search over the DAG, guided by the backward values.
 *
 * The reason this is exact rather than a beam: `best[i][state]` is the weight
 * of the best completion from that state, so `accumulated + best` is the exact
 * weight of the best sequence extending this prefix. Expanding always the
 * highest such value therefore pops complete paths in non-increasing order of
 * weight, so the n-th path popped really is the n-th most probable. A beam of
 * width w would silently drop a sequence whose prefix looked bad, and would
 * give no indication it had done so.
 *
 * Cost is O(k * L * 20 * log) after the backward pass, so asking for the top
 * hundred is cheap even on a 194,481-state trellis.
 */
#include "beam/internal.h"

/* ── A binary max-heap over partial paths ────────────────────── */

typedef struct {
    double   key;       /* accumulated weight + beta of where it sits */
    double   acc;       /* accumulated weight so far                  */
    uint32_t state;
    uint32_t depth;     /* how many residues emitted                  */
    uint32_t path;      /* index into the path arena                  */
} Item;

typedef struct {
    Item    *a;
    uint32_t n, cap;
} Heap;

/* Growth is capped below 2^31 so the doubling cannot wrap a uint32_t to zero
 * and spin forever; the reallocation of a heap that large fails long before
 * this bites, but a wrapped capacity would not fail at all. */
#define BEAM_GROW_MAX 0x40000000u          /* 2^30 items */

static bool heap_reserve(Heap *h, uint32_t need) {
    if (need <= h->cap) return true;
    if (need > BEAM_GROW_MAX) return false;
    uint32_t cap = h->cap ? h->cap * 2u : 1024u;
    while (cap < need) {
        if (cap > BEAM_GROW_MAX / 2u) { cap = need; break; }
        cap *= 2u;
    }
    Item *na = (Item *)realloc(h->a, (size_t)cap * sizeof(Item));
    if (!na) return false;
    h->a = na; h->cap = cap;
    return true;
}

static bool heap_push(Heap *h, Item it) {
    if (!heap_reserve(h, h->n + 1u)) return false;
    uint32_t i = h->n++;
    h->a[i] = it;
    while (i > 0) {
        const uint32_t p = (i - 1u) / 2u;
        if (h->a[p].key >= h->a[i].key) break;
        Item t = h->a[p]; h->a[p] = h->a[i]; h->a[i] = t;
        i = p;
    }
    return true;
}

static Item heap_pop(Heap *h) {
    Item top = h->a[0];
    h->a[0] = h->a[--h->n];
    uint32_t i = 0;
    for (;;) {
        const uint32_t l = 2u * i + 1u, r = l + 1u;
        uint32_t best = i;
        if (l < h->n && h->a[l].key > h->a[best].key) best = l;
        if (r < h->n && h->a[r].key > h->a[best].key) best = r;
        if (best == i) break;
        Item t = h->a[best]; h->a[best] = h->a[i]; h->a[i] = t;
        i = best;
    }
    return top;
}

/* ── Path arena ──────────────────────────────────────────────── */

/**
 * Partial paths share prefixes, so they are stored as a back-linked trie
 * rather than as strings: each node is one residue plus its parent. Copying
 * whole strings around the heap would cost O(L) per push and dominate.
 */
typedef struct {
    uint32_t *parent;
    uint8_t  *residue;
    uint32_t  n, cap;
} Arena;

/* Returns UINT32_MAX on failure, so the arena must never reach that index and
 * the doubling must never wrap. Both are capped here rather than left to the
 * allocator, because a wrapped capacity would loop rather than fail. */
static uint32_t arena_add(Arena *ar, uint32_t parent, uint8_t residue) {
    if (ar->n == ar->cap) {
        if (ar->cap >= BEAM_GROW_MAX) return UINT32_MAX;
        uint32_t cap = ar->cap ? ar->cap * 2u : 4096u;
        uint32_t *p = (uint32_t *)realloc(ar->parent, (size_t)cap * sizeof(uint32_t));
        uint8_t  *r = (uint8_t *)realloc(ar->residue, (size_t)cap);
        /* Exactly one of each pair is the live buffer: a realloc that returned
         * non-NULL has already released the old one, and one that returned
         * NULL has left it alone. Freeing the other of each pair is what makes
         * this neither a leak nor a double free. */
        if (!p || !r) {
            free(p ? p : ar->parent);
            free(r ? r : ar->residue);
            ar->parent = NULL; ar->residue = NULL;
            ar->n = 0; ar->cap = 0;      /* nothing survives; say so */
            return UINT32_MAX;
        }
        ar->parent = p; ar->residue = r; ar->cap = cap;
    }
    ar->parent[ar->n] = parent;
    ar->residue[ar->n] = residue;
    return ar->n++;
}

static void arena_write(const Arena *ar, uint32_t leaf, uint32_t len, char *out) {
    out[len] = '\0';
    uint32_t node = leaf;
    for (uint32_t i = len; i-- > 0; ) {
        out[i] = beam_idx_to_char[ar->residue[node]];
        node = ar->parent[node];
    }
}

/* ── Best-completion values ──────────────────────────────────── */

/**
 * `best[i][s]` is the weight of the BEST completion from state `s` at layer
 * `i`, sink term included: a max-plus sweep rather than the log-sum-exp one
 * `beam_betas` performs.
 *
 * Which one guides the search is the difference between a second and a minute.
 * `acc + beta` bounds the best completion from above but not tightly — beta
 * sums over every completion, so on a 194,481-state trellis it overshoots by
 * many nats and the frontier degenerates towards breadth-first, expanding
 * millions of prefixes that can never win. `acc + best` is the exact weight of
 * the best completion, so the pop order is the true order and every expansion
 * is one the answer needed.
 *
 * Exactness is unchanged, and so is the reported probability: at the final
 * layer there is nothing left to choose, so `best[len]` IS the sink weight,
 * exactly as `beta[len]` was.
 */
static BeamError best_completions(const BeamModel *m, uint32_t len,
                                 double **best) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    uint8_t *alive = (uint8_t *)malloc(S);
    if (!w || !alive) { free(w); free(alive);
        return BEAM_EALLOC("one layer of edge weights"); }

    /* Only states the root can reach are ever expanded by the search below,
     * so only their completions are read. This asks for no `out_max`, which
     * is the case `beam_layer_weights_live` documents as safe: dead rows are
     * provably unread, because a reached state's finite edge always lands on
     * a reached target, and an infinite one is rejected before the target is
     * touched.
     *
     * Unreached entries are filled with -inf rather than left stale. Nothing
     * should read them; if anything ever does, an impossible completion is a
     * loud wrong answer instead of a quiet one. */
    LiveHandle h;
    if (beam_live_acquire(m, len, &h) != BEAM_OK) {
        free(w); free(alive);
        return BEAM_EALLOC("the forward frontier");
    }
    const BeamLive *L = h.L;

    beam_tail_weights(m, len, best[len]);
    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        const uint32_t  nr = L->n_reached[i];
        const uint32_t *rv = L->reached[i];
        beam_reached_mask(L, (uint32_t)i, S, alive);
        beam_layer_weights_live(m, (uint32_t)i, len, w, NULL, alive);
        const double *nb = best[i + 1];
        double *cu = best[i];
        for (uint32_t t = 0; t < S; ++t) cu[t] = -INFINITY;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nr; ++k) {
            const uint32_t s = rv[k];
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            cu[s] = mx;
        }
    }
    free(w); free(alive);
    beam_live_release(&h);
    return BEAM_OK;
}

/* ── One length ──────────────────────────────────────────────── */

typedef struct {
    double   log_p;
    uint32_t length;
    uint32_t leaf;
} Found;

static int cmp_found(const void *a, const void *b) {
    const double x = ((const Found *)a)->log_p, y = ((const Found *)b)->log_p;
    return (x < y) - (x > y);              /* descending */
}

static BeamError top_k_one(const BeamModel *m, uint32_t len, uint32_t k,
                          Arena *ar, Found *found, uint32_t *n_found) {
    const uint32_t A = BEAM_ALPHABET;
    const uint32_t R = BEAM_RESIDUES;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double **beta = (double **)calloc(len + 1u, sizeof(double *));
    if (!beta) return BEAM_EALLOC("the backward-value layer table");
    for (uint32_t i = 0; i <= len; ++i) {
        beta[i] = (double *)malloc((size_t)S * sizeof(double));
        if (!beta[i]) {
            for (uint32_t j = 0; j <= i; ++j) free(beta[j]);
            free(beta);
            return BEAM_EALLOC("one layer of backward values");
        }
    }
    BeamError err = best_completions(m, len, beta);
    double *w = (double *)malloc((size_t)R * sizeof(double));
    Heap heap = {0};
    if (err != BEAM_OK || !w) {
        if (err == BEAM_OK) err = BEAM_EALLOC("one node's edge weights");
        goto done;
    }

    if (beta[0][S - 1] == -INFINITY) { err = BEAM_OK; goto done; }

    /* The constant that turns an accumulated weight into a log probability. */
    const double shift = m->log_pi[len] - m->log_z[len];

    if (!heap_push(&heap, (Item){ beta[0][S - 1], 0.0, S - 1u, 0u, UINT32_MAX })) {
        err = BEAM_EALLOC("the best-first frontier"); goto done;
    }

    while (heap.n > 0 && *n_found < k) {
        const Item it = heap_pop(&heap);

        if (it.depth == len) {
            /* The key at the final layer is `acc + best[len][state]`, and
             * `best[len]` IS the sink weight, so the key is already the
             * complete path weight. Recording `acc` would drop the tail, which
             * is the backward score of the last `order` residues. */
            found[*n_found].log_p  = it.key + shift;
            found[*n_found].length = len;
            found[*n_found].leaf   = it.path;
            (*n_found)++;
            continue;
        }

        /*
         * One node's row, not the whole layer.
         *
         * Best-first pops jump between depths in whatever order the keys
         * dictate, so a one-layer cache misses on nearly every pop, and a miss
         * cost a full 3.9-million-edge layer build to read twenty of its
         * numbers. Caching every layer instead is not an option either:
         * `n_states * 20 * len` doubles is gigabytes at order 4. Computing the
         * twenty edges actually wanted is O(1) and makes the depth order
         * irrelevant.
         */
        beam_state_weights(m, it.depth, len, it.state, w);
        const double *row = w;
        const uint32_t base = (it.state % SK) * A;

        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY) continue;
            const double nb = beta[it.depth + 1u][base + x];
            if (nb == -INFINITY) continue;
            const uint32_t leaf = arena_add(ar, it.path, (uint8_t)x);
            if (leaf == UINT32_MAX) {
                err = BEAM_EALLOC("the path arena"); goto done;
            }
            const double acc = it.acc + row[x];
            if (!heap_push(&heap, (Item){ acc + nb, acc, base + x,
                                          it.depth + 1u, leaf })) {
                err = BEAM_EALLOC("the best-first frontier"); goto done;
            }
        }
        /*
         * Re-expanding the popped node is unnecessary because the heap already
         * holds every sibling: the search is over prefixes, and each prefix is
         * pushed exactly once.
         */
    }
    err = BEAM_OK;

done:
    for (uint32_t i = 0; i <= len; ++i) free(beta[i]);
    free(beta); free(w); free(heap.a);
    return err;
}

/* ── Public entry point ──────────────────────────────────────── */

BeamError beam_model_top_k(const BeamModel *m, uint32_t k,
                              uint32_t length, char *out,
                              double *out_log_p, uint32_t *out_n) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (k == 0) { if (out_n) *out_n = 0; return BEAM_OK; }
    if (length != 0 && !beam_length_live(m, length))
        return BEAM_EBADLEN(m, length);

    const uint32_t stride = m->max_len + 1u;

    /*
     * Each length is searched for its own top k and the results merged, which
     * is exact: the k best overall must be among the k best of some length.
     */
    uint32_t n_len = 0;
    for (uint32_t L = 1; L < m->n_len_slots; ++L)
        if (length ? (L == length) : beam_length_live(m, L)) n_len++;
    if (n_len == 0) { if (out_n) *out_n = 0; return BEAM_OK; }

    Found *found = (Found *)malloc((size_t)k * n_len * sizeof(Found));
    Arena ar = {0};
    if (!found) return BEAM_EALLOC("the top-k merge buffer");

    uint32_t total = 0;
    BeamError err = BEAM_OK;
    for (uint32_t L = 1; L < m->n_len_slots && err == BEAM_OK; ++L) {
        if (length ? (L != length) : !beam_length_live(m, L)) continue;
        uint32_t got = 0;
        err = top_k_one(m, L, k, &ar, found + total, &got);
        total += got;
    }
    if (err != BEAM_OK) { free(found); free(ar.parent); free(ar.residue); return err; }

    qsort(found, total, sizeof(Found), cmp_found);
    const uint32_t n = total < k ? total : k;
    for (uint32_t i = 0; i < n; ++i) {
        arena_write(&ar, found[i].leaf, found[i].length, out + (size_t)i * stride);
        if (out_log_p) out_log_p[i] = found[i].log_p;
    }
    if (out_n) *out_n = n;

    free(found); free(ar.parent); free(ar.residue);
    return BEAM_OK;
}
