/**
 * @file positional.c
 * @brief Per-position residue marginals and expected cell usage. Both exact.
 *
 * ## The quantity
 *
 * Everything in this file is one object seen two ways: the per-layer edge
 * posterior of the length-`L` trellis,
 *
 *     gamma_i(s, x) = alpha_i(s) * w_i(s, x) * beta_{i+1}(s -> x) / Z(L)
 *
 * where `alpha` is the forward weight of reaching state `s` at layer `i`,
 * `w_i(s, x)` the edge weight of emitting `x` there, `beta` the backward
 * completion weight already used by the sampler and the top-k search, and
 * `Z(L)` the partition function. Summing `gamma` over states gives
 * `P(x_i = a | L)`; summing it into the table cell each edge would have
 * incremented during training gives the expected usage of that cell.
 *
 * ## Why this is exact and not an approximation of anything
 *
 * The DAG is layered by position and every path weight factorises over its
 * edges plus one sink term, so the three sums above are finite sums a dynamic
 * program evaluates in full. Nothing is sampled, nothing is binned, and no
 * state is pruned: the sweeps run over all `n_states` at every layer. The only
 * error is floating point, and it is at the last bits.
 *
 * That is a stronger claim than `beam_model_spectrum` can make, and the
 * difference is worth naming. The spectrum transports mass onto a GRID and
 * splits it linearly between adjacent bins at every layer, so its bins are
 * approximate even though its measure is exact. Marginals have no grid: a
 * residue is one of twenty, the axis is already discrete, and the answer lands
 * on it exactly.
 *
 * ## Clamping
 *
 * A clamp pins chosen positions to chosen residues, and the result is the
 * exact conditional `P(x_i = a | L, clamp)`. It is imposed by DELETING the
 * edges that disagree with it -- the inner loops run over `[lo, hi)` rather
 * than over all twenty residues -- in the backward pass, the forward pass and
 * the posterior alike. Deleting edges is exactly conditioning here, because
 * the set of length-`L` paths agreeing with the clamp is precisely the set of
 * paths that avoid those edges, and the weight of a path is the product over
 * the edges it uses. So the clamped `Z` is the unnormalised probability of the
 * clamped event, and dividing by it normalises the conditional.
 *
 * A clamp is NOT applied by reweighting or by masking the answer afterwards.
 * Masking would leave the other positions' marginals as their unconditional
 * values, which is a different and wrong quantity.
 *
 * ## The rescale, and why the clamp changes it
 *
 * The forward sweep runs in linear space with a per-layer rescale, as
 * `compute_log_z` does, and takes the layer maximum from the layer build for
 * free. Under a clamp that maximum is wrong in a way that matters: it is taken
 * over all twenty residues, but only one of them is summed, and if the clamped
 * residue sits 700 nats below the layer's best the single term underflows to
 * zero and the conditional comes back empty. Clamped layers therefore rescale
 * by the maximum over the residues actually summed, which costs one strided
 * pass over `n_states` and is only paid on the layers that are pinned.
 */
#include "beam/internal.h"
#include "beam/math_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* ── Clamps ──────────────────────────────────────────────────── */

/**
 * "This position is not pinned."
 *
 * Deliberately `BEAM_RESIDUE_INVALID`, so a caller can `memset` a clamp array
 * to 0xFF and mean "free everywhere", and so the one value that can never be a
 * residue index is the one that means "no residue".
 */
#define CLAMP_FREE BEAM_RESIDUE_INVALID

/**
 * The half-open range of residues layer `i` may emit.
 *
 * One value when the layer is pinned and all of them otherwise, which is what
 * lets every loop in this file impose the clamp by its bounds rather than by a
 * test in its body. A pinned layer then costs a twentieth of an unpinned one
 * instead of the same and discarding nineteen twentieths.
 */
static inline void clamp_range(const uint8_t *clamp, uint32_t i, uint32_t R,
                               uint32_t *lo, uint32_t *hi) {
    if (clamp && clamp[i] != CLAMP_FREE) {
        *lo = clamp[i];
        *hi = clamp[i] + 1u;
    } else {
        *lo = 0u;
        *hi = R;
    }
}

/** True when any position is pinned. */
static bool clamp_active(const uint8_t *clamp, uint32_t len) {
    if (!clamp) return false;
    for (uint32_t i = 0; i < len; ++i)
        if (clamp[i] != CLAMP_FREE) return true;
    return false;
}


/* ── Backward values under a clamp ───────────────────────────── */

/**
 * `beta[i][s]`, the log total weight of every completion from state `s` at
 * layer `i` that AGREES WITH THE CLAMP, sink term included.
 *
 * This is `beam_betas` with the residue loops bounded by `clamp_range`, and
 * with no clamp it performs the same operations in the same order on the same
 * values, so the two agree bit for bit. It is a separate function rather than
 * a parameter added to `beam_betas` because that function is on the sampler's
 * and the top-k search's hot path, and neither has a clamp to pass.
 */
static BeamError clamped_betas(const BeamModel *m, uint32_t len,
                               const uint8_t *clamp, double **beta) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    if (!w) return BEAM_EALLOC("one layer of edge weights");

    beam_tail_weights(m, len, beta[len]);

    for (int32_t i = (int32_t)len - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, len, w, NULL);
        uint32_t lo, hi;
        clamp_range(clamp, (uint32_t)i, R, &lo, &hi);
        const double *nb = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
            const uint32_t s = (uint32_t)s64;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double mx = -INFINITY;
            for (uint32_t x = lo; x < hi; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                const double v = row[x] + nb[base + x];
                if (v > mx) mx = v;
            }
            if (mx == -INFINITY) { beta[i][s] = -INFINITY; continue; }
            double acc = 0.0;
            for (uint32_t x = lo; x < hi; ++x) {
                if (row[x] == -INFINITY || nb[base + x] == -INFINITY) continue;
                acc += exp(row[x] + nb[base + x] - mx);
            }
            beta[i][s] = mx + log(acc);
        }
    }
    free(w);
    return BEAM_OK;
}


/* ── The forward sweep ───────────────────────────────────────── */

/**
 * Advance the LOG forward vector across one layer.
 *
 * `cur[s]` is `ln alpha_i(s)`, and on return `next` holds the same for layer
 * `i + 1`. Accumulated by TARGET, for the reason `compute_log_z` gives: a
 * target's twenty-one sources are its private business, so no two threads
 * write the same accumulator.
 *
 * There is no rescale and no running scale factor, and that is the point.
 * This sweep used to carry `alpha` linearly, renormalised once per layer by
 * the largest edge weight in it -- and no single factor per layer can hold a
 * trellis whose live states span more than 745 nats. At the C API's own
 * defaults a length-60 nucleotide sweep spans that easily, and the states that
 * underflowed to zero were not marginal: they included PINNED positions, whose
 * row must be one-hot and came back summing to 2e-54 with BEAM_OK.
 *
 * Three attempts were made to choose that factor well -- over the whole layer,
 * over edges a live path can use, and weighted by the mass each source carries
 * -- before the factor was removed instead. The backward sweep above never had
 * the problem, having always been log-space; now neither half does.
 *
 * Returns false when the layer leaves no mass at all, which under a clamp is a
 * real outcome rather than an error.
 */
static bool advance(const BeamModel *m, const double *w,
                    const double *cur, double *next,
                    uint32_t lo, uint32_t hi) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    bool any = false;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:any) \
            num_threads(beam_threads())
#endif
    for (int64_t t = 0; t < (int64_t)S; ++t) {
        const uint32_t x = (uint32_t)t % A;
        if (x < lo || x >= hi) { next[t] = -INFINITY; continue; }
        const uint32_t rest = (uint32_t)t / A;
        double mx = -INFINITY, sum = 0.0;
        for (uint32_t a = 0; a < A; ++a) {
            const uint32_t src = a * SK + rest;
            if (cur[src] == -INFINITY) continue;
            const double wv = w[(size_t)src * R + x];
            if (wv == -INFINITY) continue;
            lse_add(&mx, &sum, cur[src] + wv);
        }
        next[t] = (mx == -INFINITY) ? -INFINITY : mx + log(sum);
        if (next[t] != -INFINITY) any = true;
    }
    return any;
}


/* ── Shared forward-backward scratch ─────────────────────────── */

typedef struct {
    double **beta;    /* [len + 1][n_states], log space          */
    double  *w;       /* [n_states * n_residues], one layer      */
    double  *cur;     /* [n_states], rescaled forward vector     */
    double  *next;    /* [n_states]                              */
    double  *tail;    /* [n_states], sink weights                */
    uint32_t n_beta;  /* rows actually allocated, for the free   */
} FBWork;

static void fb_free(FBWork *k) {
    if (k->beta) {
        for (uint32_t i = 0; i < k->n_beta; ++i) free(k->beta[i]);
        free(k->beta);
    }
    free(k->w); free(k->cur); free(k->next); free(k->tail);
    memset(k, 0, sizeof(*k));
}

static BeamError fb_alloc(const BeamModel *m, uint32_t len, FBWork *k) {
    const uint32_t S = m->n_states;
    const uint32_t R = m->n_residues;
    memset(k, 0, sizeof(*k));

    /*
     * Refuse before allocating, for the reason `beam_model_new` refuses:
     * `calloc` on Linux grants lazily, so an oversized request does not fail
     * where it is made, it fails much later on first write with the OOM killer
     * choosing a victim from the whole machine.
     *
     * The whole backward field is held at once because the posterior needs
     * layer `i + 1` while the forward sweep is at layer `i`, and the forward
     * sweep only runs once the backward one is finished. That is `(len + 1) *
     * n_states` doubles, which is 44 MB on the default amino-acid model and
     * 6.6 GB on a nucleotide model at the alphabet's ceiling order -- so the
     * bound is a real one at the top of the range rather than a formality.
     */
    const uint64_t need = ((uint64_t)len + 1u) * S * sizeof(double)
                        + (uint64_t)S * (R + 3u) * sizeof(double);
    const uint64_t budget = beam_model_memory_budget();
    if (need > budget) {
        const char *nu, *bu;
        const double nv = beam_budget_scaled(need, &nu);
        const double bv = beam_budget_scaled(budget, &bu);
        return BEAM_FAIL(BEAM_ERR_ALLOC,
                        "a forward-backward sweep at length %u needs about "
                        "%.2f %s for the backward field of %u states, and the "
                        "memory budget is %.2f %s; raise it with "
                        "BEAM.set_memory_budget, or ask at a lower order",
                        (unsigned)len, nv, nu, (unsigned)S, bv, bu);
    }

    k->beta = (double **)calloc((size_t)len + 1u, sizeof(double *));
    if (!k->beta) return BEAM_EALLOC("the backward-value layer table");
    k->n_beta = len + 1u;
    for (uint32_t i = 0; i <= len; ++i) {
        k->beta[i] = (double *)malloc((size_t)S * sizeof(double));
        if (!k->beta[i]) { fb_free(k); return BEAM_EALLOC("a backward layer"); }
    }
    k->w    = (double *)malloc((size_t)S * R * sizeof(double));
    k->cur  = (double *)calloc(S, sizeof(double));
    k->next = (double *)malloc((size_t)S * sizeof(double));
    k->tail = (double *)malloc((size_t)S * sizeof(double));
    if (!k->w || !k->cur || !k->next || !k->tail) {
        fb_free(k);
        return BEAM_EALLOC("the forward-backward buffers");
    }
    return BEAM_OK;
}


/* ── Positional marginals ────────────────────────────────────── */

BeamError beam_model_positional_marginals(const BeamModel *m, uint32_t length,
                                          const uint8_t *clamp,
                                          double *out, double *out_log_z) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);

    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    for (uint32_t i = 0; i < length; ++i)
        for (uint32_t x = 0; x < R; ++x) out[(size_t)i * R + x] = 0.0;
    if (out_log_z) *out_log_z = -INFINITY;

    for (uint32_t i = 0; i < length; ++i)
        if (clamp && clamp[i] != CLAMP_FREE && clamp[i] >= R)
            return BEAM_ERANGE("positional_marginals: position %u is clamped "
                              "to residue index %u, which is not one of the "
                              "%u residues of this model's alphabet",
                              (unsigned)i, (unsigned)clamp[i], (unsigned)R);

    FBWork k;
    BeamError err = fb_alloc(m, length, &k);
    if (err != BEAM_OK) return err;

    /* Per-chunk partial rows, combined in chunk order so the answer does not
     * depend on how many threads happened to be free. See BEAM_SUM_CHUNKS. */
    double *chunk = (double *)malloc((size_t)BEAM_SUM_CHUNKS * R
                                     * sizeof(double));
    if (!chunk) { fb_free(&k); return BEAM_EALLOC("the marginal partials"); }

    err = clamped_betas(m, length, clamp, k.beta);
    if (err != BEAM_OK) goto done;

    {
    /*
     * Z from the backward pass rather than from `m->log_z`, and that is not a
     * convenience. Under a clamp `m->log_z` is the WRONG normaliser -- it is
     * the unclamped partition function -- and even unclamped, taking Z from
     * the same sweep the numerator comes from is what makes each row sum to
     * one to the last bits instead of to within the difference between a
     * forward and a backward evaluation of the same quantity.
     */
    const double logz = k.beta[0][S - 1u];
    if (!isfinite(logz)) {
        err = clamp_active(clamp, length)
            ? BEAM_ENOPATHS("the clamped positions are a configuration this "
                           "model cannot produce at this length, so the "
                           "conditional it asks for does not exist")
            : BEAM_ENOPATHS("the trellis carries no path at this length");
        goto done;
    }
    if (out_log_z) *out_log_z = logz;

    for (uint32_t s = 0; s < S; ++s) k.cur[s] = -INFINITY;
    k.cur[S - 1u] = 0.0;         /* ln 1: the all-sentinel start state */

    for (uint32_t i = 0; i < length; ++i) {
        uint32_t lo, hi;
        clamp_range(clamp, i, R, &lo, &hi);
        beam_layer_weights(m, i, length, k.w, NULL);
        const double *bn = k.beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads())
#endif
        for (int64_t c = 0; c < (int64_t)BEAM_SUM_CHUNKS; ++c) {
            double *part = chunk + (size_t)c * R;
            for (uint32_t x = 0; x < R; ++x) part[x] = 0.0;
            const uint32_t slo = (uint32_t)(((uint64_t)S * c) / BEAM_SUM_CHUNKS);
            const uint32_t shi =
                (uint32_t)(((uint64_t)S * (c + 1)) / BEAM_SUM_CHUNKS);
            for (uint32_t s = slo; s < shi; ++s) {
                if (k.cur[s] == -INFINITY) continue;
                const double la = k.cur[s] - logz;
                const double *row = k.w + (size_t)s * R;
                const uint32_t base = (s % SK) * A;
                for (uint32_t x = lo; x < hi; ++x) {
                    if (row[x] == -INFINITY) continue;
                    const double nb = bn[base + x];
                    if (nb == -INFINITY) continue;
                    part[x] += exp(la + row[x] + nb);
                }
            }
        }
        double *row_out = out + (size_t)i * R;
        for (uint32_t c = 0; c < BEAM_SUM_CHUNKS; ++c) {
            const double *part = chunk + (size_t)c * R;
            for (uint32_t x = lo; x < hi; ++x) row_out[x] += part[x];
        }

        if (!advance(m, k.w, k.cur, k.next, lo, hi)) {
            err = BEAM_ENOPATHS("the forward sweep reached a layer with no "
                               "mass although the trellis has a path");
            goto done;
        }
        double *tmp = k.cur; k.cur = k.next; k.next = tmp;
    }
    }

done:
    free(chunk);
    fb_free(&k);
    return err;
}


/* ── Expected cell usage ─────────────────────────────────────── */

/**
 * A growable map from a table's dense row index to an accumulator row.
 *
 * The alternative was an array the shape of the table itself, and it does not
 * fit: an order-4 model has 147.8 million cells, so six dense accumulators
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
} RowMap;

static void rowmap_free(RowMap *t) {
    free(t->key); free(t->at); free(t->vals); free(t->ids);
    memset(t, 0, sizeof(*t));
}

static bool rowmap_init(RowMap *t, uint32_t stride) {
    memset(t, 0, sizeof(*t));
    const uint32_t k0 = 1024u, r0 = 512u;
    t->key = (uint32_t *)malloc((size_t)k0 * sizeof(uint32_t));
    t->at  = (uint32_t *)malloc((size_t)k0 * sizeof(uint32_t));
    t->vals = (double *)calloc((size_t)r0 * stride, sizeof(double));
    t->ids  = (uint32_t *)malloc((size_t)r0 * sizeof(uint32_t));
    if (!t->key || !t->at || !t->vals || !t->ids) { rowmap_free(t); return false; }
    for (uint32_t i = 0; i < k0; ++i) t->key[i] = UINT32_MAX;
    t->mask = k0 - 1u;
    t->cap = r0;
    t->stride = stride;
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
    uint32_t *key = (uint32_t *)malloc((size_t)ncap * sizeof(uint32_t));
    uint32_t *at  = (uint32_t *)malloc((size_t)ncap * sizeof(uint32_t));
    if (!key || !at) { free(key); free(at); return false; }
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
        double *v = (double *)realloc(t->vals,
                                      (size_t)ncap * t->stride * sizeof(double));
        if (!v) return NULL;
        t->vals = v;
        memset(t->vals + (size_t)t->cap * t->stride, 0,
               (size_t)t->cap * t->stride * sizeof(double));
        uint32_t *ids = (uint32_t *)realloc(t->ids,
                                            (size_t)ncap * sizeof(uint32_t));
        if (!ids) return NULL;
        t->ids = ids;
        t->cap = ncap;
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
    uint32_t *rows = (uint32_t *)malloc((size_t)t->n * sizeof(uint32_t));
    double *counts = (double *)malloc((size_t)t->n * t->stride * sizeof(double));
    if (!rows || !counts) {
        free(rows); free(counts);
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
 * The dense ROW index of a cell, one per table. Spelled out here rather than
 * derived from `beam_ix_*` by dividing out the residue stride, because these
 * have to be the same rows `observe_one_position` writes for the answer to be
 * comparable with the counts, and the count path computes them exactly like
 * this. Deriving them a second way would be a second thing to keep in step.
 */
static inline uint32_t row_wide(const BeamModel *m, uint32_t depth,
                                uint32_t ctx) {
    return beam_depth_part(m, depth) * m->n_ctx_wide + ctx;
}
static inline uint32_t row_mid(const BeamModel *m, uint32_t depth,
                               uint32_t ctx) {
    return depth * m->n_ctx_mid + ctx;
}
static inline uint32_t row_len(const BeamModel *m, uint32_t len, uint32_t depth,
                               uint32_t ctx) {
    return (len * m->max_len + depth) * m->n_ctx_len + ctx;
}

/* The six maps, indexed [anchor][expert] exactly as BeamExpectedCounts is. */
typedef RowMap Maps[2][3];

/** Add `g` to one cell of each of the three tables of one anchor. */
static bool bump(Maps mp, uint32_t dir, uint32_t rw, uint32_t rm, uint32_t rl,
                 uint32_t x, double g) {
    double *w = rowmap_row(&mp[dir][0], rw);
    double *d = rowmap_row(&mp[dir][1], rm);
    double *l = rowmap_row(&mp[dir][2], rl);
    if (!w || !d || !l) return false;
    w[x] += g; d[x] += g; l[x] += g;
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
 * These are expected COUNTS, so they use `beam_ctx_backward_from_state`, the
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
                                     double weight, Maps mp) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t K = m->state_order;
    const uint32_t nw = m->n_ctx_wide;
    const uint32_t n3 = m->n_ctx_mid;
    const uint32_t n2 = m->n_ctx_len;

    FBWork k;
    BeamError err = fb_alloc(m, len, &k);
    if (err != BEAM_OK) return err;

    err = clamped_betas(m, len, NULL, k.beta);
    if (err != BEAM_OK) { fb_free(&k); return err; }

    {
    const double logz = k.beta[0][S - 1u];
    if (!isfinite(logz)) {           /* a live length always has one path */
        fb_free(&k);
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
            const uint32_t fw = row_wide(m, i, s % nw);
            const uint32_t fm = row_mid(m, i, s % n3);
            const uint32_t fl = row_len(m, len, i, s % n2);
            const uint32_t scored = beam_state_oldest(s, SK);
            const bool back = settle_back && scored < R;

            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY) continue;
                const double nb = bn[base + x];
                if (nb == -INFINITY) continue;
                const double g = weight * exp(la + row[x] + nb);
                if (g == 0.0) continue;
                if (!bump(mp, BEAM_FWD, fw, fm, fl, x, g)) { oom = true; break; }
                if (!back) continue;
                const uint32_t bw = row_wide(
                    m, r, beam_ctx_backward_from_state(s, x, K, m->order, A));
                const uint32_t bm = row_mid(
                    m, r, beam_ctx_backward_from_state(s, x, K, 3u, A));
                const uint32_t bl = row_len(
                    m, len, r, beam_ctx_backward_from_state(s, x, K, 2u, A));
                if (!bump(mp, BEAM_BWD, bw, bm, bl, scored, g)) {
                    oom = true; break;
                }
            }
        }
        if (oom) break;
        if (!advance(m, k.w, k.cur, k.next, 0u, R)) { lost = true; break; }
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
                uint32_t cw = 0, cm = 0, cl = 0;
                for (uint32_t j = m->order; j >= 1u; --j)
                    cw = cw * A + ((j <= rr) ? slot[pos + j] : m->none_idx);
                for (uint32_t j = 3u; j >= 1u; --j)
                    cm = cm * A + ((j <= rr) ? slot[pos + j] : m->none_idx);
                for (uint32_t j = 2u; j >= 1u; --j)
                    cl = cl * A + ((j <= rr) ? slot[pos + j] : m->none_idx);
                const uint32_t bw = row_wide(m, rr, cw);
                const uint32_t bm = row_mid(m, rr, cm);
                const uint32_t bl = row_len(m, len, rr, cl);
                if (!bump(mp, BEAM_BWD, bw, bm, bl, scored, g)) {
                    oom = true; break;
                }
            }
        }
    }

    fb_free(&k);
    if (oom) return BEAM_EALLOC("an expected-count accumulator row");
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
    Maps mp;
    memset(mp, 0, sizeof(mp));
    for (uint32_t d = 0; d < 2u; ++d)
        for (uint32_t e = 0; e < 3u; ++e)
            if (!rowmap_init(&mp[d][e], m->n_residues)) {
                for (uint32_t dd = 0; dd < 2u; ++dd)
                    for (uint32_t ee = 0; ee < 3u; ++ee)
                        rowmap_free(&mp[dd][ee]);
                return BEAM_EALLOC("the expected-count accumulators");
            }

    BeamError err = BEAM_OK;
    double total = 0.0;
    if (length != 0u) {
        err = expected_one_length(m, length, 1.0, mp);
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
            err = expected_one_length(m, L, pi, mp);
            total += pi * (double)L;
        }
    }

    for (uint32_t d = 0; d < 2u && err == BEAM_OK; ++d)
        for (uint32_t e = 0; e < 3u && err == BEAM_OK; ++e)
            err = rowmap_emit(&mp[d][e], &out->block[d][e]);

    for (uint32_t d = 0; d < 2u; ++d)
        for (uint32_t e = 0; e < 3u; ++e) rowmap_free(&mp[d][e]);

    if (err != BEAM_OK) { beam_model_expected_counts_free(out); return err; }
    out->expected_length = total;
    return BEAM_OK;
}

void beam_model_expected_counts_free(BeamExpectedCounts *ec) {
    if (!ec) return;
    for (uint32_t d = 0; d < 2u; ++d)
        for (uint32_t e = 0; e < 3u; ++e) {
            free(ec->block[d][e].rows);
            free(ec->block[d][e].counts);
        }
    memset(ec, 0, sizeof(*ec));
}
