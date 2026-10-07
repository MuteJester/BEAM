/**
 * @file spectrum.c
 * @brief Surprisal spectrum: the sweeps that move mass onto the grid.
 *
 * The transport itself -- the linear split that puts a shifted histogram onto
 * adjacent bins -- is in `internal/beam/grid.h`, because the banded positional
 * marginals run the same split in both directions and the two halves of a
 * forward-backward pair have to split mass identically to recombine.
 */
#include "beam/internal.h"
#include "beam/grid.h"
#include "beam/math_internal.h"


/*
 * This file used to carry its own `trim`, computing BACKWARD liveness into a
 * byte per state per layer. It is gone: `beam_live_acquire` in liveness.c
 * answers the same question, intersects it with forward reachability so the
 * set is the small one, and caches the result for every other analytic
 * instead of rebuilding it per spectrum.
 */

/* ── The layer transport, a group of targets at a time ─────────── */

/*
 * A target `rest * A + x` of the next layer gathers its edges from the
 * sources `a * state_shift + rest`, a = 0 .. A-1. So the up to R targets that
 * share `rest` share every source, and each source feeds that one group and
 * no other. Gathered one target at a time, each source's histogram -- up to
 * two measures of `bins` doubles -- came from memory once per target, up to
 * twenty times a layer, and the sweep was bound by memory bandwidth (as fast
 * on 20 threads as on 40). A group instead walks its destination bins in
 * tiles, every target and every source per tile, so a source is read from
 * memory about once and then from cache.
 *
 * THE SAME BITS. Every destination bin receives the same terms in the same
 * order as `beam_grid_transport` gives it, target by target: sources in
 * ascending order, and from each source the share of bin d - k - 1 before
 * that of bin d - k, each formed as value * fraction * factor. A tile only
 * changes when a bin is visited, not what is added to it. Adding the zero a
 * source bin holds, which the per-bin form skipped, leaves a sum of
 * non-negative masses unchanged. tests/native/test_spectrum_transport.c holds
 * every route to the per-bin form bit for bit.
 *
 * A destination range here covers every bin an edge could write, which can
 * be wider than the bins it did write; a row is zero outside its range either
 * way, which is all the ranges are relied on for.
 */
#define SPEC_TILE 512u

typedef double spec_v8d __attribute__((vector_size(64)));

typedef struct {
    int64_t  k;          /* whole bins of the shift                          */
    double   f, g;       /* bin b sends g to b + k and f to b + k + 1        */
    double   fac[2];     /* each measure's factor                            */
    uint32_t src;        /* the source, as an index into the group's list    */
    uint32_t lo, hi;     /* the destination bins it can write, [lo, hi)      */
} SpecEdge;

/*
 * ONE POOL OF HISTOGRAM ROWS, RECYCLED WITHIN A LAYER.
 *
 * The sweep used to hold two full planes, `cur` and `next`, each a row of
 * `hist` doubles for every live slot: on the foundation at 16,384 bins that
 * is 160,000 rows x 128 KiB, twice, and per measure. But a source row is read
 * by exactly one group (spec_group) and cleared as soon as that group is
 * done, so the rows the current layer has finished with can hold the next
 * layer's targets. A group takes rows for its targets from the pool, gathers
 * into them exactly as before, then clears its sources and gives their rows
 * back. Where a row lives changes nothing about what is added to it, so every
 * bin is the same sum in the same order, bit for bit.
 *
 * The peak is the rows alive at once: about the widest layer plus each
 * thread's group in flight, where two planes held the widest layer twice. A
 * layer whose groups take more rows than they free (one that grows) draws on
 * rows its earlier groups returned, and the pool grows by a chunk only when
 * it runs dry, so correctness never depends on that estimate.
 *
 * Every row in the free list is all zero: rows are cleared, over the range
 * they occupied, before they are given back.
 */
#define SPEC_CHUNK_SHIFT 10u
#define SPEC_CHUNK_ROWS  (1u << SPEC_CHUNK_SHIFT)

typedef struct {
    double   **chunk;            /* [cap_chunks] SPEC_CHUNK_ROWS rows each   */
    uint32_t   n_chunks, cap_chunks;
    uint32_t  *free_ids;         /* [cap_chunks * SPEC_CHUNK_ROWS] a stack   */
    uint32_t   n_free;
    size_t     hist;             /* doubles per row                          */
    int        failed;           /* a chunk could not be allocated           */
#ifdef _OPENMP
    omp_lock_t lock;
    int        lock_ready;
#endif
} SpecRows;

static inline double *spec_row(const SpecRows *p, uint32_t id) {
    return p->chunk[id >> SPEC_CHUNK_SHIFT]
         + (size_t)(id & (SPEC_CHUNK_ROWS - 1u)) * p->hist;
}

/* Under the lock. The chunk table is sized once, up front, so threads reading
 * `chunk[i]` for rows already handed out never see it move. */
static bool spec_rows_grow(SpecRows *p) {
    if (p->n_chunks == p->cap_chunks) return false;
    double *c = (double *)calloc((size_t)SPEC_CHUNK_ROWS * p->hist, sizeof(double));
    if (!c) return false;
    const uint32_t base = p->n_chunks * SPEC_CHUNK_ROWS;
    p->chunk[p->n_chunks++] = c;
    for (uint32_t r = SPEC_CHUNK_ROWS; r-- > 0; )      /* low ids pop first */
        p->free_ids[p->n_free++] = base + r;
    return true;
}

/* `n` zero rows into `out`; false (and `failed` set) if memory ran out. */
static bool spec_rows_take(SpecRows *p, uint32_t n, uint32_t *out) {
    bool ok = true;
#ifdef _OPENMP
    omp_set_lock(&p->lock);
#endif
    while (ok && p->n_free < n) ok = spec_rows_grow(p);
    if (ok) for (uint32_t i = 0; i < n; ++i) out[i] = p->free_ids[--p->n_free];
    else p->failed = 1;
#ifdef _OPENMP
    omp_unset_lock(&p->lock);
#endif
    return ok;
}

/* Return rows that are already all zero. */
static void spec_rows_give(SpecRows *p, const uint32_t *ids, uint32_t n) {
#ifdef _OPENMP
    omp_set_lock(&p->lock);
#endif
    for (uint32_t i = 0; i < n; ++i) p->free_ids[p->n_free++] = ids[i];
#ifdef _OPENMP
    omp_unset_lock(&p->lock);
#endif
}

/* What one layer's groups share. */
typedef struct {
    const uint32_t *lv_next;     /* the next layer's live states, ascending  */
    const uint64_t *mask;        /* this layer's live set                    */
    const uint32_t *sc;          /* state -> slot in this layer              */
    const double   *w;           /* this layer's edge weights                */
    SpecRows       *rows;        /* where every slot's histogram lives       */
    const uint32_t *rowc;        /* this layer's slot -> row                 */
    uint32_t       *rown;        /* next layer's slot -> row, set per group  */
    uint8_t        *done;        /* this layer's slot: row given back        */
    BeamBinRange   *rc, *rn;
    uint32_t A, R, SK, bins, nm;
    size_t   hist;
    double   ds;
    bool     counting;
} SpecLayer;

/* Per thread: a group's sources and its edges, target by target. */
typedef struct {
    uint32_t *ks, *st;           /* [A] source slots and states              */
    uint32_t *give;              /* [A] source rows to return                */
    uint32_t *first;             /* [R + 1] a target's first edge            */
    const double **src;          /* [A] source rows                          */
    double  **dst;               /* [R] target rows                          */
    SpecEdge *edge;              /* [A * R]                                  */
} SpecScratch;

/*
 * dst[d] += one edge's share of src for d in [p0, p1): bin d takes f of
 * source bin d - k - 1 and then g of source bin d - k, each only where that
 * bin lies in the source's range [slo, shi).
 */
__attribute__((always_inline))
static inline void spec_move(double *dst, const double *src, int64_t k,
                             double f, double g, double fac, uint32_t slo,
                             uint32_t shi, int64_t p0, int64_t p1) {
    const int64_t first = (int64_t)slo + k;   /* the g share of bin slo only    */
    const int64_t last  = (int64_t)shi + k;   /* the f share of bin shi-1 only  */
    if (first >= p0 && first < p1) dst[first] += src[slo] * g * fac;
    int64_t d = first + 1 > p0 ? first + 1 : p0;
    const int64_t end = last < p1 ? last : p1;
    const spec_v8d fv = {f, f, f, f, f, f, f, f};
    const spec_v8d gv = {g, g, g, g, g, g, g, g};
    const spec_v8d cv = {fac, fac, fac, fac, fac, fac, fac, fac};
    for (; d + 8 <= end; d += 8) {
        spec_v8d x, y, z;
        memcpy(&x, dst + d, sizeof x);
        memcpy(&y, src + (d - k - 1), sizeof y);
        memcpy(&z, src + (d - k), sizeof z);
        x = (x + y * fv * cv) + z * gv * cv;
        memcpy(dst + d, &x, sizeof x);
    }
    for (; d < end; ++d)
        dst[d] = (dst[d] + src[d - k - 1] * f * fac) + src[d - k] * g * fac;
    if (last >= p0 && last < p1) dst[last] += src[shi - 1u] * f * fac;
}

/* The CPU picks, at load time, a copy compiled for AVX-512, for AVX2 or for
 * the baseline; each lane is its own sum, with no fused multiply-add
 * (contraction is off under -std=c11), so the copies agree to the bit. */
#if defined(__x86_64__) && defined(__ELF__) && defined(__GNUC__)
#define SPEC_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define SPEC_CLONES
#endif

/* One group: the next layer's targets [g0, g1), which share `rest`. */
SPEC_CLONES
static void spec_group(const SpecLayer *ly, uint32_t g0, uint32_t g1,
                       const SpecScratch *sp) {
    const uint32_t A = ly->A, R = ly->R, bins = ly->bins, nm = ly->nm;
    const uint32_t rest = ly->lv_next[g0] / A;
    uint32_t ns = 0, ngive = 0;
    for (uint32_t a = 0; a < A; ++a) {
        const uint32_t s = a * ly->SK + rest;
        if (!live_bit_get(ly->mask, s)) continue;
        const uint32_t ks = ly->sc[s];
        /* Every live source belongs to this group alone: its row comes back
         * to the pool when the group is done, with mass or without. */
        ly->done[ks] = 1;
        sp->give[ngive++] = ly->rowc[ks];
        if (ly->rc[ks].lo >= ly->rc[ks].hi) continue;      /* no mass here */
        sp->src[ns] = spec_row(ly->rows, ly->rowc[ks]);
        sp->ks[ns] = ks; sp->st[ns] = s; ++ns;
    }
    /* The targets' rows are taken BEFORE any source row is given back: the
     * sources are still to be read. */
    if (!spec_rows_take(ly->rows, g1 - g0, ly->rown + g0)) return;
    for (uint32_t kk = g0; kk < g1; ++kk)
        sp->dst[kk - g0] = spec_row(ly->rows, ly->rown[kk]);
    /* Each target's edges, sources ascending, and the bins they can write. */
    uint32_t ne = 0, dlo = bins, dhi = 0;
    for (uint32_t kk = g0; kk < g1; ++kk) {
        sp->first[kk - g0] = ne;
        BeamBinRange *dr = &ly->rn[kk];
        dr->lo = bins; dr->hi = 0;
        const uint32_t x = ly->lv_next[kk] % A;
        if (x >= R) continue;
        for (uint32_t j = 0; j < ns; ++j) {
            const double wv = ly->w[(size_t)sp->st[j] * R + x];
            if (wv == -INFINITY) continue;
            const double u = -wv / ly->ds;
            const int64_t k = (int64_t)floor(u);
            const BeamBinRange sr = ly->rc[sp->ks[j]];
            int64_t lo = (int64_t)sr.lo + k, hi = (int64_t)sr.hi + k + 1;
            if (lo < 0) lo = 0;
            if (hi > (int64_t)bins) hi = (int64_t)bins;
            if (lo >= hi) continue;                      /* all off the grid */
            SpecEdge *e = &sp->edge[ne++];
            e->k = k;
            e->f = u - (double)k;
            e->g = 1.0 - e->f;
            if (nm == 1u) e->fac[0] = ly->counting ? 1.0 : exp(wv);
            else { e->fac[0] = 1.0; e->fac[1] = exp(wv); }
            e->src = j;
            e->lo = (uint32_t)lo; e->hi = (uint32_t)hi;
            if (e->lo < dr->lo) dr->lo = e->lo;
            if (e->hi > dr->hi) dr->hi = e->hi;
        }
        if (dr->lo < dlo) dlo = dr->lo;
        if (dr->hi > dhi) dhi = dr->hi;
    }
    sp->first[g1 - g0] = ne;
    for (uint32_t t0 = dlo; t0 < dhi; t0 += SPEC_TILE) {
        const uint32_t t1 = dhi - t0 > SPEC_TILE ? t0 + SPEC_TILE : dhi;
        for (uint32_t kk = g0; kk < g1; ++kk) {
            double *dst = sp->dst[kk - g0];
            for (uint32_t ei = sp->first[kk - g0]; ei < sp->first[kk - g0 + 1u]; ++ei) {
                const SpecEdge *e = &sp->edge[ei];
                const int64_t p0 = e->lo > t0 ? e->lo : t0;
                const int64_t p1 = e->hi < t1 ? e->hi : t1;
                if (p0 >= p1) continue;
                const uint32_t ks = sp->ks[e->src];
                const BeamBinRange sr = ly->rc[ks];
                const double *src = sp->src[e->src];
                for (uint32_t mi = 0; mi < nm; ++mi)
                    spec_move(dst + (size_t)mi * bins, src + (size_t)mi * bins,
                              e->k, e->f, e->g, e->fac[mi], sr.lo, sr.hi, p0, p1);
            }
        }
    }
    /* Each source fed this group only: clear it now, while it is in cache,
     * and hand its row to whichever group needs one next. */
    for (uint32_t j = 0; j < ns; ++j) {
        BeamBinRange *sr = &ly->rc[sp->ks[j]];
        double *row = (double *)sp->src[j];
        for (uint32_t mi = 0; mi < nm; ++mi)
            memset(row + (size_t)mi * bins + sr->lo, 0,
                   (size_t)(sr->hi - sr->lo) * sizeof(double));
        sr->lo = bins; sr->hi = 0;
    }
    spec_rows_give(ly->rows, sp->give, ngive);
}

/* One slice [t0, t1) of the output bins at the sink: every final state's
 * edge, in state order, so each output bin is summed in the order the
 * per-state form summed it, and threads own disjoint bins. */
SPEC_CLONES
static void spec_sink_slice(const SpecEdge *e, uint32_t ne,
                            const BeamBinRange *rc, const SpecRows *rows,
                            const uint32_t *rowc, uint32_t bins, uint32_t nm,
                            double *out_a, double *out_b, uint32_t t0,
                            uint32_t t1) {
    for (uint32_t i = 0; i < ne; ++i) {
        const int64_t p0 = e[i].lo > t0 ? e[i].lo : t0;
        const int64_t p1 = e[i].hi < t1 ? e[i].hi : t1;
        if (p0 >= p1) continue;
        const BeamBinRange sr = rc[e[i].src];
        const double *src = spec_row(rows, rowc[e[i].src]);
        spec_move(out_a, src, e[i].k, e[i].f, e[i].g, e[i].fac[0],
                  sr.lo, sr.hi, p0, p1);
        if (nm == 2u)
            spec_move(out_b, src + bins, e[i].k, e[i].f, e[i].g, e[i].fac[1],
                      sr.lo, sr.hi, p0, p1);
    }
}
#define SPEC_SINK_SLICE 256u

/* ── Scratch, shared by the lengths of one call ──────────────── */

/*
 * Everything one length of the DP needs, sized for the widest layer of every
 * length a call runs, so the lengths of a pooled spectrum share it. Its rows
 * of histograms (SpecRows) are tens of gigabytes on the foundation at 16,384
 * bins: allocated and freed per length, their pages were zeroed by the kernel
 * on first touch and unmapped again for each of 27 lengths, a tenth of the
 * sweep. The rows are all zero between lengths -- each length clears what it
 * wrote and returns every row -- so sharing them changes nothing computed,
 * and the peak is the widest length's, as before.
 */
typedef struct {
    uint32_t maxlive, nm, bins;
    int      nth;
    size_t   stride, per_edges, per_ptrs;
    double  *w, *tail;
    SpecRows rows;
    uint32_t *rowc, *rown;       /* [maxlive] slot -> row, each side         */
    uint8_t  *done;              /* [maxlive] this layer's row given back    */
    BeamBinRange *rc, *rn;
    uint32_t *sc, *sn, *gs;
    uint8_t  *av;
    unsigned char *scr;          /* [nth] a group's scratch (SpecScratch)    */
    SpecEdge *sink;              /* [maxlive] the sink's edges               */
} SpecWork;

static void spec_work_free(SpecWork *wk) {
    for (uint32_t c = 0; c < wk->rows.n_chunks; ++c) free(wk->rows.chunk[c]);
    free(wk->rows.chunk); free(wk->rows.free_ids);
#ifdef _OPENMP
    if (wk->rows.lock_ready) omp_destroy_lock(&wk->rows.lock);
#endif
    free(wk->w); free(wk->tail); free(wk->rowc); free(wk->rown); free(wk->done);
    free(wk->rc); free(wk->rn); free(wk->sc); free(wk->sn); free(wk->gs);
    free(wk->av); free(wk->scr); free(wk->sink);
    memset(wk, 0, sizeof *wk);
}

/* Rows a layer can hold at once, at most: two whole layers (when a layer that
 * grows is processed in an unlucky order) plus every thread's group in
 * flight. The chunk table is sized for that once, so it never moves; chunks
 * themselves are allocated only as rows are actually needed. */
static uint64_t spec_max_rows(uint32_t maxlive, int nth, uint32_t A, uint32_t R) {
    return 2ull * maxlive + (uint64_t)nth * (A + R) + 1u;
}

static BeamError spec_work_new(const BeamModel *m, uint32_t maxlive,
                               uint32_t nm, uint32_t bins, SpecWork *wk) {
    memset(wk, 0, sizeof *wk);
    const uint32_t A = m->n_alphabet, R = m->n_residues, S = m->n_states;
    wk->maxlive = maxlive; wk->nm = nm; wk->bins = bins;
    wk->nth = beam_threads() > 0 ? beam_threads() : 1;
    wk->per_edges = (size_t)A * R * sizeof(SpecEdge);
    wk->per_ptrs = ((size_t)A + R) * sizeof(double *);
    const size_t per_ids = (3u * (size_t)A + R + 1u) * sizeof(uint32_t);
    wk->stride = (wk->per_edges + wk->per_ptrs + per_ids + 63u) & ~(size_t)63u;

    const uint64_t max_rows = spec_max_rows(maxlive, wk->nth, A, R);
    wk->rows.hist = (size_t)nm * bins;
    wk->rows.cap_chunks = (uint32_t)((max_rows + SPEC_CHUNK_ROWS - 1u) / SPEC_CHUNK_ROWS);
    wk->rows.chunk = (double **)calloc(wk->rows.cap_chunks, sizeof(double *));
    wk->rows.free_ids = (uint32_t *)malloc((size_t)wk->rows.cap_chunks * SPEC_CHUNK_ROWS
                                           * sizeof(uint32_t));
#ifdef _OPENMP
    omp_init_lock(&wk->rows.lock);
    wk->rows.lock_ready = 1;
#endif

    wk->w    = (double *)malloc((size_t)S * R * sizeof(double));
    wk->tail = (double *)malloc((size_t)S * sizeof(double));
    wk->rowc = (uint32_t *)malloc((size_t)maxlive * sizeof(uint32_t));
    wk->rown = (uint32_t *)malloc((size_t)maxlive * sizeof(uint32_t));
    wk->done = (uint8_t *)malloc((size_t)maxlive);
    wk->rc   = (BeamBinRange *)malloc((size_t)maxlive * sizeof(BeamBinRange));
    wk->rn   = (BeamBinRange *)malloc((size_t)maxlive * sizeof(BeamBinRange));
    /* state -> slot within its own layer's live list, one map per side */
    wk->sc   = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    wk->sn   = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    wk->gs   = (uint32_t *)malloc(((size_t)maxlive + 1u) * sizeof(uint32_t));
    wk->av   = (uint8_t *)malloc((size_t)S);   /* byte mask for the build */
    wk->scr  = (unsigned char *)malloc((size_t)wk->nth * wk->stride);
    wk->sink = (SpecEdge *)malloc((size_t)maxlive * sizeof(SpecEdge));
    if (!wk->rows.chunk || !wk->rows.free_ids || !wk->w || !wk->tail || !wk->rowc
        || !wk->rown || !wk->done || !wk->rc || !wk->rn || !wk->sc || !wk->sn
        || !wk->gs || !wk->av || !wk->scr || !wk->sink) {
        spec_work_free(wk);
        return BEAM_EALLOC("the per-state surprisal histograms");
    }
    return BEAM_OK;
}

/* The widest layer of length `len`'s frontier; 0 when nothing lives. */
static BeamError spec_maxlive(const BeamModel *m, uint32_t len, uint32_t *out) {
    LiveHandle h;
    const BeamError err = beam_live_acquire(m, len, &h);
    if (err != BEAM_OK) return err;
    uint32_t mx = 0;
    for (uint32_t i = 0; i <= h.L->len; ++i)
        if (h.L->n_live[i] > mx) mx = h.L->n_live[i];
    beam_live_release(&h);
    *out = mx;
    return BEAM_OK;
}

/* ── Surprisal spectrum ──────────────────────────────────────── */

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
                             double *out_a, double *out_b, SpecWork *shared) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t nm = out_b ? 2u : 1u;      /* measures carried per state */

    /*
     * THE HISTOGRAMS ARE INDEXED BY LIVE SLOT, NOT BY STATE, and that is the
     * whole of this function's memory footprint.
     *
     * A per-state grid is `n_states * measures * bins` doubles: 796 MB at
     * order 4 with 512 bins, twice over, which is why `spectra_by_length` has
     * to cap its concurrency on a memory budget instead of on cores. The live
     * set is nothing like that big -- on a 101,373-sequence repertoire the
     * widest layer of a length-15 trellis holds 11,099 states of 194,481, and
     * the mean layer holds 3,502 -- so sizing the grids to the widest layer
     * instead of to the state space is a 17x cut.
     *
     * The frontier also replaces this file's own `trim`. That pass computed
     * BACKWARD liveness only, which on a smoothed model is very nearly every
     * state; `beam_live_acquire` intersects it with forward reachability and
     * caches the result for every other analytic. Restricting to it is
     * bit-identical: a state the walk cannot reach carries no mass, so the
     * terms dropped are exactly zero.
     */
    LiveHandle lh;
    BeamError err = beam_live_acquire(m, len, &lh);
    if (err != BEAM_OK) return err;
    const BeamLive *L = lh.L;

    uint32_t maxlive = 0;
    for (uint32_t i = 0; i <= len; ++i)
        if (L->n_live[i] > maxlive) maxlive = L->n_live[i];
    if (maxlive == 0) { beam_live_release(&lh); return BEAM_OK; }

    /* The caller's scratch when it holds this length, else this call's own. */
    SpecWork own;
    SpecWork *wk = shared;
    if (!wk || wk->rows.failed || wk->maxlive < maxlive || wk->nm != nm
        || wk->bins != bins) {
        err = spec_work_new(m, maxlive, nm, bins, &own);
        if (err != BEAM_OK) { beam_live_release(&lh); return err; }
        wk = &own;
    }
    const size_t hist = (size_t)nm * bins;    /* doubles per row */
    const int nth = wk->nth;
    double *w = wk->w, *tail = wk->tail;
    SpecRows *rows = &wk->rows;
    BeamBinRange *rc = wk->rc, *rn = wk->rn;
    uint32_t *sc = wk->sc, *sn = wk->sn, *gs = wk->gs;
    uint32_t *rowc = wk->rowc, *rown = wk->rown;
    uint8_t *av = wk->av, *done = wk->done;
    bool ok = true;

    for (uint32_t k = 0; k < maxlive; ++k) { rc[k].lo = bins; rc[k].hi = 0; }
    /* Layer 0 is the root alone, so its live list has one entry and the walk
     * starts in that slot. */
    for (uint32_t k = 0; k < L->n_live[0]; ++k) sc[L->live[0][k]] = k;
    {
        const uint32_t root = S - 1u;
        uint32_t rk = 0;
        bool have = false;
        for (uint32_t k = 0; k < L->n_live[0]; ++k)
            if (L->live[0][k] == root) { rk = k; have = true; break; }
        if (!have) {          /* the root cannot reach a sink: no paths */
            if (wk == &own) spec_work_free(&own);
            beam_live_release(&lh);
            return BEAM_OK;
        }
        /* Every slot of layer 0 has a row; the root's carries the unit mass. */
        ok = spec_rows_take(rows, L->n_live[0], rowc);
        if (ok) {
            double *r0 = spec_row(rows, rowc[rk]);
            for (uint32_t mi = 0; mi < nm; ++mi) r0[(size_t)mi * bins + 0] = 1.0;
            rc[rk].lo = 0; rc[rk].hi = 1;
        }
    }

    for (uint32_t i = 0; ok && i < len; ++i) {
        /* Pass the mask this function already computed. The transport loop
         * below tests `av_cur[s]` / `av_next[tt]` before reading a row, so the
         * "dead rows are never read" contract holds and skipping them is
         * bit-identical -- it was simply never wired up. Worth 1.36x on a
         * per-repertoire spectrum. */
        /* `beam_layer_weights_live` takes a byte per state; the frontier
         * stores membership as an ascending list, so paint the mask from it.
         * A memset over the state space per layer is far less than building
         * weights for states no path reaches. */
        memset(av, 0, S);
        for (uint32_t k = 0; k < L->n_live[i]; ++k) av[L->live[i][k]] = 1;
        beam_layer_weights_live(m, i, len, w, NULL, av);
        /*
         * By GROUPS of targets that share their sources (spec_group). Each
         * target's histogram is still one thread's alone, so nothing races,
         * and each bin is still summed in the order the per-target gather
         * summed it.
         */
        const uint32_t *lv_next = L->live[i + 1];
        const uint32_t  n_next  = L->n_live[i + 1];
        const uint32_t  n_cur   = L->n_live[i];
        for (uint32_t k = 0; k < n_next; ++k) sn[lv_next[k]] = k;
        uint32_t ng = 0;
        for (uint32_t k = 0; k < n_next; ++k)
            if (k == 0 || lv_next[k] / A != lv_next[k - 1] / A) gs[ng++] = k;
        gs[ng] = n_next;
        memset(done, 0, n_cur);
        const SpecLayer ly = {
            .lv_next = lv_next, .mask = L->mask[i], .sc = sc, .w = w,
            .rows = rows, .rowc = rowc, .rown = rown, .done = done,
            .rc = rc, .rn = rn,
            .A = A, .R = R, .SK = SK, .bins = bins, .nm = nm, .hist = hist,
            .ds = ds, .counting = counting,
        };
#ifdef _OPENMP
        #pragma omp parallel num_threads(nth)
#endif
        {
#ifdef _OPENMP
            const size_t tid = (size_t)omp_get_thread_num();
#else
            const size_t tid = 0u;
#endif
            unsigned char *base = wk->scr + tid * wk->stride;
            SpecScratch sp;
            sp.edge  = (SpecEdge *)base;
            sp.src   = (const double **)(base + wk->per_edges);
            sp.dst   = (double **)(sp.src + A);
            sp.ks    = (uint32_t *)(base + wk->per_edges + wk->per_ptrs);
            sp.st    = sp.ks + A;
            sp.give  = sp.st + A;
            sp.first = sp.give + A;
#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 1)
#endif
            for (int64_t g = 0; g < (int64_t)ng; ++g)
                spec_group(&ly, gs[g], gs[g + 1], &sp);
        }
        if (rows->failed) { ok = false; break; }
        /* Clear what no group cleared -- a source none of whose targets
         * lives -- and only what was written, so the sweep costs occupied
         * bins rather than the whole grid. Then its row goes back too. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nth)
#endif
        for (int64_t k = 0; k < (int64_t)n_cur; ++k) {
            if (done[k] || rc[k].lo >= rc[k].hi) continue;
            double *row = spec_row(rows, rowc[k]);
            for (uint32_t mi = 0; mi < nm; ++mi)
                memset(row + (size_t)mi * bins + rc[k].lo, 0,
                       (size_t)(rc[k].hi - rc[k].lo) * sizeof(double));
        }
        uint32_t nleft = 0;
        for (uint32_t k = 0; k < n_cur; ++k)
            if (!done[k]) rowc[nleft++] = rowc[k];      /* compact in place */
        spec_rows_give(rows, rowc, nleft);
        { BeamBinRange *t = rc; rc = rn; rn = t; }
        { uint32_t *t = sc; sc = sn; sn = t; }
        { uint32_t *t = rowc; rowc = rown; rown = t; }
    }

    if (ok) {
        /*
         * Sink: the backward tail, plus the per-length constant that turns the
         * accumulated weight into a surprisal. Every final state's histogram
         * lands in the one output, so the threads split the OUTPUT bins
         * between them (spec_sink_slice) and each walks the states in order:
         * the same sums as the per-state form, on every thread rather than one.
         */
        beam_tail_weights(m, len, tail);
        const double off = m->log_z[len] - m->log_pi[len];
        SpecEdge *se = wk->sink;
        uint32_t ne = 0, dlo = bins, dhi = 0;
        for (uint32_t k = 0; k < L->n_live[len]; ++k) {
            const uint32_t s = L->live[len][k];
            if (tail[s] == -INFINITY || rc[k].lo >= rc[k].hi) continue;
            const double u = (-tail[s] + off) / ds;
            const int64_t kk = (int64_t)floor(u);
            int64_t lo = (int64_t)rc[k].lo + kk, hi = (int64_t)rc[k].hi + kk + 1;
            if (lo < 0) lo = 0;
            if (hi > (int64_t)bins) hi = (int64_t)bins;
            if (lo >= hi) continue;                      /* all off the grid */
            SpecEdge *e = &se[ne++];
            e->k = kk;
            e->f = u - (double)kk;
            e->g = 1.0 - e->f;
            if (nm == 1u) e->fac[0] = counting ? 1.0 : exp(tail[s] - off);
            else { e->fac[0] = 1.0; e->fac[1] = exp(tail[s] - off); }
            e->src = k;
            e->lo = (uint32_t)lo; e->hi = (uint32_t)hi;
            if (e->lo < dlo) dlo = e->lo;
            if (e->hi > dhi) dhi = e->hi;
        }
        const int64_t nsl = dhi > dlo ? (int64_t)((dhi - dlo + SPEC_SINK_SLICE - 1u)
                                                  / SPEC_SINK_SLICE) : 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
#endif
        for (int64_t sl = 0; sl < nsl; ++sl) {
            const uint32_t t0 = dlo + (uint32_t)sl * SPEC_SINK_SLICE;
            const uint32_t t1 = dhi - t0 > SPEC_SINK_SLICE ? t0 + SPEC_SINK_SLICE : dhi;
            spec_sink_slice(se, ne, rc, rows, rowc, bins, nm, out_a, out_b, t0, t1);
        }
        /* Leave every row all zero and back in the pool, for the next length
         * that shares it. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nth)
#endif
        for (int64_t k = 0; k < (int64_t)L->n_live[len]; ++k) {
            if (rc[k].lo >= rc[k].hi) continue;
            double *row = spec_row(rows, rowc[k]);
            for (uint32_t mi = 0; mi < nm; ++mi)
                memset(row + (size_t)mi * bins + rc[k].lo, 0,
                       (size_t)(rc[k].hi - rc[k].lo) * sizeof(double));
        }
        spec_rows_give(rows, rowc, L->n_live[len]);
    }

    if (wk == &own) spec_work_free(&own);
    beam_live_release(&lh);
    if (!ok)
        return BEAM_EALLOC("the per-state surprisal histograms");
    return BEAM_OK;
}

/*
 * Every length the model generates, in ascending order, for each measure in
 * turn, all sharing one scratch. A measure is `counts[j]` (true: counting)
 * accumulated into `outs[j]`.
 */
static BeamError spectrum_pooled(const BeamModel *m, const bool *counts,
                                 double *const *outs, uint32_t n_measures,
                                 uint32_t bins, double ds) {
    uint32_t maxlive = 0;
    for (uint32_t L = 1; L < m->n_len_slots; ++L) {
        if (!beam_length_live(m, L)) continue;
        uint32_t mx = 0;
        const BeamError e = spec_maxlive(m, L, &mx);
        if (e != BEAM_OK) return e;
        if (mx > maxlive) maxlive = mx;
    }
    if (maxlive == 0) return BEAM_OK;
    SpecWork wk;
    BeamError e = spec_work_new(m, maxlive, 1u, bins, &wk);
    if (e != BEAM_OK) return e;
    for (uint32_t j = 0; j < n_measures && e == BEAM_OK; ++j)
        for (uint32_t L = 1; L < m->n_len_slots && e == BEAM_OK; ++L) {
            if (!beam_length_live(m, L)) continue;
            e = spectrum_one(m, counts[j], L, bins, ds, outs[j], NULL, &wk);
        }
    spec_work_free(&wk);
    return e;
}

/* Declared in math_internal.h; see there for why it is shared. */
BeamError beam_spectrum_spacing(const BeamModel *m, uint32_t bins,
                                 double spacing, double *out_ds) {
    if (bins < 2)
        return BEAM_ERANGE("%s: a spectrum needs at least 2 bins, got %u",
                               __func__, (unsigned)bins);
    if (spacing > 0.0) { *out_ds = spacing; return BEAM_OK; }

    const double hi = beam_model_accumulator_max(m);
    /*
     * That helper reports failure as 0.0 with the error set, so a zero bound
     * has two causes that must not be conflated: a model that generates
     * nothing, and a sweep that was refused over the memory budget. Reporting
     * the second as the first sent the user looking at their corpus for a
     * problem that was in their budget.
     */
    if (beam_last_error() != BEAM_OK) return beam_last_error();
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
    BeamError e = beam_spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(weights, 0, (size_t)bins * sizeof(double));
    if (length != 0) {
        if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
        return spectrum_one(m, counting, length, bins, ds, weights, NULL, NULL);
    }
    double *const outs[1] = {weights};
    return spectrum_pooled(m, &counting, outs, 1u, bins, ds);
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
    BeamError e = beam_spectrum_spacing(m, bins, spacing, &ds);
    if (e != BEAM_OK) return e;
    if (out_spacing) *out_spacing = ds;

    memset(counting_weights,  0, (size_t)bins * sizeof(double));
    memset(generated_weights, 0, (size_t)bins * sizeof(double));
    /*
     * One measure after the other, not both through one transport. Carrying
     * two measures per row doubled the rows' memory and saved no time -- the
     * work is per bin per measure either way (528 s fused against 2 x 265 s on
     * the foundation at 16,384 bins) -- so two passes over one shared scratch
     * hold half the peak. Both passes use the grid spacing computed once above
     * and split every edge by fractions computed from the same weight, so the
     * two measures are split identically, bit for bit, as before.
     */
    if (length != 0) {
        if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
        e = spectrum_one(m, true, length, bins, ds, counting_weights, NULL, NULL);
        if (e == BEAM_OK)
            e = spectrum_one(m, false, length, bins, ds, generated_weights, NULL, NULL);
        return e;
    }
    const bool counts[2] = {true, false};
    double *const outs[2] = {counting_weights, generated_weights};
    return spectrum_pooled(m, counts, outs, 2u, bins, ds);
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
                               uint32_t nm, uint32_t maxlive) {
    const uint64_t S = m->n_states;
    if (maxlive == 0u || maxlive > S) maxlive = (uint32_t)S;
    /* The row pool's worst case: a growing layer processed in an unlucky
     * order can hold its two neighbours' rows at once. On a full frontier it
     * holds about one layer, so this is an upper bound, kept conservative so
     * the budget is never exceeded. */
    return 2ull * maxlive * nm * bins * sizeof(double)  /* row pool     */
         + 2ull * maxlive * sizeof(BeamBinRange)
         + S * m->n_residues * sizeof(double)           /* edge weights */
         + S * sizeof(double)                           /* tail         */
         + 2ull * S * sizeof(uint32_t)                  /* slot maps    */
         + S;                                           /* byte mask    */
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
    BeamError e = beam_spectrum_spacing(m, bins, spacing, &ds);
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
    /*
     * Sized from the live sets, not from the state space. The grids are now
     * `maxlive * bins` rather than `n_states * bins`, and on a per-repertoire
     * model that is a seventeenth of the memory -- which is the difference
     * between running the lengths one at a time and running one per core.
     *
     * Measuring it means acquiring each length's frontier, which is not extra
     * work: every sweep below acquires the same frontier, and the cache hands
     * back the one built here. The widest layer is NOT the longest length --
     * on a real repertoire length 15 peaks at 11,099 live states and length 21
     * at 289 -- so this takes the max over the work list rather than guessing
     * from one.
     */
    uint32_t maxlive = 0;
    for (uint32_t k = 0; k < nw; ++k) {
        LiveHandle h;
        if (beam_live_acquire(m, work[k], &h) != BEAM_OK) continue;
        for (uint32_t i = 0; i <= h.L->len; ++i)
            if (h.L->n_live[i] > maxlive) maxlive = h.L->n_live[i];
        beam_live_release(&h);
    }

    const uint64_t need = spectrum_bytes(m, bins, 1u, maxlive);
    /*
     * The floor below used to be unconditional, so a budget under one sweep
     * ran it anyway -- the concurrency cap was honoured and the ceiling was
     * not. Refusing here is what makes the budget one number with one meaning.
     */
    if (need && !beam_budget_fits(need)) {
        const uint32_t at = nw ? work[0] : 0u;
        free(work);
        return beam_budget_refuse("a spectrum sweep", at, need);
    }
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
         * still an ACTIVE region, and would serialise every sweep inside it.
         * One after another, the lengths share one scratch. */
        SpecWork wk;
        const BeamError ew = spec_work_new(m, maxlive, 1u, bins, &wk);
        for (uint32_t k = 0; k < nw; ++k) {
            const BeamError err = spectrum_one(m, counting, work[k], bins, ds,
                                              weights + (size_t)work[k] * bins,
                                              NULL, ew == BEAM_OK ? &wk : NULL);
            if (err != BEAM_OK && (int)err > failed) failed = (int)err;
        }
        if (ew == BEAM_OK) spec_work_free(&wk);
    } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
                num_threads((int)conc)
#endif
        for (int64_t k = 0; k < (int64_t)nw; ++k) {
            const uint32_t L = work[k];
            const BeamError err = spectrum_one(m, counting, L, bins, ds,
                                              weights + (size_t)L * bins, NULL,
                                              NULL);
            if (err != BEAM_OK) { if ((int)err > failed) failed = (int)err; }
        }
    }
    free(work);
    if (failed != BEAM_OK)
        return BEAM_FAIL((BeamError)failed,
                        "spectra_by_length: a per-length sweep failed");
    return BEAM_OK;
}
