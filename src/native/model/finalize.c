/**
 * @file finalize.c
 * @brief Dense and sparse table finalization, committed atomically.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"
#include "beam/provenance.h"



/* ── Finalize: the hierarchical prior ────────────────────────── */

/*
 * Every row is the posterior mean of a Dirichlet centred on its parent:
 *
 *     P_k(x) = (n_k(x) + alpha[k] * P_{k-1}(x)) / (n_k + alpha[k])
 *
 * where level k keeps the k residues NEAREST the scored one, level 0 is the
 * depth alone, and P_{-1} is the uniform 1/R. `n_k` is the observed counts
 * summed over the residues level k drops, so no count beyond what the model
 * already keeps is ever stored, and a model stays exactly as expandable as it
 * was.
 *
 * A row nothing was observed at is exactly its parent. The sparse form stores
 * only observed rows, at every level, and `beam_lp_row` resolves the rest by
 * walking to the nearest observed ancestor; the dense form writes that
 * ancestor's row into every unobserved slot. The two are the same numbers.
 *
 * Worked depth by depth. Every level of one depth is a function of that
 * depth's top rows alone, so the scratch below is one depth's worth -- about
 * 3 MB at order 4 -- rather than a table's, and the rows come out in the
 * ascending order the finalized payloads must be written in.
 */

/* One depth's lower levels, in COUNTS packing (most-distant-first), so the
 * level-k context of a top context `c` is `c % lo_ctx[k]` in either
 * direction. Converted to a direction's finalized packing only on output. */
typedef struct {
    double *cnt[BEAM_MAX_ORDER];    /* [lo_ctx[k] * R] summed counts          */
    double *tot[BEAM_MAX_ORDER];    /* [lo_ctx[k]]     row totals             */
    double *prob[BEAM_MAX_ORDER];   /* [lo_ctx[k] * R] log P_k, where tot > 0 */
} HierScratch;

static void hier_scratch_free(HierScratch *h, uint32_t order) {
    for (uint32_t k = 0; k < order; ++k) {
        free(h->cnt[k]); free(h->tot[k]); free(h->prob[k]);
    }
}

static BeamError hier_scratch_alloc(HierScratch *h, const BeamModel *m) {
    memset(h, 0, sizeof *h);
    const uint32_t R = m->n_residues;
    for (uint32_t k = 0; k < m->order; ++k) {
        const size_t nc = m->lo_ctx[k];
        h->cnt[k]  = (double *)malloc(nc * R * sizeof(double));
        h->tot[k]  = (double *)malloc(nc * sizeof(double));
        h->prob[k] = (double *)malloc(nc * R * sizeof(double));
        if (!h->cnt[k] || !h->tot[k] || !h->prob[k]) {
            hier_scratch_free(h, m->order);
            return BEAM_EALLOC("the hierarchical prior's per-depth scratch");
        }
    }
    return BEAM_OK;
}

static void hier_scratch_clear(HierScratch *h, const BeamModel *m) {
    const uint32_t R = m->n_residues;
    for (uint32_t k = 0; k < m->order; ++k) {
        memset(h->cnt[k], 0, (size_t)m->lo_ctx[k] * R * sizeof(double));
        memset(h->tot[k], 0, (size_t)m->lo_ctx[k] * sizeof(double));
    }
}

/* Add one observed top row (counts packing `c`) into every lower level. */
static void hier_accumulate(HierScratch *h, const BeamModel *m, uint32_t c,
                            const double *counts) {
    const uint32_t R = m->n_residues;
    double tot = 0.0;
    for (uint32_t x = 0; x < R; ++x) tot += counts[x];
    for (uint32_t k = 0; k < m->order; ++k) {
        const uint32_t ck = c % m->lo_ctx[k];
        double *dst = h->cnt[k] + (size_t)ck * R;
        for (uint32_t x = 0; x < R; ++x) dst[x] += counts[x];
        h->tot[k][ck] += tot;
    }
}

/*
 * One cell of the rule, in logs:
 *
 *     log P_k(x) = log(n(x) + alpha * P_{k-1}(x)) - log(n + alpha)
 *
 * and for an unobserved residue, n(x) = 0, the same as
 * log(alpha) + log P_{k-1}(x) - log(n + alpha). Written that way because the
 * linear form underflows: a residue unobserved at every level carries
 * alpha^(k+1) / prod(n) of mass, which a small alpha drives below the smallest
 * double within a few levels -- and a cell that underflows to 0 is the closed
 * world back, silently. In logs it is an exact sum of finite terms for every
 * alpha > 0.
 */
static inline double prior_cell_log(double n_x, double a, double log_a,
                                    double log_parent, double log_z) {
    return n_x > 0.0 ? log(n_x + a * exp(log_parent)) - log_z
                     : log_a + log_parent - log_z;
}

/* log P_k for every observed lower row of this depth, bottom-up. */
static void hier_solve(HierScratch *h, const BeamModel *m) {
    const uint32_t R = m->n_residues;
    for (uint32_t k = 0; k < m->order; ++k) {
        const double a = m->cfg.alpha[k], log_a = log(a);
        for (uint32_t ck = 0; ck < m->lo_ctx[k]; ++ck) {
            const double n = h->tot[k][ck];
            if (!(n > 0.0)) continue;
            const double log_z = log(n + a);
            const double *cnt = h->cnt[k] + (size_t)ck * R;
            const double *par = k ? h->prob[k - 1]
                                    + (size_t)(ck % m->lo_ctx[k - 1]) * R
                                  : m->base_row;
            double *out = h->prob[k] + (size_t)ck * R;
            for (uint32_t x = 0; x < R; ++x)
                out[x] = prior_cell_log(cnt[x], a, log_a, par[x], log_z);
        }
    }
}

/*
 * The nearest observed ancestor of a top context `c` (counts packing), as a
 * log row, or NULL when even the depth was never observed and the answer is
 * the uniform row.
 */
static const double *hier_ancestor(const HierScratch *h, const BeamModel *m,
                                   uint32_t c) {
    for (uint32_t k = m->order; k-- > 0; ) {
        const uint32_t ck = c % m->lo_ctx[k];
        if (h->tot[k][ck] > 0.0)
            return h->prob[k] + (size_t)ck * m->n_residues;
    }
    return NULL;
}

/* Finalized packing of a level-k context: most-distant-first forward, the
 * digits reversed (nearest-first) backward. */
static inline uint32_t hier_fin_ctx(const BeamModel *m, uint32_t dir,
                                    uint32_t ck, uint32_t k) {
    return dir == BEAM_BWD ? beam_ctx_reverse(ck, k, m->n_alphabet) : ck;
}

/* Write this depth's observed lower rows into their finalized tables, in
 * ascending finalized row order. */
static BeamError hier_emit_lower(const HierScratch *h, BeamModel *st,
                                 uint32_t dir, uint32_t d) {
    const uint32_t R = st->n_residues;
    for (uint32_t k = 0; k < st->order; ++k) {
        BeamSparseRows *out = &st->sp_lo[dir][k];
        for (uint32_t j = 0; j < st->lo_ctx[k]; ++j) {
            /* j is the FINALIZED index; its counts index is the same digits
             * reversed backward, and reversal is its own inverse. */
            const uint32_t ck = hier_fin_ctx(st, dir, j, k);
            if (!(h->tot[k][ck] > 0.0)) continue;
            double *lp = beam_sparse_row_mut(out, d * st->lo_ctx[k] + j, 0u, true);
            if (!lp) return BEAM_EALLOC("a hierarchical prior row");
            const double *p = h->prob[k] + (size_t)ck * R;
            for (uint32_t x = 0; x < R; ++x) lp[x] = p[x];
        }
    }
    return BEAM_OK;
}

static int cmp_row_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/*
 * The observed top rows of one direction, as DESTINATION rows sorted
 * ascending, which is depth-major. `cnt` is a locally thawed count table or
 * NULL for the model's own.
 */
static BeamError list_top_rows(const BeamModel *m, uint32_t dir,
                               const BeamSparseRows *cnt,
                               uint32_t **rows_out, uint32_t *n_out) {
    *rows_out = NULL; *n_out = 0;
    const BeamSparseRows *ct = cnt ? cnt : &m->sp_cnt[dir];
    if (!(ct && ct->row_ids && ct->size)) return BEAM_OK;
    uint64_t cap = 0;
    for (uint32_t part = 0; part < ct->n_parts; ++part) cap += ct->size[part];
    if (!cap) return BEAM_OK;
    uint32_t *rows = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    if (!rows) return BEAM_EALLOC("the occupied-row list");
    uint32_t n = 0;
    if (!beam_sparse_occupied_rows(ct, rows, cap, &n)) {
        free(rows);
        return BEAM_FAIL(BEAM_ERR_INVALID_ARG,
                         "the count table cannot list its occupied rows, so "
                         "the finalized payload cannot be built in "
                         "destination order");
    }
    if (dir == BEAM_BWD)
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t d = rows[i] / m->n_ctx, c = rows[i] % m->n_ctx;
            rows[i] = d * m->n_ctx + beam_ctx_reverse(c, m->order, m->n_alphabet);
        }
    qsort(rows, n, sizeof(uint32_t), cmp_row_u32);
    *rows_out = rows; *n_out = n;
    return BEAM_OK;
}

/* The count row behind a destination row: its source row and counts. */
static const double *top_counts(const BeamModel *m, uint32_t dir,
                                const BeamSparseRows *cnt, uint32_t dst_row,
                                uint32_t *c_out) {
    const uint32_t d = dst_row / m->n_ctx, j = dst_row % m->n_ctx;
    const uint32_t c = dir == BEAM_BWD
        ? beam_ctx_reverse(j, m->order, m->n_alphabet) : j;
    const uint32_t src = d * m->n_ctx + c;
    *c_out = c;
    return cnt ? beam_sparse_row(cnt, src, beam_part_for_row(m, src))
               : beam_cnt_row(m, dir, src);
}

static double row_total(const double *counts, uint32_t R) {
    double t = 0.0;
    for (uint32_t x = 0; x < R; ++x) t += counts[x];
    return t;
}

/* One observed top row, blended with its level order-1 parent, as logs. */
static void blend_top(const BeamModel *m, const HierScratch *h, uint32_t c,
                      const double *counts, double tot, double *lp) {
    const uint32_t R = m->n_residues;
    const uint32_t k = m->order - 1u;
    const double *par = h->prob[k] + (size_t)(c % m->lo_ctx[k]) * R;
    const double a = m->cfg.alpha[m->order], log_a = log(a);
    const double log_z = log(tot + a);
    for (uint32_t x = 0; x < R; ++x)
        lp[x] = prior_cell_log(counts[x], a, log_a, par[x], log_z);
}

static BeamError finalize_sparse_hier(BeamModel *st, uint32_t dir,
                                      const BeamSparseRows *cnt) {
    const uint32_t R = st->n_residues;
    const uint32_t nr = beam_count_n_rows(st);
    BeamError err = beam_sparse_init(&st->sp_lp[dir], nr, 1u, R, false);
    for (uint32_t k = 0; k < st->order && err == BEAM_OK; ++k)
        err = beam_sparse_init(&st->sp_lo[dir][k], st->max_len * st->lo_ctx[k],
                               1u, R, false);
    if (err != BEAM_OK) return err;

    uint32_t *rows = NULL, n_rows = 0;
    err = list_top_rows(st, dir, cnt, &rows, &n_rows);
    if (err != BEAM_OK) return err;
    if (n_rows) {
        err = beam_sparse_reserve(&st->sp_lp[dir], 0u, n_rows);
        if (err != BEAM_OK) { free(rows); return err; }
    }
    HierScratch h;
    err = hier_scratch_alloc(&h, st);
    if (err != BEAM_OK) { free(rows); return err; }

    for (uint32_t a = 0; a < n_rows && err == BEAM_OK; ) {
        const uint32_t d = rows[a] / st->n_ctx;
        uint32_t b = a;
        while (b < n_rows && rows[b] / st->n_ctx == d) ++b;

        hier_scratch_clear(&h, st);
        for (uint32_t i = a; i < b; ++i) {
            uint32_t c;
            const double *counts = top_counts(st, dir, cnt, rows[i], &c);
            if (counts && row_total(counts, R) > 0.0)
                hier_accumulate(&h, st, c, counts);
        }
        hier_solve(&h, st);
        err = hier_emit_lower(&h, st, dir, d);
        for (uint32_t i = a; i < b && err == BEAM_OK; ++i) {
            uint32_t c;
            const double *counts = top_counts(st, dir, cnt, rows[i], &c);
            if (!counts) continue;
            const double tot = row_total(counts, R);
            if (!(tot > 0.0)) continue;
            double *lp = beam_sparse_row_mut(&st->sp_lp[dir], rows[i], 0u, true);
            if (!lp) { err = BEAM_EALLOC("a sparse probability row"); break; }
            blend_top(st, &h, c, counts, tot, lp);
        }
        a = b;
    }
    hier_scratch_free(&h, st->order);
    free(rows);
    if (err != BEAM_OK) return err;

    beam_sparse_shrink(&st->sp_lp[dir]);
    for (uint32_t k = 0; k < st->order; ++k) {
        beam_sparse_shrink(&st->sp_lo[dir][k]);
        err = beam_sparse_compact_finalized(&st->sp_lo[dir][k],
                                            st->max_len * st->lo_ctx[k],
                                            false, NULL);
        if (err != BEAM_OK) return err;
    }
    /*
     * Compaction is `storage/compaction.c`: four transforms whose order is
     * load-bearing. The lower levels above skip the plane transpose, which is
     * laid out for the edge kernel's backward sweep over the TOP table only.
     */
    return beam_sparse_compact_finalized(&st->sp_lp[dir], nr, dir == BEAM_BWD,
                                         NULL);
}

static BeamError finalize_sparse_direction(BeamModel *st, uint32_t dir) {
    /*
     * One expanded count table alive at a time, when the model arrives
     * frozen -- a load, or a re-finalize after freezing. A model whose counts
     * are already mutable has nothing to expand.
     */
    if (st->counts_frozen) {
        BeamSparseRows local;
        BeamError err = beam_model_thaw_count_table(st, dir, &local);
        if (err != BEAM_OK) return err;
        err = finalize_sparse_hier(st, dir, &local);
        beam_sparse_destroy(&local);
        return err;
    }
    return finalize_sparse_hier(st, dir, NULL);
}

/*
 * The dense form: every one of `max_len * n_ctx` rows written, observed ones
 * blended with their parent and the rest copied from their nearest observed
 * ancestor. Bit-identical to what the sparse lookup resolves. The lower
 * levels are kept as well, so both forms carry the same finalized state.
 */
static BeamError finalize_dense_direction(BeamModel *st, uint32_t dir) {
    const uint32_t R = st->n_residues, nc = st->n_ctx;
    BeamError err = BEAM_OK;
    for (uint32_t k = 0; k < st->order && err == BEAM_OK; ++k)
        err = beam_sparse_init(&st->sp_lo[dir][k], st->max_len * st->lo_ctx[k],
                               1u, R, false);
    if (err != BEAM_OK) return err;
    HierScratch h;
    err = hier_scratch_alloc(&h, st);
    if (err != BEAM_OK) return err;

    for (uint32_t d = 0; d < st->max_len && err == BEAM_OK; ++d) {
        hier_scratch_clear(&h, st);
        for (uint32_t c = 0; c < nc; ++c) {
            const double *counts = beam_cnt_row(st, dir, d * nc + c);
            if (row_total(counts, R) > 0.0) hier_accumulate(&h, st, c, counts);
        }
        hier_solve(&h, st);
        err = hier_emit_lower(&h, st, dir, d);
        for (uint32_t j = 0; j < nc && err == BEAM_OK; ++j) {
            uint32_t c;
            const double *counts = top_counts(st, dir, NULL, d * nc + j, &c);
            double *lp = st->lp[dir] + ((size_t)d * nc + j) * R;
            const double tot = row_total(counts, R);
            if (tot > 0.0) { blend_top(st, &h, c, counts, tot, lp); continue; }
            const double *anc = hier_ancestor(&h, st, c);
            for (uint32_t x = 0; x < R; ++x)
                lp[x] = anc ? anc[x] : st->base_row[x];
        }
    }
    hier_scratch_free(&h, st->order);
    if (err != BEAM_OK) return err;
    for (uint32_t k = 0; k < st->order && err == BEAM_OK; ++k) {
        beam_sparse_shrink(&st->sp_lo[dir][k]);
        err = beam_sparse_compact_finalized(&st->sp_lo[dir][k],
                                            st->max_len * st->lo_ctx[k],
                                            false, NULL);
    }
    return err;
}

/**
 * Build a complete set of log-probability tables into `st`, which must be a
 * shallow copy of the model carrying its dimensions and its COUNTS but with
 * every table pointer NULL.
 *
 * Nothing here touches the caller's model, which is the point: the tables it
 * already has stay readable and stay correct until a full replacement set
 * exists. See `beam_model_finalize` for why that matters.
 */
static BeamError finalize_tables(BeamModel *st) {
    bool ok = true;
    if (!st->cfg.sparse) {
        for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
            st->lp[d] = (double *)malloc(beam_n_table(st) * sizeof(double));
            ok = st->lp[d] != NULL;
        }
    }
    st->log_pi = (double *)malloc(st->n_len_slots * sizeof(double));
    st->log_z  = (double *)malloc(st->n_len_slots * sizeof(double));
    if (!ok || !st->log_pi || !st->log_z)
        return BEAM_FAIL(BEAM_ERR_ALLOC,
                        "finalize: out of memory allocating the "
                        "log-probability tables (%llu cells)",
                        (unsigned long long)beam_model_n_cells(st));

    /*
     * Length prior: the same Dirichlet smoothing as the tables, centred on the
     * even distribution over every length the model represents,
     *
     *     P(L) = (n_L + alpha_length / max_len) / (N + alpha_length),
     *
     * so a length the corpus never produced is improbable, not impossible.
     */
    double tot = 0.0;
    for (uint32_t L = 0; L < st->n_len_slots; ++L) tot += st->cnt_length[L];
    if (!(tot > 0.0))
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT,
                        "finalize: the length histogram is empty, so there is "
                        "nothing to build a length prior from");
    {
        const double a = st->cfg.alpha_length;
        const double even = a / (double)st->max_len;
        st->log_pi[0] = -INFINITY;
        for (uint32_t L = 1; L < st->n_len_slots; ++L)
            st->log_pi[L] = log((st->cnt_length[L] + even) / (tot + a));
    }

    /* Every cell of every level is positive: each is a blend with a parent
     * whose cells are positive, down to the uniform row. */
    st->tables_all_finite = true;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        BeamError e = st->cfg.sparse
                   ? finalize_sparse_direction(st, d)
                   : finalize_dense_direction(st, d);
        if (e != BEAM_OK)
            return BEAM_FAIL(e, "finalize: smoothing the %s tables failed",
                            d == BEAM_FWD ? "N-anchored" : "C-anchored");
    }

    /* Partition functions last: they read the finished tables. */
    st->finalized = true;         /* beam_compute_log_z reads the tables */
    {
        uint32_t *work = (uint32_t *)malloc((size_t)st->n_len_slots
                                            * sizeof(uint32_t));
        if (!work) return BEAM_EALLOC("the per-length work list");
        uint32_t nw = 0;
        /* Longest first: these are independent dynamic programs whose cost
         * grows with the length, and the longest one handed out last is the
         * one every other thread ends up waiting for. */
        for (uint32_t L = st->n_len_slots; L-- > 1; ) {
            if (isfinite(st->log_pi[L])) work[nw++] = L;
            else st->log_z[L] = -INFINITY;
        }
        st->log_z[0] = -INFINITY;

        /*
         * Lengths that run at once divide the budget, like every other
         * by-length driver in the library. This one did not, and ran at
         * `beam_threads()` regardless: sixteen concurrent sweeps of 31 MB
         * apiece is 294 MB of scratch that the budget never saw and that
         * finalize did not need -- measured, eight threads finalize the
         * foundation in 5.26 s against 5.34 s at sixteen.
         *
         * Refused HERE rather than inside the loop, for the reason the budget
         * header gives: `beam_set_error` is thread-local, so the refusal a
         * worker raises reaches this function as a bare code with its message
         * -- the one naming the figure and the way out -- already lost.
         */
        const uint64_t per_len = beam_logz_sweep_bytes(st->n_states,
                                                       st->n_residues,
                                                       st->state_shift);
        const uint32_t conc = beam_budget_concurrency(per_len, nw ? nw : 1u);
        if (nw && conc == 0u) {
            const uint32_t at = work[0];
            free(work);
            return beam_budget_refuse("the partition-function trellis",
                                      at, per_len);
        }

        int failed = BEAM_OK;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
                num_threads(conc ? (int)conc : 1)
#endif
        for (int64_t k = 0; k < (int64_t)nw; ++k) {
            double z = 0.0;
            const BeamError e = beam_compute_log_z(st, work[k], &z, NULL);
            if (e != BEAM_OK) { if ((int)e > failed) failed = (int)e; }
            else st->log_z[work[k]] = z;
        }
        free(work);
        if (failed != BEAM_OK)
            return BEAM_FAIL((BeamError)failed,
                            "finalize: a partition function could not be "
                            "computed");
    }

    /*
     * Restrict the length prior to lengths that actually have a partition
     * function, and renormalise.
     *
     * A length whose trellis has no path would otherwise keep its prior mass
     * while contributing nothing, so `sum_s P(s)` would silently stop being a
     * probability and every downstream identity (the Mellin transform at
     * q = 1, the generated spectrum, the entropy) would be wrong by exactly
     * the missing mass. With every cell positive this is a no-op, which is
     * the point: it costs one pass and removes a whole class of silent error.
     */
    double live = 0.0;
    for (uint32_t L = 1; L < st->n_len_slots; ++L)
        if (isfinite(st->log_pi[L]) && isfinite(st->log_z[L]))
            live += exp(st->log_pi[L]);
    if (!(live > 0.0))
        return BEAM_FAIL(BEAM_ERR_NO_LIVE_PATHS,
                        "finalize: no length has a finite partition function, "
                        "so the model can generate nothing");
    const double shift = log(live);
    for (uint32_t L = 0; L < st->n_len_slots; ++L) {
        if (L >= 1 && isfinite(st->log_pi[L]) && isfinite(st->log_z[L]))
            st->log_pi[L] -= shift;
        else
            st->log_pi[L] = -INFINITY;
    }
    return BEAM_OK;
}

/**
 * Smooth the counts into log-probability tables.
 *
 * The new tables are built in full before the old ones are released, so a
 * finalize that fails leaves the model exactly as it was rather than stripped
 * of its tables with `finalized` clear — a state in which `log_pseq` silently
 * returns -inf and every analytic entry point errors. That mattered most on
 * the path this exists for: `subtract` followed by `finalize`, where the
 * counts are already the only copy.
 *
 * The cost is that peak resident memory during a finalize holds two table
 * sets rather than one. See `beam_model_memory_bytes()` for the figure.
 *
 * NOT safe to call while another thread is reading the model. Building before
 * freeing removes the long window in which the tables were absent, but the
 * commit below is a run of ordinary pointer stores followed by a free, so a
 * concurrent reader can still observe a half-installed table set or a pointer
 * that has just been released. Callers that release the GIL must serialise
 * this against every reader.
 */
static BeamError finalize_body(BeamModel *m) {
    if (!m) return BEAM_FAIL(BEAM_ERR_NULL_ARG, "finalize: model must not be NULL");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (m->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "finalize: the model is compacted, so its counts are "
                        "gone and it cannot be re-finalized");
    /* Re-finalizing reads the counts, so restore the mutable form first.
     * A no-op on the first finalize, which is the only one most models see. */
    {
        /* Left frozen when `finalize_sparse_direction` will expand one table
         * at a time; expanding both here is what made the peak 2.4x the
         * finished model. */
        if (!m->cfg.sparse) {
            const BeamError thaw = beam_model_thaw_counts(m);
            if (thaw != BEAM_OK) return thaw;
        }
    }
    if (m->n_observed == 0)
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT,
                        "finalize: no usable sequence was observed (%llu were "
                        "skipped)", (unsigned long long)m->n_skipped);

    /* Dimensions and counts are shared with `m`; the tables are not. */
    BeamModel st = *m;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        st.lp[d] = NULL;
        memset(&st.sp_lp[d], 0, sizeof(st.sp_lp[d]));
        memset(st.sp_lo[d], 0, sizeof(st.sp_lo[d]));
    }
    st.log_pi = NULL; st.log_z = NULL; st.finalized = false;

    BeamError err = finalize_tables(&st);
    if (err != BEAM_OK) { beam_free_tables(&st); return err; }

    /* Commit: install the new set, then release the old one. */
    BeamModel prev = *m;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        m->lp[d] = st.lp[d];
        m->sp_lp[d] = st.sp_lp[d];
        memcpy(m->sp_lo[d], st.sp_lo[d], sizeof(st.sp_lo[d]));
    }
    m->log_pi = st.log_pi;
    m->log_z  = st.log_z;
    m->tables_all_finite = st.tables_all_finite;
    m->finalized = true;
    beam_free_tables(&prev);
    /* Keep growth slack while a builder is ingesting repeated batches; doing
     * an exact shrink after every observe/add would recopy all accumulated
     * rows and turn one-row streaming into quadratic work. Finalization is
     * the natural once-per-snapshot boundary at which to release that slack. */
    if (m->cfg.sparse) {
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            beam_sparse_shrink(&m->sp_cnt[d]);
        }
    }

    /*
     * Observation is over, so the counts move to the shape their remaining
     * consumers actually want: sorted, no directory, cells that are zero
     * costing a bit rather than eight bytes. Measured 16.4x smaller on a
     * 197,139-sequence repertoire.
     *
     * Failing to freeze is not a reason to fail the finalize. The tables are
     * built and correct at this point, and the mutable counts are a valid
     * thing to keep carrying -- it is exactly what every model did before.
     * So the error is cleared and the model keeps the larger form.
     */
    if (beam_model_freeze_counts(m) != BEAM_OK) beam_clear_error();

    /*
     * Release the mutable count tables when the frozen ones are authoritative.
     *
     * Nothing wrote them on the path above -- each table was expanded into a
     * local and destroyed with it -- but the loader allocates their
     * directories, four bytes per dense row whether or not a row exists, and
     * `beam_model_freeze_counts` returns early on an already-frozen model
     * rather than clearing them. That left 42,007,896 B of empty directory on
     * the foundation, which is exactly what the thaw-then-freeze cycle used to
     * dispose of as a side effect. Zeroed is the state that cycle left them
     * in, and what `beam_model_thaw_counts` overwrites.
     */
    if (m->cfg.sparse && m->counts_frozen) {
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            beam_sparse_destroy(&m->sp_cnt[d]);
        }
    }
    return BEAM_OK;
}

/*
 * The public entry, which exists only to time the body. Aggregated across
 * re-finalizes, because re-finalizing is work the model has had done to it.
 */
BeamError beam_model_finalize(BeamModel *m) {
    const int64_t t0 = beam_now_nanos();
    const BeamError e = finalize_body(m);
    if (m) m->finalize_nanos += beam_now_nanos() - t0;
    return e;
}
