/**
 * @file finalize.c
 * @brief Dense and sparse table finalization, committed atomically.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"


/* ── Finalize ────────────────────────────────────────────────── */

/**
 * Row-normalise counts straight into probabilities.
 *
 * The whole backoff hierarchy collapses to this when every kappa is zero:
 * `backoff_level` computes `cr[x] + kappa * pr[x]`, so a zero kappa erases
 * the parent and leaves the row's own counts. The marginals feeding those
 * parents are then built and discarded -- at order 4 that is the order-0..2
 * mid hierarchy plus a depth-aggregated copy, per direction, and it costs
 * about 1.6 SECONDS on every strict build regardless of how much data the
 * model holds. `strict` defaults kappa to zero, so every strict model paid
 * it, and so did every load, since the file stores counts and re-finalizes.
 *
 * Bit-identical, not approximately: `cr[x] + 0.0 * pr[x]` is `cr[x]` exactly
 * for any finite `pr[x]`, and the parents are probabilities in [0, 1].
 */
static void normalise_rows(const double *counts, int64_t n_rows,
                           uint32_t R, double *out) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t row = 0; row < n_rows; ++row) {
        const double *cr = counts + (size_t)row * R;
        double *orow = out + (size_t)row * R;
        double tot = 0.0;
        for (uint32_t x = 0; x < R; ++x) tot += cr[x];
        if (tot > 0.0)
            for (uint32_t x = 0; x < R; ++x) orow[x] = cr[x] / tot;
        else
            for (uint32_t x = 0; x < R; ++x) orow[x] = 0.0;
    }
}

static BeamError finalize_direction(BeamModel *m, uint32_t dir) {
    const uint32_t W  = m->max_len;
    const uint32_t A  = m->n_alphabet;
    const uint32_t R  = m->n_residues;
    const uint32_t dc = m->cfg.depth_cap;
    BeamError err = BEAM_OK;

    /* Declared together and freed on one path, so every early exit is safe.
     *
     * Sized by the largest order any alphabet allows rather than by the mid
     * expert's own 3, because they are indexed by a value derived from
     * `m->order` below. They were [4] while no order could exceed 4, and a
     * nucleotide model at its default order 6 read two slots past the end. */
    double *mid_p[BEAM_MAX_ORDER + 1] = {0};
    double *agg_p[BEAM_MAX_ORDER + 1] = {0};
    double *agg_counts   = NULL;

    /* Every backoff term is multiplied by a kappa. With all of them zero the
     * parents are dead weight, and building them is the single largest fixed
     * cost in the library. See normalise_rows above. */
    const bool no_backoff = (m->cfg.kappa[0] == 0.0 && m->cfg.kappa[1] == 0.0
                             && m->cfg.kappa[2] == 0.0
                             && m->cfg.kappa_wide == 0.0
                             && m->cfg.kappa_len == 0.0);

    if (no_backoff) {
        /* Only the top of the mid hierarchy survives, and it is just the
         * normalised counts. Nothing else in this function will read a
         * parent, so nothing else is built. */
        mid_p[3] = (double *)malloc((size_t)W * beam_ipow_u32(A, 3u) * R
                                    * sizeof(double));
        if (!mid_p[3]) {
            err = BEAM_EALLOC("a mid-expert backoff level"); goto done;
        }
        normalise_rows(m->cnt_mid[dir], (int64_t)W * beam_ipow_u32(A, 3u), R,
                       mid_p[3]);
    } else {
        /* Order 0..3 probabilities over the mid counts, at full depth. */
        for (uint32_t j = 0; j <= 3; ++j) {
            mid_p[j] = (double *)malloc((size_t)W * beam_ipow_u32(A, j) * R * sizeof(double));
            if (!mid_p[j]) {
                err = BEAM_EALLOC("a mid-expert backoff level"); goto done;
            }
        }
        err = beam_build_hierarchy(m->cnt_mid[dir], W, 3u, A, R, m->cfg.kappa, mid_p,
                              m->cfg.strict);
        if (err != BEAM_OK) goto done;
    }

    /* The wide expert's fine-depth table has one bucket for every depth from
     * depth_cap upward, so its backoff base at that bucket must come from
     * counts aggregated the same way. */
    /* The wide expert's depth-aggregated backoff base, needed only when
     * kappa_wide is non-zero. Under no_backoff nothing reads it. */
    if (!no_backoff) {
        agg_counts = (double *)calloc((size_t)m->n_ctx_mid * R, sizeof(double));
        if (!agg_counts) {
            err = BEAM_EALLOC("the depth-aggregated counts"); goto done;
        }
        {
            const int64_t ncell = (int64_t)m->n_ctx_mid * R;
            /* Summed cell by cell rather than depth by depth, so each cell is one
             * thread's own and the depths still add in ascending order. */
    #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(beam_threads())
    #endif
            for (int64_t i = 0; i < ncell; ++i) {
                double acc = 0.0;
                for (uint32_t d = dc; d < W; ++d)
                    acc += m->cnt_mid[dir][(size_t)d * ncell + i];
                agg_counts[i] = acc;
            }
        }
        for (uint32_t j = 0; j <= 3; ++j) {
            agg_p[j] = (double *)malloc((size_t)beam_ipow_u32(A, j) * R * sizeof(double));
            if (!agg_p[j]) {
                err = BEAM_EALLOC("an aggregated backoff level"); goto done;
            }
        }
        err = beam_build_hierarchy(agg_counts, 1u, 3u, A, R, m->cfg.kappa, agg_p,
                              m->cfg.strict);
        if (err != BEAM_OK) goto done;
    }

    /* ── mid expert: order 3 at full depth ── */
    memcpy(m->lp_mid[dir], mid_p[3], beam_n_mid(m) * sizeof(double));
    if (beam_to_log_inplace(m->lp_mid[dir], beam_n_mid(m), m->cfg.strict))
        m->tables_all_finite = false;

    /* ── length-conditioned expert: order 2, backing off to the order-2
     *    marginal of the mid counts at the same depth ── */
    {
        const int64_t n_rows = (int64_t)m->n_len_slots * W * m->n_ctx_len;
        const uint32_t n2 = m->n_ctx_len;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t row = 0; row < n_rows; ++row) {
            const uint32_t c = (uint32_t)(row % n2);
            const uint32_t d = (uint32_t)((row / n2) % W);
            const double *cr = m->cnt_len[dir] + (size_t)row * R;
            double *orow = m->lp_len[dir] + (size_t)row * R;
            double tot = 0.0;
            if (no_backoff) {
                /* mid_p[2] is not built in this mode; the term it feeds is
                 * multiplied by a zero kappa, so the row is its own counts. */
                for (uint32_t x = 0; x < R; ++x) { orow[x] = cr[x]; tot += cr[x]; }
            } else {
                const double *base = mid_p[2] + ((size_t)d * n2 + c) * R;
                for (uint32_t x = 0; x < R; ++x) {
                    orow[x] = cr[x] + m->cfg.kappa_len * base[x];
                    tot += orow[x];
                }
            }
            for (uint32_t x = 0; x < R; ++x)
                orow[x] = tot > 0.0 ? orow[x] / tot : 0.0;
        }
    }
    if (beam_to_log_inplace(m->lp_len[dir], beam_n_len(m), m->cfg.strict))
        m->tables_all_finite = false;

    /* ── wide expert: order K at capped depth, backing off to the order-(K-1)
     *    table at the matching depth bucket, shared across the most distant
     *    context residue ── */
    {
        /*
         * The base is the order below, capped at 3.
         *
         * The wide expert takes ONE backoff step, weighted by `kappa_wide`,
         * onto a lower-order table. That table comes from the mid hierarchy,
         * which is built over the order-3 mid counts and therefore stops at
         * order 3 -- and cannot go further, because `kappa` has three entries
         * and there is no measured strength for a fourth level. No count
         * table in the model holds order-4 context at full depth either.
         *
         * For every amino-acid order this changes nothing: the ceiling there
         * is 4, so `order - 1` is already at most 3. It is what makes a
         * nucleotide model above order 4 back off one bigger step rather than
         * index a level that was never built.
         */
        const uint32_t k1     = m->order - 1u > 3u ? 3u : m->order - 1u;
        const uint32_t n_ctx1 = beam_ipow_u32(A, k1);
        const uint32_t nw     = m->n_ctx_wide;
        const int64_t  n_rows = (int64_t)m->n_depth_wide * nw;
        /*
         * The C-anchored table is written NEAREST-first, one row per context
         * but not at the context's own index. Everything feeding this loop —
         * the counts, both backoff hierarchies — keeps the common
         * most-distant-first packing, so the relabelling is confined to the
         * destination address of the finished row and nothing upstream, the
         * file format included, has to know about it. See the long note in
         * edge_model_internal.h for what it buys.
         */
        const bool reverse = (dir == BEAM_BWD);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t row = 0; row < n_rows; ++row) {
            const uint32_t d = (uint32_t)(row / nw);
            const uint32_t c = (uint32_t)(row % nw);
            const double *cr = m->cnt_wide[dir] + (size_t)row * R;
            const uint32_t dst = reverse ? beam_ctx_reverse(c, m->order, m->n_alphabet) : c;
            double *orow = m->lp_wide[dir] + ((size_t)d * nw + dst) * R;
            double tot = 0.0;
            if (no_backoff) {
                for (uint32_t x = 0; x < R; ++x) { orow[x] = cr[x]; tot += cr[x]; }
            } else {
                const double *base_tab = (d < dc)
                    ? mid_p[k1] + (size_t)d * n_ctx1 * R
                    : agg_p[k1];
                /* Keep the k1 residues nearest the scored one, dropping
                 * the rest. That is one residue at every amino-acid order,
                 * and `order - 3` of them once the base is capped. */
                const double *base = base_tab + (size_t)(c % n_ctx1) * R;
                for (uint32_t x = 0; x < R; ++x) {
                    orow[x] = cr[x] + m->cfg.kappa_wide * base[x];
                    tot += orow[x];
                }
            }
            for (uint32_t x = 0; x < R; ++x)
                orow[x] = tot > 0.0 ? orow[x] / tot : 0.0;
        }
    }
    if (beam_to_log_inplace(m->lp_wide[dir], beam_n_wide(m), m->cfg.strict))
        m->tables_all_finite = false;

done:
    free(agg_counts);
    for (uint32_t j = 0; j <= BEAM_MAX_ORDER; ++j) {
        free(agg_p[j]); free(mid_p[j]);
    }
    return err;
}

/** Select one sparse count/log table pair. */
static BeamSparseRows *sparse_log_table(BeamModel *m, uint32_t dir,
                                            BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->sp_lp_wide[dir];
    if (kind == BEAM_TABLE_MID) return &m->sp_lp_mid[dir];
    return &m->sp_lp_len[dir];
}

/**
 * Finalize one zero-backoff strict table row by row.
 *
 * The arithmetic deliberately mirrors the dense loops: sum x=0..19, divide
 * each cell by that total, then take its log. Missing count rows are omitted
 * and therefore read as twenty -inf values. The backward wide table is the
 * sole packing exception; iterating DESTINATION rows and reversing back to
 * the count row keeps its sparse payload sorted in the order the DP scans.
 */
static int cmp_row_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static const BeamSparseRows *sparse_count_table_const(const BeamModel *m,
                                                         uint32_t dir,
                                                         BeamTableKind kind) {
    if (kind == BEAM_TABLE_WIDE) return &m->sp_cnt_wide[dir];
    if (kind == BEAM_TABLE_MID)  return &m->sp_cnt_mid[dir];
    return &m->sp_cnt_len[dir];
}

static BeamError finalize_sparse_table(BeamModel *m, uint32_t dir,
                                      BeamTableKind kind,
                                      bool reverse_wide) {
    const uint32_t nr = beam_count_n_rows(m, kind);
    BeamSparseRows *out = sparse_log_table(m, dir, kind);
    BeamError err = beam_sparse_init(out, nr, 1u, m->n_residues, false);
    if (err != BEAM_OK) return err;

    /*
     * Walk the occupied count rows, do not scan the dense index space.
     *
     * The obvious loop is `for (dst_row = 0; dst_row < nr; ++dst_row)` with a
     * miss test inside, and it costs 7,390,278 row lookups per model at order
     * 4 -- both directions, all three tables -- however little was observed.
     * On a 100-sequence model that is seven million probes to find about four
     * hundred rows, and it was 1.6 SECONDS of fixed cost on every strict
     * build and every load. Count tables are mutable and therefore carry the
     * slot -> row reverse map, so the occupied rows can simply be listed.
     *
     * The list is sorted by DESTINATION row because the finalized payload has
     * to stay in the order the dynamic programs scan it; the backward wide
     * table is the one place those differ.
     */
    const BeamSparseRows *ct = sparse_count_table_const(m, dir, kind);
    uint32_t *rows = NULL, n_rows = 0;
    if (m->cfg.sparse && ct && ct->row_ids && ct->size) {
        uint64_t cap = 0;
        for (uint32_t part = 0; part < ct->n_parts; ++part) cap += ct->size[part];
        if (cap) {
            rows = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
            if (!rows) return BEAM_EALLOC("the occupied-row list");
            for (uint32_t part = 0; part < ct->n_parts; ++part) {
                if (!ct->row_ids[part]) continue;
                for (uint32_t k = 0; k < ct->size[part]; ++k) {
                    const uint32_t src = ct->row_ids[part][k];
                    uint32_t dst = src;
                    if (reverse_wide) {
                        const uint32_t d = src / m->n_ctx_wide;
                        const uint32_t c = src % m->n_ctx_wide;
                        dst = d * m->n_ctx_wide + beam_ctx_reverse(c, m->order, m->n_alphabet);
                    }
                    rows[n_rows++] = dst;
                }
            }
            qsort(rows, n_rows, sizeof(uint32_t), cmp_row_u32);
        }
    }

    for (uint32_t idx = 0; idx < (rows ? n_rows : nr); ++idx) {
        const uint32_t dst_row = rows ? rows[idx] : idx;
        uint32_t src_row = dst_row;
        if (reverse_wide) {
            const uint32_t d = dst_row / m->n_ctx_wide;
            const uint32_t c = dst_row % m->n_ctx_wide;
            src_row = d * m->n_ctx_wide + beam_ctx_reverse(c, m->order, m->n_alphabet);
        }
        const double *counts = beam_count_row(m, dir, kind, src_row);
        if (!counts) continue;
        double tot = 0.0;
        for (uint32_t x = 0; x < m->n_residues; ++x) tot += counts[x];
        if (!(tot > 0.0)) continue;
        double *logp = beam_sparse_row_mut(out, dst_row, 0u, true);
        if (!logp) return BEAM_EALLOC("a sparse probability row");
        for (uint32_t x = 0; x < m->n_residues; ++x)
            logp[x] = log(counts[x] / tot);
    }
    free(rows);
    beam_sparse_shrink(out);

    /*
     * A table almost none of whose rows exist spends nearly all of itself on
     * a directory of absences: 28.19 MiB per generation at order 4 whether the
     * corpus was nine hundred million sequences or nine hundred. Below the
     * measured threshold, swap that directory for the rank form -- see
     * BEAM_RANK_SEAL_MAX_OCCUPANCY_DENOM for what was measured and why the
     * line is where it is.
     *
     * Sealing is exact: the same rows, the same slots, the same doubles. It
     * happens here, before `beam_compute_log_z` reads these tables, so the
     * partition functions are computed against the representation the model
     * will actually keep rather than against one it is about to discard.
     */
    const uint32_t occupied = out->size ? out->size[0] : 0u;
    if ((uint64_t)occupied * BEAM_RANK_SEAL_MAX_OCCUPANCY_DENOM <= (uint64_t)nr)
        return beam_sparse_seal_rank(out);
    return BEAM_OK;
}

static BeamError finalize_sparse_direction(BeamModel *m, uint32_t dir) {
    BeamError err = finalize_sparse_table(m, dir, BEAM_TABLE_MID, false);
    if (err != BEAM_OK) return err;
    err = finalize_sparse_table(m, dir, BEAM_TABLE_LEN, false);
    if (err != BEAM_OK) return err;
    return finalize_sparse_table(m, dir, BEAM_TABLE_WIDE,
                                 dir == BEAM_BWD);
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
            st->lp_wide[d] = (double *)malloc(beam_n_wide(st) * sizeof(double));
            st->lp_mid[d]  = (double *)malloc(beam_n_mid(st)  * sizeof(double));
            st->lp_len[d]  = (double *)malloc(beam_n_len(st)  * sizeof(double));
            ok = st->lp_wide[d] && st->lp_mid[d] && st->lp_len[d];
        }
    }
    st->log_pi = (double *)malloc(st->n_len_slots * sizeof(double));
    st->log_z  = (double *)malloc(st->n_len_slots * sizeof(double));
    if (!ok || !st->log_pi || !st->log_z)
        return BEAM_FAIL(BEAM_ERR_ALLOC,
                        "finalize: out of memory allocating the "
                        "log-probability tables (%llu cells)",
                        (unsigned long long)beam_model_n_cells(st));

    /* Length prior straight off the observed length distribution. */
    double tot = 0.0;
    for (uint32_t L = 0; L < st->n_len_slots; ++L) tot += st->cnt_length[L];
    if (!(tot > 0.0))
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT,
                        "finalize: the length histogram is empty, so there is "
                        "nothing to build a length prior from");
    for (uint32_t L = 0; L < st->n_len_slots; ++L) {
        double p = st->cnt_length[L] / tot;
        st->log_pi[L] = p > 0.0 ? log(p) : -INFINITY;
    }

    /* Sparse tables necessarily omit dead rows; the general trim must inspect
     * weights rather than assuming every trellis edge is live. */
    st->tables_all_finite = !st->cfg.sparse;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        BeamError e = st->cfg.sparse
                   ? finalize_sparse_direction(st, d)
                   : finalize_direction(st, d);
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
        int failed = BEAM_OK;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
                num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nw; ++k) {
            double z = 0.0;
            const BeamError e = beam_compute_log_z(st, work[k], &z);
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
BeamError beam_model_finalize(BeamModel *m) {
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
    if (m->n_observed == 0)
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT,
                        "finalize: no usable sequence was observed (%llu were "
                        "skipped)", (unsigned long long)m->n_skipped);

    /* Dimensions and counts are shared with `m`; the tables are not. */
    BeamModel st = *m;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        st.lp_wide[d] = NULL; st.lp_mid[d] = NULL; st.lp_len[d] = NULL;
        memset(&st.sp_lp_wide[d], 0, sizeof(st.sp_lp_wide[d]));
        memset(&st.sp_lp_mid[d], 0, sizeof(st.sp_lp_mid[d]));
        memset(&st.sp_lp_len[d], 0, sizeof(st.sp_lp_len[d]));
    }
    st.log_pi = NULL; st.log_z = NULL; st.finalized = false;

    BeamError err = finalize_tables(&st);
    if (err != BEAM_OK) { beam_free_tables(&st); return err; }

    /* Commit: install the new set, then release the old one. */
    BeamModel prev = *m;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        m->lp_wide[d] = st.lp_wide[d];
        m->lp_mid[d]  = st.lp_mid[d];
        m->lp_len[d]  = st.lp_len[d];
        m->sp_lp_wide[d] = st.sp_lp_wide[d];
        m->sp_lp_mid[d]  = st.sp_lp_mid[d];
        m->sp_lp_len[d]  = st.sp_lp_len[d];
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
            beam_sparse_shrink(&m->sp_cnt_wide[d]);
            beam_sparse_shrink(&m->sp_cnt_mid[d]);
            beam_sparse_shrink(&m->sp_cnt_len[d]);
        }
    }
    return BEAM_OK;
}
