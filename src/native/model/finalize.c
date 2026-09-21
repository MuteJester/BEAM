/**
 * @file finalize.c
 * @brief Dense and sparse table finalization, committed atomically.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"
#include "beam/provenance.h"



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
static BeamError finalize_direction(BeamModel *m, uint32_t dir) {
    const uint32_t R  = m->n_residues;
    const uint32_t nc = m->n_ctx;
    const int64_t  n_rows = (int64_t)m->max_len * nc;

    /*
     * One table per direction, so finalize is what it always was underneath:
     * normalise each row of counts into a conditional over the residues, then
     * take logs. The backoff hierarchy that used to sit here is gone with the
     * experts it served.
     *
     * The open world is a Laplace add-one at this one table rather than a
     * hierarchy. It has to be something: with no parent to fall back on, an
     * unobserved row would otherwise normalise to nothing and then floor to
     * twenty equal tiny logs that do not sum to one, which is not a
     * distribution. Add-one keeps every row normalised, costs no memory, and
     * adds no parameter, since the constant is fixed.
     *
     * The closed world is unchanged: an unobserved row stays all-zero and
     * `beam_to_log_inplace` turns it into twenty -inf, which is the support.
     */
    const double add = m->cfg.strict ? 0.0 : 1.0;

    /*
     * The C-anchored table is written NEAREST-first, one row per context but
     * not at the context's own index. The counts keep the common
     * most-distant-first packing, so the relabelling is confined to the
     * destination address of the finished row and nothing upstream, the file
     * format included, has to know about it. See the note in beam/internal.h
     * for what it buys, and note that the two context helpers are named apart
     * precisely so a call site cannot confuse them.
     */
    const bool reverse = (dir == BEAM_BWD);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t row = 0; row < n_rows; ++row) {
        const uint32_t d = (uint32_t)(row / nc);
        const uint32_t c = (uint32_t)(row % nc);
        const double *cr = m->cnt[dir] + (size_t)row * R;
        const uint32_t dst = reverse
            ? beam_ctx_reverse(c, m->order, m->n_alphabet) : c;
        double *orow = m->lp[dir] + ((size_t)d * nc + dst) * R;
        double tot = 0.0;
        for (uint32_t x = 0; x < R; ++x) {
            orow[x] = cr[x] + add;
            tot += orow[x];
        }
        for (uint32_t x = 0; x < R; ++x)
            orow[x] = tot > 0.0 ? orow[x] / tot : 0.0;
    }
    if (beam_to_log_inplace(m->lp[dir], beam_n_table(m), m->cfg.strict))
        m->tables_all_finite = false;
    return BEAM_OK;
}

/** The sparse log table of one direction. */
static BeamSparseRows *sparse_log_table(BeamModel *m, uint32_t dir) {
    return &m->sp_lp[dir];
}

/**
 * Finalize one zero-backoff strict table row by row.
 *
 * The arithmetic deliberately mirrors the dense loops: sum x=0..19, divide
 * each cell by that total, then take its log. Missing count rows are omitted
 * and therefore read as twenty -inf values. The backward table is packed
 * nearest-first; iterating DESTINATION rows and reversing back to the count
 * row keeps its sparse payload sorted in the order the DP scans.
 */
static int cmp_row_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static const BeamSparseRows *sparse_count_table_const(const BeamModel *m,
                                                      uint32_t dir) {
    return &m->sp_cnt[dir];
}

/*
 * `cnt`, when non-NULL, is the count table to read INSTEAD of the model's own.
 * Finalize passes a locally thawed one so that only a single expanded table is
 * alive at a time; see `beam_model_thaw_count_table` for why that matters.
 * NULL keeps the model's table, which is what a model whose counts are already
 * mutable wants.
 */
static BeamError finalize_sparse_table(BeamModel *m, uint32_t dir,
                                      bool reverse_wide,
                                      const BeamSparseRows *cnt) {
    const uint32_t nr = beam_count_n_rows(m);
    BeamSparseRows *out = sparse_log_table(m, dir);
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
     * to stay in the order the dynamic programs scan it; the backward table is
     * where the two packings differ.
     */
    const BeamSparseRows *ct = cnt ? cnt : sparse_count_table_const(m, dir);
    uint32_t *rows = NULL, n_rows = 0;
    if (m->cfg.sparse && ct && ct->row_ids && ct->size) {
        uint64_t cap = 0;
        for (uint32_t part = 0; part < ct->n_parts; ++part) cap += ct->size[part];
        if (cap) {
            rows = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
            if (!rows) return BEAM_EALLOC("the occupied-row list");
            /* The listing is the table's business, not this file's: it is
             * the only place that knows a mutable table carries the reverse
             * map and a finalized one does not. */
            if (!beam_sparse_occupied_rows(ct, rows, cap, &n_rows)) {
                free(rows);
                return BEAM_FAIL(BEAM_ERR_INVALID_ARG,
                                 "the count table cannot list its occupied "
                                 "rows, so the finalized payload cannot be "
                                 "built in destination order");
            }
            if (reverse_wide)
                for (uint32_t i = 0; i < n_rows; ++i) {
                    const uint32_t src = rows[i];
                    const uint32_t d = src / m->n_ctx;
                    const uint32_t c = src % m->n_ctx;
                    rows[i] = d * m->n_ctx
                            + beam_ctx_reverse(c, m->order, m->n_alphabet);
                }
            qsort(rows, n_rows, sizeof(uint32_t), cmp_row_u32);
        }
    }

    /*
     * The answer is already known: `n_rows` rows, listed and sorted above.
     * Saying so means the fill below never grows -- no second payload to copy
     * the first into, twenty-one times over on the foundation's wide table --
     * and leaves no slack for the shrink at the end to reclaim.
     *
     * It is an upper bound rather than the exact count, because a row whose
     * counts sum to zero is skipped below. That cannot happen on a model that
     * was only ever observed into, but subtraction can produce one, so the
     * shrink stays: it has nothing to do in the ordinary case and is still
     * there for the case that has something.
     */
    if (rows && n_rows) {
        const BeamError re = beam_sparse_reserve(out, 0u, n_rows);
        if (re != BEAM_OK) { free(rows); return re; }
    }

    for (uint32_t idx = 0; idx < (rows ? n_rows : nr); ++idx) {
        const uint32_t dst_row = rows ? rows[idx] : idx;
        uint32_t src_row = dst_row;
        if (reverse_wide) {
            const uint32_t d = dst_row / m->n_ctx;
            const uint32_t c = dst_row % m->n_ctx;
            src_row = d * m->n_ctx + beam_ctx_reverse(c, m->order, m->n_alphabet);
        }
        const double *counts = cnt
            ? beam_sparse_row(cnt, src_row,
                              beam_part_for_row(m, src_row))
            : beam_cnt_row(m, dir, src_row);
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
     * Compaction is `storage/compaction.c`: four transforms whose order is
     * load-bearing, now in one place with the reasons beside them. It used to
     * be four calls spread over seventy lines here, with the order held
     * together by the prose between them.
     */
    return beam_sparse_compact_finalized(out, nr, dir == BEAM_BWD, NULL);
}

static BeamError finalize_sparse_direction(BeamModel *m, uint32_t dir) {
    /*
     * One expanded count table alive at a time.
     *
     * Only when the model arrives frozen -- a load, or a re-finalize after
     * freezing. A model whose counts are already mutable has nothing to
     * expand and takes the original path.
     *
     * `beam_config_ok` REFUSES sparse storage outside the closed world,
     * because the open world's add-one gives mass to rows that were never
     * observed and a sparse table has nowhere to put them. So a sparse model
     * always satisfies this. The condition is still written out rather than
     * assumed, because the config check guards construction and a file could
     * carry a header this code would otherwise trust.
     */
    const bool lazy = m->cfg.sparse && m->counts_frozen;
    if (lazy) {
        BeamSparseRows local;
        BeamError err = beam_model_thaw_count_table(m, dir, &local);
        if (err != BEAM_OK) return err;
        err = finalize_sparse_table(m, dir, dir == BEAM_BWD, &local);
        beam_sparse_destroy(&local);
        return err;
    }
    return finalize_sparse_table(m, dir, dir == BEAM_BWD, NULL);
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
    }
    st.log_pi = NULL; st.log_z = NULL; st.finalized = false;

    BeamError err = finalize_tables(&st);
    if (err != BEAM_OK) { beam_free_tables(&st); return err; }

    /* Commit: install the new set, then release the old one. */
    BeamModel prev = *m;
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        m->lp[d] = st.lp[d];
        m->sp_lp[d] = st.sp_lp[d];
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
