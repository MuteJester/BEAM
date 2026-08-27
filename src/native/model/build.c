/**
 * @file build.c
 * @brief BEAM lifecycle, count accumulation, and finalize.
 *
 * Reference: `.private/edge/src/count.py` (counting) and
 * `.private/edge/src/smooth.py` + `foundation.py` (smoothing). The cell
 * layouts and the smoothing recurrence are reproduced exactly, so a C model
 * and a reference model built from the same sequences agree to floating-point
 * round-off rather than merely to within tolerance.
 */
#include <stdio.h>

#include "beam/internal.h"

/* ── Parallelism ─────────────────────────────────────────────── */

/*
 * Zero means "whatever the runtime would do", which is what a library should
 * default to: the caller's OMP_NUM_THREADS, or one thread per core. Anything
 * positive is a hard cap, applied by naming it in every `num_threads` clause
 * rather than by calling `omp_set_num_threads`, so BEAM never edits global
 * runtime state that the rest of the process shares.
 *
 * Plain `int` rather than an atomic: this is a knob set once at start-up, and
 * the header says so. Making it atomic would suggest a guarantee about
 * changing it mid-computation that the surrounding code does not offer.
 */
static int g_beam_threads = 0;

void beam_model_set_threads(int n) {
    g_beam_threads = n > 0 ? n : 0;
}

int beam_model_threads(void) {
    return g_beam_threads;
}

int beam_threads(void) {
#ifdef _OPENMP
    if (g_beam_threads > 0) return g_beam_threads;
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int beam_model_effective_threads(void) {
    return beam_threads();
}

/*
 * Scratch the per-length dynamic programs may hold at once.
 *
 * Only the spectrum needs this: its per-state histogram grids are 1.6 GB at
 * order 4 with 512 bins, so running one length per core would ask for
 * twenty-five gigabytes and find the swap rather than the speed-up. Every
 * other per-length program needs tens of megabytes and is capped by the
 * thread count alone.
 *
 * Four gigabytes by default: enough for full parallelism at order 2 or 3,
 * and for two or three concurrent lengths at order 4, on a machine that can
 * hold the model at all. It is a cap on CONCURRENCY, never on what a single
 * length is allowed to allocate — one length always runs, however large.
 */
static uint64_t g_beam_budget = 4ull << 30;

void beam_model_set_memory_budget(uint64_t bytes) {
    g_beam_budget = bytes ? bytes : (4ull << 30);
}

uint64_t beam_model_memory_budget(void) {
    return g_beam_budget;
}

/* ── Alphabet ────────────────────────────────────────────────── */


const uint8_t beam_char_to_idx[256] = {
    /* Filled at first use by beam_alphabet_init via a const cast would be
     * unclean; instead the table is spelled out so it is genuinely const and
     * thread-safe with no initialisation order to get wrong. 0xFF marks
     * "not an amino acid"; callers map that to BEAM_NONE. */
    ['A'] = 0,  ['C'] = 1,  ['D'] = 2,  ['E'] = 3,  ['F'] = 4,
    ['G'] = 5,  ['H'] = 6,  ['I'] = 7,  ['K'] = 8,  ['L'] = 9,
    ['M'] = 10, ['N'] = 11, ['P'] = 12, ['Q'] = 13, ['R'] = 14,
    ['S'] = 15, ['T'] = 16, ['V'] = 17, ['W'] = 18, ['Y'] = 19,
    /* Every other byte is 0, which collides with 'A'. Validity is therefore
     * decided by beam_is_residue(), not by the value alone. */
};

static inline bool beam_is_residue(unsigned char c) {
    return c == 'A' || c == 'C' || c == 'D' || c == 'E' || c == 'F'
        || c == 'G' || c == 'H' || c == 'I' || c == 'K' || c == 'L'
        || c == 'M' || c == 'N' || c == 'P' || c == 'Q' || c == 'R'
        || c == 'S' || c == 'T' || c == 'V' || c == 'W' || c == 'Y';
}

const char beam_idx_to_char[BEAM_ALPHABET] = {
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y','.'
};

bool beam_encode(const BeamModel *m, const char *seq,
                     uint8_t *out, uint32_t *out_len) {
    if (!seq) return false;
    uint32_t n = 0;
    for (const unsigned char *p = (const unsigned char *)seq; *p; ++p) {
        if (n >= m->max_len) return false;              /* too long */
        if (!beam_is_residue(*p)) return false;     /* not an amino acid */
        out[n++] = beam_char_to_idx[*p];
    }
    if (n == 0) return false;
    *out_len = n;
    return true;
}

/* ── Configuration ───────────────────────────────────────────── */

void beam_model_config_default(BeamModelConfig *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->order           = 4;
    cfg->depth_cap       = 15;
    cfg->max_length      = BEAM_MAX_LENGTH;
    cfg->coef_wide       = 0.60;
    cfg->coef_mid        = 0.25;
    cfg->coef_len        = 0.15;
    cfg->strict          = false;
    cfg->sparse          = false;
    cfg->kappa[0]        = 20.0;
    cfg->kappa[1]        = 20.0;
    cfg->kappa[2]        = 20.0;
    cfg->kappa_wide      = 20.0;
    cfg->kappa_len       = 200.0;
    cfg->weight_forward  = 0.5;
    cfg->weight_backward = 0.5;
}

static bool config_valid(const BeamModelConfig *c) {
    if (c->order < 2 || c->order > BEAM_MAX_ORDER) return false;
    if (c->max_length < 1 || c->max_length > BEAM_MAX_LENGTH) return false;
    if (c->depth_cap < 1 || c->depth_cap >= c->max_length) return false;
    if (!(c->kappa[0] >= 0.0) || !(c->kappa[1] >= 0.0) || !(c->kappa[2] >= 0.0))
        return false;
    if (!(c->kappa_wide >= 0.0) || !(c->kappa_len >= 0.0)) return false;
    /* The coefficient SUM is a cross-length calibration knob, not a free
     * parameter: moving it off 1 leaves within-length ranking untouched while
     * wrecking global ranking. Reject rather than silently mis-calibrate. */
    double s = c->coef_wide + c->coef_mid + c->coef_len;
    if (!(fabs(s - 1.0) < 1e-9)) return false;
    if (!(c->weight_forward >= 0.0) || !(c->weight_backward >= 0.0)) return false;
    /* An omitted count row is a dead probability row only in the fully
     * closed-world regime. Positive backoff can populate an unobserved child
     * row, so accepting sparse storage there would silently change scores. */
    if (c->sparse
        && (!c->strict || c->kappa[0] != 0.0 || c->kappa[1] != 0.0
            || c->kappa[2] != 0.0 || c->kappa_wide != 0.0
            || c->kappa_len != 0.0))
        return false;
    return true;
}

/* ── Lifecycle ───────────────────────────────────────────────── */

static uint32_t ipow_u32(uint32_t base, uint32_t exp) {
    uint32_t r = 1;
    while (exp--) r *= base;
    return r;
}

static void free_counts(BeamModel *m) {
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        free(m->cnt_wide[d]); m->cnt_wide[d] = NULL;
        free(m->cnt_mid[d]);  m->cnt_mid[d]  = NULL;
        free(m->cnt_len[d]);  m->cnt_len[d]  = NULL;
        beam_sparse_destroy(&m->sp_cnt_wide[d]);
        beam_sparse_destroy(&m->sp_cnt_mid[d]);
        beam_sparse_destroy(&m->sp_cnt_len[d]);
    }
    free(m->cnt_length); m->cnt_length = NULL;
}

static void free_tables(BeamModel *m) {
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        free(m->lp_wide[d]); m->lp_wide[d] = NULL;
        free(m->lp_mid[d]);  m->lp_mid[d]  = NULL;
        free(m->lp_len[d]);  m->lp_len[d]  = NULL;
        beam_sparse_destroy(&m->sp_lp_wide[d]);
        beam_sparse_destroy(&m->sp_lp_mid[d]);
        beam_sparse_destroy(&m->sp_lp_len[d]);
    }
    free(m->log_pi); m->log_pi = NULL;
    free(m->log_z);  m->log_z  = NULL;
}

BeamModel *beam_model_new(const BeamModelConfig *cfg) {
    BeamModelConfig c;
    if (cfg) c = *cfg; else beam_model_config_default(&c);
    /* A depth cap at or beyond the longest sequence resolves every depth
     * exactly, so clamping it is a no-op semantically and saves the caller
     * from having to match the two whenever max_length is reduced. */
    if (c.max_length >= 1 && c.depth_cap >= c.max_length)
        c.depth_cap = (uint8_t)(c.max_length - 1u);
    if (!config_valid(&c)) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "invalid BEAM config: order=%u (need 2..%u), "
                      "max_length=%u (need 1..%u), depth_cap=%u (need 1..%u), "
                      "coef sum=%.17g (need 1), weights=%.17g/%.17g (need >= 0)",
                      (unsigned)c.order, (unsigned)BEAM_MAX_ORDER,
                      (unsigned)c.max_length, (unsigned)BEAM_MAX_LENGTH,
                      (unsigned)c.depth_cap,
                      (unsigned)(c.max_length ? c.max_length - 1u : 0u),
                      c.coef_wide + c.coef_mid + c.coef_len,
                      c.weight_forward, c.weight_backward);
        return NULL;
    }

    BeamModel *m = (BeamModel *)calloc(1, sizeof(*m));
    if (!m) {
        beam_set_error(BEAM_ERR_ALLOC, "out of memory allocating the BEAM model");
        return NULL;
    }
    m->cfg          = c;
    m->order        = c.order;
    m->max_len      = c.max_length;
    m->n_len_slots  = (uint32_t)c.max_length + 1u;
    /* The mid expert is order 3, so the state can never be narrower. */
    /* At least 3, and `state_weights_sparse` depends on it: its backward
     * hoist assumes the order-3 context lies entirely inside the state. */
    m->state_order  = m->order > 3u ? m->order : 3u;
    m->n_states     = ipow_u32(BEAM_ALPHABET, m->state_order);
    m->state_shift  = ipow_u32(BEAM_ALPHABET, m->state_order - 1u);
    m->n_ctx_wide   = ipow_u32(BEAM_ALPHABET, m->order);
    m->n_ctx_mid    = ipow_u32(BEAM_ALPHABET, 3u);
    m->n_ctx_len    = ipow_u32(BEAM_ALPHABET, 2u);
    m->n_depth_wide = (uint32_t)c.depth_cap + 1u;

    bool ok = true;
    for (uint32_t d = 0; d < BEAM_DIRS && ok; ++d) {
        if (m->cfg.sparse) {
            ok = beam_sparse_init(
                     &m->sp_cnt_wide[d],
                     (uint32_t)(beam_n_wide(m) / BEAM_RESIDUES),
                     m->n_depth_wide, true) == BEAM_OK
              && beam_sparse_init(
                     &m->sp_cnt_mid[d],
                     (uint32_t)(beam_n_mid(m) / BEAM_RESIDUES),
                     m->n_depth_wide, true) == BEAM_OK
              && beam_sparse_init(
                     &m->sp_cnt_len[d],
                     (uint32_t)(beam_n_len(m) / BEAM_RESIDUES),
                     m->n_depth_wide, true) == BEAM_OK;
        } else {
            m->cnt_wide[d] = (double *)calloc(beam_n_wide(m), sizeof(double));
            m->cnt_mid[d]  = (double *)calloc(beam_n_mid(m),  sizeof(double));
            m->cnt_len[d]  = (double *)calloc(beam_n_len(m),  sizeof(double));
            ok = m->cnt_wide[d] && m->cnt_mid[d] && m->cnt_len[d];
        }
    }
    m->cnt_length = (double *)calloc(m->n_len_slots, sizeof(double));
    if (!ok || !m->cnt_length) {
        beam_set_error(BEAM_ERR_ALLOC,
                      "out of memory allocating BEAM count tables "
                      "(%llu logical cells at order %u, max_length %u, "
                      "sparse=%s)",
                      (unsigned long long)beam_model_n_cells(m),
                      (unsigned)m->order, (unsigned)m->max_len,
                      m->cfg.sparse ? "true" : "false");
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

void beam_model_destroy(BeamModel *m) {
    if (!m) return;
    beam_live_cache_drop(m);
    free_counts(m);
    free_tables(m);
    free(m);
}

/* ── Count accumulation ──────────────────────────────────────── */

/**
 * Record one sequence into one direction's tables.
 *
 * `depth_of(p)` is the residue's distance from this direction's anchor and
 * `ctx_of(p, k)` its order-`k` context on the anchor side. Everything else is
 * identical between the directions, which is the point of the design.
 */
static bool observe_one_position(BeamModel *m, uint32_t dir,
                                 const uint8_t *s, uint32_t len, uint32_t p,
                                 uint32_t depth, double w) {
    uint32_t cw_ix, cm_ix, cl_ix;
    if (dir == BEAM_FWD) {
        cw_ix = beam_ctx_forward(s, (int32_t)p, m->order);
        cm_ix = beam_ctx_forward(s, (int32_t)p, 3u);
        cl_ix = beam_ctx_forward(s, (int32_t)p, 2u);
    } else {
        cw_ix = beam_ctx_backward(s, (int32_t)p, len, m->order);
        cm_ix = beam_ctx_backward(s, (int32_t)p, len, 3u);
        cl_ix = beam_ctx_backward(s, (int32_t)p, len, 2u);
    }
    const uint32_t x = s[p];
    if (!m->cfg.sparse) {
        m->cnt_wide[dir][beam_ix_wide(m, depth, cw_ix, x)] += w;
        m->cnt_mid[dir][beam_ix_mid(m, depth, cm_ix, x)]   += w;
        m->cnt_len[dir][beam_ix_len(m, len, depth, cl_ix, x)] += w;
        return true;
    }

    const uint32_t wd = beam_depth_part(m, depth);
    const uint32_t rw = wd * m->n_ctx_wide + cw_ix;
    const uint32_t rm = depth * m->n_ctx_mid + cm_ix;
    const uint32_t rl = (len * m->max_len + depth) * m->n_ctx_len + cl_ix;
    double *wide = beam_cnt_wide_row_mut(m, dir, rw, true);
    double *mid  = beam_cnt_mid_row_mut(m, dir, rm, true);
    double *lng  = beam_cnt_len_row_mut(m, dir, rl, true);
    if (!wide || !mid || !lng) return false;
    wide[x] += w;
    mid[x]  += w;
    lng[x]  += w;
    return true;
}

/** Sequences encoded at a time. Bounds the scratch at a few MB whatever `n`. */
#define BEAM_OBS_CHUNK 262144u

/**
 * Count one chunk of pre-encoded sequences into the tables, in parallel.
 *
 * The parallel axis is the DEPTH, not the sequence, and that is what makes
 * this exact rather than merely fast. Every cell of every table carries the
 * depth in its index, so partitioning the depths partitions the cells: no two
 * threads can touch the same cell, no atomics are needed, and — because each
 * thread still walks the sequences in ascending index — every cell accumulates
 * its contributions in exactly the order the serial loop would have added
 * them. The counts are bit-identical, not just equivalent.
 *
 * Two details make that true rather than nearly true. The wide table caps its
 * depth index at `depth_cap`, so every depth at or beyond the cap shares one
 * cell and they must be handled by ONE task, not spread across several. And
 * within that merged task the depths have to be visited in the order the
 * serial pass met them, which is ascending position: ascending depth going
 * forwards, descending depth going backwards.
 */
static bool observe_chunk(BeamModel *m, const uint8_t *enc,
                          const uint32_t *len_of, const double *wt,
                          uint32_t n) {
    const uint32_t dc = m->cfg.depth_cap;
    const uint32_t nb = dc + 1u;
    const uint32_t stride = m->max_len;
    const int64_t ntask = 2 * (int64_t)nb;
    int failed = 0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \
            num_threads(beam_threads())
#endif
    for (int64_t task = 0; task < ntask; ++task) {
        const uint32_t dir = (uint32_t)(task / nb);
        const uint32_t b   = (uint32_t)(task % nb);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t len = len_of[i];
            if (len == 0 || b >= len) continue;
            const uint8_t *s = enc + (size_t)i * stride;
            const double w = wt[i];
            const uint32_t hi = (b == dc) ? len : b + 1u;   /* half-open */
            if (dir == BEAM_FWD) {
                for (uint32_t d = b; d < hi; ++d)
                    if (!observe_one_position(m, dir, s, len, d, d, w))
                        failed = 1;
            } else {
                for (uint32_t d = hi; d-- > b; )
                    if (!observe_one_position(m, dir, s, len,
                                              len - 1u - d, d, w))
                        failed = 1;
            }
        }
    }
    return failed == 0;
}

BeamError beam_model_observe(BeamModel *m,
                                const char *const *seqs,
                                const double *abundances,
                                uint32_t n) {
    if (!m || !seqs)
        return BEAM_FAIL(BEAM_ERR_NULL_ARG, "observe: %s must not be NULL",
                        !m ? "model" : "sequences");
    if (m->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "observe: the model is compacted, so its counts are "
                        "gone; build a fresh model instead");
    if (n == 0) return BEAM_OK;
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);

    const uint32_t stride = m->max_len;
    const uint32_t cap = n < BEAM_OBS_CHUNK ? n : BEAM_OBS_CHUNK;
    uint8_t  *enc    = (uint8_t *)malloc((size_t)cap * stride);
    uint32_t *len_of = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    double   *wt     = (double *)malloc((size_t)cap * sizeof(double));
    if (!enc || !len_of || !wt) {
        free(enc); free(len_of); free(wt);
        return BEAM_EALLOC("the observation scratch buffers");
    }

    for (uint32_t base = 0; base < n; base += cap) {
        const uint32_t cnt = (n - base) < cap ? (n - base) : cap;
        /* Validation and the scalar totals stay in sequence order: they are
         * cheap, and `total_weight` is a running sum whose value depends on
         * the order it is taken in. */
        for (uint32_t k = 0; k < cnt; ++k) {
            uint32_t len = 0;
            const double w = abundances ? abundances[base + k] : 1.0;
            if (!beam_encode(m, seqs[base + k], enc + (size_t)k * stride, &len)
                || !(w > 0.0) || !isfinite(w)) {
                len_of[k] = 0; wt[k] = 0.0; m->n_skipped++;
                continue;
            }
            len_of[k] = len;
            wt[k] = w;
            m->cnt_length[len] += w;
            m->total_weight    += w;
            m->n_observed++;
        }
        if (!observe_chunk(m, enc, len_of, wt, cnt)) {
            free(enc); free(len_of); free(wt);
            m->finalized = false;
            return BEAM_EALLOC("a sparse count-table row");
        }
    }
    free(enc); free(len_of); free(wt);
    m->finalized = false;
    return BEAM_OK;
}

BeamError beam_model_subtract(BeamModel *m, const BeamModel *o) {
    if (!m || !o)
        return BEAM_FAIL(BEAM_ERR_NULL_ARG, "subtract: %s must not be NULL",
                        !m ? "model" : "operand");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (m->compacted || o->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "subtract: the %s model is compacted, so its counts "
                        "are gone", m->compacted ? "target" : "operand");
    if (m->order != o->order || m->max_len != o->max_len
        || m->n_depth_wide != o->n_depth_wide)
        return BEAM_FAIL(BEAM_ERR_VARIANT_MISMATCH,
                        "subtract: shape mismatch, target is "
                        "order=%u max_length=%u depth_cap=%u and the operand "
                        "is order=%u max_length=%u depth_cap=%u",
                        (unsigned)m->order, (unsigned)m->max_len,
                        (unsigned)m->n_depth_wide - 1u,
                        (unsigned)o->order, (unsigned)o->max_len,
                        (unsigned)o->n_depth_wide - 1u);

    /*
     * Refuse a subtraction that would leave nothing behind, BEFORE touching a
     * single cell.
     *
     * The counts are the only copy there is: once they are decremented there
     * is nothing to roll back to, and the caller is left holding a model whose
     * `finalize` can only fail. Checking the length histogram first is cheap
     * (max_length + 1 cells) and turns an unrecoverable state into an error
     * code the caller can act on.
     */
    {
        double remaining = 0.0;
        for (uint32_t L = 0; L < m->n_len_slots; ++L) {
            const double v = m->cnt_length[L] - o->cnt_length[L];
            if (v > 0.0) remaining += v;
        }
        if (!(remaining > 0.0) || m->n_observed <= o->n_observed)
            return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT,
                            "subtract: the operand accounts for every "
                            "sequence in the target (%llu of %llu observed), "
                            "so the subtraction would empty the model; build a "
                            "fresh model instead",
                            (unsigned long long)o->n_observed,
                            (unsigned long long)m->n_observed);
    }

    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        for (uint32_t kind = BEAM_TABLE_WIDE;
             kind <= BEAM_TABLE_LEN; ++kind) {
            const BeamTableKind k = (BeamTableKind)kind;
            const uint32_t nr = beam_count_n_rows(m, k);
            for (uint32_t row = 0; row < nr; ++row) {
                double *dst = beam_count_row_mut(m, d, k, row, false);
                if (!dst) continue;
                const double *src = beam_count_row(o, d, k, row);
                if (!src) continue;
                bool any = false;
                for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) {
                    const double v = dst[x] - src[x];
                    dst[x] = v > 0.0 ? v : 0.0;
                    if (dst[x] > 0.0) any = true;
                }
                if (!any) (void)beam_count_erase(m, d, k, row);
            }
        }
    }
    for (uint32_t L = 0; L < m->n_len_slots; ++L) {
        const double v = m->cnt_length[L] - o->cnt_length[L];
        m->cnt_length[L] = v > 0.0 ? v : 0.0;
    }
    /* The sequence counters are part of the count state, so they have to move
     * with it: `add` adjusts all three, and an inverse that adjusted only one
     * would leave every leave-one-cohort-out workflow reporting a corpus size
     * that never shrinks. Floored at zero because the cells are. */
    m->total_weight = m->total_weight > o->total_weight
                    ? m->total_weight - o->total_weight : 0.0;
    m->n_observed   = m->n_observed > o->n_observed
                    ? m->n_observed - o->n_observed : 0u;
    m->n_skipped    = m->n_skipped > o->n_skipped
                    ? m->n_skipped - o->n_skipped : 0u;
    m->finalized = false;
    return BEAM_OK;
}

/* ── Bayesian posterior ──────────────────────────────────────── */

/**
 * One table's Dirichlet-multinomial update: `dst += kappa * normalise(src)`,
 * row by row.
 *
 * A row is one categorical over the 20 residues, so normalising it is what
 * turns a prior's COUNTS into the prior BELIEF that `kappa` pseudo-counts of
 * weight are then contributed to. A row the prior never saw stays untouched:
 * an empty row carries no belief, and spreading `kappa` uniformly over it
 * would invent one.
 */
static void posterior_table(double *dst, const double *src, size_t n_rows,
                            double kappa) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t r = 0; r < (int64_t)n_rows; ++r) {
        const double *sr = src + (size_t)r * BEAM_RESIDUES;
        double *dr = dst + (size_t)r * BEAM_RESIDUES;
        double tot = 0.0;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) tot += sr[x];
        if (!(tot > 0.0)) continue;
        const double scale = kappa / tot;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) dr[x] += sr[x] * scale;
    }
}

/** Row-addressed posterior used whenever either operand is sparse. */
static bool posterior_rows(BeamModel *dst_m, const BeamModel *src_m,
                           uint32_t dir, BeamTableKind kind,
                           double kappa) {
    const uint32_t nr = beam_count_n_rows(dst_m, kind);
    for (uint32_t row = 0; row < nr; ++row) {
        const double *src = beam_count_row(src_m, dir, kind, row);
        if (!src) continue;
        double tot = 0.0;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) tot += src[x];
        if (!(tot > 0.0)) continue;
        double *dst = beam_count_row_mut(dst_m, dir, kind, row, true);
        if (!dst) return false;
        const double scale = kappa / tot;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x)
            dst[x] += src[x] * scale;
    }
    return true;
}

BeamError beam_model_posterior(BeamModel *m, const BeamModel *prior,
                                  double kappa) {
    if (!m || !prior) return BEAM_ENULL(!m ? "model" : "prior");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (m->compacted || prior->compacted)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "posterior: the %s model is compacted, so its counts "
                        "are gone", m->compacted ? "target" : "prior");
    if (!(kappa >= 0.0) || !isfinite(kappa))
        return BEAM_ERANGE("posterior: kappa must be finite and >= 0, "
                               "got %.17g", kappa);
    if (m->order != prior->order || m->max_len != prior->max_len
        || m->n_depth_wide != prior->n_depth_wide)
        return BEAM_FAIL(BEAM_ERR_VARIANT_MISMATCH,
                        "posterior: shape mismatch, target is order=%u "
                        "max_length=%u depth_cap=%u and the prior is "
                        "order=%u max_length=%u depth_cap=%u",
                        (unsigned)m->order, (unsigned)m->max_len,
                        (unsigned)m->n_depth_wide - 1u,
                        (unsigned)prior->order, (unsigned)prior->max_len,
                        (unsigned)prior->n_depth_wide - 1u);
    if (kappa == 0.0) { m->finalized = false; return BEAM_OK; }

    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        if (!m->cfg.sparse && !prior->cfg.sparse) {
            /* Preserve the original dense/open-world path byte for byte. */
            posterior_table(m->cnt_wide[d], prior->cnt_wide[d],
                            beam_n_wide(m) / BEAM_RESIDUES, kappa);
            posterior_table(m->cnt_mid[d], prior->cnt_mid[d],
                            beam_n_mid(m) / BEAM_RESIDUES, kappa);
            posterior_table(m->cnt_len[d], prior->cnt_len[d],
                            beam_n_len(m) / BEAM_RESIDUES, kappa);
        } else if (!posterior_rows(m, prior, d, BEAM_TABLE_WIDE, kappa)
                   || !posterior_rows(m, prior, d, BEAM_TABLE_MID, kappa)
                   || !posterior_rows(m, prior, d, BEAM_TABLE_LEN, kappa)) {
            m->finalized = false;
            return BEAM_EALLOC("a sparse posterior row");
        }
    }
    /* The length histogram is one categorical too, over lengths rather than
     * residues, and it has to move with the rest: leaving it alone would give
     * the posterior the individual's length distribution with the prior's
     * conditionals, which is not a posterior of anything. */
    {
        double tot = 0.0;
        for (uint32_t L = 0; L < prior->n_len_slots; ++L) tot += prior->cnt_length[L];
        if (tot > 0.0) {
            const double scale = kappa / tot;
            for (uint32_t L = 0; L < m->n_len_slots; ++L)
                m->cnt_length[L] += prior->cnt_length[L] * scale;
        }
    }
    /*
     * The sequence counters do NOT move. `total_weight` and `n_observed`
     * report what was observed, and a pseudo-count is not an observation;
     * reporting kappa there would make a leave-one-out corpus size depend on
     * the smoothing strength.
     */
    m->finalized = false;
    return BEAM_OK;
}

/* ── Hierarchical smoothing ──────────────────────────────────── */

/**
 * One level of the backoff recurrence, in probability space:
 *
 *     p_j[d][ctx][x] = (c_j[d][ctx][x] + kappa * p_(j-1)[d][ctx >> A][x])
 *                      / rowsum
 *
 * where `ctx >> A` drops the MOST DISTANT context residue, so a higher-order
 * estimate falls back on the shorter context it refines. A row whose total is
 * zero cannot be produced at all and is written as exactly zero, which the
 * caller turns into -inf; that only happens when kappa is 0.
 *
 * @param counts   [n_depth][n_ctx][20] counts at this order.
 * @param parent   [n_depth][n_ctx / A][20] probabilities at the order below.
 * @param out      [n_depth][n_ctx][20] probabilities at this order.
 */
static void backoff_level(const double *counts, const double *parent,
                          double kappa, uint32_t n_depth, uint32_t n_ctx,
                          double *out) {
    const uint32_t n_parent_ctx = n_ctx / BEAM_ALPHABET;
    const int64_t n_rows = (int64_t)n_depth * n_ctx;
    /* Every row is independent of every other, so the only thing the loop
     * order decides is speed. Flattening depth and context into one index
     * keeps the largest table (the wide expert's, three million rows) from
     * being split across a dimension with sixteen values. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t row = 0; row < n_rows; ++row) {
        const uint32_t d = (uint32_t)(row / n_ctx);
        const uint32_t c = (uint32_t)(row % n_ctx);
        const double *cr = counts + (size_t)row * BEAM_RESIDUES;
        const double *pr = parent + ((size_t)d * n_parent_ctx
                                     + (c % n_parent_ctx)) * BEAM_RESIDUES;
        double *orow = out + (size_t)row * BEAM_RESIDUES;
        double tot = 0.0;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) {
            orow[x] = cr[x] + kappa * pr[x];
            tot += orow[x];
        }
        if (tot > 0.0) {
            for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) orow[x] /= tot;
        } else {
            for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) orow[x] = 0.0;
        }
    }
}

/** Sum out the most distant context residue: [n_ctx] -> [n_ctx / A]. */
static void marginalise(const double *in, uint32_t n_depth, uint32_t n_ctx,
                        double *out) {
    const uint32_t n_out = n_ctx / BEAM_ALPHABET;
    const int64_t n_rows = (int64_t)n_depth * n_out;
    /* Inverted to accumulate BY TARGET: the source form scatters `A` rows onto
     * every output row, so parallelising it would race. Gathering instead
     * makes each output row one thread's private property, and it visits the
     * sources in the same ascending order the scatter did, so the sums are
     * bit-identical rather than merely equivalent. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t row = 0; row < n_rows; ++row) {
        const uint32_t d = (uint32_t)(row / n_out);
        const uint32_t c = (uint32_t)(row % n_out);
        double *orow = out + (size_t)row * BEAM_RESIDUES;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) orow[x] = 0.0;
        for (uint32_t a = 0; a < BEAM_ALPHABET; ++a) {
            const double *ir = in + ((size_t)d * n_ctx + a * n_out + c)
                                    * BEAM_RESIDUES;
            for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) orow[x] += ir[x];
        }
    }
}

/** Order-0 level: Laplace alpha = 1 over the 20 residues, per depth. */
static void base_level(const double *c0, uint32_t n_depth, double *out,
                       bool strict) {
    for (uint32_t d = 0; d < n_depth; ++d) {
        const double *cr = c0 + (size_t)d * BEAM_RESIDUES;
        double *orow = out + (size_t)d * BEAM_RESIDUES;
        double tot = 0.0;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) {
            /*
             * The add-one is what makes zero unreachable: it is the last rung
             * of the backoff ladder, so a residue unseen everywhere still
             * leaves with 1/(total+20). Under `strict` the row is the raw MLE
             * instead, and an unseen residue stays at exactly zero so the
             * -inf survives the log.
             */
            orow[x] = strict ? cr[x] : cr[x] + 1.0;
            tot += orow[x];
        }
        if (tot <= 0.0) {
            /*
             * Strict mode only. An empty row has no observation to normalise,
             * and 0/0 would be NaN, which propagates silently through the DP
             * and is far worse than the -inf it would have replaced. Leave the
             * row at zero: every residue is then impossible from this context,
             * which is exactly what "nothing was ever seen here" means.
             */
            continue;
        }
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) orow[x] /= tot;
    }
}

/**
 * Full hierarchy over an order-`k` count table, returning probabilities at
 * every order 0..k. `out[j]` must have room for `n_depth * A^j * 20` doubles.
 * Caller owns the buffers.
 */
static BeamError build_hierarchy(const double *counts, uint32_t n_depth,
                                uint32_t order, const double *kappa,
                                double **out, bool strict) {
    /* Marginals, computed top down into scratch. */
    const double *level_counts[BEAM_MAX_ORDER + 1];
    double *scratch[BEAM_MAX_ORDER + 1];
    memset(scratch, 0, sizeof(scratch));

    level_counts[order] = counts;
    for (int32_t j = (int32_t)order - 1; j >= 0; --j) {
        uint32_t n_out = ipow_u32(BEAM_ALPHABET, (uint32_t)j);
        scratch[j] = (double *)malloc((size_t)n_depth * n_out
                                      * BEAM_RESIDUES * sizeof(double));
        if (!scratch[j]) {
            for (uint32_t t = 0; t <= order; ++t) free(scratch[t]);
            return BEAM_EALLOC("a backoff-hierarchy marginal");
        }
        marginalise(level_counts[j + 1], n_depth,
                    ipow_u32(BEAM_ALPHABET, (uint32_t)j + 1u), scratch[j]);
        level_counts[j] = scratch[j];
    }

    base_level(level_counts[0], n_depth, out[0], strict);
    for (uint32_t j = 1; j <= order; ++j) {
        backoff_level(level_counts[j], out[j - 1], kappa[j - 1], n_depth,
                      ipow_u32(BEAM_ALPHABET, j), out[j]);
    }
    for (uint32_t t = 0; t <= order; ++t) free(scratch[t]);
    return BEAM_OK;
}

/**
 * Probabilities to natural logs, reporting whether any cell came out -inf.
 *
 * The floor makes that impossible today — `log(1e-300)` is a perfectly finite
 * -690.8, which is the whole point of flooring rather than letting a zero
 * through. It is still MEASURED rather than assumed, because what depends on
 * it is not a cosmetic detail: with no -inf cell there is no dead edge, so the
 * live-state trim in analytics.c collapses to a reachability sweep and stops
 * building every layer's weights a second time. A future change that removed
 * the floor would switch that off automatically instead of silently counting
 * sequences the model cannot produce.
 */
static bool to_log_inplace(double *p, size_t n, bool strict) {
    bool nonfinite = false;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:nonfinite) \
            num_threads(beam_threads())
#endif
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        const double v = p[i];
        /*
         * The floor is the second reason zero is unreachable by default: it
         * turns a genuine 0 into log(1e-300) = -690.8, finite and therefore
         * still counted by D0. Under `strict` the value goes through
         * untouched, so 0 becomes -inf, the cell is a dead edge, and the
         * live-state trim in analytics.c switches to its general path on the
         * `tables_all_finite` flag the caller sets from this return.
         */
        const double lv = strict
            ? log(v)
            : log(v > BEAM_PROB_FLOOR ? v : BEAM_PROB_FLOOR);
        p[i] = lv;
        if (!isfinite(lv)) nonfinite = true;
    }
    return nonfinite;
}

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
static void normalise_rows(const double *counts, int64_t n_rows, double *out) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t row = 0; row < n_rows; ++row) {
        const double *cr = counts + (size_t)row * BEAM_RESIDUES;
        double *orow = out + (size_t)row * BEAM_RESIDUES;
        double tot = 0.0;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) tot += cr[x];
        if (tot > 0.0)
            for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) orow[x] = cr[x] / tot;
        else
            for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) orow[x] = 0.0;
    }
}

static BeamError finalize_direction(BeamModel *m, uint32_t dir) {
    const uint32_t W  = m->max_len;
    const uint32_t A  = BEAM_ALPHABET;
    const uint32_t R  = BEAM_RESIDUES;
    const uint32_t dc = m->cfg.depth_cap;
    BeamError err = BEAM_OK;

    /* Declared together and freed on one path, so every early exit is safe. */
    double *mid_p[4]     = {0};
    double *agg_p[4]     = {0};
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
        mid_p[3] = (double *)malloc((size_t)W * ipow_u32(A, 3u) * R
                                    * sizeof(double));
        if (!mid_p[3]) {
            err = BEAM_EALLOC("a mid-expert backoff level"); goto done;
        }
        normalise_rows(m->cnt_mid[dir], (int64_t)W * ipow_u32(A, 3u), mid_p[3]);
    } else {
        /* Order 0..3 probabilities over the mid counts, at full depth. */
        for (uint32_t j = 0; j <= 3; ++j) {
            mid_p[j] = (double *)malloc((size_t)W * ipow_u32(A, j) * R * sizeof(double));
            if (!mid_p[j]) {
                err = BEAM_EALLOC("a mid-expert backoff level"); goto done;
            }
        }
        err = build_hierarchy(m->cnt_mid[dir], W, 3u, m->cfg.kappa, mid_p,
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
            agg_p[j] = (double *)malloc((size_t)ipow_u32(A, j) * R * sizeof(double));
            if (!agg_p[j]) {
                err = BEAM_EALLOC("an aggregated backoff level"); goto done;
            }
        }
        err = build_hierarchy(agg_counts, 1u, 3u, m->cfg.kappa, agg_p,
                              m->cfg.strict);
        if (err != BEAM_OK) goto done;
    }

    /* ── mid expert: order 3 at full depth ── */
    memcpy(m->lp_mid[dir], mid_p[3], beam_n_mid(m) * sizeof(double));
    if (to_log_inplace(m->lp_mid[dir], beam_n_mid(m), m->cfg.strict))
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
    if (to_log_inplace(m->lp_len[dir], beam_n_len(m), m->cfg.strict))
        m->tables_all_finite = false;

    /* ── wide expert: order K at capped depth, backing off to the order-(K-1)
     *    table at the matching depth bucket, shared across the most distant
     *    context residue ── */
    {
        const uint32_t k1     = m->order - 1u;
        const uint32_t n_ctx1 = ipow_u32(A, k1);
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
            const uint32_t dst = reverse ? beam_ctx_reverse(c, m->order) : c;
            double *orow = m->lp_wide[dir] + ((size_t)d * nw + dst) * R;
            double tot = 0.0;
            if (no_backoff) {
                for (uint32_t x = 0; x < R; ++x) { orow[x] = cr[x]; tot += cr[x]; }
            } else {
                const double *base_tab = (d < dc)
                    ? mid_p[k1] + (size_t)d * n_ctx1 * R
                    : agg_p[k1];
                /* Drop the most distant residue to reach the base row. */
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
    if (to_log_inplace(m->lp_wide[dir], beam_n_wide(m), m->cfg.strict))
        m->tables_all_finite = false;

done:
    free(agg_counts);
    for (uint32_t j = 0; j <= 3; ++j) { free(agg_p[j]); free(mid_p[j]); }
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
    BeamError err = beam_sparse_init(out, nr, 1u, false);
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
                        dst = d * m->n_ctx_wide + beam_ctx_reverse(c, m->order);
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
            src_row = d * m->n_ctx_wide + beam_ctx_reverse(c, m->order);
        }
        const double *counts = beam_count_row(m, dir, kind, src_row);
        if (!counts) continue;
        double tot = 0.0;
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x) tot += counts[x];
        if (!(tot > 0.0)) continue;
        double *logp = beam_sparse_row_mut(out, dst_row, 0u, true);
        if (!logp) return BEAM_EALLOC("a sparse probability row");
        for (uint32_t x = 0; x < BEAM_RESIDUES; ++x)
            logp[x] = log(counts[x] / tot);
    }
    free(rows);
    beam_sparse_shrink(out);
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
    if (err != BEAM_OK) { free_tables(&st); return err; }

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
    free_tables(&prev);
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

bool beam_model_is_finalized(const BeamModel *m) {
    return m && m->finalized;
}

BeamError beam_model_compact(BeamModel *m) {
    if (!m) return BEAM_FAIL(BEAM_ERR_NULL_ARG, "compact: model must not be NULL");
    /* The live cache is a projection of the finalized tables; anything
     * that changes the counts or rebuilds the tables makes it a record of
     * a model that no longer exists. Dropped BEFORE the change, so a
     * failure part-way through cannot leave a stale one behind. */
    beam_live_cache_drop(m);
    if (!m->finalized)
        return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                        "compact: the model is not finalized, so releasing "
                        "its counts would leave nothing behind");
    free_counts(m);
    m->compacted = true;
    return BEAM_OK;
}

/* ── Introspection ───────────────────────────────────────────── */

const BeamModelConfig *beam_model_config(const BeamModel *m) {
    return m ? &m->cfg : NULL;
}
double   beam_model_total_weight(const BeamModel *m) {
    return m ? m->total_weight : 0.0;
}
uint64_t beam_model_n_observed(const BeamModel *m) {
    return m ? m->n_observed : 0;
}
uint64_t beam_model_n_skipped(const BeamModel *m) {
    return m ? m->n_skipped : 0;
}
uint32_t beam_model_n_states(const BeamModel *m) {
    return m ? m->n_states : 0;
}
uint64_t beam_model_n_cells(const BeamModel *m) {
    if (!m) return 0;
    return (uint64_t)BEAM_DIRS
         * (beam_n_wide(m) + beam_n_mid(m) + beam_n_len(m));
}
uint64_t beam_model_n_nonzero_cells(const BeamModel *m) {
    if (!m || m->compacted) return 0;
    uint64_t n = 0;
    if (m->cfg.sparse) {
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            for (uint32_t kind = BEAM_TABLE_WIDE;
                 kind <= BEAM_TABLE_LEN; ++kind) {
                const BeamSparseRows *t = beam_sparse_count_table_const(
                    m, d, (BeamTableKind)kind);
                for (uint32_t p = 0; p < t->n_parts; ++p) {
                    const double *v = t->values[p];
                    const size_t nc = (size_t)t->size[p] * BEAM_RESIDUES;
                    for (size_t i = 0; i < nc; ++i) if (v[i] > 0.0) n++;
                }
            }
        }
        return n;
    }
    for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
        size_t c;
        c = beam_n_wide(m);
        for (size_t i = 0; i < c; ++i) if (m->cnt_wide[d][i] > 0.0) n++;
        c = beam_n_mid(m);
        for (size_t i = 0; i < c; ++i) if (m->cnt_mid[d][i] > 0.0) n++;
        c = beam_n_len(m);
        for (size_t i = 0; i < c; ++i) if (m->cnt_len[d][i] > 0.0) n++;
    }
    return n;
}
uint64_t beam_model_memory_bytes(const BeamModel *m) {
    if (!m) return 0;
    if (m->cfg.sparse) {
        uint64_t n = 0;
        for (uint32_t d = 0; d < BEAM_DIRS; ++d) {
            if (!m->compacted) {
                n += beam_sparse_memory_bytes(&m->sp_cnt_wide[d]);
                n += beam_sparse_memory_bytes(&m->sp_cnt_mid[d]);
                n += beam_sparse_memory_bytes(&m->sp_cnt_len[d]);
            }
            if (m->finalized) {
                n += beam_sparse_memory_bytes(&m->sp_lp_wide[d]);
                n += beam_sparse_memory_bytes(&m->sp_lp_mid[d]);
                n += beam_sparse_memory_bytes(&m->sp_lp_len[d]);
            }
        }
        return n;
    }
    uint64_t per_dir = (uint64_t)(beam_n_wide(m) + beam_n_mid(m)
                                  + beam_n_len(m)) * sizeof(double);
    uint64_t n = 0;
    if (!m->compacted) n += per_dir * BEAM_DIRS;   /* counts */
    if (m->finalized)  n += per_dir * BEAM_DIRS;   /* log tables */
    return n;
}

double beam_model_log_length_prior(const BeamModel *m, uint32_t length) {
    if (!m || !m->finalized || length >= m->n_len_slots) return -INFINITY;
    return m->log_pi[length];
}
double beam_model_log_partition(const BeamModel *m, uint32_t length) {
    if (!m || !m->finalized || length >= m->n_len_slots) return -INFINITY;
    return m->log_z[length];
}
