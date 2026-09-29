/**
 * @file test_utils.h
 * @brief Shared test-runner macros for tests/c_unit.
 *
 * Each test file is expected to declare its own file-static counters:
 *   static int pass_count = 0, fail_count = 0;
 *
 * RUN_TEST prints the test name (left-padded), invokes it, and lets the
 * test body call PASS() or ASSERT_MSG() to update the counters.
 */
#ifndef BEAM_TEST_UTILS_H
#define BEAM_TEST_UTILS_H

#include <stdio.h>

#define RUN_TEST(fn) do { printf("  %-55s ", #fn); fn(); } while (0)

#define ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } \
} while (0)

#define PASS() do { printf("PASS\n"); pass_count++; } while (0)

#ifdef BEAM_TEST_PRIOR_REFERENCE
/*
 * The hierarchical prior restated from its definition, for the storage tests
 * that check a finalized cell against the counts it came from. Opt in by
 * defining BEAM_TEST_PRIOR_REFERENCE and including beam/internal.h first.
 *
 *     P_k(x) = (n_k(x) + alpha[k] * P_{k-1}(x)) / (n_k + alpha[k])
 *
 * Level k keeps the k residues nearest the scored one, n_k sums the counts
 * over the residues it drops, P_{-1} is 1/R, and a row with n_k == 0 IS its
 * parent. Nothing here shares the library's scratch, packing or ancestor
 * walk: one depth's counts are re-summed from `beam_cnt_row`, whose packing
 * is most-distant-first in BOTH directions, so the level-k context of a top
 * context `c` is simply `c % radix^k`. The finalized backward table's
 * nearest-first packing is the caller's business, restated in each test.
 *
 * Bit-exact against finalize, not merely close, only because the fixtures
 * weigh every sequence by an integer: every sum is then exact whatever order
 * it is taken in. Each cell is the definition in LOGS, as finalize takes it so
 * that a small alpha cannot underflow a residue unobserved at every level:
 *
 *     n(x) > 0:  log(n(x) + alpha * exp(log P_{k-1}(x))) - log(n + alpha)
 *     n(x) = 0:  log(alpha) + log P_{k-1}(x) - log(n + alpha)
 *
 * The uniform row is -log R, which is log(1/R) without first rounding 1/R.
 */
#include <math.h>
#include <stdlib.h>

typedef struct {
    const BeamModel *m;
    uint32_t dir, depth, R, order;
    uint32_t width[BEAM_MAX_ORDER + 1];      /* radix^k                      */
    double  *n[BEAM_MAX_ORDER];              /* [width[k] * R] summed counts */
    double  *tot[BEAM_MAX_ORDER];            /* [width[k]]     row totals    */
    double  *p[BEAM_MAX_ORDER];              /* [width[k] * R] log P_k       */
} PriorRef;

static void prior_ref_free(PriorRef *pr) {
    for (uint32_t k = 0; k < pr->order; ++k) {
        free(pr->n[k]); free(pr->tot[k]); free(pr->p[k]);
        pr->n[k] = pr->tot[k] = pr->p[k] = NULL;
    }
}

/* Every lower level of one (direction, depth). The model's counts must be
 * readable, i.e. thawed if finalize froze them. False on allocation failure. */
static bool prior_ref_build(PriorRef *pr, const BeamModel *m, uint32_t dir,
                            uint32_t depth) {
    memset(pr, 0, sizeof *pr);
    pr->m = m; pr->dir = dir; pr->depth = depth;
    pr->R = m->n_residues; pr->order = m->order;
    const uint32_t R = pr->R;
    pr->width[0] = 1u;
    for (uint32_t k = 1; k <= pr->order; ++k)
        pr->width[k] = pr->width[k - 1] * m->n_alphabet;
    for (uint32_t k = 0; k < pr->order; ++k) {
        pr->n[k]   = (double *)calloc((size_t)pr->width[k] * R, sizeof(double));
        pr->tot[k] = (double *)calloc(pr->width[k], sizeof(double));
        pr->p[k]   = (double *)calloc((size_t)pr->width[k] * R, sizeof(double));
        if (!pr->n[k] || !pr->tot[k] || !pr->p[k]) { prior_ref_free(pr); return false; }
    }
    const uint32_t nc = pr->width[pr->order];
    for (uint32_t c = 0; c < nc; ++c) {
        const double *cnt = beam_cnt_row(m, dir, depth * nc + c);
        if (!cnt) continue;
        for (uint32_t k = 0; k < pr->order; ++k) {
            const uint32_t ck = c % pr->width[k];
            for (uint32_t x = 0; x < R; ++x) {
                pr->n[k][(size_t)ck * R + x] += cnt[x];
                pr->tot[k][ck] += cnt[x];
            }
        }
    }
    for (uint32_t k = 0; k < pr->order; ++k) {
        const double a = m->cfg.alpha[k];
        for (uint32_t ck = 0; ck < pr->width[k]; ++ck) {
            const double *par = k ? pr->p[k - 1] + (size_t)(ck % pr->width[k - 1]) * R
                                  : NULL;
            double *out = pr->p[k] + (size_t)ck * R;
            const double *cnt = pr->n[k] + (size_t)ck * R;
            const double t = pr->tot[k][ck];
            for (uint32_t x = 0; x < R; ++x) {
                const double lparent = par ? par[x] : -log((double)R);
                out[x] = !(t > 0.0) ? lparent
                       : cnt[x] > 0.0 ? log(cnt[x] + a * exp(lparent)) - log(t + a)
                                      : log(a) + lparent - log(t + a);
            }
        }
    }
    return true;
}

/* The log row the level-k context `ck` (counts packing) should hold, k <
 * order: its own P_k if observed, else its nearest observed ancestor's, else
 * the uniform row. */
static void prior_ref_level_log(const PriorRef *pr, uint32_t k, uint32_t ck,
                                double *out) {
    for (uint32_t j = k + 1u; j-- > 0; ) {
        const uint32_t cj = ck % pr->width[j];
        if (pr->tot[j][cj] > 0.0) {
            const double *p = pr->p[j] + (size_t)cj * pr->R;
            for (uint32_t x = 0; x < pr->R; ++x) out[x] = p[x];
            return;
        }
    }
    for (uint32_t x = 0; x < pr->R; ++x) out[x] = -log((double)pr->R);
}

/* The log row the TOP context `c` (counts packing) should hold. False when
 * it was never observed, in which case `out` holds its nearest ancestor. */
static bool prior_ref_top_log(const PriorRef *pr, uint32_t c, double *out) {
    const uint32_t R = pr->R, K = pr->order;
    const uint32_t nc = pr->width[K];
    const double *cnt = beam_cnt_row(pr->m, pr->dir, pr->depth * nc + c);
    double t = 0.0;
    if (cnt) for (uint32_t x = 0; x < R; ++x) t += cnt[x];
    if (!(t > 0.0)) { prior_ref_level_log(pr, K - 1u, c % pr->width[K - 1], out);
                      return false; }
    const double a = pr->m->cfg.alpha[K];
    const double *par = pr->p[K - 1] + (size_t)(c % pr->width[K - 1]) * R;
    for (uint32_t x = 0; x < R; ++x)
        out[x] = cnt[x] > 0.0 ? log(cnt[x] + a * exp(par[x])) - log(t + a)
                              : log(a) + par[x] - log(t + a);
    return true;
}
#endif /* BEAM_TEST_PRIOR_REFERENCE */

#endif /* BEAM_TEST_UTILS_H */
