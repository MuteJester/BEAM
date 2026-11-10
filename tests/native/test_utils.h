/**
 * @file test_utils.h
 * @brief Shared test-runner macros for tests/native.
 *
 * Each test file is expected to declare its own file-static counters:
 *   static int pass_count = 0, fail_count = 0;
 *
 * RUN_TEST prints the test name (left-padded), invokes it, and lets the
 * test body call PASS() or ASSERT_MSG() to update the counters.
 *
 * Every binary also answers two questions about itself, through
 * beam_test_init(argc, argv) as main's first statement:
 *
 *   --list        print this binary's inventory, one test name per line,
 *                 and run nothing.
 *   --run NAME    run only NAME. Repeatable. A NAME this binary does not
 *                 have is an error, not an empty success.
 *
 * --list exists because the inventory used to be grepped out of these
 * sources by verification/check.py, which reads the source rather than the
 * program: a RUN_TEST line main never reaches, or one the preprocessor
 * drops, counted there as a test that ran. Asking the binary cannot make
 * that mistake.
 */
#ifndef BEAM_TEST_UTILS_H
#define BEAM_TEST_UTILS_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* argv slots naming the tests to run; empty means all of them. */
#define BEAM_TEST_MAX_SELECTED 64
static const char *beam_test_selected[BEAM_TEST_MAX_SELECTED];
static char beam_test_matched[BEAM_TEST_MAX_SELECTED];
static int beam_test_selected_count = 0;
static int beam_test_list_only = 0;

/*
 * While listing, stdout must carry the inventory and nothing else: it is the
 * input to verification/check.py and to `make list-tests`, and a parser that
 * has to pick names out of banner text can silently drop one. The 35 mains
 * print their own group banners with printf, which this header cannot edit,
 * so printf is routed through a wrapper that goes quiet while listing. The
 * names themselves are emitted with fputs, which the macro does not touch.
 */
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
static inline int beam_test_printf(const char *fmt, ...) {
    if (beam_test_list_only) return 0;
    va_list ap;
    va_start(ap, fmt);
    const int n = vprintf(fmt, ap);
    va_end(ap);
    return n;
}
#define printf(...) beam_test_printf(__VA_ARGS__)

/* Is `name` selected to run, and record that it was found. */
static inline int beam_test_wanted(const char *name) {
    if (beam_test_selected_count == 0) return 1;
    int wanted = 0;
    for (int i = 0; i < beam_test_selected_count; i++) {
        if (strcmp(name, beam_test_selected[i]) == 0) {
            beam_test_matched[i] = 1;
            wanted = 1;
        }
    }
    return wanted;
}

/*
 * A --run name this binary does not have would otherwise run nothing, report
 * "0 passed, 0 failed" and exit 0 -- a selection typo indistinguishable from
 * a suite that passed. Checked at exit because the mains own their own return
 * statement, and _Exit is what overrides the status one is about to produce.
 */
static void beam_test_report_unmatched(void) {
    if (beam_test_list_only) return;
    for (int i = 0; i < beam_test_selected_count; i++) {
        if (!beam_test_matched[i]) {
            fprintf(stderr, "NO SUCH TEST: %s\n", beam_test_selected[i]);
            fflush(NULL);
            _Exit(2);
        }
    }
}

static inline void beam_test_init(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--list") == 0) {
            beam_test_list_only = 1;
        } else if (strcmp(argv[i], "--run") == 0 && i + 1 < argc) {
            if (beam_test_selected_count == BEAM_TEST_MAX_SELECTED) {
                fprintf(stderr, "%s: more than %d --run names\n",
                        argv[0], BEAM_TEST_MAX_SELECTED);
                exit(2);
            }
            beam_test_selected[beam_test_selected_count++] = argv[++i];
        } else {
            fprintf(stderr, "%s: unrecognised argument: %s\n"
                            "usage: %s [--list] [--run NAME]...\n",
                    argv[0], argv[i], argv[0]);
            exit(2);
        }
    }
    if (beam_test_selected_count > 0) atexit(beam_test_report_unmatched);
}

#define RUN_TEST(fn) do {                                        \
    if (beam_test_list_only) { fputs(#fn "\n", stdout); break; } \
    if (!beam_test_wanted(#fn)) break;                           \
    printf("  %-55s ", #fn);                                     \
    fn();                                                        \
} while (0)

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
