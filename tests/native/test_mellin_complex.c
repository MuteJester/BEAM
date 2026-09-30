/**
 * @file test_mellin_complex.c
 * @brief The Mellin transform at complex orders, against brute force.
 *
 * `beam_model_log_mellin_complex` is the power sum `M(q) = sum_s P(s)^q` at
 * complex `q`, returned as `ln |M|` and `arg M`. It is held here to a direct
 * sum over every string the model can produce -- the fixtures are small
 * enough to enumerate -- and to the properties that do not need the sum:
 * agreement with the real transform on the real axis, `M(conj q) =
 * conj M(q)`, and the same bits whatever the company, the thread count or the
 * budget.
 *
 * Errors against the brute-force sum are judged relative to the sum of the
 * moduli, `M(Re q)`: at complex `q` the terms carry phases and cancel, and an
 * error is only meaningful against what was summed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

static char msg[512];


/* ── Fixtures ────────────────────────────────────────────────── */

#define N_SEQ 300
static char seq_buf[N_SEQ][16];
static const char *seqs[N_SEQ];
static double wts[N_SEQ];

static uint32_t lcg(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(*s >> 33);
}

typedef struct {
    const char  *name;
    BeamAlphabet alphabet;
    const char  *emit;          /* every residue the alphabet can emit */
    uint32_t     order, max_length;
    bool         sparse;
} Spec;

static const Spec SPECS[] = {
    {"nt order 3",       BEAM_ALPHABET_NT, "ACGTN",                3u, 6u, true},
    {"nt order 3 dense", BEAM_ALPHABET_NT, "ACGTN",                3u, 6u, false},
    {"aa order 2",       BEAM_ALPHABET_AA, "ACDEFGHIKLMNPQRSTVWY", 2u, 3u, true},
};
#define N_SPECS (sizeof SPECS / sizeof *SPECS)

static BeamModel *build(const Spec *sp) {
    BeamModelConfig c;
    beam_model_config_default_for(sp->alphabet, &c);
    c.order = sp->order; c.max_length = sp->max_length; c.sparse = sp->sparse;
    BeamModel *m = beam_model_new(&c);
    if (!m) return NULL;
    const char *syms = sp->alphabet == BEAM_ALPHABET_NT ? "ACGT" : "ASGLT";
    const uint32_t ns = (uint32_t)strlen(syms);
    uint64_t s = 0xc0ffeeull + sp->order;
    for (uint32_t k = 0; k < N_SEQ; ++k) {
        const uint32_t len = 2u + lcg(&s) % (sp->max_length - 1u);
        for (uint32_t i = 0; i < len; ++i) seq_buf[k][i] = syms[lcg(&s) % ns];
        seq_buf[k][len] = '\0';
        seqs[k] = seq_buf[k];
        wts[k] = 1.0 + (double)(lcg(&s) % 30u);
    }
    if (beam_model_observe(m, seqs, wts, N_SEQ) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

/* ln P of every string of length 1..max_length, in a malloc'd array. */
static double *enumerate(const BeamModel *m, const Spec *sp, size_t *out_n) {
    const size_t R = strlen(sp->emit);
    size_t total = 0, pw = 1;
    for (uint32_t L = 1; L <= sp->max_length; ++L) { pw *= R; total += pw; }
    double *lp = (double *)malloc(total * sizeof(double));
    char **strs = (char **)malloc(total * sizeof(char *));
    char *pool = (char *)malloc(total * (sp->max_length + 1u));
    if (!lp || !strs || !pool) { free(lp); free(strs); free(pool); return NULL; }
    size_t k = 0;
    for (uint32_t L = 1; L <= sp->max_length; ++L) {
        size_t n = 1;
        for (uint32_t i = 0; i < L; ++i) n *= R;
        for (size_t idx = 0; idx < n; ++idx, ++k) {
            char *p = pool + k * (sp->max_length + 1u);
            size_t v = idx;
            for (uint32_t i = 0; i < L; ++i) { p[L - 1u - i] = sp->emit[v % R]; v /= R; }
            p[L] = '\0';
            strs[k] = p;
        }
    }
    const BeamError e = beam_model_log_pseq_batch(m, (const char *const *)strs,
                                                  total, lp);
    free(strs); free(pool);
    if (e != BEAM_OK) { free(lp); return NULL; }
    *out_n = total;
    return lp;
}

static const double QRE[] = {0.5, 1.0, 0.5, 0.5, 0.3, 1.0, 2.0, -0.4, 0.0, 0.0};
static const double QIM[] = {0.0, 0.0, 3.0, -3.0, 11.0, 0.7, -1.5, 2.0, 1.0, 25.0};
#define NQ (sizeof QRE / sizeof *QRE)

static bool same_bits(double a, double b) { return memcmp(&a, &b, sizeof a) == 0; }


/* ── Against brute force ─────────────────────────────────────── */

static void test_matches_the_enumerated_sum(void) {
    for (size_t f = 0; f < N_SPECS; ++f) {
        const Spec *sp = &SPECS[f];
        BeamModel *m = build(sp);
        ASSERT_MSG(m, "a fixture failed to build");
        size_t n = 0;
        double *lp = enumerate(m, sp, &n);
        double lre[NQ], lim[NQ];
        const bool ok = lp && beam_model_log_mellin_complex(m, QRE, QIM, NQ,
                                                            lre, lim) == BEAM_OK;
        if (!ok) { free(lp); beam_model_destroy(m); ASSERT_MSG(0, "a call failed"); }
        double worst = 0.0;
        for (size_t j = 0; j < NQ; ++j) {
            double sr = 0.0, si = 0.0, sabs = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double mod = exp(QRE[j] * lp[k]);
                sr += mod * cos(QIM[j] * lp[k]);
                si += mod * sin(QIM[j] * lp[k]);
                sabs += mod;
            }
            const double gr = exp(lre[j]) * cos(lim[j]);
            const double gi = exp(lre[j]) * sin(lim[j]);
            const double err = hypot(gr - sr, gi - si) / sabs;
            if (err > worst) worst = err;
            if (!(err <= 1e-12)) {
                snprintf(msg, sizeof msg, "%s, q=%g%+gi: M=%.15g%+.15gi, "
                         "sum=%.15g%+.15gi, error %.3g of the moduli",
                         sp->name, QRE[j], QIM[j], gr, gi, sr, si, err);
                free(lp); beam_model_destroy(m);
                ASSERT_MSG(0, msg);
            }
        }
        printf("    %-18s %zu strings, worst error %.2e of the moduli\n",
               sp->name, n, worst);
        free(lp);
        beam_model_destroy(m);
    }
    PASS();
}


/* ── Properties that need no sum ─────────────────────────────── */

static void test_real_axis_and_conjugates(void) {
    for (size_t f = 0; f < N_SPECS; ++f) {
        BeamModel *m = build(&SPECS[f]);
        ASSERT_MSG(m, "a fixture failed to build");
        const double re[] = {0.0, 0.5, 1.0, 2.0, -0.3};
        const double zero[] = {0.0, 0.0, 0.0, 0.0, 0.0};
        double lre[5], lim[5];
        bool ok = beam_model_log_mellin_complex(m, re, zero, 5, lre, lim) == BEAM_OK;
        for (size_t j = 0; ok && j < 5; ++j) {
            double want;
            ok = beam_model_log_mellin(m, re[j], &want) == BEAM_OK;
            if (ok && !(fabs(lre[j] - want) <= 1e-12 * fmax(1.0, fabs(want))
                        && fabs(lim[j]) <= 1e-12)) {
                snprintf(msg, sizeof msg, "%s, q=%g: %.17g%+.3gi vs the real "
                         "transform %.17g", SPECS[f].name, re[j], lre[j],
                         lim[j], want);
                ok = false;
            }
        }
        /* M(conj q) = conj M(q): the log-modulus is the same number, the
         * phase changes sign. */
        double cre[NQ], cim[NQ], nim[NQ];
        for (size_t j = 0; j < NQ; ++j) nim[j] = -QIM[j];
        double pre[NQ], pim[NQ];
        ok = ok && beam_model_log_mellin_complex(m, QRE, QIM, NQ, pre, pim) == BEAM_OK
                && beam_model_log_mellin_complex(m, QRE, nim, NQ, cre, cim) == BEAM_OK;
        for (size_t j = 0; ok && j < NQ; ++j) {
            if (!(fabs(pre[j] - cre[j]) <= 1e-12 * fmax(1.0, fabs(pre[j]))
                  && fabs(pim[j] + cim[j]) <= 1e-9)) {
                snprintf(msg, sizeof msg, "%s, q=%g%+gi: conjugate gives "
                         "%.17g%+.17gi against %.17g%+.17gi", SPECS[f].name,
                         QRE[j], QIM[j], cre[j], cim[j], pre[j], pim[j]);
                ok = false;
            }
        }
        beam_model_destroy(m);
        ASSERT_MSG(ok, msg);
    }
    PASS();
}

static void test_company_threads_and_budget_do_not_change_the_bits(void) {
    for (size_t f = 0; f < N_SPECS; ++f) {
        BeamModel *m = build(&SPECS[f]);
        ASSERT_MSG(m, "a fixture failed to build");
        const uint64_t budget = beam_model_memory_budget();
        double wre[NQ], wim[NQ], ore[NQ], oim[NQ], tre[NQ], tim[NQ],
               bre[NQ], bim[NQ];
        bool ok = beam_model_log_mellin_complex(m, QRE, QIM, NQ, wre, wim) == BEAM_OK;
        for (size_t j = 0; ok && j < NQ; ++j)       /* one order per call */
            ok = beam_model_log_mellin_complex(m, &QRE[j], &QIM[j], 1,
                                               &ore[j], &oim[j]) == BEAM_OK;
        beam_model_set_threads(1);
        ok = ok && beam_model_log_mellin_complex(m, QRE, QIM, NQ, tre, tim) == BEAM_OK;
        beam_model_set_threads(0);
        /* A budget that fits exactly one sweep of one order: every order runs
         * in a block of its own. */
        beam_model_set_memory_budget(beam_jet_cfan_bytes(m->n_states,
                                                         m->n_residues, 1u));
        beam_model_set_threads(1);
        ok = ok && beam_model_log_mellin_complex(m, QRE, QIM, NQ, bre, bim) == BEAM_OK;
        beam_model_set_threads(0);
        beam_model_set_memory_budget(budget);
        for (size_t j = 0; ok && j < NQ; ++j) {
            if (!(same_bits(ore[j], wre[j]) && same_bits(oim[j], wim[j])
                  && same_bits(tre[j], wre[j]) && same_bits(tim[j], wim[j])
                  && same_bits(bre[j], wre[j]) && same_bits(bim[j], wim[j]))) {
                snprintf(msg, sizeof msg, "%s, q=%g%+gi: the bits moved with "
                         "the company, the threads or the budget",
                         SPECS[f].name, QRE[j], QIM[j]);
                ok = false;
            }
        }
        beam_model_destroy(m);
        ASSERT_MSG(ok, msg);
    }
    PASS();
}

/* ── The vertical-line form ──────────────────────────────────── */

static void test_line_agrees_with_the_general_form(void) {
    /* To rounding, not to the bit: the line takes each rotation by a multiply
     * from the last, the general form by a cosine and a sine. Judged against
     * M(Re q), the sum of the moduli, as everything complex here is. */
    for (size_t f = 0; f < N_SPECS; ++f) {
        BeamModel *m = build(&SPECS[f]);
        ASSERT_MSG(m, "a fixture failed to build");
        enum { K = 100 };
        const double re = 0.5, im0 = -20.0, dim = 0.4;
        double lre[K], lim[K], gre[K], gim[K], qre[K], qim[K], ref = 0.0;
        for (int k = 0; k < K; ++k) { qre[k] = re; qim[k] = im0 + k * dim; }
        bool ok = beam_model_log_mellin_line(m, re, im0, dim, K, lre, lim) == BEAM_OK
               && beam_model_log_mellin_complex(m, qre, qim, K, gre, gim) == BEAM_OK
               && beam_model_log_mellin(m, re, &ref) == BEAM_OK;
        double worst = 0.0;
        for (int k = 0; ok && k < K; ++k) {
            const double a = exp(lre[k] - ref), b = exp(gre[k] - ref);
            const double d = hypot(a * cos(lim[k]) - b * cos(gim[k]),
                                   a * sin(lim[k]) - b * sin(gim[k]));
            if (d > worst) worst = d;
        }
        printf("    %-18s line vs general: worst %.2e of M(Re q)\n",
               SPECS[f].name, worst);
        beam_model_destroy(m);
        ASSERT_MSG(ok, "a call failed");
        ASSERT_MSG(worst <= 1e-12, "the line drifted from the general form");
    }
    PASS();
}

static void test_line_bits_depend_on_the_line_alone(void) {
    /* Order k's bits: the same at every line length, thread count and a
     * budget small enough to split the line into stretches that start
     * between the recurrence's anchors. */
    BeamModel *m = build(&SPECS[0]);
    ASSERT_MSG(m, "a fixture failed to build");
    enum { K = 150 };
    const double re = 0.5, im0 = -20.0, dim = 0.3;
    double wre[K], wim[K], sre[K], sim[K], tre[K], tim[K], bre[K], bim[K];
    const uint64_t budget = beam_model_memory_budget();
    bool ok = beam_model_log_mellin_line(m, re, im0, dim, K, wre, wim) == BEAM_OK
           && beam_model_log_mellin_line(m, re, im0, dim, 77, sre, sim) == BEAM_OK;
    beam_model_set_threads(1);
    ok = ok && beam_model_log_mellin_line(m, re, im0, dim, K, tre, tim) == BEAM_OK;
    beam_model_set_threads(0);
    /* One chain a stretch: stretches start at 16, 32, 48 ... between anchors */
    beam_model_set_memory_budget(beam_jet_cfan_bytes(m->n_states,
                                                     m->n_residues,
                                                     BEAM_LINE_LANES));
    beam_model_set_threads(1);
    ok = ok && beam_model_log_mellin_line(m, re, im0, dim, K, bre, bim) == BEAM_OK;
    beam_model_set_threads(0);
    beam_model_set_memory_budget(budget);
    for (int k = 0; ok && k < K; ++k) {
        const bool same = (k >= 77 || (same_bits(sre[k], wre[k]) && same_bits(sim[k], wim[k])))
                       && same_bits(tre[k], wre[k]) && same_bits(tim[k], wim[k])
                       && same_bits(bre[k], wre[k]) && same_bits(bim[k], wim[k]);
        if (!same) {
            snprintf(msg, sizeof msg, "order %d moved with the line length, the "
                     "threads or the stretch boundaries", k);
            ok = false;
        }
    }
    beam_model_destroy(m);
    ASSERT_MSG(ok, msg);
    PASS();
}

static void test_the_calling_contract(void) {
    BeamModel *m = build(&SPECS[0]);
    ASSERT_MSG(m, "a fixture failed to build");
    double re = 0.5, im = 1.0, o1, o2;
    ASSERT_MSG(beam_model_log_mellin_complex(NULL, &re, &im, 1, &o1, &o2)
               == BEAM_ERR_NULL_ARG, "a NULL model was accepted");
    ASSERT_MSG(beam_model_log_mellin_complex(m, NULL, 0, 0, NULL, NULL) == BEAM_OK,
               "no orders is a no-op, not an error");
    const double bad[] = {NAN, INFINITY};
    for (size_t k = 0; k < 2; ++k) {
        const BeamError e1 = beam_model_log_mellin_complex(m, &bad[k], &im, 1, &o1, &o2);
        const BeamError e2 = beam_model_log_mellin_complex(m, &re, &bad[k], 1, &o1, &o2);
        if (e1 != BEAM_ERR_PARAM_OUT_OF_RANGE || e2 != BEAM_ERR_PARAM_OUT_OF_RANGE) {
            beam_model_destroy(m);
            ASSERT_MSG(0, "a non-finite order was accepted");
        }
    }
    beam_model_destroy(m);

    BeamModelConfig c;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &c);
    c.order = 3u; c.max_length = 6u;
    BeamModel *raw = beam_model_new(&c);
    ASSERT_MSG(raw, "could not create a model");
    const BeamError e = beam_model_log_mellin_complex(raw, &re, &im, 1, &o1, &o2);
    beam_model_destroy(raw);
    ASSERT_MSG(e == BEAM_ERR_NOT_BUILT, "an unfinalized model was analysed");
    PASS();
}

/* ── How a line is split over lengths, stretches and threads ─── */

static void test_a_long_line_keeps_every_thread_busy(void) {
    /* The foundation: 194,481 states, 27 lengths, 40 threads, 32 GiB. A
     * 2,048-order line used to be cut for 27 threads -- one per length --
     * so the budget then admitted only 27 stretches at once. */
    const uint64_t gib = 1ull << 30;
    const BeamLinePlan p = beam_line_plan(194481u, 20u, 2048u, 27u, 40u, 32u * gib);
    printf("    width %u, %u stretches a length, %u at once\n",
           p.width, p.n_stretches, p.concurrency);
    ASSERT_MSG(p.concurrency == 40u, "a long line left threads idle");
    ASSERT_MSG(p.concurrency * beam_jet_cfan_bytes(194481u, 20u, (p.width + 15u) / 16u * 16u)
               <= 32u * gib, "the plan overran the budget");
    PASS();
}

static void test_line_plans_cover_the_line_within_the_budget(void) {
    const uint32_t Ns[] = {1u, 7u, 100u, 128u, 512u, 2048u, 12000u};
    const uint32_t nws[] = {1u, 3u, 27u};
    const uint32_t nths[] = {1u, 8u, 40u};
    const uint64_t budgets[] = {1ull << 26, 1ull << 30, 32ull << 30, 256ull << 30};
    const uint32_t S = 194481u, R = 20u;
    for (size_t a = 0; a < sizeof Ns / sizeof *Ns; ++a)
    for (size_t b = 0; b < sizeof nws / sizeof *nws; ++b)
    for (size_t c = 0; c < sizeof nths / sizeof *nths; ++c)
    for (size_t d = 0; d < sizeof budgets / sizeof *budgets; ++d) {
        const uint32_t N = Ns[a], nw = nws[b], nth = nths[c];
        const uint64_t budget = budgets[d];
        const BeamLinePlan p = beam_line_plan(S, R, N, nw, nth, budget);
        const uint64_t need = beam_jet_cfan_bytes(S, R, (p.width + 15u) / 16u * 16u);
        const uint64_t items = (uint64_t)nw * p.n_stretches;
        snprintf(msg, sizeof msg, "N=%u nw=%u threads=%u budget=%llu: width %u, "
                 "%u stretches, %u at once", N, nw, nth,
                 (unsigned long long)budget, p.width, p.n_stretches, p.concurrency);
        if (budget < beam_jet_cfan_bytes(S, R, BEAM_LINE_LANES)) {
            ASSERT_MSG(p.concurrency == 0u, msg);    /* nothing fits: refuse */
            continue;
        }
        ASSERT_MSG(p.width >= 1u && p.n_stretches >= 1u, msg);
        ASSERT_MSG((uint64_t)p.width * p.n_stretches >= N, msg);
        ASSERT_MSG((uint64_t)p.width * (p.n_stretches - 1u) < N, msg);
        ASSERT_MSG(p.width % BEAM_LINE_LANES == 0u || p.n_stretches == 1u, msg);
        ASSERT_MSG(p.concurrency >= 1u && p.concurrency <= nth
                   && p.concurrency <= items, msg);
        ASSERT_MSG((uint64_t)p.concurrency * need <= budget, msg);
        /* Enough memory for every thread at the narrowest worthwhile width:
         * then every thread that has an item to run gets one. */
        if (budget >= (uint64_t)nth * beam_jet_cfan_bytes(S, R, 128u))
            ASSERT_MSG(p.concurrency == (items < nth ? items : nth), msg);
    }
    PASS();
}


int main(void) {
    printf("Mellin transform at complex orders\n");
    RUN_TEST(test_matches_the_enumerated_sum);
    RUN_TEST(test_real_axis_and_conjugates);
    RUN_TEST(test_company_threads_and_budget_do_not_change_the_bits);
    RUN_TEST(test_line_agrees_with_the_general_form);
    RUN_TEST(test_line_bits_depend_on_the_line_alone);
    RUN_TEST(test_the_calling_contract);
    RUN_TEST(test_a_long_line_keeps_every_thread_busy);
    RUN_TEST(test_line_plans_cover_the_line_within_the_budget);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
