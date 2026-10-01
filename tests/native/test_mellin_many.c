/**
 * @file test_mellin_many.c
 * @brief Many Mellin orders in one pass, held to the single-order bits.
 *
 * `beam_model_log_mellin_many` exists to be faster than calling
 * `beam_model_log_mellin` once per order, and for no other reason. So its
 * whole contract is that it computes the same numbers -- not numbers within a
 * tolerance, the same bits -- and every comparison here is `memcmp`. A
 * tolerance would pass a fused sweep that shared one rescale across its
 * orders, which is the cheaper design and the wrong one: it moves the last
 * bits of every order but one, and published figures quote those.
 *
 * Checked on both alphabets, sparse and dense storage and orders 2 to 6; at
 * orders on both sides of 1, at 0.999 and 1.001 where the tilt is nearly
 * flat, and at negative orders, where the layer maximum a sweep rescales by is
 * taken from the smallest weight instead of the largest. Under the prior no
 * cell is -inf, so a negative order weighs every string over the alphabet and
 * is dominated by the least probable of them: at -2 the answer sits one to
 * four hundred nats above zero, which is the case that once overflowed.
 *
 * Then what must NOT matter: which other orders share the call, the thread
 * count, and a budget small enough to split the orders into blocks or to run
 * one single-order sweep at a time.
 *
 * Every state of the trellis is live under the prior, so a sweep costs the
 * full width of the trellis at every layer. The widest two fixtures -- aa
 * order 4 and nt order 6 -- are held to the contract in the first test and
 * left out of the schedule tests, whose paths do not depend on the width.
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
static unsigned long cases_compared = 0;


/* ── Fixtures ────────────────────────────────────────────────── */

/* Generated rather than written out, so the models have many lengths and a
 * frontier worth sharing. A conserved head and tail and a random middle, the
 * shape the two anchors exist to pick up. `lo..hi` are the lengths drawn; a
 * fixture whose max_length is too short for the full head and tail gets a
 * one-residue anchor at each end instead, since a longer sequence would be
 * skipped at observe. */
#define N_SEQ 400
static char seq_buf[N_SEQ][48];
static const char *seqs[N_SEQ];
static double wts[N_SEQ];

static uint32_t lcg(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(*s >> 33);
}

static void make_corpus(BeamAlphabet a, uint32_t lo, uint32_t hi,
                        uint64_t seed) {
    const bool nt = a == BEAM_ALPHABET_NT;
    const bool short_anchor = lo < 8u;
    const char *syms = nt ? "ACGT" : "ACDEFGHIKLMNPQRSTVWY";
    const char *head = short_anchor ? (nt ? "T" : "C") : (nt ? "TGT" : "CASS");
    const char *tail = short_anchor ? (nt ? "C" : "F") : (nt ? "TTC" : "EQYF");
    const uint32_t ns = (uint32_t)strlen(syms);
    uint64_t s = seed;
    for (uint32_t k = 0; k < N_SEQ; ++k) {
        const uint32_t len = lo + lcg(&s) % (hi - lo + 1u);
        const uint32_t fill = len - (uint32_t)strlen(head) - (uint32_t)strlen(tail);
        char *p = seq_buf[k];
        strcpy(p, head);
        for (uint32_t i = 0; i < fill; ++i) p[strlen(head) + i] = syms[lcg(&s) % ns];
        strcpy(p + strlen(head) + fill, tail);
        seqs[k] = p;
        wts[k] = 1.0 + (double)(lcg(&s) % 40u);
    }
}

typedef struct {
    const char  *name;
    BeamAlphabet alphabet;
    uint32_t     order, max_length;
    uint32_t     lo, hi;              /* corpus lengths */
    bool         sparse;
    bool         wide;                /* contract test only; see the header */
} Spec;

/* Sparse is the default storage and dense the same model bit for bit, but
 * the two reach a row by different code, so the cheap fixtures run in both.
 * The wide ones are cut to short max_lengths that still reach layers of their
 * full width: every state is live, and each length added is a full sweep of
 * every order. */
static const Spec SPECS[] = {
    {"aa order 2",       BEAM_ALPHABET_AA, 2u, 16u,  8u, 15u, true,  false},
    {"aa order 3",       BEAM_ALPHABET_AA, 3u,  8u,  4u,  8u, true,  true},
    {"aa order 4",       BEAM_ALPHABET_AA, 4u,  5u,  3u,  5u, true,  true},
    {"aa order 2 dense", BEAM_ALPHABET_AA, 2u, 16u,  8u, 15u, false, false},
    {"nt order 3",       BEAM_ALPHABET_NT, 3u, 24u, 10u, 22u, true,  false},
    {"nt order 6",       BEAM_ALPHABET_NT, 6u, 12u, 10u, 12u, true,  true},
    {"nt order 3 dense", BEAM_ALPHABET_NT, 3u, 24u, 10u, 22u, false, false},
};
#define N_SPECS (sizeof SPECS / sizeof *SPECS)

static BeamModel *build(const Spec *sp) {
    BeamModelConfig c;
    beam_model_config_default_for(sp->alphabet, &c);
    c.order = sp->order; c.max_length = sp->max_length; c.sparse = sp->sparse;
    BeamModel *m = beam_model_new(&c);
    if (!m) return NULL;
    make_corpus(sp->alphabet, sp->lo, sp->hi, 0x5eedull + sp->order);
    if (beam_model_observe(m, seqs, wts, N_SEQ) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

static const double QS[] = {
    0.0, 1e-3, 0.1, 0.5, 0.9, 0.999, 1.0, 1.001, 1.1, 1.5, 2.0, 2.5, 3.0,
    5.0, -0.5, -2.0,
};
#define NQ (sizeof QS / sizeof *QS)

static bool same_bits(double a, double b) { return memcmp(&a, &b, sizeof a) == 0; }

/* The single-order values every other call is held to. */
static bool singles(const BeamModel *m, const double *qs, size_t nq, double *out) {
    for (size_t j = 0; j < nq; ++j)
        if (beam_model_log_mellin(m, qs[j], &out[j]) != BEAM_OK) return false;
    return true;
}

/* `got` against `want`, bit for bit; the first mismatch goes into `msg`. */
static bool all_same(const char *what, const Spec *sp, const double *qs,
                     const double *got, const double *want, size_t nq) {
    for (size_t j = 0; j < nq; ++j) {
        ++cases_compared;
        if (!same_bits(got[j], want[j])) {
            snprintf(msg, sizeof msg, "%s, %s, q=%g: %.17g != %.17g",
                     sp->name, what, qs[j], got[j], want[j]);
            return false;
        }
    }
    return true;
}


/* ── The contract ────────────────────────────────────────────── */

static void test_every_order_is_its_single_order_value_bitwise(void) {
    for (size_t k = 0; k < N_SPECS; ++k) {
        BeamModel *m = build(&SPECS[k]);
        ASSERT_MSG(m, "a fixture failed to build");
        double want[NQ], got[NQ];
        const bool ok = singles(m, QS, NQ, want)
                     && beam_model_log_mellin_many(m, QS, NQ, got) == BEAM_OK;
        if (!ok) { beam_model_destroy(m); ASSERT_MSG(0, "a call failed"); }
        /* A fixture whose every value is -inf would compare equal and prove
         * nothing: the total mass at q = 1 must be there. */
        const bool live = fabs(want[6]) < 1e-9;
        const bool same = all_same("one call", &SPECS[k], QS, got, want, NQ);
        beam_model_destroy(m);
        ASSERT_MSG(live, "q = 1 is not the total mass; the fixture is empty");
        ASSERT_MSG(same, msg);
    }
    PASS();
}

static void test_the_company_an_order_keeps_does_not_change_it(void) {
    /* Reversed, repeated, and one order per call. Each order's arithmetic is
     * its own, so none of these can move a bit. */
    for (size_t k = 0; k < N_SPECS; ++k) {
        if (SPECS[k].wide) continue;
        BeamModel *m = build(&SPECS[k]);
        ASSERT_MSG(m, "a fixture failed to build");
        double want[NQ], rev_q[NQ], rev[NQ], dup_q[2 * NQ], dup[2 * NQ], one[NQ];
        bool ok = singles(m, QS, NQ, want);
        for (size_t j = 0; j < NQ; ++j) {
            rev_q[j] = QS[NQ - 1 - j];
            dup_q[j] = QS[j]; dup_q[NQ + j] = QS[(j * 7u) % NQ];
        }
        ok = ok && beam_model_log_mellin_many(m, rev_q, NQ, rev) == BEAM_OK
                && beam_model_log_mellin_many(m, dup_q, 2 * NQ, dup) == BEAM_OK;
        for (size_t j = 0; ok && j < NQ; ++j)
            ok = beam_model_log_mellin_many(m, &QS[j], 1, &one[j]) == BEAM_OK;
        if (!ok) { beam_model_destroy(m); ASSERT_MSG(0, "a call failed"); }
        double rev_back[NQ], dup_tail[NQ];
        for (size_t j = 0; j < NQ; ++j) {
            rev_back[j] = rev[NQ - 1 - j];
            dup_tail[j] = want[(j * 7u) % NQ];
        }
        const bool same =
            all_same("reversed", &SPECS[k], QS, rev_back, want, NQ)
            && all_same("repeated, first copy", &SPECS[k], QS, dup, want, NQ)
            && all_same("repeated, second copy", &SPECS[k], dup_q + NQ,
                        dup + NQ, dup_tail, NQ)
            && all_same("one per call", &SPECS[k], QS, one, want, NQ);
        beam_model_destroy(m);
        ASSERT_MSG(same, msg);
    }
    PASS();
}

static void test_threads_and_budget_do_not_change_an_order(void) {
    /* Three schedules that each take a different path through the driver:
     * one thread, so every layer's targets go to one thread; a budget that
     * fits one vector of orders, so the list splits into stretches of eight;
     * and the same budget on one thread. */
    for (size_t k = 0; k < N_SPECS; ++k) {
        if (SPECS[k].wide) continue;
        BeamModel *m = build(&SPECS[k]);
        ASSERT_MSG(m, "a fixture failed to build");
        const uint32_t S = beam_model_n_states(m), R = beam_model_n_residues(m);
        const uint32_t A = m->n_alphabet;
        double want[NQ], one_thread[NQ], blocks[NQ], serial[NQ];
        bool ok = singles(m, QS, NQ, want);

        beam_model_set_threads(1);
        ok = ok && beam_model_log_mellin_many(m, QS, NQ, one_thread) == BEAM_OK;
        beam_model_set_threads(0);

        const int th = beam_model_effective_threads();
        beam_model_set_memory_budget(beam_jet_real_bytes(
            S, R, A, BEAM_REAL_LANES, th > 0 ? (uint32_t)th : 1u));
        ok = ok && beam_model_log_mellin_many(m, QS, NQ, blocks) == BEAM_OK;
        beam_model_set_memory_budget(beam_jet_real_bytes(S, R, A, BEAM_REAL_LANES, 1u));
        beam_model_set_threads(1);
        ok = ok && beam_model_log_mellin_many(m, QS, NQ, serial) == BEAM_OK;
        beam_model_set_threads(0);
        beam_model_set_memory_budget(0);

        if (!ok) { beam_model_destroy(m); ASSERT_MSG(0, "a call failed"); }
        const bool same =
            all_same("one thread", &SPECS[k], QS, one_thread, want, NQ)
            && all_same("stretches of eight", &SPECS[k], QS, blocks, want, NQ)
            && all_same("stretches of eight, one thread", &SPECS[k], QS, serial, want, NQ);
        beam_model_destroy(m);
        ASSERT_MSG(same, msg);
    }
    PASS();
}

static void test_it_refuses_only_where_one_order_would_not_fit(void) {
    BeamModel *m = build(&SPECS[0]);
    ASSERT_MSG(m, "the fixture failed to build");
    const uint32_t S = beam_model_n_states(m), R = beam_model_n_residues(m);
    const uint32_t A = m->n_alphabet;
    /* The least a sweep can hold: one vector of orders, on one thread. */
    const uint64_t one = beam_jet_real_bytes(S, R, A, BEAM_REAL_LANES, 1u);
    double out[NQ], single = 0.0;

    beam_model_set_threads(1);
    beam_model_set_memory_budget(one);
    const BeamError at = beam_model_log_mellin_many(m, QS, NQ, out);
    beam_model_set_memory_budget(one - 1u);
    const BeamError below = beam_model_log_mellin_many(m, QS, NQ, out);
    beam_model_set_memory_budget(1024u);
    const BeamError tiny_many = beam_model_log_mellin_many(m, QS, NQ, out);
    const BeamError tiny_single = beam_model_log_mellin(m, 2.0, &single);
    beam_model_set_memory_budget(0);
    beam_model_set_threads(0);
    beam_model_destroy(m);

    ASSERT_MSG(at == BEAM_OK, "a budget of exactly one single-order sweep refused");
    ASSERT_MSG(below == BEAM_ERR_BUDGET, "one byte short of a sweep was not refused");
    ASSERT_MSG(tiny_many == BEAM_ERR_BUDGET && tiny_single == BEAM_ERR_BUDGET,
               "a budget neither form fits must refuse in both");
    PASS();
}

static void test_large_orders_stay_finite_and_in_bounds(void) {
    /* At large q most states' mass underflows, and a state with none can
     * still carry an edge heavier than the order's shift, whose factor
     * exp(q w - qmax) would overflow; times its zero mass that is NaN. Every
     * value must stay finite and between the largest term and D0 times it:
     * q ln p_max <= ln M(q) <= q ln p_max + ln D0. */
    static const double BIG[] = {30.0, 90.0, 300.0, 3000.0};
    enum { NB = sizeof BIG / sizeof *BIG };
    for (size_t k = 0; k < N_SPECS; ++k) {
        if (SPECS[k].wide) continue;
        BeamModel *m = build(&SPECS[k]);
        ASSERT_MSG(m, "a fixture failed to build");
        double got[NB], lo = 0.0, hi = 0.0, d0 = 0.0;
        bool ok = beam_model_log_mellin_many(m, BIG, NB, got) == BEAM_OK
               && beam_model_surprisal_range(m, &lo, &hi) == BEAM_OK
               && beam_model_log_mellin(m, 0.0, &d0) == BEAM_OK;
        beam_model_destroy(m);
        ASSERT_MSG(ok, "a call failed");
        for (size_t j = 0; j < NB; ++j) {
            const double floor_ = -BIG[j] * lo, slack = 1e-9 * fabs(floor_) + 1e-9;
            snprintf(msg, sizeof msg, "%s, q=%g: ln M = %.17g, bounds [%.17g, %.17g]",
                     SPECS[k].name, BIG[j], got[j], floor_, floor_ + d0);
            ASSERT_MSG(isfinite(got[j]), msg);
            ASSERT_MSG(got[j] >= floor_ - slack && got[j] <= floor_ + d0 + slack, msg);
        }
    }
    PASS();
}

static void test_the_calling_contract(void) {
    BeamModel *m = build(&SPECS[0]);
    ASSERT_MSG(m, "the fixture failed to build");
    double out[2] = {0.0, 0.0};
    const double qs[2] = {0.5, 2.0};
    ASSERT_MSG(beam_model_log_mellin_many(NULL, qs, 2, out) == BEAM_ERR_NULL_ARG,
               "a NULL model was accepted");
    ASSERT_MSG(beam_model_log_mellin_many(m, NULL, 2, out) == BEAM_ERR_NULL_ARG,
               "NULL orders were accepted");
    ASSERT_MSG(beam_model_log_mellin_many(m, qs, 2, NULL) == BEAM_ERR_NULL_ARG,
               "a NULL output was accepted");
    ASSERT_MSG(beam_model_log_mellin_many(m, NULL, 0, NULL) == BEAM_OK,
               "no orders is a no-op, not an error");
    beam_model_destroy(m);

    BeamModelConfig c;
    beam_model_config_default(&c);
    c.order = 2u; c.max_length = 12u;
    BeamModel *raw = beam_model_new(&c);
    ASSERT_MSG(raw, "could not create a model");
    make_corpus(BEAM_ALPHABET_AA, 8u, 12u, 7u);
    const bool observed = beam_model_observe(raw, seqs, wts, N_SEQ) == BEAM_OK;
    const BeamError e = beam_model_log_mellin_many(raw, qs, 2, out);
    beam_model_destroy(raw);
    ASSERT_MSG(observed, "could not observe");
    ASSERT_MSG(e == BEAM_ERR_NOT_BUILT, "an unfinalized model was analysed");
    PASS();
}


int main(void) {
    printf("Mellin transform at many orders in one pass\n");
    RUN_TEST(test_every_order_is_its_single_order_value_bitwise);
    RUN_TEST(test_the_company_an_order_keeps_does_not_change_it);
    RUN_TEST(test_threads_and_budget_do_not_change_an_order);
    RUN_TEST(test_it_refuses_only_where_one_order_would_not_fit);
    RUN_TEST(test_large_orders_stay_finite_and_in_bounds);
    RUN_TEST(test_the_calling_contract);
    printf("  %lu (model, order) values compared with memcmp\n", cases_compared);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
