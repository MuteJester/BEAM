/**
 * @file test_query_automaton.c
 * @brief Tests for the constraint automaton and its boolean algebra.
 *
 * An automaton is pure data -- it needs no model, no trellis and no sweep --
 * so its language can be checked the only way a language should be: by
 * enumerating every string of a small alphabet and length and comparing the
 * automaton's verdict against a naive matcher written independently of it.
 * Over 4^6 strings and 84 motifs that is a third of a million comparisons, and
 * a Knuth-Morris-Pratt table with one wrong fallback entry cannot survive it.
 *
 * The boolean algebra is checked the same way, and additionally by identities
 * a wrong construction fails even when each operation looks plausible on its
 * own: De Morgan over the whole language, and double negation as the identity.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/query.h"

#include "../../src/native/internal/beam/query_internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

/* Small enough to enumerate exhaustively, wide enough to have structure. */
#define AR 4u          /* residues */
#define AL 6u          /* length   */
#define N_STRINGS 4096u  /* AR^AL   */

/** Decode `code` into `AL` base-`AR` digits, least significant last. */
static void decode(uint32_t code, uint8_t *out) {
    for (uint32_t i = AL; i-- > 0u; ) { out[i] = (uint8_t)(code % AR); code /= AR; }
}

/* ── Independent references ──────────────────────────────────── */

/** True when `motif` occurs anywhere in `s`. A plain scan, deliberately. */
static bool naive_contains(const uint8_t *s, uint32_t n,
                           const uint8_t *motif, uint32_t m) {
    if (m > n) return false;
    for (uint32_t i = 0; i + m <= n; ++i) {
        uint32_t k = 0;
        while (k < m && s[i + k] == motif[k]) ++k;
        if (k == m) return true;
    }
    return false;
}

/** How many positions of `s` hold a residue in `of`. */
static uint32_t naive_count(const uint8_t *s, uint32_t n, BeamResidueMask of) {
    uint32_t c = 0;
    for (uint32_t i = 0; i < n; ++i) if ((of >> s[i]) & 1u) ++c;
    return c;
}

/* ── Motif ───────────────────────────────────────────────────── */

static void test_the_motif_automaton_is_exactly_substring_search(void) {
    uint8_t s[AL], motif[3];
    uint32_t checked = 0u, motifs = 0u;

    for (uint32_t m = 1u; m <= 3u; ++m) {
        const uint32_t n_motifs = (m == 1u) ? AR : (m == 2u) ? AR * AR
                                                             : AR * AR * AR;
        for (uint32_t mc = 0; mc < n_motifs; ++mc) {
            uint32_t v = mc;
            for (uint32_t k = m; k-- > 0u; ) { motif[k] = (uint8_t)(v % AR); v /= AR; }

            BeamAutomaton *a = beam_query_automaton_motif(AL, AR, motif, m);
            ASSERT_MSG(a != NULL, "motif automaton built");
            ASSERT_MSG(beam_query_automaton_states(a) == m + 1u,
                       "one state per prefix, plus the matched state");
            motifs++;

            for (uint32_t code = 0; code < N_STRINGS; ++code) {
                decode(code, s);
                const bool want = naive_contains(s, AL, motif, m);
                const bool got  = beam_query_automaton_accepts(a, s, AL);
                if (want != got) {
                    beam_query_automaton_free(a);
                    ASSERT_MSG(false, "the automaton's language differs from "
                                      "a plain substring scan");
                }
                checked++;
            }
            beam_query_automaton_free(a);
        }
    }
    ASSERT_MSG(motifs == AR + AR * AR + AR * AR * AR, "84 motifs tried");
    ASSERT_MSG(checked == motifs * N_STRINGS, "every string against every one");
    PASS();
}

static void test_a_matched_motif_cannot_be_un_matched(void) {
    /* The accepting state must be absorbing: "contains" is a property of the
     * whole string, so a later residue cannot revoke it. */
    const uint8_t motif[2] = {1u, 2u};
    BeamAutomaton *a = beam_query_automaton_motif(AL, AR, motif, 2u);
    ASSERT_MSG(a != NULL, "built");
    const uint32_t acc = beam_query_automaton_states(a) - 1u;
    for (uint32_t x = 0; x < AR; ++x)
        ASSERT_MSG(beam_query_automaton_step(a, 0u, acc, x) == acc,
                   "the matched state absorbs every residue");
    beam_query_automaton_free(a);
    PASS();
}

static void test_a_motif_that_cannot_fit_is_refused(void) {
    const uint8_t motif[4] = {0u, 1u, 2u, 3u};
    ASSERT_MSG(beam_query_automaton_motif(3u, AR, motif, 4u) == NULL,
               "a 4-residue motif cannot occur in a length-3 sequence");
    ASSERT_MSG(beam_query_automaton_motif(AL, AR, motif, 0u) == NULL,
               "an empty motif is refused rather than accepting everything");
    ASSERT_MSG(beam_query_automaton_motif(AL, 2u, motif, 4u) == NULL,
               "a motif naming residues outside the alphabet is refused");
    PASS();
}

/* ── Count ───────────────────────────────────────────────────── */

static void test_the_count_automaton_is_exactly_a_tally(void) {
    struct { BeamResidueMask of; uint32_t lo, hi; } cases[] = {
        {0x1u, 0u, 0u},      /* none of residue 0        */
        {0x1u, 1u, 1u},      /* exactly one              */
        {0x1u, 2u, AL},      /* at least two             */
        {0x3u, 3u, 4u},      /* a band, two residues     */
        {0x5u, 0u, AL},      /* everything, always true  */
        {0xFu, AL, AL},      /* all six positions        */
        {0x2u, 5u, 3u + 99u},/* hi past the length       */
    };
    uint8_t s[AL];
    for (uint32_t t = 0; t < sizeof(cases) / sizeof(cases[0]); ++t) {
        BeamAutomaton *a = beam_query_automaton_count(
            AL, AR, cases[t].of, cases[t].lo, cases[t].hi);
        ASSERT_MSG(a != NULL, "count automaton built");
        for (uint32_t code = 0; code < N_STRINGS; ++code) {
            decode(code, s);
            const uint32_t c = naive_count(s, AL, cases[t].of);
            const bool want = (c >= cases[t].lo && c <= cases[t].hi);
            if (want != beam_query_automaton_accepts(a, s, AL)) {
                beam_query_automaton_free(a);
                ASSERT_MSG(false, "the automaton disagrees with a direct tally");
            }
        }
        beam_query_automaton_free(a);
    }
    PASS();
}

static void test_the_count_automaton_uses_no_more_states_than_it_needs(void) {
    /*
     * |Q| multiplies the product graph state for state, so a counter carrying
     * one redundant state costs a whole trellis's worth of memory in every
     * query that uses it. That is invisible to a language test -- an extra
     * non-accepting state above the band accepts nothing either -- so it is
     * asserted directly.
     */
    struct { uint32_t length, lo, hi, states; } cases[] = {
        /* hi below the length: counts 0..hi, plus one absorbing overflow. */
        {6u, 0u, 0u, 2u},      /* {0} and "more than 0"      */
        {6u, 1u, 1u, 3u},      /* {0, 1} and overflow        */
        {6u, 2u, 4u, 6u},      /* 0..4 and overflow          */
        /* hi at or above the length: no overflow is reachable. */
        {6u, 0u, 6u, 7u},      /* 0..6, nothing to overflow  */
        {6u, 2u, 99u, 7u},     /* clamped to the length      */
    };
    for (uint32_t t = 0; t < sizeof(cases) / sizeof(cases[0]); ++t) {
        BeamAutomaton *a = beam_query_automaton_count(
            cases[t].length, AR, 0x1u, cases[t].lo, cases[t].hi);
        ASSERT_MSG(a != NULL, "built");
        const uint32_t got = beam_query_automaton_states(a);
        beam_query_automaton_free(a);
        ASSERT_MSG(got == cases[t].states,
                   "a counter carries exactly the states its range needs");
    }
    PASS();
}

static void test_a_count_range_that_cannot_be_met_accepts_nothing(void) {
    uint8_t s[AL];
    /* More occurrences than there are positions. */
    BeamAutomaton *a = beam_query_automaton_count(AL, AR, 0x1u, AL + 1u,
                                                  AL + 2u);
    ASSERT_MSG(a != NULL, "an empty language is buildable");
    for (uint32_t code = 0; code < N_STRINGS; ++code) {
        decode(code, s);
        ASSERT_MSG(!beam_query_automaton_accepts(a, s, AL),
                   "and accepts nothing at all");
    }
    beam_query_automaton_free(a);
    ASSERT_MSG(beam_query_automaton_count(AL, AR, 0x1u, 3u, 2u) == NULL,
               "an inverted range is refused");
    PASS();
}

/* ── Lifting masks ───────────────────────────────────────────── */

static void test_lifting_masks_preserves_the_language(void) {
    BeamConstraint *c = beam_query_constraint_new(AL, AR);
    ASSERT_MSG(c != NULL, "constraint");
    beam_query_constrain_set(c, 1u, 0x5u);
    beam_query_constrain_position(c, 3u, 2u);
    beam_query_constrain_exclude(c, 4u, 0u);

    BeamAutomaton *a = beam_query_automaton_from_masks(c);
    ASSERT_MSG(a != NULL, "lifted");
    ASSERT_MSG(beam_query_automaton_states(a) == 2u, "alive and dead");

    uint8_t s[AL];
    for (uint32_t code = 0; code < N_STRINGS; ++code) {
        decode(code, s);
        bool want = true;
        for (uint32_t i = 0; i < AL; ++i)
            if (!((beam_query_constraint_mask(c, i) >> s[i]) & 1u)) want = false;
        ASSERT_MSG(want == beam_query_automaton_accepts(a, s, AL),
                   "the lifted automaton is the masks");
    }
    beam_query_automaton_free(a);
    beam_query_constraint_free(c);
    PASS();
}

/* ── Boolean algebra ─────────────────────────────────────────── */

/** Two structurally different automata to combine in the tests below. */
static void build_pair(BeamAutomaton **a, BeamAutomaton **b) {
    const uint8_t motif[2] = {0u, 1u};
    *a = beam_query_automaton_motif(AL, AR, motif, 2u);   /* uniform    */
    BeamConstraint *c = beam_query_constraint_new(AL, AR);
    beam_query_constrain_set(c, 2u, 0x6u);
    beam_query_constrain_exclude(c, 5u, 3u);
    *b = beam_query_automaton_from_masks(c);              /* layer-indexed */
    beam_query_constraint_free(c);
}

static void test_and_or_and_not_are_the_boolean_combinations(void) {
    BeamAutomaton *a = NULL, *b = NULL;
    build_pair(&a, &b);
    ASSERT_MSG(a && b, "operands built");

    BeamAutomaton *both   = beam_query_automaton_and(a, b);
    BeamAutomaton *either = beam_query_automaton_or(a, b);
    BeamAutomaton *not_a  = beam_query_automaton_not(a);
    ASSERT_MSG(both && either && not_a, "combinations built");
    /* One uniform operand and one layer-indexed one, so the product must be
     * layer-indexed; a construction that kept the uniform fast path here
     * would silently apply layer 0's masks at every position. */
    ASSERT_MSG(beam_query_automaton_states(both) == 2u * 3u, "|Q| multiplies");

    uint8_t s[AL];
    for (uint32_t code = 0; code < N_STRINGS; ++code) {
        decode(code, s);
        const bool ja = beam_query_automaton_accepts(a, s, AL);
        const bool jb = beam_query_automaton_accepts(b, s, AL);
        ASSERT_MSG(beam_query_automaton_accepts(both, s, AL) == (ja && jb),
                   "and");
        ASSERT_MSG(beam_query_automaton_accepts(either, s, AL) == (ja || jb),
                   "or");
        ASSERT_MSG(beam_query_automaton_accepts(not_a, s, AL) == !ja, "not");
    }
    beam_query_automaton_free(both);
    beam_query_automaton_free(either);
    beam_query_automaton_free(not_a);
    beam_query_automaton_free(a);
    beam_query_automaton_free(b);
    PASS();
}

static void test_de_morgan_holds_over_the_whole_language(void) {
    BeamAutomaton *a = NULL, *b = NULL;
    build_pair(&a, &b);
    ASSERT_MSG(a && b, "operands built");

    BeamAutomaton *na = beam_query_automaton_not(a);
    BeamAutomaton *nb = beam_query_automaton_not(b);
    /* `not` copies rather than consuming, so the conjunction has to be held
     * and freed; leaving it inline leaked it. */
    BeamAutomaton *both = beam_query_automaton_and(a, b);
    BeamAutomaton *lhs = beam_query_automaton_not(both);
    BeamAutomaton *rhs = beam_query_automaton_or(na, nb);
    ASSERT_MSG(na && nb && both && lhs && rhs, "built");

    uint8_t s[AL];
    for (uint32_t code = 0; code < N_STRINGS; ++code) {
        decode(code, s);
        ASSERT_MSG(beam_query_automaton_accepts(lhs, s, AL)
                   == beam_query_automaton_accepts(rhs, s, AL),
                   "not(a and b) is (not a) or (not b), string for string");
    }
    beam_query_automaton_free(both);
    beam_query_automaton_free(lhs);
    beam_query_automaton_free(rhs);
    beam_query_automaton_free(na);
    beam_query_automaton_free(nb);
    beam_query_automaton_free(a);
    beam_query_automaton_free(b);
    PASS();
}

static void test_double_negation_is_the_identity(void) {
    const uint8_t motif[3] = {2u, 0u, 2u};
    BeamAutomaton *a  = beam_query_automaton_motif(AL, AR, motif, 3u);
    BeamAutomaton *n  = beam_query_automaton_not(a);
    BeamAutomaton *nn = beam_query_automaton_not(n);
    ASSERT_MSG(a && n && nn, "built");
    uint8_t s[AL];
    for (uint32_t code = 0; code < N_STRINGS; ++code) {
        decode(code, s);
        ASSERT_MSG(beam_query_automaton_accepts(nn, s, AL)
                   == beam_query_automaton_accepts(a, s, AL), "not not a is a");
    }
    beam_query_automaton_free(a);
    beam_query_automaton_free(n);
    beam_query_automaton_free(nn);
    PASS();
}

static void test_combining_incompatible_automata_is_refused(void) {
    const uint8_t motif[1] = {0u};
    BeamAutomaton *a = beam_query_automaton_motif(AL, AR, motif, 1u);
    BeamAutomaton *b = beam_query_automaton_motif(AL + 1u, AR, motif, 1u);
    BeamAutomaton *d = beam_query_automaton_motif(AL, AR - 1u, motif, 1u);
    ASSERT_MSG(a && b && d, "built");
    ASSERT_MSG(beam_query_automaton_and(a, b) == NULL, "lengths must match");
    ASSERT_MSG(beam_query_automaton_or(a, d) == NULL, "alphabets must match");
    ASSERT_MSG(beam_query_automaton_and(a, NULL) == NULL, "NULL is refused");
    beam_query_automaton_free(a);
    beam_query_automaton_free(b);
    beam_query_automaton_free(d);
    PASS();
}

static void test_a_product_past_the_state_ceiling_is_refused(void) {
    uint8_t motif[40];
    for (uint32_t k = 0; k < 40u; ++k) motif[k] = (uint8_t)(k % AR);
    BeamAutomaton *a = beam_query_automaton_motif(60u, AR, motif, 40u);
    ASSERT_MSG(a != NULL && beam_query_automaton_states(a) == 41u, "41 states");
    /* 41 * 41 = 1681, past the 1024 ceiling. */
    ASSERT_MSG(beam_query_automaton_and(a, a) == NULL, "refused");
    ASSERT_MSG(strstr(beam_error_message(), "1681") != NULL,
               "and the message says how many it would have needed");
    beam_query_automaton_free(a);
    PASS();
}

/* ── Attaching to a constraint ───────────────────────────────── */

static void test_attaching_an_automaton_intersects(void) {
    const uint8_t m1[2] = {0u, 1u};
    const uint8_t m2[2] = {2u, 3u};
    BeamConstraint *c = beam_query_constraint_new(AL, AR);
    ASSERT_MSG(beam_query_constrain_automaton(
                   c, beam_query_automaton_motif(AL, AR, m1, 2u)) == BEAM_OK,
               "first attach");
    ASSERT_MSG(beam_query_constrain_automaton(
                   c, beam_query_automaton_motif(AL, AR, m2, 2u)) == BEAM_OK,
               "second attach");
    const BeamAutomaton *got = beam_query_constraint_automaton(c);
    ASSERT_MSG(got != NULL, "the constraint carries one");
    ASSERT_MSG(beam_query_automaton_states(got) == 9u,
               "and it is the product of the two, not the second alone");

    uint8_t s[AL];
    BeamAutomaton *a1 = beam_query_automaton_motif(AL, AR, m1, 2u);
    BeamAutomaton *a2 = beam_query_automaton_motif(AL, AR, m2, 2u);
    for (uint32_t code = 0; code < N_STRINGS; ++code) {
        decode(code, s);
        const bool want = beam_query_automaton_accepts(a1, s, AL)
                       && beam_query_automaton_accepts(a2, s, AL);
        ASSERT_MSG(beam_query_automaton_accepts(got, s, AL) == want,
                   "attaching twice means both, not the last one");
    }
    beam_query_automaton_free(a1);
    beam_query_automaton_free(a2);
    beam_query_constraint_free(c);
    PASS();
}

static void test_a_mismatched_automaton_is_refused_and_still_consumed(void) {
    BeamConstraint *c = beam_query_constraint_new(AL, AR);
    const uint8_t motif[1] = {0u};
    /* Wrong length. The call takes ownership on every path, so a caller that
     * frees nothing here is correct; under a leak checker this test is what
     * proves it. */
    ASSERT_MSG(beam_query_constrain_automaton(
                   c, beam_query_automaton_motif(AL + 2u, AR, motif, 1u))
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "refused");
    ASSERT_MSG(beam_query_constraint_automaton(c) == NULL,
               "and the constraint is unchanged");
    ASSERT_MSG(beam_query_constraint_is_free(c), "still free");
    beam_query_constraint_free(c);
    PASS();
}

static void test_an_automaton_makes_a_constraint_not_free(void) {
    const uint8_t motif[1] = {1u};
    BeamConstraint *c = beam_query_constraint_new(AL, AR);
    ASSERT_MSG(beam_query_constraint_is_free(c), "free to begin with");
    beam_query_constrain_automaton(c, beam_query_automaton_motif(AL, AR,
                                                                motif, 1u));
    ASSERT_MSG(!beam_query_constraint_is_free(c),
               "an automaton narrows even with every mask still full");
    /* The masks themselves are untouched, which is what selects the engine's
     * fast path and is a different question. */
    ASSERT_MSG(beam_query_all_free(c), "and the masks are still open");
    beam_query_constraint_free(c);
    PASS();
}

int main(void) {
    printf("\n[motif]\n");
    RUN_TEST(test_the_motif_automaton_is_exactly_substring_search);
    RUN_TEST(test_a_matched_motif_cannot_be_un_matched);
    RUN_TEST(test_a_motif_that_cannot_fit_is_refused);

    printf("\n[count]\n");
    RUN_TEST(test_the_count_automaton_is_exactly_a_tally);
    RUN_TEST(test_the_count_automaton_uses_no_more_states_than_it_needs);
    RUN_TEST(test_a_count_range_that_cannot_be_met_accepts_nothing);

    printf("\n[masks as automata]\n");
    RUN_TEST(test_lifting_masks_preserves_the_language);

    printf("\n[boolean algebra]\n");
    RUN_TEST(test_and_or_and_not_are_the_boolean_combinations);
    RUN_TEST(test_de_morgan_holds_over_the_whole_language);
    RUN_TEST(test_double_negation_is_the_identity);
    RUN_TEST(test_combining_incompatible_automata_is_refused);
    RUN_TEST(test_a_product_past_the_state_ceiling_is_refused);

    printf("\n[attaching to a constraint]\n");
    RUN_TEST(test_attaching_an_automaton_intersects);
    RUN_TEST(test_a_mismatched_automaton_is_refused_and_still_consumed);
    RUN_TEST(test_an_automaton_makes_a_constraint_not_free);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
