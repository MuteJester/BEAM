/**
 * @file test_query.c
 * @brief Tests for the exact probabilistic query engine.
 *
 * Three kinds of check carry the weight here.
 *
 * The first is bit-for-bit reproduction. A constraint that pins one residue at
 * one position denotes exactly what a clamp denotes, and the engine walks the
 * trellis by the same recursion in the same order, so the two must agree to
 * the last bit and not to a tolerance. Anything looser would pass while the
 * new engine quietly drifted from the one already in the verification gate.
 *
 * The second is enumeration. For a short length the whole support is walkable,
 * so a constrained probability can be checked against the brute-force sum it
 * is defined as. That is the only check here that cannot be passed by
 * accident.
 *
 * The third is an algebraic identity that ties the genuinely new capability to
 * the old one: conditioning on a residue SET must equal the mixture of the
 * conditionals on its members, weighted by their marginals. An engine that
 * built the set path independently of the singleton path fails this even when
 * both paths look reasonable on their own.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/query.h"

/* The constraint layout and the empty-position helper are internal, and the
 * tests that pin down intersection semantics need to read the masks the
 * builders leave behind. */
#include "../../src/native/internal/beam/alphabet.h"
#include "../../src/native/internal/beam/query_internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define R BEAM_AA_RESIDUES

/* Indices into "ACDEFGHIKLMNPQRSTVWY", for constraints that have to name
 * residues the corpus below actually spells. */
#define AA_A_ 0u
#define AA_C  1u
#define AA_D  2u
#define AA_F  4u
#define AA_E  3u
#define AA_G  5u
#define AA_L  9u
#define AA_P 12u
#define AA_Q 13u
#define AA_S_ 15u
#define AA_T 16u
#define AA_V 17u
#define AA_W 18u
#define AA_Y 19u

/* ── Corpora ─────────────────────────────────────────────────── */

/* CDR3-shaped, for the reproduction and identity checks. */
#define N_TRAIN 12
static const char *TRAIN[N_TRAIN] = {
    "CASSLGEQYF",  "CASSLGEQFF",  "CASSQGEQYF",  "CASSPGEQYF",
    "CASSLAEQYF",  "CASSLGETQYF", "CASSVGEQYF",  "CASSLGNTIYF",
    "CASSLREQYF",  "CASSLGEQYFF", "CASSDGEQYF",  "CASSLGEQY",
};
static const double WEIGHT[N_TRAIN] = {
    50, 30, 20, 15, 12, 25, 18, 22, 9, 7, 11, 6
};

/* Short enough that length 4 has a walkable support: 20^4 = 160,000. */
#define N_SHORT 10
static const char *SHORT_TRAIN[N_SHORT] = {
    "CASF", "CASY", "CAGF", "CGSF", "CASL",
    "CTSF", "CASQ", "CVSF", "CASE", "CAAF",
};
static const double SHORT_WEIGHT[N_SHORT] = {
    40, 25, 18, 15, 12, 10, 9, 7, 6, 5
};

static BeamModel *build(const char **seqs, const double *w, uint32_t n,
                        uint8_t order, uint8_t max_length, bool open_world) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = max_length;
    if (open_world) cfg.strict = 0;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, seqs, w, n) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/* ── Independent reference: enumerate the support ────────────── */

/**
 * `P(C | length)` by walking every string of that length and summing the ones
 * the constraint admits.
 *
 * Deliberately built from `beam_model_log_pseq` and nothing the engine
 * touches. The length prior is a common factor of numerator and denominator
 * and cancels, so this is the conditional the engine reports even though
 * `log_pseq` is a joint.
 */
static double enumerate_probability(const BeamModel *m, uint32_t length,
                                    const BeamResidueMask *mask) {
    const BeamAlphabetInfo *a = beam_alphabet_info(BEAM_ALPHABET_AA);
    char buf[16];
    uint32_t idx[16];
    if (length >= sizeof(buf)) return -1.0;

    for (uint32_t i = 0; i < length; ++i) idx[i] = 0u;
    buf[length] = '\0';

    double kept = 0.0, total = 0.0;
    for (;;) {
        bool admitted = true;
        for (uint32_t i = 0; i < length; ++i) {
            buf[i] = a->idx_to_char[idx[i]];
            if (mask && !((mask[i] >> idx[i]) & 1u)) admitted = false;
        }
        const double p = exp(beam_model_log_pseq(m, buf));
        total += p;
        if (admitted) kept += p;

        uint32_t k = length;
        while (k-- > 0u) {
            if (++idx[k] < R) break;
            idx[k] = 0u;
            if (k == 0u) goto done;
        }
    }
done:
    return total > 0.0 ? kept / total : -1.0;
}

/* ── The constraint layer, with no model in sight ────────────── */

static void test_a_new_constraint_admits_every_residue(void) {
    BeamConstraint *c = beam_query_constraint_new(6u, R);
    ASSERT_MSG(c != NULL, "constructed");
    ASSERT_MSG(beam_query_constraint_length(c) == 6u, "length recorded");
    ASSERT_MSG(beam_query_constraint_residues(c) == R, "alphabet recorded");
    ASSERT_MSG(beam_query_constraint_is_free(c), "nothing narrowed yet");
    for (uint32_t i = 0; i < 6u; ++i)
        ASSERT_MSG(beam_query_constraint_mask(c, i) == BEAM_RESIDUES_ALL(R),
                   "every residue permitted");
    ASSERT_MSG(beam_query_empty_position(c) == 6u, "no empty position");
    beam_query_constraint_free(c);
    PASS();
}

static void test_builders_intersect_rather_than_replace(void) {
    BeamConstraint *c = beam_query_constraint_new(4u, R);
    ASSERT_MSG(beam_query_constrain_set(c, 1u, 0x0Fu) == BEAM_OK, "set ok");
    ASSERT_MSG(beam_query_constrain_set(c, 1u, 0x33u) == BEAM_OK, "set ok");
    /* 0x0F & 0x33 = 0x03. A replacing builder would leave 0x33. */
    ASSERT_MSG(beam_query_constraint_mask(c, 1u) == 0x03u,
               "the second set intersected with the first");
    ASSERT_MSG(!beam_query_constraint_is_free(c), "narrowing is visible");
    beam_query_constraint_free(c);
    PASS();
}

static void test_builders_commute(void) {
    BeamConstraint *a = beam_query_constraint_new(5u, R);
    BeamConstraint *b = beam_query_constraint_new(5u, R);
    beam_query_constrain_set(a, 2u, 0x1Fu);
    beam_query_constrain_exclude(a, 2u, 1u);
    beam_query_constrain_exclude(b, 2u, 1u);
    beam_query_constrain_set(b, 2u, 0x1Fu);
    for (uint32_t i = 0; i < 5u; ++i)
        ASSERT_MSG(beam_query_constraint_mask(a, i)
                   == beam_query_constraint_mask(b, i),
                   "order of application does not matter");
    beam_query_constraint_free(a);
    beam_query_constraint_free(b);
    PASS();
}

static void test_contradictory_builders_leave_an_empty_position(void) {
    BeamConstraint *c = beam_query_constraint_new(4u, R);
    ASSERT_MSG(beam_query_constrain_position(c, 2u, 3u) == BEAM_OK, "pin");
    ASSERT_MSG(beam_query_constrain_position(c, 2u, 5u) == BEAM_OK,
               "a contradiction is buildable, not an error to build");
    ASSERT_MSG(beam_query_constraint_mask(c, 2u) == BEAM_RESIDUES_NONE,
               "the intersection is empty");
    ASSERT_MSG(beam_query_empty_position(c) == 2u,
               "and the empty position is named");
    beam_query_constraint_free(c);
    PASS();
}

static void test_a_refused_run_leaves_the_constraint_untouched(void) {
    BeamConstraint *c = beam_query_constraint_new(5u, R);
    /* The last element is out of range, so the whole run must be refused --
     * and positions 0 and 1 must not have been narrowed on the way past. */
    const uint8_t bad[3] = {1u, 2u, (uint8_t)R};
    ASSERT_MSG(beam_query_constrain_run(c, 0u, bad, 3u)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "refused");
    ASSERT_MSG(beam_query_constraint_is_free(c),
               "a refused run is not a partly applied one");
    beam_query_constraint_free(c);
    PASS();
}

static void test_a_run_applies_as_a_prefix_and_as_a_suffix(void) {
    BeamConstraint *c = beam_query_constraint_new(6u, R);
    const uint8_t pre[2] = {1u, 0u};
    const uint8_t suf[2] = {4u, 5u};
    ASSERT_MSG(beam_query_constrain_run(c, 0u, pre, 2u) == BEAM_OK, "prefix");
    ASSERT_MSG(beam_query_constrain_run(c, 4u, suf, 2u) == BEAM_OK, "suffix");
    ASSERT_MSG(beam_query_constraint_mask(c, 0u) == (1u << 1), "pos 0 pinned");
    ASSERT_MSG(beam_query_constraint_mask(c, 5u) == (1u << 5), "pos 5 pinned");
    ASSERT_MSG(beam_query_constraint_mask(c, 2u) == BEAM_RESIDUES_ALL(R),
               "the middle is untouched");
    beam_query_constraint_free(c);
    PASS();
}

static void test_a_run_near_the_integer_ceiling_does_not_wrap(void) {
    BeamConstraint *c = beam_query_constraint_new(4u, R);
    const uint8_t one[1] = {0u};
    /* start + n overflows 32 bits if it is added narrow, and the wrapped sum
     * is small enough to look like a legal range. */
    ASSERT_MSG(beam_query_constrain_run(c, 0xFFFFFFFFu, one, 2u)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "refused, not wrapped");
    ASSERT_MSG(beam_query_constraint_is_free(c), "and nothing was written");
    beam_query_constraint_free(c);
    PASS();
}

static void test_a_position_past_the_end_is_refused_by_name(void) {
    BeamConstraint *c = beam_query_constraint_new(4u, R);
    ASSERT_MSG(beam_query_constrain_position(c, 4u, 0u)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "refused");
    ASSERT_MSG(strstr(beam_error_message(), "position 4") != NULL,
               "the message names the position");
    beam_query_constraint_free(c);
    PASS();
}

static void test_a_residue_outside_the_alphabet_is_refused_by_name(void) {
    BeamConstraint *c = beam_query_constraint_new(4u, R);
    ASSERT_MSG(beam_query_constrain_position(c, 0u, R)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "refused");
    ASSERT_MSG(strstr(beam_error_message(), "residue index 20") != NULL,
               "the message names the residue");
    ASSERT_MSG(beam_query_constrain_set(c, 0u, 1u << R)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "a set out of range too");
    beam_query_constraint_free(c);
    PASS();
}

static void test_a_bad_shape_is_refused_at_construction(void) {
    ASSERT_MSG(beam_query_constraint_new(0u, R) == NULL, "length 0 refused");
    ASSERT_MSG(beam_query_constraint_new(4u, 0u) == NULL, "0 residues refused");
    ASSERT_MSG(beam_query_constraint_new(4u, BEAM_MAX_RESIDUES + 1u) == NULL,
               "more residues than a mask can address is refused");
    PASS();
}

/* ── Bit-for-bit reproduction of the clamped sweep ───────────── */

static void test_an_unconstrained_query_reproduces_the_marginals(void) {
    /*
     * Bit for bit. Both engines carry their forward values as logarithms now,
     * accumulated by the same streaming log-sum-exp in the same order, so
     * there is no rounding for them to differ over.
     */
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;

    double *want = (double *)malloc((size_t)L * R * sizeof(double));
    double *got  = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(want && got, "alloc");

    double zw = 0.0, zg = 0.0;
    ASSERT_MSG(beam_model_positional_marginals(m, L, NULL, want, &zw)
               == BEAM_OK, "reference sweep");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_marginals(m, c, got, &zg) == BEAM_OK, "query sweep");

    ASSERT_MSG(zw == zg, "the two agree on ln Z exactly");
    for (uint32_t k = 0; k < L * R; ++k)
        ASSERT_MSG(want[k] == got[k], "and on every cell, to the last bit");
    for (uint32_t i = 0; i < L; ++i) {
        double s = 0.0;
        for (uint32_t x = 0; x < R; ++x) s += got[(size_t)i * R + x];
        ASSERT_MSG(fabs(s - 1.0) < 1e-12, "every row is a distribution");
    }

    beam_query_constraint_free(c);
    free(want); free(got);
    beam_model_destroy(m);
    PASS();
}

/**
 * Every pin, compared cell by cell against the clamp that denotes the same
 * thing, and each disagreement classified.
 *
 * `*n_live` counts pins compared; `*n_old_broken` pins where they disagree
 * because `positional_marginals` returned a row that does not sum to one; and
 * `*n_bad` pins where they disagree MATERIALLY while both answers are still
 * valid distributions. Only the last is a defect.
 *
 * Differences of a few times 1e-13 with both rows summing to one are neither:
 * the two engines choose their per-layer rescale differently, so the last bits
 * move everywhere. That is the price of the fix below and it is accounted for
 * rather than hidden -- `n_bad` still fires on any disagreement large enough
 * to be a wrong answer.
 *
 * That third case is real and was found while fixing this engine. Under a
 * clamp, `beam_model_positional_marginals` rescales each layer by the whole
 * layer's maximum, which can be attained by an edge no surviving path uses;
 * every path that IS admitted then underflows. On an open-world model at
 * length 10, pinning position 3 leaves row 7 summing to 5.9e-264 instead of
 * 1. This engine takes its rescale over edges a live path can actually use
 * and returns 1, which an independent log-space route -- exp(ln P(both) -
 * ln P(pin)), where nothing can underflow -- confirms is right.
 */
static void compare_every_pin(BeamModel *m, uint32_t L, uint32_t *n_live,
                              uint32_t *n_bad, uint32_t *n_old_broken,
                              uint32_t *n_identical) {
    double *want = (double *)malloc((size_t)L * R * sizeof(double));
    double *got  = (double *)malloc((size_t)L * R * sizeof(double));
    uint8_t *clamp = (uint8_t *)malloc(L);
    *n_live = 0u; *n_bad = 0u; *n_old_broken = 0u; *n_identical = 0u;
    if (!want || !got || !clamp) { *n_bad = UINT32_MAX; return; }

    /*
     * Every pin is attempted, including those whose unconstrained marginal is
     * exactly 0.0. In an open-world model that zero is an underflow, not an
     * absence -- log_pseq("AAAAAAAAAA") is about -6908 -- so those pins
     * condition on something 6,900 nats below the mode, which is precisely
     * where two implementations of one recursion part company.
     */
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t a = 0; a < R; ++a) {
            memset(clamp, BEAM_RESIDUE_INVALID, L);
            clamp[i] = (uint8_t)a;
            double zw = 0.0, zg = 0.0;
            const BeamError ref =
                beam_model_positional_marginals(m, L, clamp, want, &zw);
            BeamConstraint *c = beam_query_constraint_new(L, R);
            beam_query_constrain_position(c, i, a);
            const BeamError qry = beam_query_marginals(m, c, got, &zg);
            beam_query_constraint_free(c);

            /* They must agree about whether the question has an answer,
             * before they are asked to agree about what it is. */
            if (ref != qry) { (*n_bad)++; continue; }
            if (ref != BEAM_OK) continue;
            (*n_live)++;

            bool same = (zw == zg);
            for (uint32_t k = 0; same && k < L * R; ++k)
                if (want[k] != got[k]) same = false;
            if (same) { (*n_identical)++; continue; }

            /* Disagreement. Whose rows are still distributions, and by how
             * much do the cells actually differ? */
            bool old_ok = true, new_ok = true;
            double worst = 0.0;
            for (uint32_t r = 0; r < L; ++r) {
                double sw = 0.0, sg = 0.0;
                for (uint32_t k = 0; k < R; ++k) {
                    sw += want[(size_t)r * R + k];
                    sg += got[(size_t)r * R + k];
                    const double d = fabs(want[(size_t)r * R + k]
                                          - got[(size_t)r * R + k]);
                    if (d > worst) worst = d;
                }
                if (fabs(sw - 1.0) > 1e-9) old_ok = false;
                if (fabs(sg - 1.0) > 1e-9) new_ok = false;
            }
            if (!old_ok && new_ok) { (*n_old_broken)++; continue; }
            if (old_ok && new_ok && worst < 1e-9) continue;   /* rounding */
            (*n_bad)++;
        }
    }
    free(want); free(got); free(clamp);
}

static void test_a_pinned_position_reproduces_a_clamp(void) {
    /*
     * Bit for bit, on every pin, in both worlds. The two are separate
     * implementations of one recursion -- this engine's over a residue SET and
     * `positional_marginals`' over a residue RANGE -- and both now carry their
     * forward values as logarithms accumulated in the same order, so agreement
     * to the last bit is the right expectation and anything less is a defect.
     *
     * It was not always so. While both rescaled each layer by a single factor
     * the claim held on amino acids and failed on nucleotides, and while only
     * one of them had been fixed it failed on 160 open-world cells -- every
     * one of them `positional_marginals` returning a row that summed to
     * 5.9e-264 instead of 1. The classification below survives from that
     * period deliberately: if the two ever diverge again it says WHICH is
     * wrong, rather than only that they differ.
     */
    uint32_t live = 0u, bad = 0u, old_broken = 0u, identical = 0u;

    BeamModel *shut = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(shut != NULL, "closed-world model built");
    compare_every_pin(shut, 10u, &live, &bad, &old_broken, &identical);
    beam_model_destroy(shut);
    ASSERT_MSG(bad == 0u, "no closed-world pin disagreed with its clamp");
    ASSERT_MSG(old_broken == 0u, "and neither engine lost a distribution");
    ASSERT_MSG(identical == live, "every closed-world pin is bit-identical");
    ASSERT_MSG(live == 17u, "the 17 the corpus spells were all compared");

    BeamModel *open = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, true);
    ASSERT_MSG(open != NULL, "open-world model built");
    compare_every_pin(open, 10u, &live, &bad, &old_broken, &identical);
    beam_model_destroy(open);
    ASSERT_MSG(bad == 0u, "no open-world pin disagreed either");
    ASSERT_MSG(old_broken == 0u,
               "and the underflow that used to lose six rows here is gone");
    ASSERT_MSG(identical == live,
               "every open-world pin is bit-identical too, across a trellis "
               "spanning thousands of nats");
    ASSERT_MSG(live == 10u * R,
               "the open world carries a path for all 200 pins");
    PASS();
}

static void test_a_pinned_probability_reproduces_log_clamp_probability(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    uint8_t *clamp = (uint8_t *)malloc(L);
    double *sink = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(clamp && sink, "alloc");

    for (uint32_t a = 0; a < R; ++a) {
        memset(clamp, BEAM_RESIDUE_INVALID, L);
        clamp[3] = (uint8_t)a;
        double zc = 0.0;
        if (beam_model_positional_marginals(m, L, clamp, sink, &zc) != BEAM_OK)
            continue;                                    /* zero-mass pin */
        const double want = zc - beam_model_log_partition(m, L);

        BeamConstraint *c = beam_query_constraint_new(L, R);
        beam_query_constrain_position(c, 3u, a);
        double got = 0.0;
        ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
        beam_query_constraint_free(c);
        ASSERT_MSG(want == got, "log P identical to the clamped route");
    }
    free(clamp); free(sink);
    beam_model_destroy(m);
    PASS();
}

/* ── Enumeration ─────────────────────────────────────────────── */

static void test_a_constrained_probability_matches_enumeration(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    /* A pin, a set, and a two-position conjunction: three shapes of mask. */
    struct { uint32_t pos; BeamResidueMask set; } cases[] = {
        {1u, (BeamResidueMask)1u << 0},
        {1u, 0x0Du},
        {2u, 0x2Au},
    };
    for (uint32_t t = 0; t < 3u; ++t) {
        BeamConstraint *c = beam_query_constraint_new(L, R);
        ASSERT_MSG(beam_query_constrain_set(c, cases[t].pos, cases[t].set)
                   == BEAM_OK, "set");
        double got = 0.0;
        ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
        const double want = enumerate_probability(m, L, c->mask);
        beam_query_constraint_free(c);
        ASSERT_MSG(want > 0.0, "the reference found mass");
        ASSERT_MSG(fabs(exp(got) - want) < 1e-12,
                   "constrained probability matches the enumerated sum");
    }

    /* Two positions at once, which no single-position shortcut can fake. */
    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_set(c, 1u, 0x0Du);
    beam_query_constrain_set(c, 3u, 0x30u);
    double got = 0.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
    const double want = enumerate_probability(m, L, c->mask);
    beam_query_constraint_free(c);
    ASSERT_MSG(want > 0.0, "the reference found mass");
    ASSERT_MSG(fabs(exp(got) - want) < 1e-12,
               "a two-position conjunction matches enumeration");

    beam_model_destroy(m);
    PASS();
}

/* ── The identity that ties sets to singletons ───────────────── */

static void test_a_set_conditions_as_the_mixture_of_its_members(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u, PIN = 4u;

    double *base = (double *)malloc((size_t)L * R * sizeof(double));
    double *one  = (double *)malloc((size_t)L * R * sizeof(double));
    double *mix  = (double *)calloc((size_t)L * R, sizeof(double));
    double *got  = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(base && one && mix && got, "alloc");
    ASSERT_MSG(beam_model_positional_marginals(m, L, NULL, base, NULL)
               == BEAM_OK, "unconstrained marginals");

    /* The first three residues that carry mass at PIN. */
    BeamResidueMask set = 0u;
    double weight_total = 0.0;
    uint32_t members = 0u;
    for (uint32_t a = 0; a < R && members < 3u; ++a) {
        const double pa = base[(size_t)PIN * R + a];
        if (pa <= 0.0) continue;
        set |= (BeamResidueMask)1u << a;
        weight_total += pa;
        members++;

        BeamConstraint *c = beam_query_constraint_new(L, R);
        beam_query_constrain_position(c, PIN, a);
        ASSERT_MSG(beam_query_marginals(m, c, one, NULL) == BEAM_OK, "member");
        beam_query_constraint_free(c);
        for (uint32_t k = 0; k < L * R; ++k) mix[k] += pa * one[k];
    }
    ASSERT_MSG(members == 3u, "found three live residues to mix");

    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_set(c, PIN, set) == BEAM_OK, "set");
    ASSERT_MSG(beam_query_marginals(m, c, got, NULL) == BEAM_OK, "set query");
    beam_query_constraint_free(c);

    for (uint32_t k = 0; k < L * R; ++k)
        ASSERT_MSG(fabs(got[k] - mix[k] / weight_total) < 1e-12,
                   "conditioning on a set is the marginal-weighted mixture "
                   "of conditioning on its members");

    free(base); free(one); free(mix); free(got);
    beam_model_destroy(m);
    PASS();
}

static void test_a_set_probability_is_the_sum_over_its_members(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    /* Position 4, which spells five residues; a position with one would make
     * the sum trivially 1.0 and the test vacuous. */
    const uint32_t L = 10u, PIN = 4u;

    double *base = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(base != NULL, "alloc");
    ASSERT_MSG(beam_model_positional_marginals(m, L, NULL, base, NULL)
               == BEAM_OK, "marginals");

    /* L and P, two of the five, so the answer is a proper fraction. */
    const BeamResidueMask set = (1u << AA_L) | (1u << AA_P);
    double want = base[(size_t)PIN * R + AA_L] + base[(size_t)PIN * R + AA_P];
    ASSERT_MSG(want > 0.05 && want < 0.95, "a non-trivial fraction");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_set(c, PIN, set);
    double got = 0.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
    beam_query_constraint_free(c);
    ASSERT_MSG(fabs(exp(got) - want) < 1e-12,
               "P(x in S) is the sum of the marginals over S");

    free(base);
    beam_model_destroy(m);
    PASS();
}

static void test_marginal_rows_sum_to_one_under_a_constraint(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *out = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(out != NULL, "alloc");

    /* Position 4 carries D L P Q V and position 8 carries F Y, so this asks
     * for something the corpus spells; a set drawn from residues the layer
     * cannot emit would be refused, and rightly. */
    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_set(c, 4u, (1u << AA_L) | (1u << AA_Q) | (1u << AA_V));
    beam_query_constrain_exclude(c, 8u, AA_F);
    ASSERT_MSG(beam_query_marginals(m, c, out, NULL) == BEAM_OK, "query");
    beam_query_constraint_free(c);

    for (uint32_t i = 0; i < L; ++i) {
        double s = 0.0;
        for (uint32_t a = 0; a < R; ++a) s += out[(size_t)i * R + a];
        ASSERT_MSG(fabs(s - 1.0) < 1e-12, "every row is a distribution");
    }
    free(out);
    beam_model_destroy(m);
    PASS();
}

static void test_the_probability_of_everything_is_exactly_one(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    BeamConstraint *c = beam_query_constraint_new(10u, R);
    double got = 1.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
    /* Exactly, not nearly: an unconstrained probability that came back as
     * 0.9999999999999998 would mean the two partition functions were reached
     * by different routes. */
    ASSERT_MSG(got == 0.0, "ln P(everything) is 0.0 exactly");
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

/* ── Impossible constraints ──────────────────────────────────── */

static void test_an_impossible_constraint_has_probability_zero(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    BeamConstraint *c = beam_query_constraint_new(10u, R);
    beam_query_constrain_position(c, 2u, 1u);
    beam_query_constrain_position(c, 2u, 4u);       /* contradiction */
    double got = 0.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK,
               "asking the probability of the impossible is not an error");
    ASSERT_MSG(got == -INFINITY, "and the answer is zero");
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

static void test_conditioning_on_the_impossible_is_refused(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *out = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(out != NULL, "alloc");

    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_position(c, 2u, 1u);
    beam_query_constrain_position(c, 2u, 4u);
    ASSERT_MSG(beam_query_marginals(m, c, out, NULL)
               == BEAM_ERR_NO_LIVE_PATHS,
               "a conditional on an impossible event does not exist");
    ASSERT_MSG(strstr(beam_error_message(), "no residue") != NULL,
               "and the message says why");
    beam_query_constraint_free(c);

    /* Satisfiable as a mask, unsatisfiable in this trellis: a closed-world
     * model does not spell every letter at every position. */
    BeamConstraint *d = beam_query_constraint_new(L, R);
    beam_query_constrain_position(d, 0u, 0u);
    beam_query_constrain_position(d, 1u, 0u);
    beam_query_constrain_position(d, 2u, 0u);
    beam_query_constrain_position(d, 3u, 0u);
    const BeamError e = beam_query_marginals(m, d, out, NULL);
    ASSERT_MSG(e == BEAM_ERR_NO_LIVE_PATHS,
               "a mask the trellis cannot satisfy is refused too");
    beam_query_constraint_free(d);

    free(out);
    beam_model_destroy(m);
    PASS();
}

/* ── Shape and alphabet mismatches ───────────────────────────── */

static void test_a_constraint_over_the_wrong_alphabet_is_refused(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    double out[10 * R];
    BeamConstraint *c = beam_query_constraint_new(10u, BEAM_NT_RESIDUES);
    ASSERT_MSG(beam_query_marginals(m, c, out, NULL)
               == BEAM_ERR_PARAM_OUT_OF_RANGE,
               "residue indices would mean different letters");
    ASSERT_MSG(strstr(beam_error_message(), "different letters") != NULL,
               "and the message says so");
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

static void test_a_length_the_model_never_generates_is_refused(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    double out[3 * R];
    BeamConstraint *c = beam_query_constraint_new(3u, R);
    ASSERT_MSG(beam_query_marginals(m, c, out, NULL)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "refused");
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

/* ── The product engine ──────────────────────────────────────── */

/** True when `motif` occurs anywhere in `s`. Independent of the automaton. */
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

/** `P(C | length)` by enumeration, with the predicate supplied by the caller. */
static double enumerate_predicate(const BeamModel *m, uint32_t length,
                                  bool (*keep)(const uint8_t *, uint32_t,
                                               const void *),
                                  const void *ctx) {
    const BeamAlphabetInfo *a = beam_alphabet_info(BEAM_ALPHABET_AA);
    char buf[16];
    uint8_t idx[16];
    if (length >= sizeof(buf)) return -1.0;
    for (uint32_t i = 0; i < length; ++i) idx[i] = 0u;
    buf[length] = '\0';

    double kept = 0.0, total = 0.0;
    for (;;) {
        for (uint32_t i = 0; i < length; ++i) buf[i] = a->idx_to_char[idx[i]];
        const double p = exp(beam_model_log_pseq(m, buf));
        total += p;
        if (keep(idx, length, ctx)) kept += p;

        uint32_t k = length;
        while (k-- > 0u) {
            if (++idx[k] < R) break;
            idx[k] = 0u;
            if (k == 0u) goto done;
        }
    }
done:
    return total > 0.0 ? kept / total : -1.0;
}

struct motif_ctx { const uint8_t *motif; uint32_t n; };
static bool keep_contains(const uint8_t *s, uint32_t n, const void *ctx) {
    const struct motif_ctx *c = (const struct motif_ctx *)ctx;
    return naive_contains(s, n, c->motif, c->n);
}

struct count_ctx { BeamResidueMask of; uint32_t lo, hi; };
static bool keep_count(const uint8_t *s, uint32_t n, const void *ctx) {
    const struct count_ctx *c = (const struct count_ctx *)ctx;
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; ++i) if ((c->of >> s[i]) & 1u) ++k;
    return k >= c->lo && k <= c->hi;
}

/**
 * An automaton that accepts everything, with one state the run never enters.
 *
 * Counting members of the empty set gives zero everywhere, and zero is in
 * range, so the language is universal -- but `|Q|` is 2, so the product
 * machinery is genuinely exercised rather than degenerating to the mask sweep.
 */
static BeamAutomaton *accept_all(uint32_t length) {
    return beam_query_automaton_count(length, R, 0u, 0u, 0u);
}

static void test_a_universal_automaton_reproduces_the_mask_sweep(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *want = (double *)malloc((size_t)L * R * sizeof(double));
    double *got  = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(want && got, "alloc");

    /*
     * Two independent implementations of one recursion, required to agree to
     * the last bit. The product path carries an automaton dimension the mask
     * path has never heard of, and at a universal automaton every term of the
     * sum but one is zero -- so if the two ever diverge, the product path has
     * started doing different arithmetic rather than more of it.
     */
    for (uint32_t trial = 0; trial < 2u; ++trial) {
        BeamConstraint *plain = beam_query_constraint_new(L, R);
        BeamConstraint *prod  = beam_query_constraint_new(L, R);
        if (trial == 1u) {           /* and again with the masks narrowed */
            beam_query_constrain_set(plain, 4u,
                                     (1u << AA_L) | (1u << AA_Q));
            beam_query_constrain_set(prod, 4u,
                                     (1u << AA_L) | (1u << AA_Q));
        }
        ASSERT_MSG(beam_query_constrain_automaton(prod, accept_all(L))
                   == BEAM_OK, "attached");

        double zw = 0.0, zg = 0.0;
        ASSERT_MSG(beam_query_marginals(m, plain, want, &zw) == BEAM_OK,
                   "mask sweep");
        ASSERT_MSG(beam_query_marginals(m, prod, got, &zg) == BEAM_OK,
                   "product sweep");
        ASSERT_MSG(zw == zg, "log Z identical across the two engines");
        for (uint32_t k = 0; k < L * R; ++k)
            ASSERT_MSG(want[k] == got[k], "every cell identical, not close");

        double pw = 0.0, pg = 0.0;
        ASSERT_MSG(beam_query_log_probability(m, plain, &pw) == BEAM_OK, "p");
        ASSERT_MSG(beam_query_log_probability(m, prod, &pg) == BEAM_OK, "p");
        if (trial == 0u) {
            /*
             * Not bit-for-bit here, and that is right. An unnarrowed
             * constraint short-circuits to `ln 1 = 0.0` exactly, but a
             * UNIVERSAL AUTOMATON is not recognised as unnarrowed -- doing so
             * means deciding whether a DFA accepts everything, which is
             * decidable and pointless. So the product path takes the sweep and
             * lands within the difference between a forward and a backward
             * evaluation of the same partition function.
             */
            ASSERT_MSG(pw == 0.0, "the free constraint still short-circuits");
            ASSERT_MSG(fabs(pg) < 1e-12,
                       "and the sweep route agrees to floating point");
        } else {
            /* Both took the sweep, from log Z values already shown identical,
             * so here anything but equality would be a defect. */
            ASSERT_MSG(pw == pg, "two sweep routes agree to the last bit");
        }

        beam_query_constraint_free(plain);
        beam_query_constraint_free(prod);
    }
    free(want); free(got);
    beam_model_destroy(m);
    PASS();
}

static void test_a_motif_probability_matches_enumeration(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    const uint8_t motifs[3][2] = {{AA_A_, AA_S_}, {AA_S_, AA_F}, {AA_C, AA_A_}};
    const uint32_t lens[3] = {2u, 2u, 2u};
    for (uint32_t t = 0; t < 3u; ++t) {
        BeamConstraint *c = beam_query_constraint_new(L, R);
        ASSERT_MSG(beam_query_constrain_automaton(
                       c, beam_query_automaton_motif(L, R, motifs[t], lens[t]))
                   == BEAM_OK, "attached");
        double got = 0.0;
        ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
        beam_query_constraint_free(c);

        struct motif_ctx ctx = {motifs[t], lens[t]};
        const double want = enumerate_predicate(m, L, keep_contains, &ctx);
        ASSERT_MSG(want > 0.0 && want < 1.0,
                   "the reference found a non-trivial fraction");
        ASSERT_MSG(fabs(exp(got) - want) < 1e-12,
                   "P(contains motif) matches the enumerated sum");
    }
    beam_model_destroy(m);
    PASS();
}

static void test_a_count_probability_matches_enumeration(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    struct count_ctx cases[3] = {
        {(1u << AA_S_), 1u, 1u},
        {(1u << AA_S_) | (1u << AA_A_), 2u, 3u},
        {(1u << AA_F), 0u, 0u},
    };
    for (uint32_t t = 0; t < 3u; ++t) {
        BeamConstraint *c = beam_query_constraint_new(L, R);
        ASSERT_MSG(beam_query_constrain_automaton(
                       c, beam_query_automaton_count(L, R, cases[t].of,
                                                     cases[t].lo, cases[t].hi))
                   == BEAM_OK, "attached");
        double got = 0.0;
        ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
        beam_query_constraint_free(c);

        const double want = enumerate_predicate(m, L, keep_count, &cases[t]);
        ASSERT_MSG(want > 0.0 && want < 1.0, "non-trivial");
        ASSERT_MSG(fabs(exp(got) - want) < 1e-12,
                   "P(count in range) matches the enumerated sum");
    }
    beam_model_destroy(m);
    PASS();
}

struct both_ctx { struct motif_ctx mo; BeamResidueMask pin_set; uint32_t pin; };
static bool keep_both(const uint8_t *s, uint32_t n, const void *ctx) {
    const struct both_ctx *c = (const struct both_ctx *)ctx;
    if (!((c->pin_set >> s[c->pin]) & 1u)) return false;
    return naive_contains(s, n, c->mo.motif, c->mo.n);
}

static void test_masks_and_an_automaton_compose(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    const uint8_t motif[2] = {AA_A_, AA_S_};

    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_set(c, 0u, (1u << AA_C) | (1u << AA_V))
               == BEAM_OK, "mask");
    ASSERT_MSG(beam_query_constrain_automaton(
                   c, beam_query_automaton_motif(L, R, motif, 2u)) == BEAM_OK,
               "automaton");
    double got = 0.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");
    beam_query_constraint_free(c);

    struct both_ctx ctx = {{motif, 2u}, (1u << AA_C) | (1u << AA_V), 0u};
    const double want = enumerate_predicate(m, L, keep_both, &ctx);
    ASSERT_MSG(want > 0.0 && want < 1.0, "non-trivial");
    ASSERT_MSG(fabs(exp(got) - want) < 1e-12,
               "the two tiers intersect rather than one overriding the other");
    beam_model_destroy(m);
    PASS();
}

static void test_a_language_and_its_complement_partition_the_mass(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    const uint8_t motif[2] = {AA_L, AA_G};

    BeamConstraint *a = beam_query_constraint_new(L, R);
    BeamConstraint *b = beam_query_constraint_new(L, R);
    BeamAutomaton *m1 = beam_query_automaton_motif(L, R, motif, 2u);
    BeamAutomaton *m2 = beam_query_automaton_motif(L, R, motif, 2u);
    ASSERT_MSG(m1 && m2, "built");
    ASSERT_MSG(beam_query_constrain_automaton(a, m1) == BEAM_OK, "a");
    ASSERT_MSG(beam_query_constrain_automaton(b, beam_query_automaton_not(m2))
               == BEAM_OK, "not a");
    beam_query_automaton_free(m2);

    double pa = 0.0, pb = 0.0;
    ASSERT_MSG(beam_query_log_probability(m, a, &pa) == BEAM_OK, "P(A)");
    ASSERT_MSG(beam_query_log_probability(m, b, &pb) == BEAM_OK, "P(not A)");
    ASSERT_MSG(exp(pa) > 0.01 && exp(pa) < 0.99, "a non-trivial split");
    ASSERT_MSG(fabs(exp(pa) + exp(pb) - 1.0) < 1e-12,
               "P(A) + P(not A) is 1; a product graph that lost or "
               "double-counted a path could not manage this");
    beam_query_constraint_free(a);
    beam_query_constraint_free(b);
    beam_model_destroy(m);
    PASS();
}

static double prob_of(BeamModel *m, uint32_t L, BeamAutomaton *a) {
    BeamConstraint *c = beam_query_constraint_new(L, R);
    if (beam_query_constrain_automaton(c, a) != BEAM_OK) {
        beam_query_constraint_free(c);
        return -1.0;
    }
    double p = 0.0;
    const BeamError e = beam_query_log_probability(m, c, &p);
    beam_query_constraint_free(c);
    return e == BEAM_OK ? exp(p) : -1.0;
}

static void test_union_obeys_inclusion_exclusion(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    const uint8_t u[2] = {AA_L, AA_G};
    const uint8_t v[2] = {AA_E, AA_Q};

    BeamAutomaton *a = beam_query_automaton_motif(L, R, u, 2u);
    BeamAutomaton *b = beam_query_automaton_motif(L, R, v, 2u);
    ASSERT_MSG(a && b, "built");
    BeamAutomaton *both   = beam_query_automaton_and(a, b);
    BeamAutomaton *either = beam_query_automaton_or(a, b);
    ASSERT_MSG(both && either, "combined");

    const double pa = prob_of(m, L, a);
    const double pb = prob_of(m, L, b);
    const double pab = prob_of(m, L, both);
    const double pab_or = prob_of(m, L, either);
    ASSERT_MSG(pa > 0.0 && pb > 0.0 && pab >= 0.0 && pab_or > 0.0, "all ran");
    ASSERT_MSG(pab_or > pa && pab_or < 1.0 + 1e-12, "a real union");
    ASSERT_MSG(fabs(pab_or - (pa + pb - pab)) < 1e-12,
               "P(A or B) = P(A) + P(B) - P(A and B), which a product "
               "construction that mixed up its accepting sets cannot satisfy");
    beam_model_destroy(m);
    PASS();
}

static void test_a_motif_the_model_cannot_spell_is_impossible(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *out = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(out != NULL, "alloc");
    /* The closed-world corpus spells no run of four tryptophans. */
    const uint8_t motif[4] = {AA_W, AA_W, AA_W, AA_W};

    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_automaton(c,
        beam_query_automaton_motif(L, R, motif, 4u));
    double p = 1.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &p) == BEAM_OK,
               "asking is not an error");
    ASSERT_MSG(p == -INFINITY, "and the answer is zero");
    ASSERT_MSG(beam_query_marginals(m, c, out, NULL) == BEAM_ERR_NO_LIVE_PATHS,
               "but conditioning on it has no answer");
    beam_query_constraint_free(c);
    free(out);
    beam_model_destroy(m);
    PASS();
}

static void test_a_motif_conditional_matches_enumeration(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    const uint8_t motif[2] = {AA_A_, AA_S_};

    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_automaton(
                   c, beam_query_automaton_motif(L, R, motif, 2u)) == BEAM_OK,
               "attached");
    double *got = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(got != NULL, "alloc");
    ASSERT_MSG(beam_query_marginals(m, c, got, NULL) == BEAM_OK, "query");
    beam_query_constraint_free(c);

    /* The same conditional built by walking the support directly. */
    const BeamAlphabetInfo *ai = beam_alphabet_info(BEAM_ALPHABET_AA);
    double *want = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(want != NULL, "alloc");
    char buf[8];
    uint8_t idx[8];
    for (uint32_t i = 0; i < L; ++i) idx[i] = 0u;
    buf[L] = '\0';
    double mass = 0.0;
    for (;;) {
        for (uint32_t i = 0; i < L; ++i) buf[i] = ai->idx_to_char[idx[i]];
        if (naive_contains(idx, L, motif, 2u)) {
            const double p = exp(beam_model_log_pseq(m, buf));
            mass += p;
            for (uint32_t i = 0; i < L; ++i) want[(size_t)i * R + idx[i]] += p;
        }
        uint32_t k = L;
        while (k-- > 0u) {
            if (++idx[k] < R) break;
            idx[k] = 0u;
            if (k == 0u) goto done;
        }
    }
done:
    ASSERT_MSG(mass > 0.0, "the reference found mass");
    for (uint32_t k = 0; k < L * R; ++k)
        ASSERT_MSG(fabs(got[k] - want[k] / mass) < 1e-12,
                   "every conditional cell matches the enumerated one");
    free(got); free(want);
    beam_model_destroy(m);
    PASS();
}

struct or_ctx { uint32_t pa, pb; BeamResidueMask sa, sb; };
static bool keep_either(const uint8_t *s, uint32_t n, const void *ctx) {
    const struct or_ctx *c = (const struct or_ctx *)ctx;
    (void)n;
    return ((c->sa >> s[c->pa]) & 1u) || ((c->sb >> s[c->pb]) & 1u);
}

static void test_a_layer_indexed_automaton_answers_a_disjunction(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, 6u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    /* Positions 1 and 3 both vary across the corpus. Position 0 does not --
     * it is C in every training sequence -- so a disjunction naming it would
     * be satisfied by almost all the mass and would test nothing. */
    const BeamResidueMask sa = (1u << AA_G) | (1u << AA_T);
    const BeamResidueMask sb = (1u << AA_Y) | (1u << AA_L);

    /*
     * "position 0 is C or V, OR position 2 is S or A" is not a per-position
     * restriction and no mask can express it: the union of two restrictions is
     * not a restriction. Lifting each to an automaton and taking their
     * disjunction gives a LAYER-INDEXED transition function, which is the
     * whole reason the automaton is layer-indexed rather than uniform.
     */
    BeamConstraint *ca = beam_query_constraint_new(L, R);
    BeamConstraint *cb = beam_query_constraint_new(L, R);
    beam_query_constrain_set(ca, 1u, sa);
    beam_query_constrain_set(cb, 3u, sb);
    BeamAutomaton *aa = beam_query_automaton_from_masks(ca);
    BeamAutomaton *ab = beam_query_automaton_from_masks(cb);
    ASSERT_MSG(aa && ab, "lifted");
    BeamAutomaton *either = beam_query_automaton_or(aa, ab);
    ASSERT_MSG(either != NULL, "disjoined");
    beam_query_automaton_free(aa);
    beam_query_automaton_free(ab);
    beam_query_constraint_free(ca);
    beam_query_constraint_free(cb);

    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_automaton(c, either) == BEAM_OK, "attached");
    double got = 0.0;
    ASSERT_MSG(beam_query_log_probability(m, c, &got) == BEAM_OK, "query");

    double *gm = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(gm != NULL, "alloc");
    ASSERT_MSG(beam_query_marginals(m, c, gm, NULL) == BEAM_OK, "marginals");
    beam_query_constraint_free(c);

    struct or_ctx ctx = {1u, 3u, sa, sb};
    const double want = enumerate_predicate(m, L, keep_either, &ctx);
    ASSERT_MSG(want > 0.05 && want < 0.95,
               "the disjunction selects a real fraction of the mass, so the "
               "comparison below can fail");
    ASSERT_MSG(fabs(exp(got) - want) < 1e-12,
               "a disjunction across positions matches enumeration");
    for (uint32_t i = 0; i < L; ++i) {
        double s = 0.0;
        for (uint32_t a = 0; a < R; ++a) s += gm[(size_t)i * R + a];
        ASSERT_MSG(fabs(s - 1.0) < 1e-12, "its conditional is a distribution");
    }
    free(gm);
    beam_model_destroy(m);
    PASS();
}

/** Two-pass log-sum-exp: find the maximum, then sum. The obvious way. */
static double reference_lse(const double *v, uint32_t n) {
    double mx = -INFINITY;
    for (uint32_t k = 0; k < n; ++k) if (v[k] > mx) mx = v[k];
    if (mx == -INFINITY) return -INFINITY;
    double s = 0.0;
    for (uint32_t k = 0; k < n; ++k)
        if (v[k] != -INFINITY) s += exp(v[k] - mx);
    return mx + log(s);
}

static void test_the_streaming_log_sum_exp_matches_a_two_pass_reference(void) {
    /*
     * `lse_add` is the numerical core of both forward sweeps: it accumulates a
     * target's sources in ONE pass, raising a running maximum and rescaling
     * the running sum when a larger term arrives. The two-pass form above is
     * written independently and is the thing it must agree with.
     *
     * The cases that matter are the ones a per-layer rescale used to fail:
     * terms spanning thousands of nats, and terms arriving smallest-first so
     * the running maximum is raised repeatedly. Exponentiating any of these
     * directly gives zero or infinity.
     */
    double v[64];
    for (uint32_t trial = 0; trial < 6u; ++trial) {
        uint32_t n = 0u;
        switch (trial) {
        case 0:  /* a single term */
            v[n++] = -1234.5; break;
        case 1:  /* ascending: the maximum is raised at every step */
            for (uint32_t k = 0; k < 40u; ++k) v[n++] = -3000.0 + 50.0 * k;
            break;
        case 2:  /* descending: the maximum is set once */
            for (uint32_t k = 0; k < 40u; ++k) v[n++] = -50.0 * k;
            break;
        case 3:  /* spanning far more than a double can hold at once */
            v[n++] = -4000.0; v[n++] = -1.0; v[n++] = -3999.0; v[n++] = -2.0;
            break;
        case 4:  /* equal terms: the answer is ln n above them */
            for (uint32_t k = 0; k < 16u; ++k) v[n++] = -700.0;
            break;
        default: /* -inf mixed in, which must contribute nothing */
            v[n++] = -INFINITY; v[n++] = -20.0; v[n++] = -INFINITY;
            v[n++] = -21.0; v[n++] = -INFINITY;
            break;
        }
        double mx = -INFINITY, sum = 0.0;
        for (uint32_t k = 0; k < n; ++k) lse_add(&mx, &sum, v[k]);
        const double got = (mx == -INFINITY) ? -INFINITY : mx + log(sum);
        const double want = reference_lse(v, n);
        ASSERT_MSG(fabs(got - want) < 1e-12 * (fabs(want) + 1.0),
                   "the streaming accumulator matches a two-pass log-sum-exp");
    }

    /* Sixteen equal terms sit exactly ln 16 above one of them, which pins the
     * rescaling rather than only the ordering. */
    double mx = -INFINITY, sum = 0.0;
    for (uint32_t k = 0; k < 16u; ++k) lse_add(&mx, &sum, -700.0);
    ASSERT_MSG(fabs((mx + log(sum)) - (-700.0 + log(16.0))) < 1e-12,
               "and sixteen equal terms land ln 16 above one of them");

    /* Nothing at all stays nothing. */
    mx = -INFINITY; sum = 0.0;
    lse_add(&mx, &sum, -INFINITY);
    ASSERT_MSG(mx == -INFINITY, "an impossible term adds nothing");
    PASS();
}

static void test_the_estimate_grows_with_the_automaton(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    const uint8_t motif[3] = {AA_L, AA_G, AA_E};

    BeamConstraint *plain = beam_query_constraint_new(L, R);
    BeamConstraint *prod  = beam_query_constraint_new(L, R);
    beam_query_constrain_automaton(prod,
        beam_query_automaton_motif(L, R, motif, 3u));

    BeamQueryCost a, b;
    ASSERT_MSG(beam_query_estimate(m, plain, &a) == BEAM_OK, "plain");
    ASSERT_MSG(beam_query_estimate(m, prod, &b) == BEAM_OK, "product");
    /* A 3-residue motif is 4 automaton states, and every one multiplies. */
    ASSERT_MSG(b.states == 4ull * a.states, "states multiply by |Q|");
    ASSERT_MSG(b.edges  == 4ull * a.edges,  "so do edge relaxations");
    /*
     * Checked against arithmetic done here rather than against another call
     * to the sizing function, which would agree with itself however wrong it
     * was. The backward field alone is (L + 1) * n_states * |Q| doubles, so
     * dropping |Q| from the budget formula cannot clear this bar -- while
     * "bigger than the plain one" could, on the sink buffer alone.
     */
    const uint64_t beta_only = 11ull * beam_model_n_states(m) * 4ull
                             * sizeof(double);
    ASSERT_MSG(b.bytes >= beta_only,
               "the budget covers a backward field four times as tall");
    ASSERT_MSG(b.bytes  >  a.bytes, "and the sweep needs more memory");
    beam_query_constraint_free(plain);
    beam_query_constraint_free(prod);
    beam_model_destroy(m);
    PASS();
}

static void test_a_dominant_rejected_path_cannot_collapse_the_answer(void) {
    /*
     * Regression for a defect found by review. Three sequences, the rejected
     * one weighted a hundred times the two accepted ones, with mixing weights
     * and backoff strengths turned up so the trellis spans thousands of nats.
     *
     * The forward sweep used to rescale each layer by the layer's own maximum
     * whenever the mask was full, on the reasoning that an automaton removes
     * (state, q) pairs rather than edges. But the heavy REJECTED edge still
     * set that maximum, so every accepted path underflowed and the last row
     * came back summing to zero -- a distribution that was not one, returned
     * with BEAM_OK. The rescale is now taken over edges a live path can use.
     */
    const char *seqs[3] = {"AAAA", "TTTT", "CCCC"};
    const double w[3] = {1.0, 1.0, 100.0};
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 2; cfg.max_length = 4; cfg.depth_cap = 3; cfg.strict = 0;
    cfg.kappa[0] = cfg.kappa[1] = cfg.kappa[2] = 20.0;
    cfg.kappa_wide = 20.0; cfg.kappa_len = 200.0;
    cfg.weight_forward = cfg.weight_backward = 100.0;
    BeamModel *m = beam_model_new(&cfg);
    ASSERT_MSG(m != NULL, "model built");
    ASSERT_MSG(beam_model_observe(m, seqs, w, 3u) == BEAM_OK, "observed");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalized");

    const uint32_t L = 4u;
    /* Accept exactly AAAA or TTTT: two all-pinned masks, disjoined. */
    BeamConstraint *ca = beam_query_constraint_new(L, R);
    BeamConstraint *ct = beam_query_constraint_new(L, R);
    for (uint32_t i = 0; i < L; ++i) {
        beam_query_constrain_position(ca, i, AA_A_);
        beam_query_constrain_position(ct, i, AA_T);
    }
    BeamAutomaton *aa = beam_query_automaton_from_masks(ca);
    BeamAutomaton *at = beam_query_automaton_from_masks(ct);
    ASSERT_MSG(aa && at, "lifted");
    BeamAutomaton *either = beam_query_automaton_or(aa, at);
    ASSERT_MSG(either != NULL, "disjoined");
    beam_query_automaton_free(aa);
    beam_query_automaton_free(at);
    beam_query_constraint_free(ca);
    beam_query_constraint_free(ct);

    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_automaton(c, either) == BEAM_OK, "attached");

    double *marg = (double *)malloc((size_t)L * R * sizeof(double));
    double *h    = (double *)malloc((size_t)L * sizeof(double));
    double *mi   = (double *)malloc((size_t)L * L * sizeof(double));
    ASSERT_MSG(marg && h && mi, "alloc");
    ASSERT_MSG(beam_query_marginals(m, c, marg, NULL) == BEAM_OK, "marginals");
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");

    /* The two admitted sequences are equally weighted, so every position is
     * an even split between A and T -- including the last, which is the one
     * that used to come back empty. */
    for (uint32_t i = 0; i < L; ++i) {
        double s = 0.0;
        for (uint32_t x = 0; x < R; ++x) s += marg[(size_t)i * R + x];
        ASSERT_MSG(fabs(s - 1.0) < 1e-9, "every row is still a distribution");
        ASSERT_MSG(fabs(marg[(size_t)i * R + AA_A_] - 0.5) < 1e-9, "half A");
        ASSERT_MSG(fabs(marg[(size_t)i * R + AA_T] - 0.5) < 1e-9, "half T");
        ASSERT_MSG(fabs(h[i] - 0.69314718055994531) < 1e-9,
                   "and carries exactly one bit");
        ASSERT_MSG(fabs(mi[(size_t)i * L + i] - 0.69314718055994531) < 1e-9,
                   "which the matrix diagonal agrees with");
    }
    double hs = 0.0;
    ASSERT_MSG(beam_query_entropy(m, c, &hs, NULL) == BEAM_OK, "H(seq)");
    ASSERT_MSG(fabs(hs - 0.69314718055994531) < 1e-9,
               "two admitted sequences are one bit of sequence entropy");

    double joint[400];
    ASSERT_MSG(beam_query_pair_joint(m, c, 3u, 3u, joint) == BEAM_OK, "joint");
    double js = 0.0;
    for (uint32_t k = 0; k < R * R; ++k) js += joint[k];
    ASSERT_MSG(fabs(js - 1.0) < 1e-9, "and the last position's self-joint too");

    beam_query_constraint_free(c);
    free(marg); free(h); free(mi);
    beam_model_destroy(m);
    PASS();
}

/* ── The nucleotide alphabet, end to end ─────────────────────── */

/**
 * Every quantity the engine reports, on a nucleotide model, against brute
 * force over the whole support.
 *
 * The alphabet was almost untested across this subsystem before review: five
 * residues instead of twenty, a radix of six instead of twenty-one, and case
 * folding on input. `5^5` is 3,125 sequences, so the entire support is
 * walkable and nothing has to be taken on trust.
 */
static void test_the_whole_engine_on_a_nucleotide_model(void) {
    const char *seqs[6] = {"ACGTA","ACGTT","AGGTA","ACCTA","ACGAA","TCGTA"};
    const double w[6] = {40.0, 25.0, 18.0, 12.0, 9.0, 6.0};
    const uint32_t NR = BEAM_NT_RESIDUES, L = 5u;

    for (uint32_t world = 0; world < 2u; ++world) {
        BeamModelConfig cfg;
        beam_model_config_default(&cfg);
        cfg.alphabet = BEAM_ALPHABET_NT;
        cfg.order = 2; cfg.max_length = 8;
        cfg.strict = (world == 0);
        BeamModel *m = beam_model_new(&cfg);
        ASSERT_MSG(m != NULL, "nucleotide model built");
        ASSERT_MSG(beam_model_observe(m, seqs, w, 6u) == BEAM_OK, "observed");
        ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalized");
        ASSERT_MSG(beam_model_n_residues(m) == NR, "five residues");

        /* Brute force: every string of length 5 over ACGTN. */
        const BeamAlphabetInfo *ai = beam_alphabet_info(BEAM_ALPHABET_NT);
        double *pm = (double *)calloc((size_t)L * NR, sizeof(double));
        ASSERT_MSG(pm != NULL, "alloc");
        char buf[8];
        uint8_t idx[8];
        double total = 0.0, kept = 0.0, hseq = 0.0;
        const BeamResidueMask keep = (1u << 0) | (1u << 2);   /* A or G at 1 */
        for (uint32_t i = 0; i < L; ++i) idx[i] = 0u;
        buf[L] = '\0';
        for (;;) {
            for (uint32_t i = 0; i < L; ++i) buf[i] = ai->idx_to_char[idx[i]];
            const double p = exp(beam_model_log_pseq(m, buf));
            if (p > 0.0) {
                total += p;
                for (uint32_t i = 0; i < L; ++i) pm[(size_t)i * NR + idx[i]] += p;
                if ((keep >> idx[1]) & 1u) kept += p;
            }
            uint32_t k = L;
            while (k-- > 0u) {
                if (++idx[k] < NR) break;
                idx[k] = 0u;
                if (k == 0u) goto done;
            }
        }
done:
        ASSERT_MSG(total > 0.0, "the support is non-empty");
        for (uint32_t i = 0; i < L; ++i)
            for (uint32_t a = 0; a < NR; ++a) pm[(size_t)i * NR + a] /= total;
        /* Sequence entropy, from the same walk. */
        for (uint32_t i = 0; i < L; ++i) idx[i] = 0u;
        for (;;) {
            for (uint32_t i = 0; i < L; ++i) buf[i] = ai->idx_to_char[idx[i]];
            const double p = exp(beam_model_log_pseq(m, buf)) / total;
            if (p > 0.0) hseq -= p * log(p);
            uint32_t k = L;
            while (k-- > 0u) {
                if (++idx[k] < NR) break;
                idx[k] = 0u;
                if (k == 0u) goto done2;
            }
        }
done2:;

        BeamConstraint *c = beam_query_constraint_new(L, NR);
        double *got = (double *)malloc((size_t)L * NR * sizeof(double));
        double *h   = (double *)malloc((size_t)L * sizeof(double));
        double *mi  = (double *)malloc((size_t)L * L * sizeof(double));
        ASSERT_MSG(got && h && mi, "alloc");

        ASSERT_MSG(beam_query_marginals(m, c, got, NULL) == BEAM_OK, "marginals");
        for (uint32_t k2 = 0; k2 < L * NR; ++k2)
            ASSERT_MSG(fabs(got[k2] - pm[k2]) < 1e-12,
                       "every marginal matches the enumerated one");

        double hs = 0.0;
        ASSERT_MSG(beam_query_entropy(m, c, &hs, NULL) == BEAM_OK, "H(seq)");
        ASSERT_MSG(fabs(hs - hseq) < 1e-12, "and so does the sequence entropy");

        ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "H per pos");
        for (uint32_t i = 0; i < L; ++i) {
            double want = 0.0;
            for (uint32_t a = 0; a < NR; ++a)
                if (pm[(size_t)i * NR + a] > 0.0)
                    want -= pm[(size_t)i * NR + a] * log(pm[(size_t)i * NR + a]);
            ASSERT_MSG(fabs(h[i] - want) < 1e-12, "per-position entropy too");
        }
        ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");
        for (uint32_t i = 0; i < L; ++i)
            ASSERT_MSG(fabs(mi[(size_t)i * L + i] - h[i]) < 1e-12,
                       "and the matrix diagonal is the position entropy");
        beam_query_constraint_free(c);

        /* A set mask, and a motif automaton, both against the same walk. */
        BeamConstraint *d = beam_query_constraint_new(L, NR);
        ASSERT_MSG(beam_query_constrain_set(d, 1u, keep) == BEAM_OK, "mask");
        double lp = 0.0;
        ASSERT_MSG(beam_query_log_probability(m, d, &lp) == BEAM_OK, "P(mask)");
        ASSERT_MSG(fabs(exp(lp) - kept / total) < 1e-12,
                   "a residue set over five residues matches enumeration");
        beam_query_constraint_free(d);

        const uint8_t motif[2] = {0u, 2u};        /* AG */
        BeamConstraint *e = beam_query_constraint_new(L, NR);
        ASSERT_MSG(beam_query_constrain_automaton(
                       e, beam_query_automaton_motif(L, NR, motif, 2u))
                   == BEAM_OK, "motif over nucleotides");
        double lm = 0.0;
        ASSERT_MSG(beam_query_log_probability(m, e, &lm) == BEAM_OK, "P(motif)");
        double wantm = 0.0;
        for (uint32_t i = 0; i < L; ++i) idx[i] = 0u;
        for (;;) {
            bool has = false;
            for (uint32_t i = 0; i + 1u < L; ++i)
                if (idx[i] == 0u && idx[i + 1u] == 2u) has = true;
            if (has) {
                for (uint32_t i = 0; i < L; ++i) buf[i] = ai->idx_to_char[idx[i]];
                wantm += exp(beam_model_log_pseq(m, buf));
            }
            uint32_t k = L;
            while (k-- > 0u) {
                if (++idx[k] < NR) break;
                idx[k] = 0u;
                if (k == 0u) goto done3;
            }
        }
done3:
        ASSERT_MSG(fabs(exp(lm) - wantm / total) < 1e-12,
                   "and so does a motif on the product engine");
        beam_query_constraint_free(e);

        free(pm); free(got); free(h); free(mi);
        beam_model_destroy(m);
    }
    PASS();
}

/* ── Cost estimation ─────────────────────────────────────────── */

static void test_the_estimate_matches_what_the_sweep_allocates(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    BeamConstraint *c = beam_query_constraint_new(10u, R);
    BeamQueryCost cost;
    ASSERT_MSG(beam_query_estimate(m, c, &cost) == BEAM_OK, "estimated");
    /*
     * Against arithmetic done here, not against another call to the sizing
     * helper -- a helper agrees with itself however wrong it is, and that is
     * how the beta pointer table went unaccounted for until review.
     */
    const uint64_t S = beam_model_n_states(m);
    const uint64_t expect = 11ull * S * sizeof(double)          /* beta      */
                          + 11ull * sizeof(double *)            /* its rows  */
                          + S * R * sizeof(double)              /* one layer */
                          + S * sizeof(double)                  /* the sink  */
                          + 3ull * S * sizeof(double)   /* cur, next, scratch */
                          + 256ull * R * sizeof(double);        /* chunks    */
    ASSERT_MSG(cost.bytes == expect,
               "the guarded figure is every buffer the sweep allocates");
    /* And a probability query really is charged less, rather than the flag
     * being carried around without changing anything. */
    ASSERT_MSG(beam_query_sweep_bytes(beam_model_n_states(m), 1u, R, 10u,
                                      false) < cost.bytes,
               "a partition function is not charged for forward buffers");
    ASSERT_MSG(cost.states == 11ull * beam_model_n_states(m), "states");
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

static void test_a_pinned_layer_is_estimated_cheaper_than_a_free_one(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    BeamConstraint *free_c = beam_query_constraint_new(10u, R);
    BeamConstraint *pin_c  = beam_query_constraint_new(10u, R);
    beam_query_constrain_position(pin_c, 5u, 2u);

    BeamQueryCost a, b;
    ASSERT_MSG(beam_query_estimate(m, free_c, &a) == BEAM_OK, "free");
    ASSERT_MSG(beam_query_estimate(m, pin_c, &b) == BEAM_OK, "pinned");
    ASSERT_MSG(b.edges < a.edges, "a pinned layer relaxes fewer edges");
    /* One layer of twenty residues becomes one layer of one. */
    ASSERT_MSG(a.edges - b.edges
               == 2ull * beam_model_n_states(m) * (R - 1u),
               "by exactly the residues the pin removed");
    beam_query_constraint_free(free_c);
    beam_query_constraint_free(pin_c);
    beam_model_destroy(m);
    PASS();
}

/* ── Main ────────────────────────────────────────────────────── */

int main(void) {
    printf("\n[constraints]\n");
    RUN_TEST(test_a_new_constraint_admits_every_residue);
    RUN_TEST(test_builders_intersect_rather_than_replace);
    RUN_TEST(test_builders_commute);
    RUN_TEST(test_contradictory_builders_leave_an_empty_position);
    RUN_TEST(test_a_refused_run_leaves_the_constraint_untouched);
    RUN_TEST(test_a_run_applies_as_a_prefix_and_as_a_suffix);
    RUN_TEST(test_a_run_near_the_integer_ceiling_does_not_wrap);
    RUN_TEST(test_a_position_past_the_end_is_refused_by_name);
    RUN_TEST(test_a_residue_outside_the_alphabet_is_refused_by_name);
    RUN_TEST(test_a_bad_shape_is_refused_at_construction);

    printf("\n[bit-for-bit reproduction of the clamped sweep]\n");
    RUN_TEST(test_an_unconstrained_query_reproduces_the_marginals);
    RUN_TEST(test_a_pinned_position_reproduces_a_clamp);
    RUN_TEST(test_a_pinned_probability_reproduces_log_clamp_probability);

    printf("\n[enumeration]\n");
    RUN_TEST(test_a_constrained_probability_matches_enumeration);

    printf("\n[residue sets]\n");
    RUN_TEST(test_a_set_conditions_as_the_mixture_of_its_members);
    RUN_TEST(test_a_set_probability_is_the_sum_over_its_members);
    RUN_TEST(test_marginal_rows_sum_to_one_under_a_constraint);
    RUN_TEST(test_the_probability_of_everything_is_exactly_one);

    printf("\n[impossible and malformed questions]\n");
    RUN_TEST(test_an_impossible_constraint_has_probability_zero);
    RUN_TEST(test_conditioning_on_the_impossible_is_refused);
    RUN_TEST(test_a_constraint_over_the_wrong_alphabet_is_refused);
    RUN_TEST(test_a_length_the_model_never_generates_is_refused);

    printf("\n[the product engine]\n");
    RUN_TEST(test_a_universal_automaton_reproduces_the_mask_sweep);
    RUN_TEST(test_a_motif_probability_matches_enumeration);
    RUN_TEST(test_a_count_probability_matches_enumeration);
    RUN_TEST(test_masks_and_an_automaton_compose);
    RUN_TEST(test_a_motif_conditional_matches_enumeration);
    RUN_TEST(test_a_language_and_its_complement_partition_the_mass);
    RUN_TEST(test_union_obeys_inclusion_exclusion);
    RUN_TEST(test_a_motif_the_model_cannot_spell_is_impossible);
    RUN_TEST(test_a_layer_indexed_automaton_answers_a_disjunction);
    RUN_TEST(test_the_streaming_log_sum_exp_matches_a_two_pass_reference);
    RUN_TEST(test_the_estimate_grows_with_the_automaton);
    RUN_TEST(test_a_dominant_rejected_path_cannot_collapse_the_answer);
    RUN_TEST(test_the_whole_engine_on_a_nucleotide_model);

    printf("\n[cost]\n");
    RUN_TEST(test_the_estimate_matches_what_the_sweep_allocates);
    RUN_TEST(test_a_pinned_layer_is_estimated_cheaper_than_a_free_one);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
