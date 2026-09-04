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
#define AA_D  2u
#define AA_F  4u
#define AA_L  9u
#define AA_P 12u
#define AA_Q 13u
#define AA_V 17u
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

    ASSERT_MSG(zw == zg, "log Z identical, not merely close");
    for (uint32_t k = 0; k < L * R; ++k)
        ASSERT_MSG(want[k] == got[k], "every cell identical, not merely close");

    beam_query_constraint_free(c);
    free(want); free(got);
    beam_model_destroy(m);
    PASS();
}

/**
 * Every pin that carries mass, compared cell by cell against the clamp that
 * denotes the same thing.
 *
 * `*n_live` counts the pins that were compared and `*n_bad` the ones that
 * disagreed, reported separately on purpose: a helper that returned one number
 * would let "the engine matched on every pin" and "there were almost no pins"
 * fail the caller identically, and those need different fixes.
 */
static void compare_every_pin(BeamModel *m, uint32_t L,
                              uint32_t *n_live, uint32_t *n_bad) {

    double *want = (double *)malloc((size_t)L * R * sizeof(double));
    double *got  = (double *)malloc((size_t)L * R * sizeof(double));
    uint8_t *clamp = (uint8_t *)malloc(L);
    *n_live = 0u; *n_bad = 0u;
    if (!want || !got || !clamp) { *n_bad = UINT32_MAX; return; }

    /*
     * Every pin is attempted, including those whose unconstrained marginal is
     * exactly 0.0. In an open-world model that zero is an underflow, not an
     * absence -- log_pseq("AAAAAAAAAA") is about -6908, which is a real path
     * whose linear-space mass is far below the smallest double. Those pins
     * condition on something 6,900 nats below the mode, which is precisely
     * where two implementations of the same recursion are most likely to
     * disagree, so skipping them would have skipped the interesting half.
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

            /* The two must agree about whether the question has an answer,
             * before they are asked to agree about what it is. */
            if (ref != qry) { (*n_bad)++; continue; }
            if (ref != BEAM_OK) continue;

            (*n_live)++;
            if (zw != zg) { (*n_bad)++; continue; }
            for (uint32_t k = 0; k < L * R; ++k)
                if (want[k] != got[k]) { (*n_bad)++; break; }
        }
    }
    free(want); free(got); free(clamp);
}

static void test_a_pinned_position_reproduces_a_clamp(void) {
    /*
     * Both worlds, because they exercise different halves of the sweep. The
     * closed-world trellis spells only what the corpus spells, so most layers
     * admit one residue and the pins that exist sit on a very narrow graph.
     * The open-world one carries all twenty residues at every position, so it
     * is where the arithmetic is actually dense.
     */
    uint32_t live = 0u, bad = 0u;

    BeamModel *shut = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(shut != NULL, "closed-world model built");
    compare_every_pin(shut, 10u, &live, &bad);
    beam_model_destroy(shut);
    ASSERT_MSG(bad == 0u, "no closed-world pin disagreed with its clamp");
    const uint32_t shut_live = live;

    BeamModel *open = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, true);
    ASSERT_MSG(open != NULL, "open-world model built");
    compare_every_pin(open, 10u, &live, &bad);
    beam_model_destroy(open);
    ASSERT_MSG(bad == 0u, "no open-world pin disagreed with its clamp");
    ASSERT_MSG(live == 10u * R,
               "and the open world carries a path for all 200 pins");
    ASSERT_MSG(shut_live == 17u,
               "while the closed world carries one for the 17 it spells");
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

static void test_the_mask_engine_refuses_an_automaton_it_cannot_honour(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 10u;
    double *out = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(out != NULL, "alloc");

    const uint8_t motif[2] = {AA_L, AA_Q};
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_automaton(
                   c, beam_query_automaton_motif(L, R, motif, 2u)) == BEAM_OK,
               "attached");

    /*
     * Until the product graph lands, an engine that reads only the masks would
     * answer this as though the motif were not there -- a confidently wrong
     * probability, which is worse than no answer. All three entry points must
     * refuse it rather than one.
     */
    double p = 0.0;
    BeamQueryCost cost;
    ASSERT_MSG(beam_query_marginals(m, c, out, NULL)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "marginals refuse");
    ASSERT_MSG(strstr(beam_error_message(), "product graph") != NULL,
               "and say what is missing");
    ASSERT_MSG(beam_query_log_probability(m, c, &p)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "probability refuses");
    ASSERT_MSG(beam_query_estimate(m, c, &cost)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "estimate refuses");

    beam_query_constraint_free(c);
    free(out);
    beam_model_destroy(m);
    PASS();
}

/* ── Cost estimation ─────────────────────────────────────────── */

static void test_the_estimate_matches_what_the_sweep_allocates(void) {
    BeamModel *m = build(TRAIN, WEIGHT, N_TRAIN, 3u, 14u, false);
    ASSERT_MSG(m != NULL, "model built");
    BeamConstraint *c = beam_query_constraint_new(10u, R);
    BeamQueryCost cost;
    ASSERT_MSG(beam_query_estimate(m, c, &cost) == BEAM_OK, "estimated");
    ASSERT_MSG(cost.bytes == beam_query_sweep_bytes(beam_model_n_states(m),
                                                    R, 10u, true),
               "the guarded figure is the allocated figure");
    /* And a probability query really is charged less, rather than the flag
     * being carried around without changing anything. */
    ASSERT_MSG(beam_query_sweep_bytes(beam_model_n_states(m), R, 10u, false)
               < cost.bytes,
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

    RUN_TEST(test_the_mask_engine_refuses_an_automaton_it_cannot_honour);

    printf("\n[cost]\n");
    RUN_TEST(test_the_estimate_matches_what_the_sweep_allocates);
    RUN_TEST(test_a_pinned_layer_is_estimated_cheaper_than_a_free_one);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
