/**
 * @file test_query_information.c
 * @brief Tests for per-position entropy, pairwise joints and mutual information.
 *
 * Two corpora here have answers known before the model is built, which is the
 * best kind of test available. In one, positions 1 and 2 always agree, so
 * knowing either determines the other and the mutual information between them
 * must be exactly one bit -- `ln 2` nats. In the other, all four combinations
 * appear equally, so the two positions are independent and their mutual
 * information must be exactly zero. Neither number is read off the model; both
 * are what information theory requires, and the engine has to produce them.
 *
 * Beyond that, the whole matrix is compared against one assembled by
 * enumerating the support and counting, and against identities a wrong
 * assembly fails: the diagonal must be the position entropy, the matrix must
 * be symmetric although the two halves are computed by conditioning on
 * different positions, and the total correlation `sum_i H(X_i) - H(X)` must
 * come out non-negative and must agree with the matrix on the two corpora
 * where it is known.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/query.h"

#include "../../src/native/internal/beam/alphabet.h"
#include "../../src/native/internal/beam/query_internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

#define R BEAM_AA_RESIDUES
#define AA_A_  0u
#define AA_S_ 15u
#define LN2 0.69314718055994530942

#define N_SHORT 10
static const char *SHORT_TRAIN[N_SHORT] = {
    "CASF","CASY","CAGF","CGSF","CASL","CTSF","CASQ","CVSF","CASE","CAAF",
};
static const double SHORT_WEIGHT[N_SHORT] = {40,25,18,15,12,10,9,7,6,5};

static BeamModel *build(const char **seqs, const double *w, uint32_t n,
                        uint8_t order, bool open_world) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = 6u;
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

/* ── Independent reference: the matrix by counting ───────────── */

/**
 * The whole information matrix, assembled by walking the support and
 * accumulating every pairwise joint directly.
 *
 * Shares nothing with the engine: no sweep, no constraint, no conditioning --
 * only `beam_model_log_pseq` and arithmetic. `keep` filters the support when
 * the comparison is against a constrained query.
 */
static void enumerate_matrix(const BeamModel *m, uint32_t length,
                             bool (*keep)(const uint8_t *, uint32_t),
                             double *out_mi, double *out_h) {
    const BeamAlphabetInfo *ai = beam_alphabet_info(BEAM_ALPHABET_AA);
    const size_t JN = (size_t)length * length * R * R;
    double *joint = (double *)calloc(JN, sizeof(double));
    double *marg  = (double *)calloc((size_t)length * R, sizeof(double));
    char buf[8];
    uint8_t idx[8];
    double total = 0.0;

    for (uint32_t i = 0; i < length; ++i) idx[i] = 0u;
    buf[length] = '\0';
    for (;;) {
        for (uint32_t i = 0; i < length; ++i) buf[i] = ai->idx_to_char[idx[i]];
        if (!keep || keep(idx, length)) {
            const double p = exp(beam_model_log_pseq(m, buf));
            if (p > 0.0) {
                total += p;
                for (uint32_t i = 0; i < length; ++i) {
                    marg[(size_t)i * R + idx[i]] += p;
                    for (uint32_t j = 0; j < length; ++j)
                        joint[(((size_t)i * length + j) * R + idx[i]) * R
                              + idx[j]] += p;
                }
            }
        }
        uint32_t k = length;
        while (k-- > 0u) {
            if (++idx[k] < R) break;
            idx[k] = 0u;
            if (k == 0u) goto done;
        }
    }
done:
    for (uint32_t i = 0; i < length; ++i) {
        double h = 0.0;
        for (uint32_t a = 0; a < R; ++a) {
            const double p = marg[(size_t)i * R + a] / total;
            if (p > 0.0) h -= p * log(p);
        }
        out_h[i] = h;
    }
    for (uint32_t i = 0; i < length; ++i) {
        for (uint32_t j = 0; j < length; ++j) {
            double hij = 0.0;
            for (uint32_t a = 0; a < R; ++a)
                for (uint32_t b = 0; b < R; ++b) {
                    const double p =
                        joint[(((size_t)i * length + j) * R + a) * R + b]
                        / total;
                    if (p > 0.0) hij -= p * log(p);
                }
            out_mi[(size_t)i * length + j] = out_h[i] + out_h[j] - hij;
        }
    }
    free(joint); free(marg);
}

/* ── Corpora with answers known in advance ───────────────────── */

static void test_a_deterministic_dependency_carries_exactly_one_bit(void) {
    /* Position 2 always repeats position 1, so each tells you the other
     * completely and the information they share is the whole of either. */
    const char *seqs[2] = {"CAAF", "CGGF"};
    const double w[2] = {1.0, 1.0};
    BeamModel *m = build(seqs, w, 2u, 3u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    double *mi = (double *)malloc((size_t)L * L * sizeof(double));
    double *h  = (double *)malloc((size_t)L * sizeof(double));
    ASSERT_MSG(mi && h, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");

    ASSERT_MSG(fabs(h[1] - LN2) < 1e-12, "position 1 carries one bit");
    ASSERT_MSG(fabs(mi[1 * L + 2] - LN2) < 1e-12,
               "and shares all of it with position 2");
    ASSERT_MSG(fabs(h[0]) < 1e-12 && fabs(h[3]) < 1e-12,
               "the fixed positions carry none");
    ASSERT_MSG(fabs(mi[0 * L + 1]) < 1e-12,
               "and a constant shares nothing with anything");

    double hs = 0.0;
    ASSERT_MSG(beam_query_entropy(m, c, &hs, NULL) == BEAM_OK, "H(seq)");
    ASSERT_MSG(fabs(hs - LN2) < 1e-12,
               "two equally likely sequences are one bit of sequence entropy");
    /* The redundancy: two positions of one bit each, carrying one bit
     * between them, so exactly one bit is double-counted. */
    double sum_h = 0.0;
    for (uint32_t i = 0; i < L; ++i) sum_h += h[i];
    ASSERT_MSG(fabs((sum_h - hs) - LN2) < 1e-12,
               "the gap between the summed position entropies and the "
               "sequence entropy is the bit they share");
    beam_query_constraint_free(c);
    free(mi); free(h);
    beam_model_destroy(m);
    PASS();
}

static void test_independent_positions_share_no_information(void) {
    /* All four combinations, equally weighted: knowing one position says
     * nothing whatever about the other. */
    const char *seqs[4] = {"CAAF", "CAGF", "CGAF", "CGGF"};
    const double w[4] = {1.0, 1.0, 1.0, 1.0};
    BeamModel *m = build(seqs, w, 4u, 3u, false);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    double *mi = (double *)malloc((size_t)L * L * sizeof(double));
    double *h  = (double *)malloc((size_t)L * sizeof(double));
    ASSERT_MSG(mi && h, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");

    ASSERT_MSG(fabs(h[1] - LN2) < 1e-12 && fabs(h[2] - LN2) < 1e-12,
               "each varying position still carries its bit");
    ASSERT_MSG(fabs(mi[1 * L + 2]) < 1e-12,
               "but they share none of it");

    double hs = 0.0, sum_h = 0.0;
    ASSERT_MSG(beam_query_entropy(m, c, &hs, NULL) == BEAM_OK, "H(seq)");
    for (uint32_t i = 0; i < L; ++i) sum_h += h[i];
    ASSERT_MSG(fabs(hs - 2.0 * LN2) < 1e-12, "four sequences are two bits");
    ASSERT_MSG(fabs(sum_h - hs) < 1e-12,
               "and with nothing shared the position entropies add up to the "
               "sequence entropy exactly");
    beam_query_constraint_free(c);
    free(mi); free(h);
    beam_model_destroy(m);
    PASS();
}

/* ── The matrix against enumeration ──────────────────────────── */

static void test_the_information_matrix_matches_enumeration(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    double *mi = (double *)malloc((size_t)L * L * sizeof(double));
    double *h  = (double *)malloc((size_t)L * sizeof(double));
    double *wm = (double *)malloc((size_t)L * L * sizeof(double));
    double *wh = (double *)malloc((size_t)L * sizeof(double));
    ASSERT_MSG(mi && h && wm && wh, "alloc");

    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");
    beam_query_constraint_free(c);
    enumerate_matrix(m, L, NULL, wm, wh);

    double biggest = 0.0;
    for (uint32_t i = 0; i < L; ++i) {
        ASSERT_MSG(fabs(h[i] - wh[i]) < 1e-12, "position entropy matches");
        for (uint32_t j = 0; j < L; ++j) {
            const double want = wm[(size_t)i * L + j];
            if (want > biggest) biggest = want;
            ASSERT_MSG(fabs(mi[(size_t)i * L + j] - want) < 1e-12,
                       "every entry matches the enumerated matrix");
        }
    }
    ASSERT_MSG(biggest > 0.05, "the model has real structure to find");
    free(mi); free(h); free(wm); free(wh);
    beam_model_destroy(m);
    PASS();
}

static bool keep_has_as(const uint8_t *s, uint32_t n) {
    for (uint32_t i = 0; i + 1u < n; ++i)
        if (s[i] == AA_A_ && s[i + 1u] == AA_S_) return true;
    return false;
}

static void test_the_matrix_under_a_constraint_matches_enumeration(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    const uint8_t motif[2] = {AA_A_, AA_S_};

    double *mi = (double *)malloc((size_t)L * L * sizeof(double));
    double *h  = (double *)malloc((size_t)L * sizeof(double));
    double *wm = (double *)malloc((size_t)L * L * sizeof(double));
    double *wh = (double *)malloc((size_t)L * sizeof(double));
    ASSERT_MSG(mi && h && wm && wh, "alloc");

    /*
     * Conditioning on a motif is conditioning INSIDE that language: every
     * sweep the matrix takes must carry the automaton as well as the pin, or
     * the answer is the unconditioned matrix wearing a constraint's name.
     */
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_constrain_automaton(
                   c, beam_query_automaton_motif(L, R, motif, 2u)) == BEAM_OK,
               "attached");
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");
    beam_query_constraint_free(c);
    enumerate_matrix(m, L, keep_has_as, wm, wh);

    for (uint32_t i = 0; i < L; ++i) {
        ASSERT_MSG(fabs(h[i] - wh[i]) < 1e-12, "conditional entropy matches");
        for (uint32_t j = 0; j < L; ++j)
            ASSERT_MSG(fabs(mi[(size_t)i * L + j] - wm[(size_t)i * L + j])
                       < 1e-12, "every conditional entry matches");
    }
    free(mi); free(h); free(wm); free(wh);
    beam_model_destroy(m);
    PASS();
}

/* ── Identities the assembly must satisfy ────────────────────── */

static void test_the_matrix_obeys_the_identities_information_must(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    double *mi = (double *)malloc((size_t)L * L * sizeof(double));
    double *h  = (double *)malloc((size_t)L * sizeof(double));
    ASSERT_MSG(mi && h, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");

    uint32_t off_diagonal = 0u;
    for (uint32_t i = 0; i < L; ++i) {
        ASSERT_MSG(fabs(mi[(size_t)i * L + i] - h[i]) < 1e-12,
                   "a position tells you exactly itself, so the diagonal is "
                   "the position entropy");
        for (uint32_t j = 0; j < L; ++j) {
            const double v = mi[(size_t)i * L + j];
            ASSERT_MSG(v >= 0.0, "information is never negative");
            /*
             * Symmetric although the two halves are computed by conditioning
             * on DIFFERENT positions -- entry (i, j) comes from pinning i and
             * entry (j, i) from pinning j -- so this is two routes meeting,
             * not a property of how the array was filled.
             */
            ASSERT_MSG(fabs(v - mi[(size_t)j * L + i]) < 1e-12, "symmetric");
            const double cap = h[i] < h[j] ? h[i] : h[j];
            ASSERT_MSG(v <= cap + 1e-12,
                       "and no more than either position has to give");
            if (i != j && v > 1e-9) off_diagonal++;
            /* Conditional entropy, recovered by subtraction, stays a real
             * entropy. */
            const double cond = mi[(size_t)i * L + i] - v;
            ASSERT_MSG(cond >= -1e-12, "H(X_i | X_j) is non-negative");
        }
    }
    ASSERT_MSG(off_diagonal > 0u,
               "some pair actually shares information, so the bounds above "
               "were not satisfied by a matrix of zeros");
    beam_query_constraint_free(c);
    free(mi); free(h);
    beam_model_destroy(m);
    PASS();
}

static void test_summed_position_entropies_never_undercut_the_sequence(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    double *h = (double *)malloc((size_t)L * sizeof(double));
    ASSERT_MSG(h != NULL, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    double hs = 0.0, sum = 0.0;
    ASSERT_MSG(beam_query_entropy(m, c, &hs, NULL) == BEAM_OK, "H(seq)");
    for (uint32_t i = 0; i < L; ++i) sum += h[i];
    /* Subadditivity, across two engines that share no accumulator: the
     * moment sweep produced H(seq) and the posterior sweep the marginals. */
    ASSERT_MSG(sum >= hs - 1e-12,
               "the position entropies cannot sum to less than the sequence "
               "entropy");
    ASSERT_MSG(sum > hs + 1e-6,
               "and on a model with structure they sum to strictly more");
    beam_query_constraint_free(c);
    free(h);
    beam_model_destroy(m);
    PASS();
}

/* ── Pair joints ─────────────────────────────────────────────── */

static void test_a_pair_joint_is_a_distribution_with_the_right_marginals(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    double *joint = (double *)malloc((size_t)R * R * sizeof(double));
    double *marg  = (double *)malloc((size_t)L * R * sizeof(double));
    ASSERT_MSG(joint && marg, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_marginals(m, c, marg, NULL) == BEAM_OK, "marginals");

    ASSERT_MSG(beam_query_pair_joint(m, c, 1u, 2u, joint) == BEAM_OK, "joint");
    double total = 0.0;
    for (uint32_t k = 0; k < R * R; ++k) total += joint[k];
    ASSERT_MSG(fabs(total - 1.0) < 1e-12, "the joint is a distribution");
    for (uint32_t a = 0; a < R; ++a) {
        double row = 0.0;
        for (uint32_t b = 0; b < R; ++b) row += joint[(size_t)a * R + b];
        ASSERT_MSG(fabs(row - marg[(size_t)1 * R + a]) < 1e-12,
                   "summing out the second position gives the first's marginal");
    }
    for (uint32_t b = 0; b < R; ++b) {
        double col = 0.0;
        for (uint32_t a = 0; a < R; ++a) col += joint[(size_t)a * R + b];
        ASSERT_MSG(fabs(col - marg[(size_t)2 * R + b]) < 1e-12,
                   "and summing out the first gives the second's");
    }

    /* A position with itself is diagonal, which is the correct joint and not
     * a degenerate case to be special-cased away. */
    ASSERT_MSG(beam_query_pair_joint(m, c, 2u, 2u, joint) == BEAM_OK, "self");
    for (uint32_t a = 0; a < R; ++a)
        for (uint32_t b = 0; b < R; ++b)
            if (a != b)
                ASSERT_MSG(joint[(size_t)a * R + b] == 0.0,
                           "a position cannot disagree with itself");
    ASSERT_MSG(beam_query_pair_joint(m, c, 0u, L, joint)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "a position past the end");
    beam_query_constraint_free(c);
    free(joint); free(marg);
    beam_model_destroy(m);
    PASS();
}

static void test_the_matrix_agrees_with_the_pair_joints(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;

    double *mi    = (double *)malloc((size_t)L * L * sizeof(double));
    double *h     = (double *)malloc((size_t)L * sizeof(double));
    double *joint = (double *)malloc((size_t)R * R * sizeof(double));
    ASSERT_MSG(mi && h && joint, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    ASSERT_MSG(beam_query_position_entropy(m, c, h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_query_information_matrix(m, c, mi) == BEAM_OK, "matrix");

    /*
     * Two public routes to one number, reaching it differently. The matrix
     * accumulates joint entropies a conditioning residue at a time and never
     * holds a joint; `beam_query_pair_joint` builds the whole R x R table and
     * hands it over. Requiring them to agree is not a restatement of either.
     */
    uint32_t nonzero = 0u;
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
            ASSERT_MSG(beam_query_pair_joint(m, c, i, j, joint) == BEAM_OK,
                       "pair joint");
            double hij = 0.0;
            for (uint32_t k = 0; k < R * R; ++k)
                if (joint[k] > 0.0) hij -= joint[k] * log(joint[k]);
            const double want = h[i] + h[j] - hij;
            ASSERT_MSG(fabs(mi[(size_t)i * L + j] - (want > 0.0 ? want : 0.0))
                       < 1e-12,
                       "the matrix entry is the entropy of the pair joint");
            if (i != j && mi[(size_t)i * L + j] > 1e-9) nonzero++;
        }
    }
    ASSERT_MSG(nonzero > 0u, "and some pairs actually share information");
    beam_query_constraint_free(c);
    free(mi); free(h); free(joint);
    beam_model_destroy(m);
    PASS();
}

/* ── Cost ────────────────────────────────────────────────────── */

static void test_the_information_cost_reports_the_whole_job(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    BeamConstraint *c = beam_query_constraint_new(L, R);
    BeamQueryCost one, all;
    ASSERT_MSG(beam_query_estimate(m, c, &one) == BEAM_OK, "one sweep");
    ASSERT_MSG(beam_query_information_cost(m, c, &all) == BEAM_OK, "matrix");
    /* One sweep per (position, residue), plus the one that finds out which
     * of them carry mass. */
    ASSERT_MSG(all.edges == one.edges * (L * R + 1u),
               "the matrix is costed as the sweeps it will actually take");
    /*
     * The sweeps run one at a time, but the matrix HOLDS its accumulators
     * across all of them, so the peak is a sweep plus those. An earlier
     * version of this test asserted equality with one sweep, which enshrined
     * an undercount: a caller could set the budget to the reported figure and
     * still exceed it. Review caught the test as well as the code.
     */
    const uint64_t held = 2ull * L * R * sizeof(double)
                        + (uint64_t)L * L * sizeof(double)
                        + L * sizeof(double)
                        + L * sizeof(BeamResidueMask);
    ASSERT_MSG(all.bytes == one.bytes + held,
               "and the peak counts what is held across them");
    ASSERT_MSG(all.bytes > one.bytes, "which is strictly more than one sweep");
    beam_query_constraint_free(c);
    beam_model_destroy(m);
    PASS();
}

static void test_information_refuses_what_it_cannot_answer(void) {
    BeamModel *m = build(SHORT_TRAIN, SHORT_WEIGHT, N_SHORT, 2u, true);
    ASSERT_MSG(m != NULL, "model built");
    const uint32_t L = 4u;
    double *mi = (double *)malloc((size_t)L * L * sizeof(double));
    ASSERT_MSG(mi != NULL, "alloc");
    BeamConstraint *c = beam_query_constraint_new(L, R);
    beam_query_constrain_position(c, 1u, 1u);
    beam_query_constrain_position(c, 1u, 2u);      /* contradiction */
    ASSERT_MSG(beam_query_information_matrix(m, c, mi)
               == BEAM_ERR_NO_LIVE_PATHS,
               "there is no information in a distribution with no support");
    beam_query_constraint_free(c);
    free(mi);
    beam_model_destroy(m);
    PASS();
}

int main(void) {
    printf("\n[answers known before the model is built]\n");
    RUN_TEST(test_a_deterministic_dependency_carries_exactly_one_bit);
    RUN_TEST(test_independent_positions_share_no_information);

    printf("\n[enumeration]\n");
    RUN_TEST(test_the_information_matrix_matches_enumeration);
    RUN_TEST(test_the_matrix_under_a_constraint_matches_enumeration);

    printf("\n[identities]\n");
    RUN_TEST(test_the_matrix_obeys_the_identities_information_must);
    RUN_TEST(test_summed_position_entropies_never_undercut_the_sequence);

    printf("\n[pair joints]\n");
    RUN_TEST(test_a_pair_joint_is_a_distribution_with_the_right_marginals);

    RUN_TEST(test_the_matrix_agrees_with_the_pair_joints);

    printf("\n[cost and refusals]\n");
    RUN_TEST(test_the_information_cost_reports_the_whole_job);
    RUN_TEST(test_information_refuses_what_it_cannot_answer);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
