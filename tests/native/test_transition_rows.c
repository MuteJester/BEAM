/**
 * @file test_transition_rows.c
 * @brief Normalized forward conditionals of the whole BEAM law.
 *
 * The reference sums the independently scored sequence law over its entire
 * nucleotide support. It never reads a trellis edge or a backward field. In
 * particular, normalizing the forward expert's own rows would pass a row-sum
 * test but fail the joint/marginal comparison and full-sequence reconstruction.
 * Short lengths exercise sentinel padding, while order 2 at length 4 exercises
 * dropping an older prefix residue. Order 4 at length 6 also visits every full
 * context, including contexts the training corpus never contained.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "beam/common.h"
#include "beam/model.h"

static int pass_count = 0, fail_count = 0;
#include "test_utils.h"

#define R BEAM_NT_RESIDUES
static const char LETTERS[] = "ACGTN";

static const char *TRAIN[] = {
    "A", "C", "AG", "CT", "ACG", "TNA", "ACGT", "AGGT",
    "CCGA", "TCGT", "ACGTA", "NCGTT", "ACGTAC", "TCNAGN",
};
static const double WEIGHTS[] = {
    3, 2, 8, 5, 17, 6, 40, 19, 13, 11, 7, 4, 21, 9,
};

static uint32_t pow_r(uint32_t n) {
    uint32_t result = 1u;
    while (n--) result *= R;
    return result;
}

static void decode(uint32_t code, uint32_t length, uint32_t *digits, char *s) {
    for (uint32_t p = length; p-- > 0u;) {
        digits[p] = code % R;
        s[p] = LETTERS[digits[p]];
        code /= R;
    }
    s[length] = '\0';
}

static uint32_t context_code(const uint32_t *digits, uint32_t position,
                             uint32_t order) {
    const uint32_t first = position > order ? position - order : 0u;
    uint32_t code = 0u;
    for (uint32_t i = first; i < position; ++i) code = code * R + digits[i];
    return code;
}

static BeamModel *build(uint8_t order, bool sparse, bool finalize) {
    BeamModelConfig cfg;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &cfg);
    cfg.order = order;
    cfg.max_length = 6u;
    cfg.sparse = sparse;
    cfg.weight_forward = 0.7;
    cfg.weight_backward = 0.3;
    cfg.alpha_auto = false;
    for (uint32_t k = 0u; k <= order; ++k) cfg.alpha[k] = 1.5 + k;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN, WEIGHTS,
                           (uint32_t)(sizeof TRAIN / sizeof *TRAIN)) != BEAM_OK
        || (finalize && beam_model_finalize(m) != BEAM_OK)) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/* Compare every conditional with P(context, next)/P(context), where both
 * masses come only from direct scoring and exhaustive summation. */
static bool compare_enumeration(const BeamModel *m, uint32_t order,
                                uint32_t length) {
    const uint32_t n = pow_r(length);
    double *prob = malloc((size_t)n * sizeof *prob);
    double *rows[4] = {NULL, NULL, NULL, NULL};
    bool ok = false;
    if (!prob) goto done;
    double total = 0.0;
    uint32_t digits[7];
    char s[8];
    for (uint32_t code = 0u; code < n; ++code) {
        decode(code, length, digits, s);
        prob[code] = exp(beam_model_log_pseq(m, s));
        total += prob[code];
    }
    if (!(total > 0.0) || !isfinite(total)) goto done;

    for (uint32_t p = 0u; p < length; ++p) {
        const uint32_t nrows = pow_r(p < order ? p : order);
        const size_t cells = (size_t)nrows * R;
        double *joint = calloc(cells, sizeof *joint);
        rows[p] = malloc(cells * sizeof *rows[p]);
        if (!joint || !rows[p]) { free(joint); goto done; }
        if (beam_model_transition_rows(m, length, p, false, rows[p]) != BEAM_OK) {
            free(joint); goto done;
        }
        for (uint32_t code = 0u; code < n; ++code) {
            decode(code, length, digits, s);
            const uint32_t c = context_code(digits, p, order);
            joint[(size_t)c * R + digits[p]] += prob[code];
        }
        for (uint32_t c = 0u; c < nrows; ++c) {
            double denominator = 0.0;
            for (uint32_t a = 0u; a < R; ++a)
                denominator += joint[(size_t)c * R + a];
            if (!(denominator > 0.0)) { free(joint); goto done; }
            for (uint32_t a = 0u; a < R; ++a) {
                const size_t j = (size_t)c * R + a;
                const double expected = joint[j] / denominator;
                if (!isfinite(rows[p][j])
                    || fabs(rows[p][j] - expected) > 2e-12) {
                    printf("conditional mismatch (k=%u,L=%u,p=%u,c=%u,a=%u): "
                           "%.17g vs %.17g\n", order, length, p, c, a,
                           rows[p][j], expected);
                    free(joint); goto done;
                }
            }
        }
        free(joint);
    }

    /* The conditional product must reconstruct the normalized sequence law,
     * including when the context drops residues from the prefix. */
    for (uint32_t code = 0u; code < n; ++code) {
        decode(code, length, digits, s);
        double reconstructed = 0.0;
        for (uint32_t p = 0u; p < length; ++p) {
            const uint32_t c = context_code(digits, p, order);
            reconstructed += log(rows[p][(size_t)c * R + digits[p]]);
        }
        const double expected = log(prob[code] / total);
        if (!isfinite(reconstructed) || fabs(reconstructed - expected) > 2e-11) {
            printf("reconstruction mismatch (k=%u,L=%u,s=%s): %.17g vs %.17g\n",
                   order, length, s, reconstructed, expected);
            goto done;
        }
    }
    ok = true;
done:
    for (uint32_t p = 0u; p < 4u; ++p) free(rows[p]);
    free(prob);
    return ok;
}

static void test_matches_exhaustive_sequence_law(void) {
    for (uint8_t order = 2u; order <= 4u; order += 2u) {
        for (uint32_t sparse = 0u; sparse < 2u; ++sparse) {
            BeamModel *m = build(order, sparse != 0u, true);
            ASSERT_MSG(m, "model built");
            bool ok = true;
            for (uint32_t length = 1u; length <= 4u && ok; ++length)
                ok = compare_enumeration(m, order, length);
            beam_model_destroy(m);
            ASSERT_MSG(ok, "conditional rows or reconstruction disagree with enumeration");
        }
    }
    PASS();
}

static void test_order_four_rows_are_normalized_and_log_consistent(void) {
    BeamModel *m = build(4u, true, true);
    ASSERT_MSG(m, "model built");
    const size_t max_cells = (size_t)pow_r(4u) * R;
    double *linear = malloc(max_cells * sizeof *linear);
    double *logs = malloc(max_cells * sizeof *logs);
    ASSERT_MSG(linear && logs, "output buffers allocated");
    bool ok = true;
    for (uint32_t length = 1u; length <= 6u && ok; ++length) {
        for (uint32_t p = 0u; p < length && ok; ++p) {
            const uint32_t nrows = pow_r(p < 4u ? p : 4u);
            if (beam_model_transition_rows(m, length, p, false, linear) != BEAM_OK
                || beam_model_transition_rows(m, length, p, true, logs) != BEAM_OK) {
                ok = false; break;
            }
            for (uint32_t c = 0u; c < nrows && ok; ++c) {
                double total = 0.0;
                for (uint32_t a = 0u; a < R; ++a) {
                    const size_t j = (size_t)c * R + a;
                    if (!(linear[j] > 0.0) || !isfinite(logs[j])
                        || fabs(exp(logs[j]) - linear[j]) > 2e-14) ok = false;
                    total += linear[j];
                }
                if (fabs(total - 1.0) > 2e-12) ok = false;
            }
        }
    }
    free(linear); free(logs);
    beam_model_destroy(m);
    ASSERT_MSG(ok, "all reachable context rows sum to one and agree in both domains");
    PASS();
}

static void test_invalid_inputs_are_reported(void) {
    BeamModel *m = build(2u, true, true);
    BeamModel *unfinished = build(2u, true, false);
    double out[R * R * R];
    ASSERT_MSG(m && unfinished, "models built");
    ASSERT_MSG(beam_model_transition_rows(NULL, 4u, 0u, false, out)
               == BEAM_ERR_NULL_ARG, "null model refused");
    ASSERT_MSG(beam_model_transition_rows(m, 4u, 0u, false, NULL)
               == BEAM_ERR_NULL_ARG, "null output refused");
    ASSERT_MSG(beam_model_transition_rows(unfinished, 4u, 0u, false, out)
               == BEAM_ERR_NOT_BUILT, "unfinished model refused");
    ASSERT_MSG(beam_model_transition_rows(m, 0u, 0u, false, out)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "zero length refused");
    ASSERT_MSG(beam_model_transition_rows(m, 7u, 0u, false, out)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "length past model maximum refused");
    ASSERT_MSG(beam_model_transition_rows(m, 4u, 4u, false, out)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "position at length refused");
    ASSERT_MSG(beam_model_transition_rows(m, 4u, UINT32_MAX, false, out)
               == BEAM_ERR_PARAM_OUT_OF_RANGE, "overflow-size position refused");
    ASSERT_MSG(beam_error_message()[0] != '\0', "refusal has an error message");
    beam_model_destroy(unfinished);
    beam_model_destroy(m);
    PASS();
}

static void test_budget_refusal_does_not_poison_the_model(void) {
    BeamModel *m = build(2u, true, true);
    ASSERT_MSG(m, "model built");
    double out[R * R * R];
    const uint64_t previous_budget = beam_model_memory_budget();
    beam_model_set_memory_budget(1u);
    const BeamError refused = beam_model_transition_rows(m, 6u, 4u, false, out);
    beam_model_set_memory_budget(previous_budget);
    const BeamError recovered = beam_model_transition_rows(m, 6u, 4u, false, out);
    beam_model_destroy(m);
    ASSERT_MSG(refused == BEAM_ERR_BUDGET, "scratch is refused under a one-byte budget");
    ASSERT_MSG(recovered == BEAM_OK, "same model answers once the budget is restored");
    PASS();
}

int main(void) {
    printf("Normalized transition rows\n");
    RUN_TEST(test_matches_exhaustive_sequence_law);
    RUN_TEST(test_order_four_rows_are_normalized_and_log_consistent);
    RUN_TEST(test_invalid_inputs_are_reported);
    RUN_TEST(test_budget_refusal_does_not_poison_the_model);
    printf("\n%d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
