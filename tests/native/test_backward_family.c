/**
 * @file test_backward_family.c
 * @brief The three backward sweeps are one family, and this is what says so.
 *
 * `beam_betas`, `beam_betas_tilted` and `beam_best_completions` share a
 * skeleton -- the `beam_tail_weights` seed, the descending layer loop, the
 * `base = (s % SK) * A` decomposition of a state into its successor block, and
 * the test that skips an edge or a target that cannot be used. They differ in
 * the semiring, in whether the weights are tilted, and in whether they visit
 * every state or only the ones the forward frontier reached.
 *
 * Two of the three lived in `dp/backward.c` and the third in
 * `algorithms/top_k.c`, so a change to the shared shape was a change to two
 * files, and nothing anywhere related their outputs. They are in one file now,
 * which helps a reader and does not help a compiler. What follows is the part
 * that holds under a rewrite: a RELATION between the answers that only holds
 * if the skeleton really is shared.
 *
 * The relation is that a maximum cannot exceed a total over the same set. Both
 * sweeps run over the completions of state `s` at layer `i`;
 * `beam_best_completions` takes the largest and `beam_betas` takes the
 * log of their sum, so
 *
 *     best[i][s] <= beta[i][s]        everywhere
 *     best[len]  == beta[len]         exactly, both being the sink weights
 *
 * and the two agree to the bit wherever exactly one completion exists. That is
 * enough to catch a divergence in the seed, in the layer order, in the state
 * decomposition, or in the edge-skip test -- each of which would break the
 * inequality or the seed equality rather than merely change a number.
 *
 * It is deliberately NOT an assertion that the two produce the same values.
 * They must not: `beam_best_completions` visits only reached states and leaves
 * `-INFINITY` elsewhere, where `beam_betas` computes a finite value for states
 * the forward pass never gets to. The inequality survives that, which is why
 * it is the right relation to pin.
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

#define N_TRAIN 24
static const char *TRAIN[N_TRAIN] = {
    "CASSF",    "CASSY",    "CAGSF",    "CGSSF",    "CASSL",    "CTSSF",
    "CASSQEV",  "CASRWMT",  "CAWSFGN",  "CAYSLKQ",  "CASSFGH",  "CASSYWM",
    "CAGSFKPD", "CGSSFPDE", "CASSLNRT", "CTSSFRWY", "CASSQEVI", "CASRWMTL",
    "CAWSFGNE", "CAYSLKQD", "CASSLGGYE", "CASSLGGTE", "CASRLGGYE", "CASSPGGYE",
};
static const double WEIGHT[N_TRAIN] = {
    40, 25, 18, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 12, 9, 7, 5,
};

static BeamModel *build(uint32_t order, bool strict) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = order;
    cfg.max_length = 12u;
    cfg.strict = strict;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN, WEIGHT, N_TRAIN) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/** `len + 1` rows of `n_states` doubles, as both sweeps require. */
static double **field(uint32_t len, uint32_t n_states) {
    double **f = (double **)calloc((size_t)len + 1u, sizeof(double *));
    if (!f) return NULL;
    for (uint32_t i = 0; i <= len; ++i) {
        f[i] = (double *)malloc((size_t)n_states * sizeof(double));
        if (!f[i]) { for (uint32_t j = 0; j < i; ++j) free(f[j]);
                     free(f); return NULL; }
    }
    return f;
}

static void field_free(double **f, uint32_t len) {
    if (!f) return;
    for (uint32_t i = 0; i <= len; ++i) free(f[i]);
    free(f);
}

/**
 * The best completion never exceeds the total, at every layer and state.
 *
 * Run over every live length of the model rather than one, because the layer
 * loop is where a descending sweep goes wrong and a single length can be too
 * short to show it.
 */
static void test_the_best_completion_never_exceeds_the_total(void) {
    /* OPEN world: with backoff on, every residue keeps a finite weight, so
     * every reached state has a full complement of completions and the
     * inequality is compared over eight thousand states per layer rather
     * than a handful. */
    BeamModel *m = build(3u, false);
    ASSERT_MSG(m, "model did not build");
    const uint32_t S = m->n_states;

    uint32_t lengths = 0, compared = 0, seeds = 0;
    static char msg[256];
    for (uint32_t len = 1; len <= m->max_len; ++len) {
        if (!beam_length_live(m, len)) continue;
        double **beta = field(len, S);
        double **best = field(len, S);
        ASSERT_MSG(beta && best, "allocation");
        BeamMemAccount acct;
        memset(&acct, 0, sizeof(acct));
        ASSERT_MSG(beam_betas(m, len, beta, &acct) == BEAM_OK,
                   beam_error_message());
        ASSERT_MSG(beam_best_completions(m, len, best, &acct) == BEAM_OK,
                   beam_error_message());
        ++lengths;

        /* The seed layer is the sink weight in both, so it must agree to the
         * bit: nothing has been chosen or summed yet. */
        for (uint32_t s = 0; s < S; ++s) {
            if (best[len][s] != beta[len][s]) {
                snprintf(msg, sizeof msg,
                         "the seed layers differ at length %u state %u: "
                         "best %.17g, beta %.17g",
                         (unsigned)len, (unsigned)s, best[len][s], beta[len][s]);
                field_free(beta, len); field_free(best, len);
                beam_model_destroy(m);
                ASSERT_MSG(false, msg);
            }
            ++seeds;
        }

        for (uint32_t i = 0; i < len; ++i) {
            for (uint32_t s = 0; s < S; ++s) {
                const double a = best[i][s], b = beta[i][s];
                if (a == -INFINITY) continue;      /* unreached, or no path */
                ++compared;
                /* A sum of positive terms is at least its largest, so in logs
                 * the total is at least the maximum. The slack is the tiny
                 * rounding of `mx + log(acc)` when `acc` is exactly 1. */
                if (!(a <= b + 1e-9)) {
                    snprintf(msg, sizeof msg,
                             "a best completion exceeds the total at length %u "
                             "layer %u state %u: best %.17g > beta %.17g",
                             (unsigned)len, (unsigned)i, (unsigned)s, a, b);
                    field_free(beta, len); field_free(best, len);
                    beam_model_destroy(m);
                    ASSERT_MSG(false, msg);
                }
            }
        }
        field_free(beta, len);
        field_free(best, len);
    }
    beam_model_destroy(m);

    ASSERT_MSG(lengths >= 2, "fewer than two live lengths were compared");
    ASSERT_MSG(seeds > 0 && compared > 0,
               "no state was actually compared, so this proves nothing");
    PASS();
}

/**
 * Each sweep computes ITS OWN semiring, checked against a reference derived
 * here rather than from the other sweep.
 *
 * The inequality above pins the shared skeleton -- seed, layer order, state
 * decomposition, edge test -- and cannot pin the semiring: make
 * `beam_best_completions` total its completions instead of maximising them and
 * `best <= beta` still holds, with equality. That mutation passes the test
 * above, which is how this one came to be written.
 *
 * So the last layer before the sink is recomputed here from the edge weights
 * and the sink values, both ways, and each sweep is required to match its own.
 * The reference shares no code with either.
 *
 * Strict inequality would have been the cheaper check and is not safe: with
 * backoff on, an unobserved cell sits near `ln(1e-300)`, so one completion can
 * dominate the rest by seven hundred nats and `log(acc)` rounds to zero. The
 * total is then equal to the maximum in floating point, legitimately.
 */
static void test_each_sweep_computes_its_own_semiring(void) {
    BeamModel *m = build(3u, false);
    ASSERT_MSG(m, "model did not build");
    const uint32_t S = m->n_states;
    const uint32_t R = m->n_residues;
    const uint32_t A = m->n_alphabet;
    const uint32_t SK = m->state_shift;

    uint32_t len = 0;
    for (uint32_t l = 1; l <= m->max_len; ++l) if (beam_length_live(m, l)) len = l;
    ASSERT_MSG(len >= 2, "no live length");

    double **beta = field(len, S);
    double **best = field(len, S);
    double *w = (double *)malloc((size_t)S * R * sizeof(double));
    ASSERT_MSG(beta && best && w, "allocation");
    BeamMemAccount acct;
    memset(&acct, 0, sizeof(acct));
    ASSERT_MSG(beam_betas(m, len, beta, &acct) == BEAM_OK,
               beam_error_message());
    ASSERT_MSG(beam_best_completions(m, len, best, &acct) == BEAM_OK,
               beam_error_message());
    beam_layer_weights(m, len - 1u, len, w, NULL);

    uint32_t checked = 0, multi = 0;
    static char msg[256];
    for (uint32_t s = 0; s < S; ++s) {
        if (best[len - 1u][s] == -INFINITY) continue;   /* unreached */
        const double *row = w + (size_t)s * R;
        const uint32_t base = (s % SK) * A;

        double ref_max = -INFINITY;
        uint32_t live = 0;
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY || beta[len][base + x] == -INFINITY) continue;
            const double v = row[x] + beta[len][base + x];
            if (v > ref_max) ref_max = v;
            ++live;
        }
        double acc = 0.0;
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY || beta[len][base + x] == -INFINITY) continue;
            acc += exp(row[x] + beta[len][base + x] - ref_max);
        }
        const double ref_sum = (live == 0u) ? -INFINITY : ref_max + log(acc);

        ++checked;
        if (live > 1u) ++multi;
        if (fabs(best[len - 1u][s] - ref_max) > 1e-12) {
            snprintf(msg, sizeof msg,
                     "state %u: the best-completion sweep gave %.17g where the "
                     "maximum over its completions is %.17g",
                     (unsigned)s, best[len - 1u][s], ref_max);
            goto fail;
        }
        if (fabs(beta[len - 1u][s] - ref_sum) > 1e-9) {
            snprintf(msg, sizeof msg,
                     "state %u: the backward sweep gave %.17g where the total "
                     "over its completions is %.17g",
                     (unsigned)s, beta[len - 1u][s], ref_sum);
            goto fail;
        }
    }
    free(w); field_free(beta, len); field_free(best, len);
    beam_model_destroy(m);
    ASSERT_MSG(checked > 0, "no reached state was compared");
    ASSERT_MSG(multi > 0,
               "every state had at most one completion, so a sweep that "
               "totalled instead of maximising would pass this");
    PASS();
    return;
fail:
    free(w); field_free(beta, len); field_free(best, len);
    beam_model_destroy(m);
    ASSERT_MSG(false, msg);
}

int main(void) {
    printf("the backward family\n");
    RUN_TEST(test_the_best_completion_never_exceeds_the_total);
    RUN_TEST(test_each_sweep_computes_its_own_semiring);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
