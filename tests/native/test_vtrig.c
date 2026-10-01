/**
 * @file test_vtrig.c
 * @brief The line sweeps' sine, cosine and exponential against long double.
 *
 * `beam_sincos_many` replaces libm's sincos in the line sweep, where each edge
 * needs a handful per stretch and libm's had become a third of the line's
 * time. It must be as good as libm where the sweep uses it: angles up to
 * about 1e5 radians (t up to ~1e3, an edge's log-weight up to ~1e2), within
 * two units in the last place of 1, and exactly right where the sweep leans
 * on exactness -- sin 0 = 0 and cos 0 = 1, which is what makes t = 0 real.
 * It must also give the same bits however many angles a call is handed,
 * since a call covers one target's edges and that count varies.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;
static char msg[256];

#include "test_utils.h"

static double worst_error(const double *x, size_t n) {
    double *s = malloc(n * sizeof *s), *c = malloc(n * sizeof *c);
    beam_sincos_many(x, s, c, n);
    double worst = 0.0;
    for (size_t k = 0; k < n; ++k) {
        const long double xs = sinl((long double)x[k]), xc = cosl((long double)x[k]);
        const double es = (double)fabsl((long double)s[k] - xs);
        const double ec = (double)fabsl((long double)c[k] - xc);
        if (es > worst) { worst = es; snprintf(msg, sizeof msg, "sin(%.17g)", x[k]); }
        if (ec > worst) { worst = ec; snprintf(msg, sizeof msg, "cos(%.17g)", x[k]); }
    }
    free(s); free(c);
    return worst;
}

static void test_random_angles_within_two_ulp(void) {
    enum { N = 400000 };
    double *x = malloc(N * sizeof *x);
    uint64_t st = 12345u;
    const double scales[] = {1e-8, 1.0, 10.0, 1e3, 1e5};
    for (size_t k = 0; k < N; ++k) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        const double u = (double)(st >> 11) / 9007199254740992.0;   /* [0,1) */
        x[k] = (2.0 * u - 1.0) * scales[k % 5];
    }
    const double worst = worst_error(x, N);
    printf("    worst %.2e (%.2f ulp of 1) at %s\n", worst, worst / DBL_EPSILON, msg);
    free(x);
    ASSERT_MSG(worst <= 2.0 * DBL_EPSILON, "an angle lost more than two ulp");
    PASS();
}

static void test_near_multiples_of_half_pi(void) {
    /* The reduction's hard cases: x within a few ulp of k pi/2, where the
     * reduced angle is tiny and everything rests on pi/2's lower bits. */
    enum { K = 2000, D = 9 };
    double *x = malloc((size_t)K * D * sizeof *x);
    size_t n = 0;
    for (int k = -K / 2; k < K / 2; ++k) {
        const double m = (double)k * 1.57079632679489661923 * 37.0;
        double y = m;
        for (int d = 0; d < 4; ++d) y = nextafter(y, -INFINITY);
        for (int d = 0; d < D; ++d) { x[n++] = y; y = nextafter(y, INFINITY); }
    }
    const double worst = worst_error(x, n);
    printf("    worst %.2e (%.2f ulp of 1) at %s\n", worst, worst / DBL_EPSILON, msg);
    free(x);
    ASSERT_MSG(worst <= 2.0 * DBL_EPSILON, "a near-multiple of pi/2 lost digits");
    PASS();
}

static void test_zero_is_exact(void) {
    const double x[] = {0.0, -0.0};
    double s[2], c[2];
    beam_sincos_many(x, s, c, 2);
    ASSERT_MSG(s[0] == 0.0 && s[1] == 0.0 && c[0] == 1.0 && c[1] == 1.0,
               "sin 0 or cos 0 is not exact");
    PASS();
}

static void test_bits_do_not_depend_on_the_call_size(void) {
    enum { N = 67 };
    double x[N], s1[N], c1[N], s2[N], c2[N];
    for (int k = 0; k < N; ++k) x[k] = 0.37 * k * k - 11.0 * k + 0.5;
    beam_sincos_many(x, s1, c1, N);
    for (int k = 0; k < N; ++k) beam_sincos_many(x + k, s2 + k, c2 + k, 1);
    ASSERT_MSG(memcmp(s1, s2, sizeof s1) == 0 && memcmp(c1, c2, sizeof c1) == 0,
               "an angle's bits moved with how many came with it");
    PASS();
}

static void test_beyond_the_reduction_range(void) {
    /* Past the three-part reduction's range the result must still be right
     * (it falls back to libm), never silently wrong. */
    const double x[] = {3e6, -7.5e7, 1e12, 1e300};
    double s[4], c[4];
    beam_sincos_many(x, s, c, 4);
    int ok = 1;
    for (int k = 0; k < 4; ++k) ok &= (s[k] == sin(x[k]) && c[k] == cos(x[k]));
    ASSERT_MSG(ok, "a large angle was not handed to libm");
    PASS();
}

/* ── exp, eight at a time ────────────────────────────────────── */

static void test_exp_within_two_ulp(void) {
    /* The real-order sweep's edge factors exp(q w - qmax) lie in
     * [-inf, 0]; anything below the smallest normal double is flushed. */
    enum { N = 400000 };
    double *x = malloc(N * sizeof *x), *y = malloc(N * sizeof *y);
    uint64_t st = 777u;
    for (size_t k = 0; k < N; ++k) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        const double u = (double)(st >> 11) / 9007199254740992.0;
        const double span = (k % 3 == 0) ? 1.0 : (k % 3 == 1) ? 40.0 : 708.0;
        x[k] = -u * span + (k % 7 == 0 ? 0.5 : 0.0);
    }
    beam_exp_many(x, y, N);
    double worst = 0.0;
    for (size_t k = 0; k < N; ++k) {
        const long double want = expl((long double)x[k]);
        const double rel = (double)fabsl(((long double)y[k] - want) / want);
        if (rel > worst) { worst = rel; snprintf(msg, sizeof msg, "exp(%.17g)", x[k]); }
    }
    printf("    worst %.2e relative (%.2f ulp) at %s\n", worst, worst / DBL_EPSILON, msg);
    free(x); free(y);
    ASSERT_MSG(worst <= 2.0 * DBL_EPSILON, "an exponential lost more than two ulp");
    PASS();
}

static void test_exp_is_exact_at_zero_and_flushes_below_the_normals(void) {
    const double x[] = {0.0, -0.0, -708.5, -800.0, -INFINITY, -708.3};
    double y[6];
    beam_exp_many(x, y, 6);
    ASSERT_MSG(y[0] == 1.0 && y[1] == 1.0, "exp 0 is not exactly 1");
    ASSERT_MSG(y[2] == 0.0 && y[3] == 0.0 && y[4] == 0.0,
               "below the smallest normal double is not flushed to 0");
    ASSERT_MSG(y[5] >= DBL_MIN && fabs(y[5] / exp(-708.3) - 1.0) < 1e-15,
               "just above the flush the value is wrong");
    PASS();
}

static void test_exp_bits_do_not_depend_on_the_call_size(void) {
    enum { N = 67 };
    double x[N], y1[N], y2[N];
    for (int k = 0; k < N; ++k) x[k] = -0.37 * k * k + 0.25;
    beam_exp_many(x, y1, N);
    for (int k = 0; k < N; ++k) beam_exp_many(x + k, y2 + k, 1);
    ASSERT_MSG(memcmp(y1, y2, sizeof y1) == 0,
               "an exponential's bits moved with how many came with it");
    PASS();
}

int main(void) {
    printf("Vector sine and cosine for the line sweep\n");
    RUN_TEST(test_random_angles_within_two_ulp);
    RUN_TEST(test_near_multiples_of_half_pi);
    RUN_TEST(test_zero_is_exact);
    RUN_TEST(test_bits_do_not_depend_on_the_call_size);
    RUN_TEST(test_beyond_the_reduction_range);
    RUN_TEST(test_exp_within_two_ulp);
    RUN_TEST(test_exp_is_exact_at_zero_and_flushes_below_the_normals);
    RUN_TEST(test_exp_bits_do_not_depend_on_the_call_size);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
