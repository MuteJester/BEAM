/**
 * @file publicness.c
 * @brief Poisson-binomial repertoire-occupancy PMF: moments, PGF, truncation.
 */
#include <math.h>
#include <stdlib.h>

#include "beam/publicness.h"

/* 2*pi to the last representable bit; the codebase spells such constants out
 * rather than relying on M_PI, which is not in C11 proper. */
static const double TWO_PI = 6.283185307179586;

/* ── Shared validation ─────────────────────────────────────── */

static BeamError publicness_check_groups(const double *depths,
                                        const double *multiplicity,
                                        uint32_t n_groups) {
    if (!depths || !multiplicity) return BEAM_ERR_NULL_ARG;
    if (n_groups == 0)
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT, "no depth groups given");

    for (uint32_t b = 0; b < n_groups; b++) {
        if (!(depths[b] >= 0.0) || !isfinite(depths[b]))
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "depth group %u has depth %g; must be finite "
                            "and non-negative", b, depths[b]);
        if (!(multiplicity[b] >= 0.0) ||
            multiplicity[b] != floor(multiplicity[b]) ||
            multiplicity[b] > 4294967295.0)
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "depth group %u has multiplicity %g; must be a "
                            "non-negative integer below 2^32", b,
                            multiplicity[b]);
    }
    return BEAM_OK;
}

static BeamError publicness_check_probability(double p, uint32_t index) {
    if (!(p >= 0.0) || !(p <= 1.0))
        return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                        "probability %g at index %u is outside [0, 1]",
                        p, index);
    return BEAM_OK;
}

/**
 * Detection probability 1 - (1 - p)^N, evaluated through log1p/expm1 so that
 * it keeps its relative accuracy for the p ~ 1e-80 atoms that dominate the
 * counting spectrum. The guards matter: a zero-depth group would otherwise
 * reach 0 * -inf = NaN at p = 1.
 */
BEAM_INLINE double publicness_detection(double p, double depth) {
    double pi;
    if (!(p > 0.0) || !(depth > 0.0)) return 0.0;
    if (p >= 1.0) return 1.0;
    pi = -expm1(depth * log1p(-p));
    if (pi < 0.0) return 0.0;
    return pi > 1.0 ? 1.0 : pi;
}

/* ── Closed-form moments ───────────────────────────────────── */

BeamError beam_publicness_moments(const double *p, uint32_t n_atoms,
                                const double *depths,
                                const double *multiplicity,
                                uint32_t n_groups,
                                double *out_mean, double *out_variance) {
    BeamError err;

    if (!p || !out_mean || !out_variance) return BEAM_ERR_NULL_ARG;
    if (n_atoms == 0) return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT, "no probabilities");

    err = publicness_check_groups(depths, multiplicity, n_groups);
    if (err != BEAM_OK) return err;

    for (uint32_t a = 0; a < n_atoms; a++) {
        double mean = 0.0, variance = 0.0;

        err = publicness_check_probability(p[a], a);
        if (err != BEAM_OK) return err;

        for (uint32_t b = 0; b < n_groups; b++) {
            double pi = publicness_detection(p[a], depths[b]);
            mean += multiplicity[b] * pi;
            variance += multiplicity[b] * pi * (1.0 - pi);
        }

        out_mean[a] = mean;
        out_variance[a] = variance < 0.0 ? 0.0 : variance;
    }

    return BEAM_OK;
}

/* ── Generating function ───────────────────────────────────── */

/**
 * (br + i*bi)^m by binary exponentiation.
 *
 * The obvious alternative -- accumulating m_b * arg(base) across groups and
 * exponentiating once at the end -- is both slower and far less accurate:
 * m_b reaches the repertoire count, so the accumulated angle reaches 1e5
 * radians and carries an absolute error near 1e-11 before cos/sin ever see
 * it. Squaring holds the relative error to about 2*log2(m)*eps. Every factor
 * has modulus at most one, so nothing here can overflow; a factor whose
 * modulus underflows to zero was below 1e-308 and contributes nothing.
 */
static void publicness_cpow(double br, double bi, uint32_t m,
                            double *out_re, double *out_im) {
    double rr = 1.0, ri = 0.0;

    while (m) {
        if (m & 1u) {
            double nr = rr * br - ri * bi;
            double ni = rr * bi + ri * br;
            rr = nr;
            ri = ni;
        }
        m >>= 1;
        if (m) {
            double nr = br * br - bi * bi;
            double ni = 2.0 * br * bi;
            br = nr;
            bi = ni;
        }
    }

    *out_re = rr;
    *out_im = ri;
}

BeamError beam_publicness_pgf(const double *p, uint32_t n_atoms,
                            const double *depths,
                            const double *multiplicity,
                            uint32_t n_groups, uint32_t k_fft,
                            double *out) {
    double *pi = NULL, *root_re = NULL, *root_im = NULL;
    uint32_t n_roots;
    BeamError err;

    if (!p || !out) return BEAM_ERR_NULL_ARG;
    if (n_atoms == 0) return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT, "no probabilities");
    if (k_fft < 2 || (k_fft & 1u))
        return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                        "k_fft=%u must be even and at least 2", k_fft);

    err = publicness_check_groups(depths, multiplicity, n_groups);
    if (err != BEAM_OK) return err;

    n_roots = k_fft / 2u + 1u;
    if ((size_t)n_atoms > SIZE_MAX / (size_t)n_groups / sizeof(double))
        return BEAM_FAIL(BEAM_ERR_ALLOC, "publicness PGF workspace is too large");
#if SIZE_MAX < UINT64_MAX
    if ((size_t)n_roots > SIZE_MAX / sizeof(double))
        return BEAM_FAIL(BEAM_ERR_ALLOC, "publicness root grid is too large");
#endif
    pi = malloc((size_t)n_atoms * (size_t)n_groups * sizeof(double));
    root_re = malloc((size_t)n_roots * sizeof(double));
    root_im = malloc((size_t)n_roots * sizeof(double));
    if (!pi || !root_re || !root_im) {
        free(pi);
        free(root_re);
        free(root_im);
        return BEAM_ERR_ALLOC;
    }

    for (uint32_t a = 0; a < n_atoms; a++) {
        err = publicness_check_probability(p[a], a);
        if (err != BEAM_OK) {
            free(pi);
            free(root_re);
            free(root_im);
            return err;
        }

        for (uint32_t b = 0; b < n_groups; b++)
            pi[(size_t)a * n_groups + b] =
                publicness_detection(p[a], depths[b]);
    }

    /* Roots are common to every atom. Besides avoiding redundant trig, this
     * makes atom rows completely independent and therefore safe to evaluate
     * in parallel when the extension was built with OpenMP. Each thread owns
     * a disjoint output row; the subsequent weighted reduction remains serial
     * and deterministic. */
    for (uint32_t k = 0; k < n_roots; k++) {
        double theta = TWO_PI * (double)k / (double)k_fft;
        root_re[k] = cos(theta);
        root_im[k] = sin(theta);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_atoms >= 4u)
#endif
    for (uint32_t a = 0; a < n_atoms; a++) {
        double *row = out + (size_t)a * 2u * (size_t)k_fft;
        const double *atom_pi = pi + (size_t)a * n_groups;

        /* G has real coefficients, so G(conj z) = conj(G(z)): only the first
         * half of the circle has to be evaluated, and the rest is a mirror. */
        for (uint32_t k = 0; k <= k_fft / 2u; k++) {
            double zr = root_re[k], zi = root_im[k];
            double gr = 1.0, gi = 0.0;

            for (uint32_t b = 0; b < n_groups; b++) {
                double fr, fi, nr, ni;
                uint32_t m = (uint32_t)multiplicity[b];

                if (m == 0u || atom_pi[b] == 0.0) continue;

                publicness_cpow(1.0 - atom_pi[b] + atom_pi[b] * zr,
                                atom_pi[b] * zi, m,
                                &fr, &fi);
                nr = gr * fr - gi * fi;
                ni = gr * fi + gi * fr;
                gr = nr;
                gi = ni;
            }

            row[2u * k] = gr;
            row[2u * k + 1u] = gi;
            if (k > 0u && k < k_fft - k) {
                row[2u * (k_fft - k)] = gr;
                row[2u * (k_fft - k) + 1u] = -gi;
            }
        }
    }

    free(pi);
    free(root_re);
    free(root_im);
    return BEAM_OK;
}

/* ── Truncation and aggregation ────────────────────────────── */

/** First bin whose right edge lies strictly above `level`. */
static uint32_t publicness_first_bin(const double *edges, uint32_t n_bins,
                                     double level) {
    uint32_t lo = 0, hi = n_bins;

    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2u;
        if (edges[mid + 1u] > level) hi = mid; else lo = mid + 1u;
    }
    return lo;
}

static BeamError publicness_accumulate_impl(
    const double *pmf, uint32_t n_atoms, uint32_t k_fft,
    const double *mean, const double *variance, const double *weight,
    const uint32_t *group, uint32_t n_groups_out,
    const double *edges, uint32_t n_bins,
    double tail_sigma, double tail_floor, double mass_tol,
    double *out_counts, double *out_retained_mass) {
    if (!pmf || !mean || !variance || !weight || !edges || !out_counts)
        return BEAM_ERR_NULL_ARG;
    if (n_atoms == 0 || k_fft == 0 || n_bins == 0 || n_groups_out == 0)
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT, "empty batch or bin set");
    if (!(tail_sigma >= 0.0) || !(tail_floor >= 0.0) || !(mass_tol > 0.0))
        return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                        "tail_sigma=%g, tail_floor=%g and mass_tol=%g must be "
                        "non-negative, non-negative and positive",
                        tail_sigma, tail_floor, mass_tol);

    for (uint32_t i = 0; i <= n_bins; i++) {
        if (!(edges[i] >= 0.0) || edges[i] > (double)k_fft ||
            edges[i] != floor(edges[i]))
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "bin edge %u is %g; must be an integer in "
                            "[0, %u]", i, edges[i], k_fft);
        if (i > 0 && edges[i] < edges[i - 1])
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "bin edges must be non-decreasing, but edge %u "
                            "(%g) is below edge %u (%g)",
                            i, edges[i], i - 1, edges[i - 1]);
    }

    for (uint32_t a = 0; a < n_atoms; a++) {
        const double *row = pmf + (size_t)a * (size_t)k_fft;
        double sd, half, retained = 0.0;
        double lo_f, hi_f;
        uint32_t lo, hi, bin;

        if (group && group[a] >= n_groups_out)
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "atom %u has output group %u, but only %u groups "
                            "were allocated", a, group[a], n_groups_out);

        if (!isfinite(mean[a]) || !isfinite(variance[a]) || variance[a] < 0.0)
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "atom %u has mean=%g variance=%g; both must be "
                            "finite and the variance non-negative",
                            a, mean[a], variance[a]);

        /* The window comes from the closed-form moments, never from the PMF
         * itself: outside it the PMF holds nothing but inversion roundoff,
         * and reading the roundoff to decide what to keep would be circular. */
        sd = sqrt(variance[a]);
        half = tail_sigma * sd + tail_floor;
        lo_f = floor(mean[a] - half);
        hi_f = ceil(mean[a] + half) + 1.0;
        lo = lo_f <= 0.0 ? 0u : (lo_f >= (double)k_fft ? k_fft
                                                       : (uint32_t)lo_f);
        hi = hi_f <= (double)lo ? lo : (hi_f >= (double)k_fft ? k_fft
                                                             : (uint32_t)hi_f);

        /* Zeroing outside the window happens by never reading outside it.
         * Clipping negatives only ever applies inside, so the far-tail
         * roundoff is discarded rather than rectified into a positive floor. */
        for (uint32_t j = lo; j < hi; j++)
            if (row[j] > 0.0) retained += row[j];

        if (!(fabs(retained - 1.0) <= mass_tol))
            return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                            "atom %u retains %.12f of its occupancy mass over "
                            "levels [%u, %u) (mean=%g, sd=%g); the truncation "
                            "is discarding real mass, so widen tail_sigma",
                            a, retained, lo, hi, mean[a], sd);

        if (out_retained_mass) out_retained_mass[a] = retained;

        for (bin = publicness_first_bin(edges, n_bins, (double)lo);
             bin < n_bins && edges[bin] < (double)hi; bin++) {
            uint32_t from = edges[bin] > (double)lo ? (uint32_t)edges[bin] : lo;
            uint32_t to = edges[bin + 1u] < (double)hi
                              ? (uint32_t)edges[bin + 1u] : hi;
            double mass = 0.0;

            for (uint32_t j = from; j < to; j++)
                if (row[j] > 0.0) mass += row[j];

            const size_t output_index = group
                ? (size_t)group[a] * n_bins + bin
                : bin;
            out_counts[output_index] += weight[a] * mass;
        }
    }

    return BEAM_OK;
}

BeamError beam_publicness_accumulate(const double *pmf, uint32_t n_atoms,
                                   uint32_t k_fft,
                                   const double *mean, const double *variance,
                                   const double *weight,
                                   const double *edges, uint32_t n_bins,
                                   double tail_sigma, double tail_floor,
                                   double mass_tol,
                                   double *out_counts,
                                   double *out_retained_mass) {
    return publicness_accumulate_impl(
        pmf, n_atoms, k_fft, mean, variance, weight,
        NULL, 1, edges, n_bins, tail_sigma, tail_floor, mass_tol,
        out_counts, out_retained_mass);
}

BeamError beam_publicness_accumulate_grouped(
    const double *pmf, uint32_t n_atoms, uint32_t k_fft,
    const double *mean, const double *variance, const double *weight,
    const uint32_t *group, uint32_t n_groups_out,
    const double *edges, uint32_t n_bins,
    double tail_sigma, double tail_floor, double mass_tol,
    double *out_counts, double *out_retained_mass) {
    if (!group) return BEAM_ERR_NULL_ARG;
    return publicness_accumulate_impl(
        pmf, n_atoms, k_fft, mean, variance, weight,
        group, n_groups_out, edges, n_bins,
        tail_sigma, tail_floor, mass_tol,
        out_counts, out_retained_mass);
}
