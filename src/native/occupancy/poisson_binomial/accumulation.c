/**
 * @file accumulation.c
 * @brief Truncation, binning and accumulation into level counts.
 */
#include <math.h>
#include <stdlib.h>

#include "beam/publicness.h"
#include "beam/error_macros.h"
#include "beam/occupancy_internal.h"

/* 2*pi to the last representable bit; the codebase spells such constants out
 * rather than relying on M_PI, which is not in C11 proper. */

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
        return BEAM_ENULL("one of pmf, mean, variance, weight, edges or "
                          "out_counts");
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
    if (!group) return BEAM_ENULL("group");
    return publicness_accumulate_impl(
        pmf, n_atoms, k_fft, mean, variance, weight,
        group, n_groups_out, edges, n_bins,
        tail_sigma, tail_floor, mass_tol,
        out_counts, out_retained_mass);
}
