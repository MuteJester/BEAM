/**
 * @file pgf.c
 * @brief The probability generating function at roots of unity.
 */
#include <math.h>
#include <stdlib.h>

#include "beam/publicness.h"
#include "beam/occupancy_internal.h"

/* 2*pi to the last representable bit; the codebase spells such constants out
 * rather than relying on M_PI, which is not in C11 proper. */

static const double TWO_PI = 6.283185307179586;

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
