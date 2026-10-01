/**
 * @file lgamma_ratio.c
 * @brief ln Gamma(w) - ln Gamma(w + z) for many real w and complex z.
 *
 * The contour kernels of the exact occupancy engine are built from this ratio
 * at every node of a line and every depth asked for: w = n + 1 from 1 to
 * 1e82, z from the line's real part to far out along it. Taken in numpy, as a
 * ninety-term polygamma series with a full matrix per term, it had become most
 * of the non-sweep time of the figure steps that ask for tens of thousands of
 * depths.
 *
 * METHOD. Stirling's series for the DIFFERENCE, arranged so nothing cancels:
 *
 *   ln G(w+z) - ln G(w) = (w + z - 1/2) L + z ln w - z
 *                         + sum_k c_k w^(1-2k) ((1 + z/w)^(1-2k) - 1),
 *
 * with L = log1p(z / w) and c_k = B_2k / (2k (2k-1)), eight terms. Where
 * w < 15 it is shifted up first, ln G(x) = ln G(x + N) - sum_j ln(x + j),
 * each ln(w + j + z) - ln(w + j) taken as log1p(z / (w + j)). At w = 1e82 the
 * direct difference of two log-gammas would keep no digit of the answer; this
 * keeps them all. The first dropped term is below 1e-20 of w^-15.
 *
 * ACCURACY. A few rounding units, ABSOLUTE, of the ratio's natural size
 * |z| ln(w + |z|) -- which is what the engine's bounds assume of a kernel --
 * and the imaginary part modulo 2 pi, since only its exponential is used.
 * Held to 30 digits by tests/python/analytics/test_log_gamma_ratio.py.
 *
 * Real and imaginary parts are written out: C's complex multiply calls a
 * runtime routine to handle infinities, which cannot arise here.
 */
#include "beam/internal.h"

/* B_2k / (2k (2k - 1)), k = 1 .. 8. */
static const double LGR_C[8] = {
    1.0 / 12.0, -1.0 / 360.0, 1.0 / 1260.0, -1.0 / 1680.0,
    1.0 / 1188.0, -691.0 / 360360.0, 1.0 / 156.0, -3617.0 / 122400.0,
};
#define LGR_W0 15.0

/* log(1 + a + i b), accurate for small a + i b. */
static inline void lgr_log1p(double a, double b, double *re, double *im) {
    *re = 0.5 * log1p(2.0 * a + a * a + b * b);
    *im = atan2(b, 1.0 + a);
}

static void lgr_one(double w, double zr, double zi, double *out_re,
                    double *out_im) {
    if (zr == 0.0 && zi == 0.0) { *out_re = 0.0; *out_im = 0.0; return; }
    double dr = 0.0, di = 0.0;
    double ww = w;
    if (w < LGR_W0) {
        const double N = ceil(LGR_W0 - w);
        for (double j = 0.0; j < N; j += 1.0) {
            double lr, li;
            lgr_log1p(zr / (w + j), zi / (w + j), &lr, &li);
            dr -= lr; di -= li;
        }
        ww = w + N;
    }
    const double ur = zr / ww, ui = zi / ww;
    double Lr, Li;
    lgr_log1p(ur, ui, &Lr, &Li);
    const double ar = ww + zr - 0.5;
    const double lw = log(ww);
    dr += ar * Lr - zi * Li - zr + zr * lw;
    di += ar * Li + zi * Lr - zi + zi * lw;
    /* r = 1 / (1 + u); the corrections c_k w^(1-2k) (r^(2k-1) - 1). */
    const double den = (1.0 + ur) * (1.0 + ur) + ui * ui;
    const double rr = (1.0 + ur) / den, ri = -ui / den;
    const double r2r = rr * rr - ri * ri, r2i = 2.0 * rr * ri;
    double pr = rr, pi = ri, wp = 1.0 / ww;
    const double iw2 = 1.0 / (ww * ww);
    for (int k = 0; k < 8; ++k) {
        dr += LGR_C[k] * wp * (pr - 1.0);
        di += LGR_C[k] * wp * pi;
        const double t = pr * r2r - pi * r2i;
        pi = pr * r2i + pi * r2r;
        pr = t;
        wp *= iw2;
    }
    *out_re = -dr;
    *out_im = -di;
}

void beam_log_gamma_ratio(const double *w, size_t nw, const double *zre,
                          const double *zim, size_t nz, double *out_re,
                          double *out_im) {
    const int64_t total = (int64_t)(nw * nz);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads()) \
            if (total > 4096)
#endif
    for (int64_t x = 0; x < total; ++x) {
        const size_t i = (size_t)x / nw, k = (size_t)x % nw;
        lgr_one(w[k], zre[i], zim[i], out_re + x, out_im + x);
    }
}
