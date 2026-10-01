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

/* ── A line's integral and tail for a Beta-type kernel ─────────── */

/*
 * The contour sums of `ExactOccupancy._integrate_many` for kernels of the form
 *     K(z) = exp(lg(z) + ln G(w) - ln G(w + z)),   lg = ln G(z + s), z + s given,
 * over a line's nodes t: for each parameter k,
 *     val[k]  = sum_t wt_t Re(K M_t)
 *     absi[k] = sum_t wt_t |K M_t| (1 + t ln n_k)      (the rounding of the sum)
 *     absk[k] = sum_t wt_t |K| dph_t                   (the rounding of M)
 * in log space -- K and M are never formed on their own, so neither can
 * overflow where their product does not -- and with no matrix: each parameter
 * is one thread's loop over the nodes, in node order, so the sums do not
 * depend on the thread count.
 */
void beam_beta_contour_sums(size_t nt, const double *zr, const double *zi,
                            const double *lgr, const double *lgi,
                            const double *lmr, const double *lmi,
                            const double *wt, const double *tt,
                            const double *dph, size_t nk, const double *w,
                            const double *lnns, double *out_val,
                            double *out_absi, double *out_absk) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16) num_threads(beam_threads()) \
            if (nk * nt > 4096)
#endif
    for (int64_t k = 0; k < (int64_t)nk; ++k) {
        double sv = 0.0, sa = 0.0, sk = 0.0;
        for (size_t t = 0; t < nt; ++t) {
            double rr, ri;
            lgr_one(w[k], zr[t], zi[t], &rr, &ri);
            const double er = lgr[t] + rr, ei = lgi[t] + ri;
            const double absk = exp(er);
            const double absi = exp(er + lmr[t]);
            sv += wt[t] * absi * cos(ei + lmi[t]);
            sa += wt[t] * absi * (1.0 + tt[t] * lnns[k]);
            sk += wt[t] * absk * dph[t];
        }
        out_val[k] = sv; out_absi[k] = sa; out_absk[k] = sk;
    }
}

/*
 * `ExactOccupancy._kernel_tail_many` for the same kernels: from |K| at the
 * points ts (beyond a line's reach, ascending), per parameter, the area under
 * the suffix maximum of |K| up to the last point, plus a closing term that
 * extends the last decade's decay to infinity -- or inf where that decay is
 * slower than 1/t, and the area alone where |K| has vanished. A non-finite
 * |K| counts as inf. `lgr` is Re lg at the points.
 */
void beam_beta_contour_tail(size_t nt, const double *zr, const double *zi,
                            const double *lgr, const double *ts, size_t nk,
                            const double *w, double *out) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16) num_threads(beam_threads()) \
            if (nk * nt > 4096)
#endif
    for (int64_t k = 0; k < (int64_t)nk; ++k) {
        double env = 0.0, upper = 0.0, env_last = 0.0, env_tail = 0.0;
        for (size_t s = nt; s-- > 0; ) {
            double rr, ri;
            lgr_one(w[k], zr[s], zi[s], &rr, &ri);
            double v = exp(lgr[s] + rr);
            if (!isfinite(v)) v = INFINITY;
            if (s == nt - 1u || v > env) env = v;      /* suffix maximum */
            if (s == nt - 1u) env_last = env;
            if (nt >= 61u && s == nt - 61u) env_tail = env;
            if (s + 1u < nt) upper += env * (ts[s + 1u] - ts[s]);
        }
        const double a = log10((env_tail > 1e-300 ? env_tail : 1e-300)
                               / (env_last > 1e-300 ? env_last : 1e-300));
        if (env_last == 0.0) out[k] = upper;
        else if (a > 1.0) out[k] = upper + env_last * ts[nt - 1u] / (a - 1.0);
        else out[k] = INFINITY;
    }
}

/* ── The ratio's shift in w: the cohort private kernel ─────────── */

/*
 * D(w, d, z) = [ln G(w-d) - ln G(w-d+z)] - [ln G(w) - ln G(w+z)]
 *            = G(w, z) - G(v, z),   v = w - d,   G(x, z) = ln G(x+z) - ln G(x),
 *
 * which `ExactOccupancy.private_at_depth` needs at every node and depth:
 * its kernel is -Gamma(z) exp(ln G(w) - ln G(w+z)) expm1(D). D is of size
 * d |z| / w where each ratio is |z| ln w, so it cannot be their difference;
 * the numpy form took it as a Taylor series in d, a matrix per power. Here
 * Stirling's series for G is differenced term by term, each difference
 * written so that nothing cancels:
 *
 *   (w+z-1/2) log1p(z/w) - (v+z-1/2) log1p(z/v)
 *       = (w+z-1/2) log1p(-d z / (w (v+z))) + d log1p(z/v)
 *   z ln w - z ln v = -z log1p(-d/w)
 *   c_k [w^(1-2k) (r_w^(2k-1) - 1) - v^(1-2k) (r_v^(2k-1) - 1)],  r_x = 1/(1+z/x),
 *
 * with r^m - 1 = (r - 1)(1 + r + ... + r^(m-1)) and r - 1 = -r z/x, so each
 * term keeps its relative digits. The error is a few rounding units of
 * |D| + d |z| / v (tests/python/analytics/test_log_gamma_ratio.py, against
 * mpmath). Needs v >= 15, where the series holds unshifted; below that the
 * result is NaN.
 */
static void lgr_corrections(double x, double zr, double zi, double *cr, double *ci) {
    const double ur = zr / x, ui = zi / x;
    const double den = (1.0 + ur) * (1.0 + ur) + ui * ui;
    const double rr = (1.0 + ur) / den, ri = -ui / den;          /* r = 1/(1+u) */
    const double gr = -(rr * ur - ri * ui), gi = -(rr * ui + ri * ur); /* r - 1 */
    const double r2r = rr * rr - ri * ri, r2i = 2.0 * rr * ri;
    double sr = 1.0, si = 0.0;            /* S_m = 1 + r + ... + r^(m-1), m = 1 */
    double pr = rr, pi = ri;              /* r^m */
    double xp = 1.0 / x;
    const double ix2 = 1.0 / (x * x);
    double accr = 0.0, acci = 0.0;
    for (int k = 0; k < 8; ++k) {
        /* c_k x^(1-2k) (r - 1) S_m */
        const double tr = gr * sr - gi * si, ti = gr * si + gi * sr;
        accr += LGR_C[k] * xp * tr;
        acci += LGR_C[k] * xp * ti;
        /* S_(m+2) = S_m + r^m + r^(m+1); r^(m+2) */
        const double qr = pr * rr - pi * ri, qi = pr * ri + pi * rr;
        sr += pr + qr; si += pi + qi;
        const double t = pr * r2r - pi * r2i;
        pi = pr * r2i + pi * r2r;
        pr = t;
        xp *= ix2;
    }
    *cr = accr; *ci = acci;
}

/* `lb` = log1p(-d / w), which depends on the depth only. */
static void lgr_shift(double w, double d, double lb, double zr, double zi,
                      double *out_re, double *out_im) {
    const double v = w - d;
    if (!(v >= LGR_W0)) { *out_re = NAN; *out_im = NAN; return; }
    if (zr == 0.0 && zi == 0.0) { *out_re = 0.0; *out_im = 0.0; return; }
    /* -d z / (w (v + z)) */
    const double vr = v + zr;
    const double den = vr * vr + zi * zi;
    const double f = -(d / w) / den;
    const double ar = f * (zr * vr + zi * zi), ai = f * (zi * v);
    double l1r, l1i, l2r, l2i;
    lgr_log1p(ar, ai, &l1r, &l1i);
    lgr_log1p(zr / v, zi / v, &l2r, &l2i);
    const double cr0 = w + zr - 0.5;
    double dr = cr0 * l1r - zi * l1i + d * l2r - zr * lb;
    double di = cr0 * l1i + zi * l1r + d * l2i - zi * lb;
    double cwr, cwi, cvr, cvi;
    lgr_corrections(w, zr, zi, &cwr, &cwi);
    lgr_corrections(v, zr, zi, &cvr, &cvi);
    *out_re = dr + (cwr - cvr);
    *out_im = di + (cwi - cvi);
}

void beam_log_gamma_ratio_shift(double w, const double *d, size_t nd,
                                const double *zre, const double *zim, size_t nz,
                                double *out_re, double *out_im) {
    const int64_t total = (int64_t)(nd * nz);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads()) \
            if (total > 4096)
#endif
    for (int64_t x = 0; x < total; ++x) {
        const size_t i = (size_t)x / nd, k = (size_t)x % nd;
        lgr_shift(w, d[k], log1p(-d[k] / w), zre[i], zim[i], out_re + x, out_im + x);
    }
}

/*
 * The contour sums of `ExactOccupancy._integrate_many` for the cohort private
 * kernel K(z) = -B(z) expm1(D(w, d_k, z)), B = Gamma(z) Gamma(w) / Gamma(w+z),
 * given ln B at the nodes (lbr, lbi): the same three sums as
 * `beam_beta_contour_sums`, with |B M| and |B| taken in log space and
 * expm1(D) on its own, so no digit of a small D is lost to the 1 it is
 * added to. One thread's loop per depth, in node order.
 */
void beam_cohort_private_sums(size_t nt, const double *zr, const double *zi,
                              const double *lbr, const double *lbi,
                              const double *lmr, const double *lmi,
                              const double *wt, const double *tt,
                              const double *dph, double w, size_t nk,
                              const double *d, const double *lnns,
                              double *out_val, double *out_absi,
                              double *out_absk) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16) num_threads(beam_threads()) \
            if (nk * nt > 4096)
#endif
    for (int64_t k = 0; k < (int64_t)nk; ++k) {
        const double lb = log1p(-d[k] / w);
        double sv = 0.0, sa = 0.0, sk = 0.0;
        for (size_t t = 0; t < nt; ++t) {
            double ar, ai;
            lgr_shift(w, d[k], lb, zr[t], zi[t], &ar, &ai);
            /* expm1(ar + i ai) */
            const double em = expm1(ar), sh = sin(0.5 * ai);
            const double c = cos(ai), s = sin(ai);
            const double er = em * c - 2.0 * sh * sh;
            const double ei = (em + 1.0) * s;
            const double ae = hypot(er, ei);
            const double bm = exp(lbr[t] + lmr[t]);
            const double ph = lbi[t] + lmi[t];
            sv -= wt[t] * bm * (cos(ph) * er - sin(ph) * ei);
            sa += wt[t] * bm * ae * (1.0 + tt[t] * lnns[k]);
            sk += wt[t] * exp(lbr[t]) * ae * dph[t];
        }
        out_val[k] = sv; out_absi[k] = sa; out_absk[k] = sk;
    }
}
