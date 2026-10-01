/**
 * @file ztp_moments.c
 * @brief Tilted zero-truncated-Poisson bin moments, one pass per sequence.
 *
 * Every Newton step of `ExactOccupancy`-style repertoire scoring
 * (BEAM/likelihood/fixed_total.py) needs, for each sequence the repertoire
 * saw, the tilted probability that its count falls in each abundance bin and
 * the first two moments of that count. The numpy form built a
 * (sequences x counts) matrix per step and scattered it into bins with
 * `np.add.at`: on a 100,000-sequence repertoire with bins up to 101 reads
 * that was 11 of the 12 seconds a donor cost, and all of it was thrown away
 * to produce eight numbers per sequence.
 *
 * Here each sequence walks its counts ONCE, accumulating straight into its
 * bins, and only the sums over sequences are returned -- which is all the
 * caller ever used. No matrix is formed.
 *
 * THE WEIGHTS. A sequence seen at least once, with Poisson mean `lam`,
 * tilted by `tau` on the count and `t_b` on its bin, gives count r the
 * weight mu^r / r! e^{t_b(r)}, mu = lam e^tau, normalised over r >= 1. The
 * powers come from the recursion u_{r+1} = u_r mu / (r+1), one multiply per
 * count, started from exp(log mu - m) where m is the series' own peak -- so
 * the largest term is about 1 and nothing overflows, whatever mu is. The
 * bin tilts enter as B exponentials per call, not one per count.
 *
 * THE OPEN TOP BIN is summed by the same recursion past the last edge rather
 * than from an incomplete gamma, until the terms are negligible past the
 * peak. That is where the numpy form spent its third second (scipy's
 * `gammainc`, three calls per step).
 *
 * THE SAME SUMS WHATEVER THE THREAD COUNT. Sequences are reduced in fixed
 * chunks, each chunk summed alone and the chunks added in order, so the
 * answer does not depend on how many threads ran.
 */
#include "beam/internal.h"

#define ZTP_CHUNK 4096u          /* must match BEAM_ZTP_MAX_BINS below */          /* sequences per deterministic partial sum */
#define ZTP_TINY  1e-18          /* a term this far below the peak is dropped */
#define ZTP_RMAX  2000000u       /* hard cap on the open bin's walk */
#define BEAM_ZTP_OPEN 0xffffffffu /* the last bin's upper count: unbounded */

/* log(e^x - 1), without overflowing for large x. */
static inline double ztp_log_expm1(double x) {
    return x > 30.0 ? x + log1p(-exp(-x)) : log(expm1(x));
}

/* Each count's bin, and each bin's lowest and highest count. */
static void ztp_bin_bounds(const double *edges, uint32_t B, uint32_t top,
                           uint32_t *rbin, uint32_t *blo, uint32_t *bhi) {
    for (uint32_t r = 1u; r <= top; ++r) {
        uint32_t b = 0;
        while (b + 1u < B && r >= (uint32_t)edges[b + 1u]) ++b;
        rbin[r] = b;
    }
    for (uint32_t b = 0; b < B; ++b) {
        blo[b] = (uint32_t)edges[b];
        bhi[b] = b + 1u < B ? (uint32_t)edges[b + 1u] - 1u : BEAM_ZTP_OPEN;
    }
}

/* The peak of r log mu - lgamma(r+1) over integer r >= 1, as a bound. */
static inline double ztp_peak(double mu, double log_mu) {
    if (!(mu > 1.0)) return log_mu;                  /* r = 1 */
    const double lo = floor(mu), hi = lo + 1.0;
    const double a = lo * log_mu - lgamma(lo + 1.0);
    const double b = hi * log_mu - lgamma(hi + 1.0);
    return a > b ? a : b;
}

/* One sequence's bin sums. `et` holds e^{t_b - tmax} per bin, zero where the
 * bin is not allowed; `blo`/`bhi` are each bin's lowest and highest count,
 * the last bin's `bhi` being BEAM_ZTP_OPEN.
 *
 * THE WINDOW. The terms mu^r / r! are a Poisson shape peaking at r = mu with
 * width sqrt(mu), so starting at r = 1 can start a thousand orders of
 * magnitude below the peak: at mu = 1058 the first term is exp(-1041), which
 * underflows to zero and silently loses the whole sequence. The walk is
 * therefore centred on the largest term the ALLOWED bins can reach -- not on
 * the peak, since the normaliser runs over the allowed counts only, and a
 * profile with no large-count bin leaves a big clone's mass entirely in its
 * far lower tail -- and runs twelve standard deviations either side, where
 * the term is about exp(-72) of the largest whatever mu is.
 */
static void ztp_one(double lam, double tau, const uint32_t *rbin,
                    const uint32_t *blo, const uint32_t *bhi, uint32_t top,
                    uint32_t B, const double *et, double peak_shift,
                    double *pb, double *rb, double *logZ, double *ec,
                    double *ec2) {
    const double mu = lam * exp(tau);
    if (!(mu > 0.0)) return;
    const double log_mu = log(mu);
    const double sd = sqrt(mu) + 1.0;
    const double W = ceil(12.0 * sd) + 40.0;

    /* the largest term an allowed bin can reach, and where */
    double m = -INFINITY;
    uint32_t r_pk = 1u, a_lo = 0u, a_hi = 0u;
    int open = 0;
    for (uint32_t b = 0; b < B; ++b) {
        if (et[b] <= 0.0) continue;
        const uint32_t lo = blo[b];
        const uint32_t hi = bhi[b];
        double c = mu < (double)lo ? (double)lo : floor(mu);
        if (hi != BEAM_ZTP_OPEN && c > (double)hi) c = (double)hi;
        if (c < 1.0) c = 1.0;
        for (int k = 0; k < 2; ++k) {          /* floor and ceiling of the clamp */
            const double rr = c + (double)k;
            if (rr < (double)lo) continue;
            if (hi != BEAM_ZTP_OPEN && rr > (double)hi) continue;
            const double f = rr * log_mu - lgamma(rr + 1.0);
            if (f > m) { m = f; r_pk = (uint32_t)rr; }
        }
        if (a_lo == 0u || lo < a_lo) a_lo = lo;
        if (hi == BEAM_ZTP_OPEN) open = 1;
        else if (hi > a_hi) a_hi = hi;
    }
    if (!isfinite(m)) return;                  /* no bin can hold it */

    uint32_t r_start = (double)r_pk > W ? (uint32_t)((double)r_pk - W) : 1u;
    if (r_start < a_lo) r_start = a_lo;
    if (r_start < 1u) r_start = 1u;
    double fhi = (double)r_pk + W;
    if (!open && (double)a_hi < fhi) fhi = (double)a_hi;
    if (fhi > (double)ZTP_RMAX) fhi = (double)ZTP_RMAX;
    const uint32_t r_end = (uint32_t)fhi;

    double u = exp((double)r_start * log_mu - lgamma((double)r_start + 1.0) - m);
    double z = 0.0, s1 = 0.0, s2 = 0.0, log_scale = 0.0;
    for (uint32_t r = r_start; r <= r_end; ++r) {
        const uint32_t b = r < top ? rbin[r] : B - 1u;
        if (et[b] > 0.0) {
            const double w = u * et[b];
            const double rw = (double)r * w;
            z += w; s1 += rw; s2 += (double)r * rw;
            pb[b] += w; rb[b] += rw;
        }
        if ((double)r > mu && u < ZTP_TINY) break;
        u *= mu / (double)(r + 1u);
        if (u > 1e250) {                       /* keep the walk in range */
            const double f = 1e-250;
            u *= f; z *= f; s1 *= f; s2 *= f;
            for (uint32_t k = 0; k < B; ++k) { pb[k] *= f; rb[k] *= f; }
            log_scale += log(1e250);
        }
    }
    if (!(z > 0.0)) return;                          /* every bin refused it */
    const double inv = 1.0 / z;
    for (uint32_t b = 0; b < B; ++b) { pb[b] *= inv; rb[b] *= inv; }
    *logZ = m + peak_shift + log_scale + log(z) - ztp_log_expm1(lam);
    *ec = s1 * inv;
    *ec2 = s2 * inv;
}

/*
 * out, in order: sum logZ, sum of the bin probabilities (B), sum of
 * E[count 1_bin] (B), sum E[count], sum E[count^2], then the (B+1)x(B+1)
 * Gram matrix of (bin probabilities, E[count]) summed over sequences, which
 * is the Newton step's outer-product term.
 */
void beam_ztp_bin_reduce(const double *lam, size_t n, const double *t_bins,
                         const double *edges, const double *allowed,
                         size_t nb, double tau, double *out) {
    const uint32_t B = (uint32_t)nb;
    const uint32_t top = (uint32_t)edges[nb - 1u];
    const size_t width = 2u * (size_t)B + 3u + (size_t)(B + 1u) * (B + 1u);
    memset(out, 0, width * sizeof(double));
    if (n == 0u || B == 0u || B > BEAM_ZTP_MAX_BINS) return;

    uint32_t *rbin = (uint32_t *)malloc((size_t)(top + 1u) * sizeof(uint32_t));
    uint32_t *blo = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
    uint32_t *bhi = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
    double *et = (double *)malloc((size_t)B * sizeof(double));
    const size_t nchunk = (n + ZTP_CHUNK - 1u) / ZTP_CHUNK;
    double *part = (double *)calloc(nchunk * width, sizeof(double));
    if (!rbin || !blo || !bhi || !et || !part) {
        free(rbin); free(blo); free(bhi); free(et); free(part); return;
    }
    ztp_bin_bounds(edges, B, top, rbin, blo, bhi);
    double tmax = -INFINITY;
    for (uint32_t b = 0; b < B; ++b)
        if (allowed[b] != 0.0 && t_bins[b] > tmax) tmax = t_bins[b];
    if (!isfinite(tmax)) { free(rbin); free(et); free(part); return; }
    for (uint32_t b = 0; b < B; ++b)
        et[b] = allowed[b] != 0.0 ? exp(t_bins[b] - tmax) : 0.0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads()) \
            if (n > ZTP_CHUNK)
#endif
    for (int64_t c = 0; c < (int64_t)nchunk; ++c) {
        double *acc = part + (size_t)c * width;
        double *gram = acc + 2u * (size_t)B + 3u;
        double pb[BEAM_ZTP_MAX_BINS], rb[BEAM_ZTP_MAX_BINS], s[BEAM_ZTP_MAX_BINS + 1u];
        const size_t lo = (size_t)c * ZTP_CHUNK;
        const size_t hi = lo + ZTP_CHUNK < n ? lo + ZTP_CHUNK : n;
        for (size_t i = lo; i < hi; ++i) {
            double logZ = 0.0, ec = 0.0, ec2 = 0.0;
            for (uint32_t b = 0; b < B; ++b) { pb[b] = 0.0; rb[b] = 0.0; }
            ztp_one(lam[i], tau, rbin, blo, bhi, top, B, et, tmax, pb, rb,
                    &logZ, &ec, &ec2);
            acc[0] += logZ;
            for (uint32_t b = 0; b < B; ++b) {
                acc[1u + b] += pb[b];
                acc[1u + B + b] += rb[b];
                s[b] = pb[b];
            }
            s[B] = ec;
            acc[1u + 2u * B] += ec;
            acc[2u + 2u * B] += ec2;
            for (uint32_t j = 0; j <= B; ++j)
                for (uint32_t k = 0; k <= B; ++k)
                    gram[(size_t)j * (B + 1u) + k] += s[j] * s[k];
        }
    }
    for (size_t c = 0; c < nchunk; ++c)          /* in chunk order, always */
        for (size_t j = 0; j < width; ++j)
            out[j] += part[c * width + j];
    free(rbin); free(blo); free(bhi); free(et); free(part);
}

/*
 * E[how many sequences fall in each bin | the set seen, `total` reads]:
 * the tilt that makes the expected read total `total`, then each sequence's
 * bin probabilities reweighted by the chance the others make up the rest,
 * as a normal in the sequence's own count. Returns B numbers.
 */
void beam_ztp_bin_expect(const double *lam, size_t n, const double *edges,
                         size_t nb, double total, double *out_F) {
    const uint32_t B = (uint32_t)nb;
    const uint32_t top = (uint32_t)edges[nb - 1u];
    memset(out_F, 0, nb * sizeof(double));
    if (n == 0u || B == 0u || B > BEAM_ZTP_MAX_BINS) return;
    const size_t width = 2u * (size_t)B + 3u + (size_t)(B + 1u) * (B + 1u);
    double *red = (double *)malloc(width * sizeof(double));
    double *t0 = (double *)calloc((size_t)B, sizeof(double));
    double *ones = (double *)malloc((size_t)B * sizeof(double));
    uint32_t *rbin = (uint32_t *)malloc((size_t)(top + 1u) * sizeof(uint32_t));
    uint32_t *blo = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
    uint32_t *bhi = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
    const size_t nchunk = (n + ZTP_CHUNK - 1u) / ZTP_CHUNK;
    double *part = (double *)calloc(nchunk * (size_t)B, sizeof(double));
    if (!red || !t0 || !ones || !rbin || !blo || !bhi || !part) {
        free(red); free(t0); free(ones); free(rbin); free(blo); free(bhi);
        free(part); return;
    }
    for (uint32_t b = 0; b < B; ++b) ones[b] = 1.0;
    ztp_bin_bounds(edges, B, top, rbin, blo, bhi);
    /* the tilt that puts the expected read total on `total` */
    double tau = 0.0, V = 0.0;
    for (int it = 0; it < 200; ++it) {
        beam_ztp_bin_reduce(lam, n, t0, edges, ones, nb, tau, red);
        const double ec = red[1u + 2u * B], ec2 = red[2u + 2u * B];
        V = ec2 - red[2u * (size_t)B + 3u + (size_t)B * (B + 1u) + B];
        const double f = ec - total;
        if (fabs(f) <= 1e-9 * (total > 1.0 ? total : 1.0) || !(V > 0.0)) break;
        double step = f / V;
        if (step > 20.0) step = 20.0;
        if (step < -20.0) step = -20.0;
        tau -= step;
    }
    if (!(V > 0.0)) V = 1e-300;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads()) \
            if (n > ZTP_CHUNK)
#endif
    for (int64_t c = 0; c < (int64_t)nchunk; ++c) {
        double *acc = part + (size_t)c * (size_t)B;
        double pb[BEAM_ZTP_MAX_BINS], rb[BEAM_ZTP_MAX_BINS];
        const size_t lo = (size_t)c * ZTP_CHUNK;
        const size_t hi = lo + ZTP_CHUNK < n ? lo + ZTP_CHUNK : n;
        for (size_t i = lo; i < hi; ++i) {
            double logZ = 0.0, ec = 0.0, ec2 = 0.0;
            for (uint32_t b = 0; b < B; ++b) { pb[b] = 0.0; rb[b] = 0.0; }
            ztp_one(lam[i], tau, rbin, blo, bhi, top, B, ones, 0.0, pb, rb,
                    &logZ, &ec, &ec2);
            const double v = ec2 - ec * ec;
            double rest = V - v;
            if (!(rest > 1e-300)) rest = 1e-300;
            /* the open bin stands at its own mean count */
            const double pt = pb[B - 1u] - 0.0;
            const double r_top = pt > 0.0 ? rb[B - 1u] / pt : 0.0;
            /* walk the counts again, now weighted by the normal factor;
             * the open bin stands as one atom at its own mean count */
            const double mu = lam[i] * exp(tau);
            double wb[BEAM_ZTP_MAX_BINS];
            for (uint32_t b = 0; b < B; ++b) wb[b] = 0.0;
            double zz = 0.0;
            if (mu > 0.0) {
                const double log_mu = log(mu);
                const double peak = ztp_peak(mu, log_mu);
                const double sd = sqrt(mu) + 1.0;
                const double flo = floor(mu - 12.0 * sd) - 1.0;
                const double fhi = ceil(mu + 12.0 * sd) + 40.0;
                const uint32_t r_lo = flo < 1.0 ? 1u : (uint32_t)flo;
                const uint32_t r_hi = fhi > (double)ZTP_RMAX ? ZTP_RMAX : (uint32_t)fhi;
                double u = exp((double)r_lo * log_mu - lgamma((double)r_lo + 1.0) - peak);
                double tail = 0.0;
                for (uint32_t r = r_lo; r <= r_hi; ++r) {
                    if (r < top) {
                        const double d = (double)r - ec;
                        const double w = u * exp(-d * d / (2.0 * rest));
                        wb[rbin[r]] += w;
                        zz += w;
                    } else {
                        tail += u;
                    }
                    if ((double)r > mu && u < ZTP_TINY) break;
                    u *= mu / (double)(r + 1u);
                }
                if (tail > 0.0) {
                    const double d = r_top - ec;
                    const double w = tail * exp(-d * d / (2.0 * rest));
                    wb[B - 1u] += w;
                    zz += w;
                }
            }
            if (zz > 0.0)
                for (uint32_t b = 0; b < B; ++b) acc[b] += wb[b] / zz;
        }
    }
    for (size_t c = 0; c < nchunk; ++c)
        for (uint32_t b = 0; b < B; ++b)
            out_F[b] += part[c * (size_t)B + b];
    free(red); free(t0); free(ones); free(rbin); free(blo); free(bhi); free(part);
}

/* ── The clone-size spectrum, exactly ──────────────────────────── */

/*
 * E[F_k] and Var[F_k] for each requested count k: how many of the sequences
 * the repertoire was seen to hold the model expects at exactly k reads.
 *
 * Given the set seen, each sequence's count is an independent zero-truncated
 * Poisson, so these are plain sums over sequences -- no tilt, no saddlepoint,
 * no bins:
 *
 *     p_ik = mu_i^k e^-mu_i / (k! (1 - e^-mu_i))
 *     E[F_k] = sum_i p_ik           Var[F_k] = sum_i p_ik (1 - p_ik)
 *
 * which is why this is exact where the chain's binned profile piece is not.
 * Summed in fixed chunks combined in order, so the thread count cannot
 * change the answer.
 */
void beam_ztp_count_spectrum(const double *lam, size_t n, const double *ks,
                             size_t nk, double *out_e, double *out_v) {
    memset(out_e, 0, nk * sizeof(double));
    memset(out_v, 0, nk * sizeof(double));
    if (n == 0u || nk == 0u) return;
    double *lgam = (double *)malloc(nk * sizeof(double));
    const size_t nchunk = (n + ZTP_CHUNK - 1u) / ZTP_CHUNK;
    double *part = (double *)calloc(nchunk * nk * 2u, sizeof(double));
    if (!lgam || !part) { free(lgam); free(part); return; }
    for (size_t j = 0; j < nk; ++j) lgam[j] = lgamma(ks[j] + 1.0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads()) \
            if (n > ZTP_CHUNK)
#endif
    for (int64_t c = 0; c < (int64_t)nchunk; ++c) {
        double *pe = part + (size_t)c * nk * 2u;
        double *pv = pe + nk;
        const size_t lo = (size_t)c * ZTP_CHUNK;
        const size_t hi = lo + ZTP_CHUNK < n ? lo + ZTP_CHUNK : n;
        for (size_t i = lo; i < hi; ++i) {
            const double mu = lam[i];
            if (!(mu > 0.0)) continue;
            const double log_mu = log(mu);
            const double lz = ztp_log_expm1(mu) - mu;   /* log(1 - e^-mu) */
            for (size_t j = 0; j < nk; ++j) {
                const double l = ks[j] * log_mu - mu - lgam[j] - lz;
                if (l < -745.0) continue;               /* nothing to add */
                const double p = exp(l);
                pe[j] += p;
                pv[j] += p * (1.0 - p);
            }
        }
    }
    for (size_t c = 0; c < nchunk; ++c)            /* in chunk order, always */
        for (size_t j = 0; j < nk; ++j) {
            out_e[j] += part[c * nk * 2u + j];
            out_v[j] += part[c * nk * 2u + nk + j];
        }
    free(lgam); free(part);
}
