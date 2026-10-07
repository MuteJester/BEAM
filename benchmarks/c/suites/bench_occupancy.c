#include "bench_framework.h"
#include "beam/publicness.h"

/*
 * Poisson-binomial publicness kernels on synthetic atoms: a cohort of depth
 * groups and a batch of sequence probabilities, sized like one batch of a
 * publicness run. No model is involved.
 */

#define N_GROUPS 10u
static const double DEPTHS[N_GROUPS] = {1e3, 2e3, 5e3, 1e4, 2e4, 3e4, 4e4, 5e4, 7.5e4, 1e5};
static const double MULT[N_GROUPS] = {5, 8, 12, 15, 10, 8, 6, 4, 2, 1};   /* 71 repertoires */

BEAM_BENCHMARK(occupancy, moments, "Occupancy mean and variance for 1,000 atoms over a 71-repertoire cohort") {
    const uint32_t n = 1000u;
    BENCH_SCRATCH(ctx, double, buf, 3u * (size_t)n);
    double *p = buf, *mean = buf + n, *var = buf + 2u * n;
    for (uint32_t i = 0; i < n; ++i) p[i] = 1e-7 * (double)(i + 1u);
    bench_set_items(ctx, n, "atoms");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_publicness_moments(p, n, DEPTHS, MULT, N_GROUPS, mean, var));
    }
}

BEAM_BENCHMARK(occupancy, pgf, "PGF at 256 roots of unity for 100 atoms (71 repertoires)") {
    const uint32_t n = 100u, k_fft = 256u;
    BENCH_SCRATCH(ctx, double, buf, (size_t)n + (size_t)n * k_fft * 2u);
    double *p = buf, *out = buf + n;
    for (uint32_t i = 0; i < n; ++i) p[i] = 1e-6 * (double)(i + 1u);
    bench_set_items(ctx, n, "atoms");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_publicness_pgf(p, n, DEPTHS, MULT, N_GROUPS, k_fft, out));
    }
}

/* PMFs that are a point mass at 50 repertoires, with matching moments, so
 * every atom's retained mass is exactly 1 and the tolerance check passes. */
typedef struct AccumInput {
    double *pmf, *mean, *var, *weight, *out;
    uint32_t *group;
    double edges[21];
} AccumInput;

static bool accum_input(BenchContext *ctx, AccumInput *in, uint32_t n, uint32_t k_fft, uint32_t n_out) {
    const size_t doubles = (size_t)n * k_fft + 4u * (size_t)n + n_out;
    double *buf = (double *)bench_scratch(ctx, doubles * sizeof(double) + n * sizeof(uint32_t));
    if (!buf) return false;
    in->pmf = buf;
    in->mean = in->pmf + (size_t)n * k_fft;
    in->var = in->mean + n;
    in->weight = in->var + n;
    in->out = in->weight + n;
    in->group = (uint32_t *)(in->out + n_out);
    for (uint32_t i = 0; i <= 20u; ++i) in->edges[i] = 10.0 * i;
    for (uint32_t a = 0; a < n; ++a) {
        in->mean[a] = 50.0;
        in->var[a] = 10.0;
        in->weight[a] = 1.0;
        in->group[a] = a % 4u;
        in->pmf[(size_t)a * k_fft + 50u] = 1.0;
    }
    return true;
}

BEAM_BENCHMARK(occupancy, accumulate, "Truncate, check and bin 500 occupancy PMFs (k = 256, 20 bins)") {
    const uint32_t n = 500u, k_fft = 256u, n_bins = 20u;
    AccumInput in;
    if (!accum_input(ctx, &in, n, k_fft, n_bins)) return;
    bench_set_items(ctx, n, "atoms");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_publicness_accumulate(in.pmf, n, k_fft, in.mean, in.var, in.weight,
                                                    in.edges, n_bins, BEAM_PUBLICNESS_TAIL_SIGMA,
                                                    BEAM_PUBLICNESS_TAIL_FLOOR, 1e-4, in.out, NULL));
    }
}

BEAM_BENCHMARK(occupancy, accumulate_grouped, "As accumulate, binned into 4 groups") {
    const uint32_t n = 500u, k_fft = 256u, n_bins = 20u, n_groups = 4u;
    AccumInput in;
    if (!accum_input(ctx, &in, n, k_fft, n_groups * n_bins)) return;
    bench_set_items(ctx, n, "atoms");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_publicness_accumulate_grouped(in.pmf, n, k_fft, in.mean, in.var, in.weight,
                                                            in.group, n_groups, in.edges, n_bins,
                                                            BEAM_PUBLICNESS_TAIL_SIGMA,
                                                            BEAM_PUBLICNESS_TAIL_FLOOR, 1e-4, in.out, NULL));
    }
}
