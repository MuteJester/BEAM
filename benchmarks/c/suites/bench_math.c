#include "bench_framework.h"

/*
 * Whole-model analytics. Most of them sweep every length 1..max_length and
 * acquire each length's live frontier on the way (the cold call builds those;
 * warm calls reuse them). Output sizes follow the header contracts, which
 * depend on the model's max_length W, not on the per-length argument L.
 */

/* Spectrum grids with automatic spacing require bins > W + 9. */
static uint32_t spectrum_bins(const BenchContext *ctx) {
    const uint32_t need = (uint32_t)ctx->max_length + 16u;
    return need > 512u ? need : 512u;
}

BEAM_BENCHMARK(math, log_mellin, "log sum_s P(s)^q at q = 2 (every length)") {
    BENCH_MODEL(ctx, m);
    double out = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_log_mellin(m, 2.0, &out));
    }
    BENCH_KEEP(out);
}

BEAM_BENCHMARK(math, log_mellin_many, "Mellin transform at four real orders q = 0.5, 1.5, 2, 3") {
    BENCH_MODEL(ctx, m);
    const double qs[4] = {0.5, 1.5, 2.0, 3.0};
    double out[4] = {0};
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_log_mellin_many(m, qs, 4, out));
    }
    BENCH_KEEP(out[0]);
}

BEAM_BENCHMARK(math, log_mellin_complex, "Mellin transform at four complex orders 1 + i{0.1, 0.5, 1, 2}") {
    BENCH_MODEL(ctx, m);
    const double q_re[4] = {1.0, 1.0, 1.0, 1.0};
    const double q_im[4] = {0.1, 0.5, 1.0, 2.0};
    double out_re[4] = {0}, out_im[4] = {0};
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_log_mellin_complex(m, q_re, q_im, 4, out_re, out_im));
    }
    BENCH_KEEP(out_re[0] + out_im[0]);
}

BEAM_BENCHMARK(math, log_mellin_line, "Mellin transform along a vertical line, 16 points") {
    BENCH_MODEL(ctx, m);
    double out_re[16] = {0}, out_im[16] = {0};
    bench_set_items(ctx, 16, "points");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_log_mellin_line(m, 1.0, 0.1, 0.1, 16, out_re, out_im));
    }
    BENCH_KEEP(out_re[0] + out_im[0]);
}

BEAM_BENCHMARK(math, entropy, "Exact Shannon entropy of the model distribution") {
    BENCH_MODEL(ctx, m);
    double h = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_entropy(m, &h));
    }
    BENCH_KEEP(h);
}

BEAM_BENCHMARK(math, hill_number, "Hill number at alpha = 2") {
    BENCH_MODEL(ctx, m);
    double d = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_hill_number(m, 2.0, &d));
    }
    BENCH_KEEP(d);
}

BEAM_BENCHMARK(math, surprisal_moments, "Raw surprisal moments 0..4 at q = 1") {
    BENCH_MODEL(ctx, m);
    double out[5] = {0};
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_surprisal_moments(m, 1.0, 4, out));
    }
    BENCH_KEEP(out[1]);
}

BEAM_BENCHMARK(math, surprisal_cumulants, "Surprisal cumulants 1..4 at q = 1") {
    BENCH_MODEL(ctx, m);
    double out[4] = {0};
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_surprisal_cumulants(m, 1.0, 4, out));
    }
    BENCH_KEEP(out[0]);
}

BEAM_BENCHMARK(math, length_derivatives, "Per-length surprisal moments 0..4 at q = 1") {
    BENCH_MODEL(ctx, m);
    BENCH_SCRATCH(ctx, double, out, ((size_t)ctx->max_length + 1u) * 5u);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_length_derivatives(m, 1.0, 4, out));
    }
}

BEAM_BENCHMARK(math, surprisal_tilt, "Log-Mellin, mean and variance of the q = 1.5 tilt in one sweep") {
    BENCH_MODEL(ctx, m);
    double lm = 0, mean = 0, var = 0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_surprisal_tilt(m, 1.5, &lm, &mean, &var));
    }
    BENCH_KEEP(lm + mean + var);
}

BEAM_BENCHMARK(math, surprisal_by_depth, "Surprisal split across positions at q = 1") {
    BENCH_MODEL(ctx, m);
    BENCH_SCRATCH(ctx, double, out, ctx->max_length);
    double lconst = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_surprisal_by_depth(m, 1.0, out, &lconst));
    }
    BENCH_KEEP(lconst);
}

BEAM_BENCHMARK(math, surprisal_range, "Smallest and largest surprisal (min-plus / max-plus over every length)") {
    BENCH_MODEL(ctx, m);
    double lo = 0.0, hi = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_surprisal_range(m, &lo, &hi));
    }
    BENCH_KEEP(lo + hi);
}

BEAM_BENCHMARK(math, topology, "Live states and edges of the trellis at length L") {
    BENCH_MODEL(ctx, m);
    BENCH_SCRATCH(ctx, uint32_t, layer_nodes, (size_t)ctx->length + 1u);
    double nodes = 0.0, edges = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_topology(m, ctx->length, &nodes, &edges, layer_nodes));
    }
    BENCH_KEEP(nodes + edges);
}

BEAM_BENCHMARK(math, length_marginals, "Length distribution: counting, generated, surprisal mean and variance") {
    BENCH_MODEL(ctx, m);
    const size_t n = (size_t)ctx->max_length + 1u;
    BENCH_SCRATCH(ctx, double, c, 4u * n);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_length_marginals(m, c, c + n, c + 2u * n, c + 3u * n));
    }
}

BEAM_BENCHMARK(math, positional_marginals, "Per-position residue marginals at length L (forward-backward)") {
    BENCH_MODEL(ctx, m);
    BENCH_SCRATCH(ctx, double, out, (size_t)ctx->length * beam_model_n_residues(m));
    double log_z = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_positional_marginals(m, ctx->length, NULL, out, &log_z));
    }
    BENCH_KEEP(log_z);
}

BEAM_BENCHMARK(math, spectrum, "Generated-measure surprisal spectrum at length L (auto spacing)") {
    BENCH_MODEL(ctx, m);
    const uint32_t bins = spectrum_bins(ctx);
    BENCH_SCRATCH(ctx, double, w, bins);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_spectrum(m, false, ctx->length, bins, 0.0, w, NULL));
    }
}

BEAM_BENCHMARK(math, spectrum_pair, "Counting and generated spectra at length L together") {
    BENCH_MODEL(ctx, m);
    const uint32_t bins = spectrum_bins(ctx);
    BENCH_SCRATCH(ctx, double, cw, 2u * (size_t)bins);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_spectrum_pair(m, ctx->length, bins, 0.0, cw, cw + bins, NULL));
    }
}

BEAM_BENCHMARK(math, spectra_by_length, "Generated spectrum for every length 1..max_length") {
    BENCH_MODEL(ctx, m);
    const uint32_t bins = spectrum_bins(ctx);
    BENCH_SCRATCH(ctx, double, w, ((size_t)ctx->max_length + 1u) * bins);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_spectra_by_length(m, false, ctx->max_length, bins, 0.0, w, NULL));
    }
}

BEAM_BENCHMARK(math, positional_banded, "Positional marginals split into 2 surprisal bands at length L") {
    BENCH_MODEL(ctx, m);
    const uint32_t bins = spectrum_bins(ctx);
    const double edges[3] = {0.0, 25.0, 50.0};
    const uint32_t n_bands = 2u;
    const size_t lr = (size_t)ctx->length * beam_model_n_residues(m);
    BENCH_SCRATCH(ctx, double, marg, n_bands * lr + n_bands + n_bands * (size_t)ctx->length);
    double *band_mass = marg + n_bands * lr;
    double *pos_mass = band_mass + n_bands;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_positional_banded(m, ctx->length, false, bins, 0.0, edges, 3,
                                                      marg, band_mass, pos_mass, NULL));
    }
}

BEAM_BENCHMARK(math, expected_counts, "Expected cell counts at length L (result allocated and freed)") {
    BENCH_MODEL(ctx, m);
    BEAM_TIMED_BLOCK(ctx) {
        /* The call memsets its output on entry, so it is freed every time;
         * the free is part of what a caller pays for the result. */
        BeamExpectedCounts ec;
        memset(&ec, 0, sizeof ec);
        BENCH_CHECK(ctx, beam_model_expected_counts(m, ctx->length, &ec));
        beam_model_expected_counts_free(&ec);
    }
}
