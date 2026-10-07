#include "bench_framework.h"
#include "beam/rng.h"

/* Both write fixed-stride strings of max_length + 1 bytes, whatever L is. */

BEAM_BENCHMARK(algorithms, sample_100, "Draw 100 sequences of length L exactly in proportion to P") {
    BENCH_MODEL(ctx, m);
    const uint32_t n = 100u;
    BENCH_SCRATCH(ctx, char, out, (size_t)n * ((size_t)ctx->max_length + 1u) + n * sizeof(uint32_t));
    uint32_t *lens = (uint32_t *)(out + (size_t)n * ((size_t)ctx->max_length + 1u));
    BeamRng rng;
    beam_rng_seed(&rng, 42u);
    bench_set_items(ctx, n, "sequences");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_sample(m, n, ctx->length, &rng, out, lens));
    }
}

BEAM_BENCHMARK(algorithms, sample_100_any_length, "Draw 100 sequences, length drawn from pi(L)") {
    BENCH_MODEL(ctx, m);
    const uint32_t n = 100u;
    BENCH_SCRATCH(ctx, char, out, (size_t)n * ((size_t)ctx->max_length + 1u) + n * sizeof(uint32_t));
    uint32_t *lens = (uint32_t *)(out + (size_t)n * ((size_t)ctx->max_length + 1u));
    BeamRng rng;
    beam_rng_seed(&rng, 42u);
    bench_set_items(ctx, n, "sequences");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_sample(m, n, 0u, &rng, out, lens));
    }
}

BEAM_BENCHMARK(algorithms, top_k_20, "The 20 most probable sequences of length L (exact best-first search)") {
    BENCH_MODEL(ctx, m);
    const uint32_t k = 20u;
    /* log probabilities first, so they are aligned; the strings follow. */
    BENCH_SCRATCH(ctx, double, log_p, k + ((size_t)k * ((size_t)ctx->max_length + 1u) + 7u) / 8u);
    char *out = (char *)(log_p + k);
    uint32_t found = 0;
    bench_set_items(ctx, k, "sequences");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_top_k(m, k, ctx->length, out, log_p, &found));
    }
    BENCH_REQUIRE(ctx, found == k, "top_k found %u of %u sequences", found, k);
}
