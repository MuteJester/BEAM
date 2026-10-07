#include "bench_framework.h"
#include "beam/mixture.h"

/*
 * Every mixture benchmark runs on TWO DISTINCT models of the workload's own
 * configuration (trained on different corpora) at the workload's length L.
 * The first version used a fixed order-3 model at L = 12 whatever the run was
 * configured for, which understated these costs about 60-fold at K=4, L=27.
 */

#define BENCH_PAIR(ctx, models)                                                              \
    BENCH_MODEL_VARIANT(ctx, _ma, BENCH_VARIANT_PRIMARY);                                    \
    BENCH_MODEL_VARIANT(ctx, _mb, BENCH_VARIANT_SECOND);                                     \
    const BeamModel *models[2] = {_ma, _mb}

BEAM_BENCHMARK(mixture, inner_product, "log <P_A, P_B> over sequences of length L") {
    BENCH_PAIR(ctx, models);
    double out = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_log_inner_product(models, 2, ctx->length, &out));
    }
    BENCH_KEEP(out);
}

BEAM_BENCHMARK(mixture, inner_product_by_length, "log <P_A, P_B> for every length and in total") {
    BENCH_PAIR(ctx, models);
    BENCH_SCRATCH(ctx, double, by_length, (size_t)ctx->max_length + 1u);
    double total = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_log_inner_product_by_length(models, 2, by_length, &total));
    }
    BENCH_KEEP(total);
}

BEAM_BENCHMARK(mixture, power_product, "Bhattacharyya product sum P_A^0.5 P_B^0.5 at length L") {
    BENCH_PAIR(ctx, models);
    const double expo[2] = {0.5, 0.5};
    double out = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_log_power_product(models, expo, 2, ctx->length, &out));
    }
    BENCH_KEEP(out);
}

BEAM_BENCHMARK(mixture, power_product_by_length, "Bhattacharyya product for every length and in total") {
    BENCH_PAIR(ctx, models);
    const double expo[2] = {0.5, 0.5};
    BENCH_SCRATCH(ctx, double, by_length, (size_t)ctx->max_length + 1u);
    double total = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_log_power_product_by_length(models, expo, 2, by_length, &total));
    }
    BENCH_KEEP(total);
}

BEAM_BENCHMARK(mixture, power_product_by_position, "Bhattacharyya product decomposed by position at length L") {
    BENCH_PAIR(ctx, models);
    const double expo[2] = {0.5, 0.5};
    BENCH_SCRATCH(ctx, double, by_pos, ctx->length);
    double sink = 0.0, konst = 0.0, total = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_log_power_product_by_position(models, expo, 2, ctx->length,
                                                                    by_pos, &sink, &konst, &total));
    }
    BENCH_KEEP(total);
}

BEAM_BENCHMARK(mixture, gram_2x2, "Inner-product Gram matrix of two models, every length") {
    BENCH_PAIR(ctx, models);
    double gram[4] = {0};
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_gram(models, 2, gram));
    }
    BENCH_KEEP(gram[1]);
}

BEAM_BENCHMARK(mixture, power_gram_2x2, "Bhattacharyya (a = 0.5) Gram matrix of two models, every length") {
    BENCH_PAIR(ctx, models);
    double gram[4] = {0};
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_power_gram(models, 2, 0.5, gram));
    }
    BENCH_KEEP(gram[1]);
}

BEAM_BENCHMARK(mixture, power_pairs, "Bhattacharyya products for two listed pairs, every length") {
    BENCH_PAIR(ctx, models);
    const uint32_t pairs[4] = {0, 1, 1, 1};
    double out[2] = {0};
    bench_set_items(ctx, 2, "pairs");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_power_pairs(models, 2, 0.5, pairs, 2, out));
    }
    BENCH_KEEP(out[0]);
}

BEAM_BENCHMARK(mixture, support_union, "Size of the union of two models' supports at length L") {
    BENCH_PAIR(ctx, models);
    double out = 0.0;
    uint64_t peak = 0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_support_union(models, 2, ctx->length, 0, &out, &peak));
    }
    BENCH_KEEP(out);
}

BEAM_BENCHMARK(mixture, support_union_by_length, "Support union for every length and in total") {
    BENCH_PAIR(ctx, models);
    BENCH_SCRATCH(ctx, double, by_length, (size_t)ctx->max_length + 1u);
    double total = 0.0;
    uint64_t peak = 0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_support_union_by_length(models, 2, 0, by_length, &total, &peak));
    }
    BENCH_KEEP(total);
}

BEAM_BENCHMARK(mixture, check_shape, "Shape and finalization check across two models") {
    BENCH_PAIR(ctx, models);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_mixture_check_shape(models, 2));
    }
}
