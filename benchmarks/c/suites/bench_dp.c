#include "bench_framework.h"
#include "beam/dp.h"

/*
 * Trellis kernels. layer_weights and layer_max are internal (beam/dp.h, not a
 * public header) and validate nothing, so the preconditions -- a finalized
 * model, layer < length <= max_length, S*R doubles of output -- are met here.
 *
 * The forward sweep that computes Z(L) runs inside finalize and
 * log_partition_by_layer (model suite); beam_model_log_partition itself is an
 * O(1) read of the stored value and is timed as model::log_partition_lookup.
 */

BEAM_BENCHMARK(dp, layer_weights, "Edge weights of one trellis layer (N- and C-anchored experts combined)") {
    BENCH_MODEL(ctx, m);
    const size_t cells = (size_t)beam_model_n_states(m) * beam_model_n_residues(m);
    BENCH_SCRATCH(ctx, double, out, cells);
    double max_val = 0.0;
    const uint32_t layer = ctx->length / 2u;
    BEAM_TIMED_BLOCK(ctx) {
        beam_layer_weights(m, layer, ctx->length, out, &max_val);
    }
    BENCH_KEEP(max_val);
}

BEAM_BENCHMARK(dp, layer_max, "Largest edge weight of one trellis layer") {
    BENCH_MODEL(ctx, m);
    double max_val = 0.0;
    const uint32_t layer = ctx->length / 2u;
    BEAM_TIMED_BLOCK(ctx) {
        max_val += beam_layer_max(m, layer, ctx->length);
    }
    BENCH_KEEP(max_val);
}

BEAM_BENCHMARK(dp, path_count, "Number of sequences of length L the model can produce") {
    BENCH_MODEL(ctx, m);
    double paths = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        paths += beam_model_path_count(m, ctx->length);
    }
    BENCH_KEEP(paths);
}

BEAM_BENCHMARK(dp, path_count_total, "Support size summed over every length") {
    BENCH_MODEL(ctx, m);
    double total = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        total += beam_model_path_count_total(m);
    }
    BENCH_KEEP(total);
}

BEAM_BENCHMARK(dp, accumulator_max, "Max-plus sweep over every length (bound used by the spectrum grid)") {
    BENCH_MODEL(ctx, m);
    double hi = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        hi += beam_model_accumulator_max(m);
    }
    BENCH_KEEP(hi);
}
