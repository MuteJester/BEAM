#include "bench_framework.h"

#include <stdio.h>
#include <unistd.h>

/* A fresh model of the workload's configuration, observed on a corpus
 * variant `times` times. NULL (benchmark failed) on any error. */
static BeamModel *observed_model(BenchContext *ctx, uint32_t variant, int times) {
    BeamModelConfig cfg;
    bench_workload_config(&cfg);
    BeamModel *m = beam_model_new(&cfg);
    if (!m) {
        bench_fail(ctx, (int)beam_last_error(), "beam_model_new: %s", beam_error_message());
        return NULL;
    }
    size_t n = 0;
    const char *const *seqs = bench_corpus(variant, &n);
    for (int t = 0; t < times; ++t) {
        if (beam_model_observe(m, seqs, NULL, (uint32_t)n) != BEAM_OK) {
            bench_fail(ctx, (int)beam_last_error(), "setup observe: %s", beam_error_message());
            beam_model_destroy(m);
            return NULL;
        }
    }
    return m;
}

static BeamModel *finalized_model(BenchContext *ctx, uint32_t variant) {
    BeamModel *m = observed_model(ctx, variant, 1);
    if (m && beam_model_finalize(m) != BEAM_OK) {
        bench_fail(ctx, (int)beam_last_error(), "setup finalize: %s", beam_error_message());
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/* ── Lifecycle ─────────────────────────────────────────────────── */

BEAM_BENCHMARK(model, new_destroy, "Model allocation, config validation and teardown") {
    BeamModelConfig cfg;
    bench_workload_config(&cfg);
    BEAM_TIMED_BLOCK(ctx) {
        BeamModel *m = beam_model_new(&cfg);
        if (!m && !ctx->failed) bench_fail(ctx, (int)beam_last_error(), "beam_model_new: %s", beam_error_message());
        beam_model_destroy(m);
    }
}

BEAM_BENCHMARK_EX(model, observe_batch, "Count the whole training corpus into an empty model", BENCH_SINGLE_SHOT) {
    BeamModelConfig cfg;
    bench_workload_config(&cfg);
    size_t n = 0;
    const char *const *seqs = bench_corpus(BENCH_VARIANT_PRIMARY, &n);
    BeamModel *m = beam_model_new(&cfg);
    BENCH_REQUIRE(ctx, m, "beam_model_new: %s", beam_error_message());
    bench_set_items(ctx, (double)n, "sequences");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_observe(m, seqs, NULL, (uint32_t)n));
    }
    beam_model_destroy(m);
}

BEAM_BENCHMARK(model, finalize, "Finalize: hierarchical prior estimation, log tables, Z(L) for every length") {
    BeamModel *m = observed_model(ctx, BENCH_VARIANT_PRIMARY, 1);
    if (!m) return;
    /* Repeatable: finalize has no early-out and rebuilds every table. */
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_finalize(m));
    }
    beam_model_destroy(m);
}

BEAM_BENCHMARK(model, set_prior, "Set explicit prior strengths and re-finalize from the counts") {
    BeamModel *m = observed_model(ctx, BENCH_VARIANT_PRIMARY, 1);
    if (!m) return;
    double alpha[BEAM_MAX_ORDER + 1];
    for (uint32_t i = 0; i <= BEAM_MAX_ORDER; ++i) alpha[i] = 1.0 + (double)i;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_set_prior(m, alpha, (uint32_t)ctx->order + 1u, 3.0));
    }
    beam_model_destroy(m);
}

BEAM_BENCHMARK_EX(model, add, "Add one model's counts into another (corpus-sized operands)", BENCH_SINGLE_SHOT) {
    BeamModel *a = observed_model(ctx, BENCH_VARIANT_PRIMARY, 1);
    if (!a) return;
    BeamModel *b = observed_model(ctx, BENCH_VARIANT_SECOND, 1);
    if (!b) { beam_model_destroy(a); return; }
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_add(a, b));
    }
    beam_model_destroy(a);
    beam_model_destroy(b);
}

BEAM_BENCHMARK_EX(model, subtract, "Subtract a cohort's counts (leave-one-out), target holds it twice", BENCH_SINGLE_SHOT) {
    BeamModel *a = observed_model(ctx, BENCH_VARIANT_PRIMARY, 2);
    if (!a) return;
    BeamModel *b = observed_model(ctx, BENCH_VARIANT_PRIMARY, 1);
    if (!b) { beam_model_destroy(a); return; }
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_subtract(a, b));
    }
    beam_model_destroy(a);
    beam_model_destroy(b);
}

BEAM_BENCHMARK_EX(model, posterior, "Dirichlet-multinomial posterior update from a prior model (kappa = 10)", BENCH_SINGLE_SHOT) {
    BeamModel *a = observed_model(ctx, BENCH_VARIANT_PRIMARY, 1);
    if (!a) return;
    BeamModel *prior = observed_model(ctx, BENCH_VARIANT_SECOND, 1);
    if (!prior) { beam_model_destroy(a); return; }
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_posterior(a, prior, 10.0));
    }
    beam_model_destroy(a);
    beam_model_destroy(prior);
}

BEAM_BENCHMARK_EX(model, temper, "Temper a finalized model (t = 1.5): rescale tables, recompute every Z(L)", BENCH_SINGLE_SHOT) {
    BeamModel *m = finalized_model(ctx, BENCH_VARIANT_PRIMARY);
    if (!m) return;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_temper(m, 1.5));
    }
    beam_model_destroy(m);
}

BEAM_BENCHMARK_EX(model, compact, "Release a finalized model's count tables", BENCH_SINGLE_SHOT) {
    BeamModel *m = finalized_model(ctx, BENCH_VARIANT_PRIMARY);
    if (!m) return;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_compact(m));
    }
    beam_model_destroy(m);
}

BEAM_BENCHMARK(model, save_load, "Serialise to a file (CRC32C) and load it back (load re-finalizes)") {
    BENCH_MODEL(ctx, m);
    char path[64];
    snprintf(path, sizeof path, "/tmp/beam_bench_%d.beam", (int)getpid());
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_save(m, path));
        BeamModel *loaded = beam_model_load(path);
        if (!loaded && !ctx->failed) bench_fail(ctx, (int)beam_last_error(), "beam_model_load: %s", beam_error_message());
        beam_model_destroy(loaded);
    }
    unlink(path);
}

BEAM_BENCHMARK_EX(model, drop_caches, "Free a populated live cache (every length's frontier)", BENCH_SINGLE_SHOT) {
    BENCH_MODEL(ctx, m);
    /* path_count_total acquires every length's live set; log_partition would
     * fill nothing, which is how the first version timed a no-op. */
    const double paths = beam_model_path_count_total(m);
    BENCH_REQUIRE(ctx, beam_last_error() == BEAM_OK, "populating the cache: %s", beam_error_message());
    BENCH_REQUIRE(ctx, beam_model_live_cache_bytes(m) > 0, "live cache still empty after path_count_total");
    BENCH_KEEP(paths);
    BEAM_TIMED_BLOCK(ctx) {
        beam_model_drop_caches(m);
    }
}

/* ── Scoring ───────────────────────────────────────────────────── */

BEAM_BENCHMARK(model, log_pseq, "Log-probability of one typical training sequence") {
    BENCH_MODEL(ctx, m);
    const char *seq = bench_typical_sequence();
    double p = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        p = beam_model_log_pseq(m, seq);
    }
    BENCH_REQUIRE(ctx, p > -1e300, "log_pseq returned -inf for a training sequence");
    BENCH_KEEP(p);
}

#define SCORE_BATCH 1000u

BEAM_BENCHMARK(model, log_pseq_batch, "Batch log-probability of 1,000 training sequences") {
    BENCH_MODEL(ctx, m);
    size_t n = 0;
    const char *const *seqs = bench_corpus(BENCH_VARIANT_PRIMARY, &n);
    const uint32_t k = (uint32_t)(n < SCORE_BATCH ? n : SCORE_BATCH);
    BENCH_SCRATCH(ctx, double, out, k);
    bench_set_items(ctx, k, "sequences");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_log_pseq_batch(m, seqs, k, out));
    }
}

BEAM_BENCHMARK(model, log_pseq_packed, "Packed-buffer log-probability of 1,000 training sequences") {
    BENCH_MODEL(ctx, m);
    size_t n = 0;
    const char *const *seqs = bench_corpus(BENCH_VARIANT_PRIMARY, &n);
    const uint32_t k = (uint32_t)(n < SCORE_BATCH ? n : SCORE_BATCH);
    size_t bytes = 0;
    for (uint32_t i = 0; i < k; ++i) bytes += strlen(seqs[i]);
    /* One scratch block: k doubles out, k + 1 offsets, then the packed text. */
    char *base = (char *)bench_scratch(ctx, k * sizeof(double) + (k + 1u) * sizeof(int64_t) + bytes + 1u);
    if (!base) return;
    double *out = (double *)base;
    int64_t *offsets = (int64_t *)(base + k * sizeof(double));
    char *packed = (char *)(offsets + k + 1u);
    int64_t off = 0;
    for (uint32_t i = 0; i < k; ++i) {
        const size_t l = strlen(seqs[i]);
        offsets[i] = off;
        memcpy(packed + off, seqs[i], l);
        off += (int64_t)l;
    }
    offsets[k] = off;
    bench_set_items(ctx, k, "sequences");
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_log_pseq_packed(m, packed, (uint64_t)off, offsets, k, out));
    }
}

BEAM_BENCHMARK(model, explain, "Per-residue forward/backward score decomposition of one sequence") {
    BENCH_MODEL(ctx, m);
    const char *seq = bench_typical_sequence();
    BENCH_SCRATCH(ctx, double, fwd, 2u * ctx->max_length);
    double *bwd = fwd + ctx->max_length;
    uint32_t out_len = 0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_explain(m, seq, fwd, bwd, &out_len));
    }
}

/* ── Per-length values and introspection ─────────────────────── */

BEAM_BENCHMARK(model, log_partition_lookup, "Read Z(L) and pi(L) for every length (O(1) lookups of values finalize computed)") {
    BENCH_MODEL(ctx, m);
    double sink = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        for (uint32_t l = 1; l <= ctx->max_length; ++l)
            sink += beam_model_log_partition(m, l) + beam_model_log_length_prior(m, l);
    }
    BENCH_KEEP(sink);
}

BEAM_BENCHMARK(model, log_partition_by_layer, "Z(L) split across positions: one forward sweep at length L") {
    BENCH_MODEL(ctx, m);
    BENCH_SCRATCH(ctx, double, out, ctx->length);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_log_partition_by_layer(m, ctx->length, out));
    }
}

BEAM_BENCHMARK(model, depth_occupancy, "Stored rows and cells per depth, both directions") {
    BENCH_MODEL(ctx, m);
    BENCH_SCRATCH(ctx, uint64_t, rows, 4u * ctx->max_length);
    uint64_t *cells = rows + 2u * ctx->max_length;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_depth_occupancy(m, rows, cells));
    }
}

BEAM_BENCHMARK(model, transition_rows, "Conditional transition rows at position min(K, L-1)") {
    BENCH_MODEL(ctx, m);
    uint32_t pos = ctx->order < ctx->length ? ctx->order : ctx->length - 1u;
    uint32_t r = beam_model_n_residues(m);
    size_t rows = 1;
    for (uint32_t i = 0; i < pos && i < ctx->order; ++i) rows *= r;
    BENCH_SCRATCH(ctx, double, out, rows * r);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_transition_rows(m, ctx->length, pos, false, out));
    }
}

BEAM_BENCHMARK(model, accessors, "Eight O(1) accessors: sizes, counts, weights, memory") {
    BENCH_MODEL(ctx, m);
    uint64_t sum = 0;
    BEAM_TIMED_BLOCK(ctx) {
        sum += beam_model_n_states(m);
        sum += beam_model_n_cells(m);
        sum += (uint64_t)beam_model_total_weight(m);
        sum += beam_model_n_observed(m);
        sum += beam_model_n_skipped(m);
        sum += beam_model_memory_bytes(m);
        sum += beam_model_live_cache_bytes(m);
        sum += beam_model_is_finalized(m) ? 1u : 0u;
    }
    BENCH_KEEP(sum);
}

BEAM_BENCHMARK(model, n_nonzero_cells, "Count non-zero cells (scans the count tables)") {
    BENCH_MODEL(ctx, m);
    uint64_t nz = 0;
    BEAM_TIMED_BLOCK(ctx) {
        nz += beam_model_n_nonzero_cells(m);
    }
    BENCH_KEEP(nz);
}

BEAM_BENCHMARK(model, config_accessors, "Config presets and structural accessors") {
    BENCH_MODEL(ctx, m);
    BeamModelConfig cfg;
    uint32_t sum = 0;
    BEAM_TIMED_BLOCK(ctx) {
        beam_model_config_default(&cfg);
        beam_model_config_default_for(ctx->alphabet, &cfg);
        sum += beam_model_config(m)->order;
        sum += (uint32_t)beam_model_alphabet(m);
        sum += beam_model_n_residues(m);
        sum += beam_model_radix(m);
    }
    BENCH_KEEP(sum);
}

BEAM_BENCHMARK(model, provenance, "Training provenance and file provenance span") {
    BENCH_MODEL(ctx, m);
    int64_t created = 0, obs_ns = 0, fin_ns = 0;
    size_t span_offset = 0, span_bytes = 0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_model_provenance(m, &created, &obs_ns, &fin_ns));
        beam_model_file_provenance_span(&span_offset, &span_bytes);
    }
    BENCH_KEEP(created + obs_ns + fin_ns + (int64_t)span_offset + (int64_t)span_bytes);
}

BEAM_BENCHMARK(model, threads_and_budget, "Global thread-cap and memory-budget setters and getters") {
    const int saved_threads = beam_model_threads();
    const uint64_t saved_budget = beam_model_memory_budget();
    uint64_t sum = 0;
    BEAM_TIMED_BLOCK(ctx) {
        beam_model_set_threads(saved_threads);
        sum += (uint64_t)beam_model_threads();
        sum += (uint64_t)beam_model_effective_threads();
        beam_model_set_memory_budget(saved_budget);
        sum += beam_model_memory_budget();
    }
    beam_model_set_threads(saved_threads);
    beam_model_set_memory_budget(saved_budget);
    BENCH_KEEP(sum);
}
