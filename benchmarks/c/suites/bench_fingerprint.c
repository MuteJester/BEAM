#include "bench_framework.h"
#include "beam/fingerprint.h"

/* The configuration BEAM.analytics.fingerprint.compare uses by default. */
static void default_config(BeamFingerprintConfig *c) {
    c->max_leaves = 16;
    c->max_depth = 4;
    c->lookahead = 2;
    c->beam_width = 32;
    c->max_queries = 256;
    c->min_mass = 1e-5;
    c->min_gain = 1e-7;
}

BEAM_BENCHMARK(fingerprint, search, "Fingerprint search between two models at length L (Python defaults; result freed)") {
    BENCH_MODEL_VARIANT(ctx, p, BENCH_VARIANT_PRIMARY);
    BENCH_MODEL_VARIANT(ctx, q, BENCH_VARIANT_SECOND);
    BeamFingerprintConfig cfg;
    default_config(&cfg);
    BEAM_TIMED_BLOCK(ctx) {
        BeamFingerprintResult *res = NULL;
        BENCH_CHECK(ctx, beam_fingerprint_search(p, q, ctx->length, &cfg, &res));
        beam_fingerprint_free(res);
    }
}
