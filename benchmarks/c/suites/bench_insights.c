#include "bench_framework.h"
#include "beam/insights.h"

BEAM_BENCHMARK(insights, logo_cost, "Default logo request and its predicted cost (arithmetic, no sweep)") {
    BENCH_MODEL(ctx, m);
    BeamQueryCost cost;
    BEAM_TIMED_BLOCK(ctx) {
        BeamLogoRequest req;
        beam_logo_request_default(&req);
        req.length = ctx->length;
        BENCH_CHECK(ctx, beam_insights_logo_cost(m, &req, &cost));
    }
}

BEAM_BENCHMARK(insights, unconditional_logo, "Exact generated-measure sequence logo at length L (result freed)") {
    BENCH_MODEL(ctx, m);
    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = ctx->length;
    BEAM_TIMED_BLOCK(ctx) {
        /* logos overwrites its output without freeing it, so each call's set
         * is released inside the window. */
        BeamLogoSet set;
        memset(&set, 0, sizeof set);
        BENCH_CHECK(ctx, beam_insights_logos(m, &req, &set));
        beam_insights_logo_set_free(&set);
    }
}
