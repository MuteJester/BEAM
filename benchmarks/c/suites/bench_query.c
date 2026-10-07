#include "bench_framework.h"
#include "beam/query.h"

/*
 * Query engine. Model queries run on a constraint of the workload's length L
 * with position 0 pinned, so every query does a real sweep (a free constraint
 * short-circuits some of them). The residue count comes from the model --
 * 20 for amino acids, 5 for nucleotides (A C G T N) -- never a constant.
 */

static BeamConstraint *pinned_constraint(BenchContext *ctx, const BeamModel *m) {
    const uint32_t r = beam_model_n_residues(m);
    BeamConstraint *c = beam_query_constraint_new(ctx->length, r);
    if (!c) {
        bench_fail(ctx, (int)beam_last_error(), "constraint_new: %s", beam_error_message());
        return NULL;
    }
    if (beam_query_constrain_position(c, 0, 1u % r) != BEAM_OK) {
        bench_fail(ctx, (int)beam_last_error(), "constrain_position: %s", beam_error_message());
        beam_query_constraint_free(c);
        return NULL;
    }
    return c;
}

#define BENCH_CONSTRAINT(ctx, m, c)                                                          \
    BeamConstraint *c = pinned_constraint((ctx), (m));                                       \
    if (!c) return;

BEAM_BENCHMARK(query, log_probability, "P(constraint) at length L: one backward sweep") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    double log_p = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_log_probability(m, c, &log_p));
    }
    beam_query_constraint_free(c);
    BENCH_KEEP(log_p);
}

BEAM_BENCHMARK(query, marginals, "Per-position residue posterior under the constraint") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    BENCH_SCRATCH(ctx, double, out, (size_t)ctx->length * beam_model_n_residues(m));
    double log_z = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_marginals(m, c, out, &log_z));
    }
    beam_query_constraint_free(c);
    BENCH_KEEP(log_z);
}

BEAM_BENCHMARK(query, entropy, "Conditional entropy and its variance under the constraint") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    double h = 0.0, var = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_entropy(m, c, &h, &var));
    }
    beam_query_constraint_free(c);
    BENCH_KEEP(h + var);
}

BEAM_BENCHMARK(query, position_entropy, "Per-position entropy H(X_i | constraint)") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    BENCH_SCRATCH(ctx, double, out, ctx->length);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_position_entropy(m, c, out));
    }
    beam_query_constraint_free(c);
}

BEAM_BENCHMARK(query, pair_joint, "Joint distribution of positions 0 and min(2, L-1)") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    const uint32_t r = beam_model_n_residues(m);
    BENCH_SCRATCH(ctx, double, out, (size_t)r * r);
    const uint32_t j = ctx->length > 2u ? 2u : ctx->length - 1u;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_pair_joint(m, c, 0, j, out));
    }
    beam_query_constraint_free(c);
}

BEAM_BENCHMARK(query, information_matrix, "Mutual information between every pair of positions (L*R + 1 marginal sweeps)") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    BENCH_SCRATCH(ctx, double, out, (size_t)ctx->length * ctx->length);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_information_matrix(m, c, out));
    }
    beam_query_constraint_free(c);
}

BEAM_BENCHMARK(query, expect, "Mean and variance of the log weight under the constraint") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    BeamFunctional f = {.kind = BEAM_FUNCTIONAL_LOG_WEIGHT, .table = NULL};
    BeamQueryMoments mom;
    memset(&mom, 0, sizeof mom);
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_expect(m, c, &f, &mom));
    }
    beam_query_constraint_free(c);
}

BEAM_BENCHMARK(query, estimate, "Predicted cost of a query (arithmetic, no sweep)") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    BeamQueryCost cost;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_estimate(m, c, &cost));
    }
    beam_query_constraint_free(c);
}

BEAM_BENCHMARK(query, information_cost, "Predicted cost of the information matrix (arithmetic, no sweep)") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    BeamQueryCost cost;
    BEAM_TIMED_BLOCK(ctx) {
        BENCH_CHECK(ctx, beam_query_information_cost(m, c, &cost));
    }
    beam_query_constraint_free(c);
}

BEAM_BENCHMARK(query, insertions, "Every one-residue insertion of a sequence, scored (result freed)") {
    BENCH_MODEL(ctx, m);
    const char *seq = bench_typical_sequence();
    BEAM_TIMED_BLOCK(ctx) {
        BeamNeighbourhood nb;
        memset(&nb, 0, sizeof nb);
        BENCH_CHECK(ctx, beam_query_insertions(m, seq, &nb));
        beam_query_neighbourhood_free(&nb);
    }
}

BEAM_BENCHMARK(query, deletions, "Every one-residue deletion of a sequence, scored (result freed)") {
    BENCH_MODEL(ctx, m);
    const char *seq = bench_typical_sequence();
    BEAM_TIMED_BLOCK(ctx) {
        BeamNeighbourhood nb;
        memset(&nb, 0, sizeof nb);
        BENCH_CHECK(ctx, beam_query_deletions(m, seq, &nb));
        beam_query_neighbourhood_free(&nb);
    }
}

/* ── Constraints and automata (no model sweep) ─────────────────── */

BEAM_BENCHMARK(query, automaton_motif, "Compile a 3-residue motif automaton for length L") {
    BENCH_MODEL(ctx, m);
    const uint32_t r = beam_model_n_residues(m);
    const uint8_t motif[3] = {0, (uint8_t)(1u % r), (uint8_t)(2u % r)};
    BEAM_TIMED_BLOCK(ctx) {
        BeamAutomaton *a = beam_query_automaton_motif(ctx->length, r, motif, 3);
        if (!a && !ctx->failed) bench_fail(ctx, (int)beam_last_error(), "automaton_motif: %s", beam_error_message());
        beam_query_automaton_free(a);
    }
}

BEAM_BENCHMARK(query, automaton_count, "Compile a counter automaton (1..4 occurrences of two residues)") {
    BENCH_MODEL(ctx, m);
    const uint32_t r = beam_model_n_residues(m);
    const BeamResidueMask mask = (1u << 0) | (1u << 1);
    BEAM_TIMED_BLOCK(ctx) {
        BeamAutomaton *a = beam_query_automaton_count(ctx->length, r, mask, 1, 4);
        if (!a && !ctx->failed) bench_fail(ctx, (int)beam_last_error(), "automaton_count: %s", beam_error_message());
        beam_query_automaton_free(a);
    }
}

BEAM_BENCHMARK(query, automaton_ops, "Automaton AND, OR and NOT of two motif automata") {
    BENCH_MODEL(ctx, m);
    const uint32_t r = beam_model_n_residues(m);
    const uint8_t m1[2] = {0, (uint8_t)(1u % r)};
    const uint8_t m2[2] = {(uint8_t)(1u % r), (uint8_t)(2u % r)};
    BeamAutomaton *a1 = beam_query_automaton_motif(ctx->length, r, m1, 2);
    BeamAutomaton *a2 = beam_query_automaton_motif(ctx->length, r, m2, 2);
    if (!a1 || !a2) {
        bench_fail(ctx, (int)beam_last_error(), "automaton_motif: %s", beam_error_message());
        beam_query_automaton_free(a1);
        beam_query_automaton_free(a2);
        return;
    }
    BEAM_TIMED_BLOCK(ctx) {
        BeamAutomaton *x = beam_query_automaton_and(a1, a2);
        BeamAutomaton *y = beam_query_automaton_or(a1, a2);
        BeamAutomaton *z = beam_query_automaton_not(a1);
        if ((!x || !y || !z) && !ctx->failed)
            bench_fail(ctx, (int)beam_last_error(), "automaton algebra: %s", beam_error_message());
        beam_query_automaton_free(x);
        beam_query_automaton_free(y);
        beam_query_automaton_free(z);
    }
    beam_query_automaton_free(a1);
    beam_query_automaton_free(a2);
}

BEAM_BENCHMARK(query, automaton_accepts, "Run a motif automaton over one length-L string") {
    BENCH_MODEL(ctx, m);
    const uint32_t r = beam_model_n_residues(m);
    const uint8_t motif[3] = {0, (uint8_t)(1u % r), (uint8_t)(2u % r)};
    BeamAutomaton *a = beam_query_automaton_motif(ctx->length, r, motif, 3);
    BENCH_REQUIRE(ctx, a, "automaton_motif: %s", beam_error_message());
    BENCH_SCRATCH(ctx, uint8_t, seq, ctx->length);
    for (uint32_t i = 0; i < ctx->length; ++i) seq[i] = (uint8_t)(i % r);
    uint64_t sink = 0;
    BEAM_TIMED_BLOCK(ctx) {
        sink += beam_query_automaton_states(a);
        sink += beam_query_automaton_accepts(a, seq, ctx->length) ? 1u : 0u;
    }
    beam_query_automaton_free(a);
    BENCH_KEEP(sink);
}

BEAM_BENCHMARK(query, automaton_from_masks, "Build an automaton from a constraint's residue masks") {
    BENCH_MODEL(ctx, m);
    BENCH_CONSTRAINT(ctx, m, c);
    BEAM_TIMED_BLOCK(ctx) {
        BeamAutomaton *a = beam_query_automaton_from_masks(c);
        if (!a && !ctx->failed) bench_fail(ctx, (int)beam_last_error(), "automaton_from_masks: %s", beam_error_message());
        beam_query_automaton_free(a);
    }
    beam_query_constraint_free(c);
}

BEAM_BENCHMARK(query, constraint_ops, "Constraint create, pin, set, exclude, read back, free") {
    BENCH_MODEL(ctx, m);
    const uint32_t r = beam_model_n_residues(m);
    uint64_t sink = 0;
    BEAM_TIMED_BLOCK(ctx) {
        BeamConstraint *c = beam_query_constraint_new(ctx->length, r);
        if (!c) {
            if (!ctx->failed) bench_fail(ctx, (int)beam_last_error(), "constraint_new: %s", beam_error_message());
            continue;
        }
        BENCH_CHECK(ctx, beam_query_constrain_position(c, 0, 1u % r));
        if (ctx->length > 1u) BENCH_CHECK(ctx, beam_query_constrain_set(c, 1, (1u << 0) | (1u << 1)));
        if (ctx->length > 2u) BENCH_CHECK(ctx, beam_query_constrain_exclude(c, 2, 2u % r));
        sink += beam_query_constraint_length(c);
        sink += beam_query_constraint_residues(c);
        sink += beam_query_constraint_mask(c, 0);
        sink += beam_query_constraint_is_free(c) ? 1u : 0u;
        beam_query_constraint_free(c);
    }
    BENCH_KEEP(sink);
}

BEAM_BENCHMARK(query, constrain_run, "Constrain a contiguous 3-residue run") {
    BENCH_MODEL(ctx, m);
    const uint32_t r = beam_model_n_residues(m);
    BENCH_SKIP_IF(ctx, ctx->length < 3u, "length %u is shorter than the run", ctx->length);
    const uint8_t run[3] = {0, (uint8_t)(1u % r), (uint8_t)(2u % r)};
    BEAM_TIMED_BLOCK(ctx) {
        BeamConstraint *c = beam_query_constraint_new(ctx->length, r);
        if (!c) {
            if (!ctx->failed) bench_fail(ctx, (int)beam_last_error(), "constraint_new: %s", beam_error_message());
            continue;
        }
        BENCH_CHECK(ctx, beam_query_constrain_run(c, 0, run, 3));
        beam_query_constraint_free(c);
    }
}

BEAM_BENCHMARK(query, constrain_automaton, "Attach a motif automaton to a constraint (ownership transfers)") {
    BENCH_MODEL(ctx, m);
    const uint32_t r = beam_model_n_residues(m);
    const uint8_t motif[2] = {0, (uint8_t)(1u % r)};
    uint64_t sink = 0;
    BEAM_TIMED_BLOCK(ctx) {
        BeamConstraint *c = beam_query_constraint_new(ctx->length, r);
        BeamAutomaton *a = beam_query_automaton_motif(ctx->length, r, motif, 2);
        if (!c || !a) {
            if (!ctx->failed) bench_fail(ctx, (int)beam_last_error(), "setup: %s", beam_error_message());
            beam_query_automaton_free(a);
            beam_query_constraint_free(c);
            continue;
        }
        BENCH_CHECK(ctx, beam_query_constrain_automaton(c, a));
        sink += beam_query_constraint_automaton(c) != NULL;
        beam_query_constraint_free(c);
    }
    BENCH_KEEP(sink);
}
