/**
 * @file transitions.c
 * @brief Bulk conditionals of the normalized fixed-length sequence law.
 *
 * If b_i(s) is the log total weight of all suffixes from state s at layer i,
 * the normalized transition is
 *
 *   K_i(x | s) = exp(w_i(s,x) + b_{i+1}(next(s,x)) - b_i(s)).
 *
 * The edge includes both experts' contributions when they become available,
 * and the suffix includes the delayed backward scores paid at the sink.
 * Normalizing an expert row or an edge row alone would give another law.
 *
 * This entry point answers ONE layer. Two backward vectors suffice: descend
 * only as far as position + 1, then normalize the requested rows directly.
 * Neither the whole backward field nor consecutive block tables are kept.
 */
#include "beam/internal.h"
#include "beam/model_build.h"

/* Convert a residue-only base-R row index to the trellis's base-A state.
 * At early positions the unavailable leading context is padded with the
 * sentinel, exactly as it is on a path starting at state S - 1. */
static uint32_t transition_state(uint32_t row, uint32_t width, uint32_t R,
                                 uint32_t A, uint32_t prefix) {
    uint32_t packed = 0, place = 1;
    for (uint32_t j = 0; j < width; ++j) {
        packed += (row % R) * place;
        row /= R;
        place *= A;
    }
    return prefix + packed;
}

/* One log-sum-exp suffix step. Independent rows make results independent of
 * the OpenMP thread count; the residue sum always has the same order. */
static bool transition_suffix(const BeamModel *m, const double *w,
                               const double *next, double *current) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    bool invalid = false;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:invalid) \
            num_threads(beam_threads())
#endif
    for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
        const uint32_t s = (uint32_t)s64;
        const double *row = w + (size_t)s * R;
        const uint32_t base = (s % SK) * A;
        double mx = -INFINITY;
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY || next[base + x] == -INFINITY)
                continue;
            const double v = row[x] + next[base + x];
            if (!isfinite(v)) { invalid = true; continue; }
            if (v > mx) mx = v;
        }
        if (mx == -INFINITY) { current[s] = -INFINITY; continue; }
        double sum = 0.0;
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY || next[base + x] == -INFINITY)
                continue;
            sum += exp(row[x] + next[base + x] - mx);
        }
        current[s] = mx + log(sum);
        if (!isfinite(current[s])) invalid = true;
    }
    return !invalid;
}

BeamError beam_model_transition_rows(const BeamModel *m, uint32_t length,
                                     uint32_t position, bool log_space,
                                     double *out) {
    if (!m || !out) return BEAM_ENULL(!m ? "model" : "output buffer");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
    if (position >= length)
        return BEAM_ERANGE("transition_rows: position %u is outside a "
                           "length-%u sequence (positions are zero-based)",
                           (unsigned)position, (unsigned)length);
    if (!isfinite(m->cfg.weight_forward) || !isfinite(m->cfg.weight_backward))
        return BEAM_ERANGE("transition_rows: both expert weights must be finite");

    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t width = position < m->order ? position : m->order;
    const uint32_t n_rows = beam_ipow_u32(R, width);
    const uint32_t prefix = S - beam_ipow_u32(A, width);

    /* Charge the caller-owned output too: it remains live while all scratch
     * buffers are held. The public API describes this total explicitly. */
    const uint64_t edge_bytes = (uint64_t)S * R * sizeof(double);
    const uint64_t vector_bytes = (uint64_t)S * sizeof(double);
    const uint64_t output_bytes = (uint64_t)n_rows * R * sizeof(double);
    const uint64_t need = edge_bytes + 2u * vector_bytes + output_bytes;
    BEAM_BUDGET_OR_REFUSE("bulk normalized transition rows", length, need);
    if (edge_bytes > SIZE_MAX || vector_bytes > SIZE_MAX || output_bytes > SIZE_MAX)
        return BEAM_ERANGE("transition_rows: the requested buffers exceed "
                           "this platform's addressable size");

    double *w = (double *)malloc((size_t)edge_bytes);
    double *next = (double *)malloc((size_t)vector_bytes);
    double *current = (double *)malloc((size_t)vector_bytes);
    if (!w || !next || !current) {
        free(w); free(next); free(current);
        return BEAM_EALLOC("the rolling transition-row backward buffers");
    }

    BeamError err = BEAM_OK;
    beam_tail_weights(m, length, next);
    for (uint32_t layer = length; layer > position + 1u; ) {
        --layer;
        beam_layer_weights(m, layer, length, w, NULL);
        if (!transition_suffix(m, w, next, current)) {
            err = BEAM_ERANGE("transition_rows: non-finite suffix arithmetic "
                              "at length %u, layer %u; the weights exceed "
                              "the floating-point operating range",
                              (unsigned)length, (unsigned)layer);
            goto done;
        }
        double *swap = next; next = current; current = swap;
    }

    beam_layer_weights(m, position, length, w, NULL);
    bool invalid = false;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:invalid) \
            num_threads(beam_threads())
#endif
    for (int64_t r64 = 0; r64 < (int64_t)n_rows; ++r64) {
        const uint32_t r = (uint32_t)r64;
        const uint32_t s = transition_state(r, width, R, A, prefix);
        const uint32_t base = (s % SK) * A;
        const double *row = w + (size_t)s * R;
        double *result = out + (size_t)r * R;
        double mx = -INFINITY;
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] == -INFINITY || next[base + x] == -INFINITY) {
                result[x] = -INFINITY;
                continue;
            }
            const double v = row[x] + next[base + x];
            result[x] = v;
            if (!isfinite(v)) { invalid = true; continue; }
            if (v > mx) mx = v;
        }
        if (mx == -INFINITY) {
            for (uint32_t x = 0; x < R; ++x)
                result[x] = log_space ? -INFINITY : 0.0;
            continue;
        }
        double sum = 0.0;
        for (uint32_t x = 0; x < R; ++x) sum += exp(result[x] - mx);
        const double log_sum = log(sum);
        for (uint32_t x = 0; x < R; ++x) {
            /* Subtract the shared offset before the small log normalizer,
             * retaining more precision than result - (mx + log_sum). */
            const double lp = (result[x] - mx) - log_sum;
            result[x] = log_space ? lp : exp(lp);
        }
    }
    if (invalid)
        err = BEAM_ERANGE("transition_rows: non-finite normalization at "
                          "length %u, position %u; the weights exceed the "
                          "floating-point operating range",
                          (unsigned)length, (unsigned)position);

done:
    free(w); free(next); free(current);
    return err;
}
