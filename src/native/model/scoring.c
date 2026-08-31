/**
 * @file scoring.c
 * @brief Direct sequence scoring and the mixed per-residue weight.
 */
#include <string.h>

#include "beam/internal.h"
#include "beam/mixing.h"

#ifdef _OPENMP
#include <omp.h>
#endif



double beam_mix(const BeamModel *m, uint32_t dir, uint32_t len,
                    uint32_t depth, uint32_t ctx_wide, uint32_t ctx_mid,
                    uint32_t ctx_len, uint32_t x) {
    return m->cfg.sparse
         ? mix_sparse(m, dir, len, depth, ctx_wide, ctx_mid, ctx_len, x)
         : mix_dense(m, dir, len, depth, ctx_wide, ctx_mid, ctx_len, x);
}

/* ── Direct sequence scoring ─────────────────────────────────── */

/**
 * Sum of both directions' contributions over every residue, before the length
 * prior and the partition function. This is the quantity the DAG's edge
 * weights plus its tail weight also sum to.
 */
double beam_raw_log_weight(const BeamModel *m, const uint8_t *s,
                           uint32_t len) {
    double total = 0.0;
    for (uint32_t p = 0; p < len; ++p) {
        const uint32_t x = s[p];

        const double f = beam_mix(
            m, BEAM_FWD, len, p,
            beam_ctx_forward(s, (int32_t)p, m->order, m->n_alphabet),
            beam_ctx_forward(s, (int32_t)p, 3u, m->n_alphabet),
            beam_ctx_forward(s, (int32_t)p, 2u, m->n_alphabet), x);
        if (f == -INFINITY) return -INFINITY;

        const uint32_t r = len - 1u - p;
        const double b = beam_mix(
            m, BEAM_BWD, len, r,
            beam_ctx_backward_wide(s, (int32_t)p, len, m->order,
                                   m->n_alphabet),
            beam_ctx_backward(s, (int32_t)p, len, 3u, m->n_alphabet),
            beam_ctx_backward(s, (int32_t)p, len, 2u, m->n_alphabet), x);
        if (b == -INFINITY) return -INFINITY;

        total += m->cfg.weight_forward * f + m->cfg.weight_backward * b;
    }
    return total;
}

double beam_model_log_pseq(const BeamModel *m, const char *seq) {
    if (!m || !m->finalized || !seq) return -INFINITY;
    uint8_t buf[BEAM_MAX_LENGTH_CEILING];
    uint32_t len = 0;
    if (!beam_encode(m, seq, buf, &len)) return -INFINITY;
    if (!isfinite(m->log_pi[len]) || !isfinite(m->log_z[len])) return -INFINITY;

    const double raw = beam_raw_log_weight(m, buf, len);
    if (raw == -INFINITY) return -INFINITY;
    return raw + m->log_pi[len] - m->log_z[len];
}

BeamError beam_model_log_pseq_batch(const BeamModel *m,
                                       const char *const *seqs,
                                       uint32_t n, double *out) {
    if (!m || !seqs || !out)
        return BEAM_ENULL(!m ? "model" : (!seqs ? "sequences" : "output buffer"));
    if (!m->finalized) return BEAM_ENOTBUILT();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        out[i] = beam_model_log_pseq(m, seqs[i]);
    }
    return BEAM_OK;
}

BeamError beam_model_explain(const BeamModel *m, const char *seq,
                                double *out_forward, double *out_backward,
                                uint32_t *out_len) {
    if (!m || !seq) return BEAM_ENULL(!m ? "model" : "sequence");
    if (!m->finalized) return BEAM_ENOTBUILT();
    uint8_t buf[BEAM_MAX_LENGTH_CEILING];
    uint32_t len = 0;
    if (!beam_encode(m, seq, buf, &len))
        return BEAM_FAIL(BEAM_ERR_INVALID_SEQUENCE,
                        "%s: the sequence is empty, longer than max_length "
                        "(%u), or contains a character outside the 20 amino "
                        "acids", __func__, (unsigned)m->max_len);

    for (uint32_t p = 0; p < len; ++p) {
        const uint32_t x = buf[p];
        if (out_forward) {
            out_forward[p] = beam_mix(
                m, BEAM_FWD, len, p,
                beam_ctx_forward(buf, (int32_t)p, m->order, m->n_alphabet),
                beam_ctx_forward(buf, (int32_t)p, 3u, m->n_alphabet),
                beam_ctx_forward(buf, (int32_t)p, 2u, m->n_alphabet), x);
        }
        if (out_backward) {
            const uint32_t r = len - 1u - p;
            out_backward[p] = beam_mix(
                m, BEAM_BWD, len, r,
                beam_ctx_backward_wide(buf, (int32_t)p, len, m->order,
                                   m->n_alphabet),
                beam_ctx_backward(buf, (int32_t)p, len, 3u, m->n_alphabet),
                beam_ctx_backward(buf, (int32_t)p, len, 2u, m->n_alphabet), x);
        }
    }
    if (out_len) *out_len = len;
    return BEAM_OK;
}
