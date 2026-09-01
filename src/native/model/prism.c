/**
 * @file prism.c
 * @brief PRISM: is this sequence better explained as a model draw, or as a
 *        model draw with one edit applied to it?
 *
 * The metric is a Bayes factor `B = Q/P`. The alternative hypothesis is that a
 * one-edit REPAIR of the observed sequence is what the model actually
 * generated. `B` is the PRISM factor; the per-site arrays are the PRISM
 * posterior.
 *
 * The name pairs with BEAM and describes what it does: a prism takes a beam
 * and splits it into components, and this splits one score into per-position
 * evidence. Both outputs come from the same pass, because computing `Q` means
 * visiting every edit history exactly once.
 *
 * "Prism explains it better" is deliberately weaker than "this is noise". A
 * sequencing error produces a large `B`, and so does a genuinely rare
 * biological sequence sitting one edit from a common one. The model cannot
 * separate those and the name should not imply it can.
 *
 * WHY IT IS A SEPARATE TRANSLATION UNIT. It is not a variant of `log_pseq`; it
 * is a hypothesis test built on top of it, with its own alternative, its own
 * normalisation and its own calibration story (there isn't one -- see below).
 * Keeping it beside the scorer would bury that.
 *
 * THE PROPERTY THAT MAKES IT WORTH HAVING. `B` is a likelihood ratio, so under
 * the null
 *
 *     E_P[B] = sum_s P(s) Q(s)/P(s) = Q(support(P)) <= 1
 *
 * making it an e-value: Markov gives `P(B >= t) <= 1/t` directly. A threshold
 * is therefore a false-positive bound with NO calibration table, NO simulated
 * null and nothing fitted, which is what the three-channel index it replaces
 * needed four separate nulls to achieve.
 */
#include <math.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/mixing.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── the one-edit Bayes factor ──────────────────────────────────────────────
 *
 * Two hypotheses about an observed sequence `s`:
 *
 *   H0   s was drawn from the model.
 *   H1   a clean `x` was drawn from the model, one of the three edit types was
 *        chosen with probability 1/3, a legal site was chosen uniformly, a
 *        residue was chosen uniformly where one was needed, and the single
 *        edit produced `s`.
 *
 * Recovering `x` from `s` needs the INVERSE operation, which is why the
 * substitution channel sums over substitutions of `s`, the insertion channel
 * over deletions FROM `s`, and the deletion channel over insertions INTO `s`:
 *
 *   Qsub = 1/(19L)  * sum_i sum_{a != s_i} P(s with s_i replaced by a)
 *   Qins = 1/(20L)  * sum_i P(s with residue i removed)
 *   Qdel = 1/(L+1)  * sum_j sum_a P(s with a inserted at gap j)
 *   Q    = (Qsub + Qins + Qdel) / 3          B = Q / P(s)
 *
 * The denominators are the exact counts of forward edit operations, not tuned
 * penalties; they are what make each Q a probability law and therefore B a
 * genuine likelihood ratio. Repairs are summed WITH MULTIPLICITY: two edit
 * histories that happen to produce the same string are two histories, and
 * deduplicating them would understate Q.
 *
 * WHY THIS IS WORTH A C PATH RATHER THAN A LOOP IN PYTHON. B is a likelihood
 * ratio, so under the null
 *
 *     E_P[B] = sum_s P(s) Q(s)/P(s) = Q(support(P)) <= 1
 *
 * making it an e-value: Markov gives P(B >= t) <= 1/t with NO calibration
 * table and NO simulated null. That is the whole appeal, and it means B is
 * meant to be computed for every sequence in a repertoire rather than for a
 * sampled subset. It costs 40L+20 scorings each, so the Python form spends
 * most of its time building candidate strings; here the candidates are edited
 * in place in the encoded buffer and never materialise.
 *
 * The per-site arrays are the localisation posterior, unnormalised: dividing
 * them by Q gives P(edit history | s, H1). They come free, because computing
 * Q means visiting every history exactly once.
 */

static inline double lse2(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    const double hi = a > b ? a : b, lo = a > b ? b : a;
    return hi + log1p(exp(lo - hi));
}

/* log P of an already-encoded buffer, or -inf if the model cannot spell it. */
static double log_pseq_buf(const BeamModel *m, const uint8_t *buf,
                           uint32_t len) {
    if (len == 0u || len > m->max_len) return -INFINITY;
    if (!isfinite(m->log_pi[len]) || !isfinite(m->log_z[len])) return -INFINITY;
    const double raw = beam_raw_log_weight(m, buf, len);
    if (raw == -INFINITY) return -INFINITY;
    return raw + m->log_pi[len] - m->log_z[len];
}

/* ── omega: how many edits away we look ────────────────────────────────────
 *
 * Q is the one-inverse-edit operator T applied to P:
 *
 *     (T R)(s) = 1/3 [ 1/(19L) sum_{i,a!=s_i} R(s[i<-a])
 *                    + 1/(20L) sum_i       R(s[-i])
 *                    + 1/(L+1) sum_{j,a}   R(s[+j,a]) ]
 *
 * so looking `omega` edits away is just `Q_omega = T^omega P`, and the whole
 * derivation carries over unchanged: T maps a probability law to a probability
 * law, so `E[B] <= 1` still holds and `B >= t` is still a 1/t false-positive
 * bound with nothing calibrated. Only the alternative widens.
 *
 * THE COST IS THE REASON omega=1 IS THE DEFAULT. Each level multiplies the
 * work by 40L+20, so omega=2 costs about (40L+20)^2 -- roughly 380,000
 * scorings at length 14 against 580. That is tenths of a second per sequence
 * rather than tens of thousands per second, which makes omega=2 a tool for a
 * flagged subset, not for a repertoire.
 *
 * The per-site arrays are only filled at omega=1. At omega=2 a "site" is a
 * PAIR of edits and the posterior is over pairs, which is a different object
 * and is not what the arrays describe; filling them with a marginal would
 * invite reading a pair posterior as a site posterior.
 */
static double prism_transport(const BeamModel *m, uint8_t *buf, uint32_t len,
                              uint32_t depth, double *out_sub,
                              double *out_ins, double *out_del) {
    if (depth == 0u) return log_pseq_buf(m, buf, len);

    uint8_t tmp[BEAM_MAX_LENGTH_CEILING];
    const uint32_t A = m->n_residues;
    double q_sub = -INFINITY, q_ins = -INFINITY, q_del = -INFINITY;

    memcpy(tmp, buf, len);
    for (uint32_t i = 0; i < len; ++i) {
        const uint8_t keep = tmp[i];
        double at_i = -INFINITY;
        for (uint32_t a = 0; a < A; ++a) {
            if ((uint8_t)a == keep) continue;
            tmp[i] = (uint8_t)a;
            at_i = lse2(at_i, prism_transport(m, tmp, len, depth - 1u,
                                              NULL, NULL, NULL));
        }
        tmp[i] = keep;
        if (out_sub) out_sub[i] = at_i;
        q_sub = lse2(q_sub, at_i);
    }
    /* One substitution channel per (site, other residue): A - 1 of them at
     * each of `len` sites, where A is the alphabet's residue count. Twenty
     * amino acids give 19; five bases give 4. */
    if (len > 0u) q_sub -= log((double)((A - 1u) * len));

    if (len > 1u) {
        for (uint32_t i = 0; i < len; ++i) {
            memcpy(tmp, buf, i);
            memcpy(tmp + i, buf + i + 1, len - i - 1u);
            const double v = prism_transport(m, tmp, len - 1u, depth - 1u,
                                             NULL, NULL, NULL);
            if (out_ins) out_ins[i] = v;
            q_ins = lse2(q_ins, v);
        }
        /* An insertion can place any of the A residues before each site. */
        q_ins -= log((double)(A * len));
    } else if (out_ins) {
        for (uint32_t i = 0; i < len; ++i) out_ins[i] = -INFINITY;
    }

    if (len < m->max_len) {
        for (uint32_t j = 0; j <= len; ++j) {
            double at_j = -INFINITY;
            memcpy(tmp, buf, j);
            memcpy(tmp + j + 1, buf + j, len - j);
            for (uint32_t a = 0; a < A; ++a) {
                tmp[j] = (uint8_t)a;
                at_j = lse2(at_j, prism_transport(m, tmp, len + 1u, depth - 1u,
                                                  NULL, NULL, NULL));
            }
            if (out_del) out_del[j] = at_j;
            q_del = lse2(q_del, at_j);
        }
        q_del -= log((double)(len + 1u));
    } else if (out_del) {
        for (uint32_t j = 0; j <= len; ++j) out_del[j] = -INFINITY;
    }

    return lse2(lse2(q_sub, q_ins), q_del) - log(3.0);
}

BeamError beam_prism_factor(const BeamModel *m, const char *seq,
                            uint32_t omega,
                            double *out_log_b, double *out_log_q,
                            double *out_log_p0, double *out_sub,
                            double *out_ins, double *out_del,
                            uint32_t *out_len) {
    if (!m || !seq) return BEAM_ENULL(!m ? "model" : "sequence");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (omega == 0u || omega > BEAM_PRISM_MAX_OMEGA)
        return BEAM_FAIL(BEAM_ERR_INVALID_ARG,
                        "%s: omega must be between 1 and %u; it is the number "
                        "of edits the alternative allows, and each level costs "
                        "about (40L+20) times the last",
                        __func__, (unsigned)BEAM_PRISM_MAX_OMEGA);

    uint8_t buf[BEAM_MAX_LENGTH_CEILING];
    uint32_t len = 0;
    if (!beam_encode(m, seq, buf, &len))
        return BEAM_FAIL(BEAM_ERR_INVALID_SEQUENCE,
                        "%s: the sequence is empty, longer than max_length "
                        "(%u), or contains a character outside the %u %s "
                        "residues (%s)", __func__, (unsigned)m->max_len,
                        (unsigned)m->n_residues, m->alpha->name,
                        m->alpha->idx_to_char);

    const double log_p0 = log_pseq_buf(m, buf, len);
    /* Site arrays describe SINGLE edits, so they are only filled at omega=1.
     * See the note on prism_transport. */
    const bool sites = (omega == 1u);
    const double log_q = prism_transport(m, buf, len, omega,
                                         sites ? out_sub : NULL,
                                         sites ? out_ins : NULL,
                                         sites ? out_del : NULL);
    if (!sites) {
        if (out_sub) for (uint32_t i = 0; i < len; ++i) out_sub[i] = NAN;
        if (out_ins) for (uint32_t i = 0; i < len; ++i) out_ins[i] = NAN;
        if (out_del) for (uint32_t j = 0; j <= len; ++j) out_del[j] = NAN;
    }

    if (out_log_q) *out_log_q = log_q;
    if (out_log_p0) *out_log_p0 = log_p0;
    if (out_len) *out_len = len;
    if (out_log_b) {
        if (log_p0 == -INFINITY && log_q == -INFINITY) *out_log_b = NAN;
        else if (log_p0 == -INFINITY)                  *out_log_b = INFINITY;
        else                                           *out_log_b = log_q - log_p0;
    }
    return BEAM_OK;
}

BeamError beam_prism_factor_batch(const BeamModel *m,
                                  const char *const *seqs, uint32_t n,
                                  uint32_t omega, double *out_log_b) {
    if (!m || !seqs || !out_log_b)
        return BEAM_ENULL(!m ? "model" : (!seqs ? "sequences" : "output buffer"));
    if (!m->finalized) return BEAM_ENOTBUILT();
    /* omega is validated ONCE, up front, and refused. The NaN-on-failure rule
     * below is for bad DATA -- a repertoire row this model cannot encode must
     * not lose the other million -- and a bad PARAMETER is not data. Letting
     * it through would return a column of NaN with no error, which reads as
     * "nothing was detectable" rather than "you called it wrong". */
    if (omega == 0u || omega > BEAM_PRISM_MAX_OMEGA)
        return BEAM_FAIL(BEAM_ERR_INVALID_ARG,
                        "%s: omega must be between 1 and %u",
                        __func__, (unsigned)BEAM_PRISM_MAX_OMEGA);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16) num_threads(beam_threads())
#endif
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        double b = NAN;
        /* A sequence the encoder rejects scores NaN rather than aborting the
         * batch: a repertoire file legitimately contains rows this model
         * cannot represent, and one of them must not lose the other million. */
        (void)beam_prism_factor(m, seqs[i], omega, &b, NULL, NULL,
                                NULL, NULL, NULL, NULL);
        out_log_b[i] = b;
    }
    return BEAM_OK;
}

