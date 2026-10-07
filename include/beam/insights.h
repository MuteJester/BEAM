/**
 * @file insights.h
 * @brief Derived, plottable views of a finalized model.
 *
 * Nothing here computes anything new about a model. Every quantity is built
 * from `beam_model_positional_marginals` and `beam_model_positional_banded`,
 * which are exact and verified; what this header adds is the shaping — the
 * arithmetic a consumer would otherwise do by hand, and the refusals that
 * stop a badly posed question from coming back as a confident number.
 *
 * It never draws. It returns arrays, and the caller plots them.
 *
 * ## Sequence logos
 *
 * For each position `i` of a length-`L` sequence, under a condition on the
 * sequence's own probability:
 *
 *     p[i][a]        = P(x_i = a | L, condition)
 *     H[i]           = -sum_a p log p
 *     information[i] = ln(reference_residues) - H[i]
 *     height[i][a]   = p[i][a] * information[i]
 *
 * In bits that is the Schneider-Stephens logo every plotting package expects.
 *
 * `information` IS a relative entropy, against a uniform background:
 * `log R - H` is exactly `D_KL(p_i || uniform over R)`. Saying so is not
 * pedantry — a reader who does not know which background was chosen cannot
 * tell whether one was.
 *
 * No small-sample correction is applied and none is wanted. The classic
 * `e_n = (R-1)/(2 ln2 . n)` corrects a logo built from `n` observed
 * sequences; this one is analytic over the whole support, so `n` is unbounded
 * and the correction is zero.
 *
 * ## The probability a band is cut on
 *
 * Bands are given in SURPRISAL, `-ln P(s)`, ascending — so they ascend in
 * rarity, and a caller thinking in probabilities passes `-ln p`, which
 * reverses the order.
 *
 * That `P(s)` is the JOINT probability, the one `beam_model_log_pseq`
 * returns, and it carries the length prior. It is the right axis because it
 * is what a user means by Pseq, but the consequence has to be stated: at a
 * length the model rarely produces, EVERY sequence has a small Pseq, so one
 * fixed threshold selects a different fraction at every length and at a rare
 * length may select nothing at all.
 *
 * Bands are half-open in surprisal, `[lo, hi)`. In probability that makes the
 * low-surprisal side strict and the high-surprisal side not — `Pseq > t` and
 * `Pseq <= t` — which is exactly what makes two bands a partition rather than
 * a double count. It is also below the resolution of the method: a sequence
 * within one bin of an edge is split across it, so the strictness is a
 * convention, not an observable.
 */
#ifndef BEAM_INSIGHTS_H
#define BEAM_INSIGHTS_H

#include "beam/common.h"
#include "beam/model.h"
#include "beam/query.h"      /* BeamQueryCost, shared rather than re-invented */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Request ─────────────────────────────────────────────────── */

/**
 * What logos to build.
 *
 * Zero-initialising this and setting `length` asks for the unconditional
 * logo of that length on the generated measure, which is the exact route.
 * That is deliberate: `beam_model_config_default` opens with a memset for the
 * same reason, and a caller who zeroes a struct has meant the ordinary thing.
 */
typedef struct {
    /** The sequence length. Logos are per length; 0 is refused, because
     *  position 5 of a length-12 sequence and position 5 of a length-18 one
     *  are different distances from the C terminus. */
    uint32_t length;

    /**
     * Band edges in surprisal (nats), STRICTLY ASCENDING, giving
     * `n_edges - 1` logos over `[edges[j], edges[j+1])`.
     *
     * `-INFINITY` and `+INFINITY` are accepted for the outer edges and are
     * resolved against the grid. That is the whole reason this lives in C:
     * the top of the grid is `bins * spacing`, and when the spacing is chosen
     * automatically only the library knows it. A caller outside would have to
     * pass a magic large number or re-derive the spacing formula, which is
     * the mistake `budget.h` forbids for guards and the same mistake here.
     */
    const double *edges;

    /** 0 or 1 asks for the unconditional logo; `n_edges - 1` logos otherwise. */
    uint32_t n_edges;

    /**
     * true weights every producible sequence 1, false weights it by its own
     * probability -- the two measures `beam_model_spectrum` offers.
     *
     * BEWARE the counting measure: under the hierarchical prior every cell is
     * positive, so every sequence is producible, every position's counting
     * marginal is uniform, and the logo is flat everywhere. That is the
     * correct answer to a badly posed question and it looks exactly like a
     * bug. The generated measure is the one that says something.
     */
    bool counting;

    /** Surprisal grid, used only when a band is asked for. 0 chooses a
     *  default from the model; `spacing <= 0` chooses the spacing the
     *  spectrum would choose, so the two land on the same grid. */
    uint32_t bins;
    double   spacing;

    /**
     * Alphabet size the information content is measured against, so that
     * `max_entropy` is `ln(reference_residues)`. 0 means the model's own
     * `n_residues`.
     *
     * It exists for nucleotides. `BEAM_NT_RESIDUES` is 5, because N is an
     * emitted residue in BEAM and not a wildcard, so the honest maximum is
     * `log2(5) = 2.32` bits where every DNA logo ever published tops out at
     * 2. Passing 4 gets the conventional axis. `information` may then be
     * NEGATIVE at a position that spells N with real probability, and that is
     * information about the model rather than an error, so it is not clamped.
     */
    uint32_t reference_residues;

    /**
     * Refuse rather than return a logo whose grid error exceeds this.
     * `<= 0` or NaN does not check.
     *
     * A logo at 3% resolution error is a picture of the grid, not of the
     * model, and a caller has to remember to look at `resolution_error` to
     * find that out. This turns it into the refusal the rest of the library
     * would give: a question with no exact answer is refused.
     */
    double max_resolution_error;
} BeamLogoRequest;

/** Zero the request and set what is not zero. */
void beam_logo_request_default(BeamLogoRequest *r);


/* ── Result ──────────────────────────────────────────────────── */

/** One band's logo. Arrays are owned by the enclosing `BeamLogoSet`. */
typedef struct {
    uint32_t  length;
    uint32_t  n_residues;

    /** `[length * n_residues]`, row-major. Each row sums to 1. Under the
     *  hierarchical prior no residue is exactly 0: one the corpus never put
     *  here is small, not absent. */
    double   *probability;

    /** `[length * n_residues]`, `probability * information`. The stack
     *  heights a logo is drawn from. Both units are the same numbers scaled
     *  by `ln 2`; the bits are what every plotting package expects. */
    double   *height_bits;
    double   *height_nats;

    /** `[length]` */
    double   *entropy_bits;
    double   *entropy_nats;

    /** `[length]`, `max_entropy - entropy`. */
    double   *information_bits;
    double   *information_nats;

    /** `[length]`, how many residues carry any probability at all.
     *
     *  Under the hierarchical prior every residue carries some probability,
     *  so this is `R` wherever the model is defined; it was the disclosure
     *  that the `log R` reference could be partly fictional when a model
     *  could spell only a few residues at a position.
     *  It is reported rather than used: a per-position reference would make
     *  positions incomparable, which is worse than a generous one. */
    uint32_t *n_support;

    /** `[length]`, the argmax residue and its probability. */
    uint32_t *consensus;
    double   *consensus_probability;

    /** `[length]`, this band's mass measured independently at each position,
     *  or NULL on the exact route. Every position is the same number in
     *  exact arithmetic; `resolution_error` is their disagreement. */
    double   *position_mass;

    /** The band, after `INFINITY` was resolved against the grid. */
    double    lo_nats, hi_nats;

    /** Mass of this band in the chosen measure. `log_mass` is primary because
     *  a far-tail band underflows a double, as everything else in this
     *  library that can underflow is carried in logs. */
    double    log_mass;
    double    mass;

    /** `mass` over the mass the grid actually held, so the bands of one call
     *  sum to 1. Compare `on_grid_total` with `length_total` on the set for
     *  the other question: whether the grid held everything. */
    double    fraction;

    /** Cross-position disagreement in this band's mass, relative. NaN when
     *  the set is exact -- never 0.0, which would be the stronger and false
     *  claim that the grid error was measured and found to be nil. */
    double    resolution_error;
} BeamLogo;

/** One or more logos of one length, and what they were built from. */
typedef struct {
    uint32_t     n_logos;
    BeamLogo    *logo;

    BeamAlphabet alphabet;
    uint32_t     reference_residues;
    double       max_entropy_bits;
    double       max_entropy_nats;

    /** Mass the grid held, and the length's whole mass in the same measure
     *  (`exp(log_length_prior)` generated, `path_count` counting). They
     *  differ when the grid is too SHORT and mass ran off the top -- a
     *  different fault from a grid that is too COARSE, which
     *  `resolution_error` reports. */
    double       on_grid_total;
    double       length_total;

    /** The grid, or NaN when no grid was involved. */
    double       spacing;

    /**
     * True when no grid error is possible.
     *
     * It describes the answer, not the route. A single band covering the
     * whole grid is exact whichever function computed it: linear splitting
     * moves mass between bins and never destroys it, so with nothing to cross
     * an edge, nothing can be misclassified. Mass running off the TOP of the
     * grid is still possible and is reported by `on_grid_total`, not here.
     */
    bool         exact;
    bool         counting;
} BeamLogoSet;

/**
 * Build the logos a request asks for.
 *
 * @param out  caller-allocated; filled on success, untouched on failure.
 *             Release with `beam_insights_logo_set_free`.
 *
 * @return BEAM_ERR_PARAM_OUT_OF_RANGE for a length the model does not
 *         generate, edges that do not ascend or are NaN, a reference wider
 *         than the alphabet, a band outside the surprisal this length can
 *         attain, or a grid coarser than `max_resolution_error` allows;
 *         BEAM_ERR_NO_LIVE_PATHS for a band holding no mass, which is a real
 *         question with no answer rather than a zero to be returned. That
 *         last one matters more than it looks: an empty band gives zero
 *         probabilities, hence zero entropy, hence the MAXIMUM information
 *         content -- a blank plot beside a summary claiming every position is
 *         perfectly conserved.
 */
BeamError beam_insights_logos(const BeamModel *m, const BeamLogoRequest *req,
                              BeamLogoSet *out);

/**
 * What that call will cost, without running it.
 *
 * Exposed for the same reason `beam_query_estimate` is: an unconditional
 * generated logo is one forward-backward sweep and a conditioned one is a
 * multi-gigabyte grid, and the caller cannot see that from the question.
 * `bytes` is the figure that will be checked against the memory budget, from
 * the same expression, so a caller can compare before asking instead of
 * catching a refusal after. For a banded sweep it is exact -- a budget of
 * `bytes` runs it and one byte less refuses it -- because the estimate runs
 * the sweep's own integer pass that sizes each backward row.
 */
BeamError beam_insights_logo_cost(const BeamModel *m,
                                  const BeamLogoRequest *req,
                                  BeamQueryCost *out);

/** Release a set and zero it. Safe on an already-freed set. */
void beam_insights_logo_set_free(BeamLogoSet *set);

#ifdef __cplusplus
}
#endif

#endif /* BEAM_INSIGHTS_H */
