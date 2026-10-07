/**
 * @file mixture.h
 * @brief Several fitted models read as ONE distribution over sequences.
 *
 * The object is the probability mixture
 *
 *     pbar(s) = sum_i w_i P_i(s),   w_i >= 0,   sum_i w_i = 1,
 *
 * which is "pick a donor by w, then pick a sequence from that donor". It is
 * what beta diversity between individuals is defined on, and it is NOT what
 * adding counts produces.
 *
 * ## `beam_model_add` is not this, and the difference is measurable
 *
 * Summing edge counts and refinalizing gives the model you would get by
 * TRAINING on the pooled corpus, which is not a mixture of the members: a
 * pooled row blends every donor's counts before the prior is applied, so the
 * merged model scores spliced sequences -- an edge learned from one donor
 * followed by an edge learned from another -- above what any mixture of the
 * members can.
 *
 * So nothing in this file is implemented in terms of the count algebra.
 * Under the hierarchical prior every member's support is the whole space, so
 * `beam_mixture_support_union` equals `beam_model_path_count_total` of the
 * merged model; the difference between the two constructions lives in the
 * probabilities, not in the supports.
 *
 * ## What is exact here, and what is not
 *
 * The three Hill orders are not equally tractable and this header does not
 * pretend they are.
 *
 *   - INTEGER q >= 2 is EXACT and is what this file is for. The power sum
 *     expands into a sum over q-tuples of members,
 *
 *         sum_s pbar(s)^q = sum_{f1..fq} w_f1..w_fq <P_f1, .., P_fq>,
 *
 *     and each inner product is ONE forward sweep over the shared trellis
 *     whose edge weight is the SUM of the members' edge weights.
 *     `beam_mixture_log_inner_product` is that sweep. q = 2 needs M(M+1)/2 of
 *     them.
 *
 *   - q = 0 is the size of the UNION of the supports, which needs the members
 *     tracked separately along the path. `beam_mixture_support_union` carries
 *     a bitmask of the members still consistent, so its cost is set by how
 *     many distinct bitmasks survive per state rather than by M alone.
 *
 *   - q = 1 IS NOT COMPUTABLE HERE AT ALL, and no function in this header
 *     returns it. `log sum_i w_i P_i(s)` does not factor over edges, so the
 *     entropy of a mixture has no forward recursion. The callers get exact
 *     bounds from the members' own entropies and a sampled estimate with a
 *     standard error; both live in the Python layer, because both are
 *     arithmetic on quantities that already exist.
 *
 * ## The state space is shared, and that is what makes this cheap
 *
 * A state is a position and the last `order` residues, so the state
 * INDEX is a function of the alphabet and the order and not of the data. Two
 * models that agree on alphabet, order and max_length therefore
 * index the same trellis, cell for cell, and a sweep can read both models'
 * weights at one state index without any translation. Every entry point here
 * enforces that agreement with the same shape test `beam_model_add` uses and
 * refuses with BEAM_ERR_VARIANT_MISMATCH otherwise.
 */
#ifndef BEAM_MIXTURE_H
#define BEAM_MIXTURE_H

#include "beam/common.h"
#include "beam/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Most members `beam_mixture_support_union` accepts.
 *
 * The union sweep carries one BIT per member in a `uint64_t`, so this is a
 * hard structural ceiling rather than a tuning choice. Its PRACTICAL ceiling
 * is usually lower and is set by how many distinct bitmasks survive per state,
 * which is data-dependent.
 *
 * IT APPLIES ONLY TO THE UNION. The inner product never distinguishes the
 * members it multiplies -- it sums their edge weights into one accumulator --
 * so it has no such limit and now enforces none.
 *
 * It used to. The bound was shared by both, and by `beam_mixture_check_shape`
 * with it, so that "how many models may I pass" had one answer. That tidiness
 * cost more than it bought: a mixture of 256 repertoires was refused AT
 * CONSTRUCTION, and with it every quantity that has nothing to do with the
 * bitmask -- q >= 2, the alpha diversity, the q = 1 bounds. The limit now sits
 * on the one sweep that cannot exceed it.
 */
#define BEAM_MIXTURE_MAX_MEMBERS 64u

/**
 * Most members any other entry point accepts.
 *
 * Not a structural limit; a guard against an argument that is obviously a
 * mistake, and a bound on what the per-member scratch arrays may reach. The
 * Gram matrix of M members is M(M+1)/2 sweeps, so anything near this ceiling
 * is a compute-farm job rather than a call -- but that is the caller's
 * arithmetic to do, not this library's to refuse.
 */
#define BEAM_MIXTURE_MAX_PRODUCT_MEMBERS 65536u

/**
 * `ln sum_s prod_f P_f(s)` over every sequence of length `len`, exactly.
 *
 * The multiset of `k` models may repeat a model: passing one model twice gives
 * `sum_s P(s)^2`, whose reciprocal is that model's own Hill number at q = 2.
 * That identity is the cross-check this kernel is tested by, because it puts
 * two independent implementations on one number at full scale.
 *
 * Returns -INFINITY, with BEAM_OK, when no sequence of that length is spelled
 * by all `k` members at once. That is an ANSWER, not a failure; under the
 * hierarchical prior it happens only at a length some member cannot represent,
 * since every member spells every string up to its max_length.
 *
 * NUMERICS: a product of k per-edge probabilities over ~27 layers underflows a
 * double long before the sweep ends, so the accumulator is rescaled every
 * layer and the logarithm of the scale is carried alongside, exactly as
 * `jet_sweep` does. Relative precision is held at any magnitude; inner
 * products of 1e-300 are returned as ordinary finite logarithms.
 *
 * ORDER: the k weights at an edge are summed in the order the models are
 * given, so a permutation of `models` may move the last bit. Callers that need
 * an exactly symmetric Gram matrix should compute one triangle and mirror it.
 */
BeamError beam_mixture_log_inner_product(const BeamModel *const *models,
                                         uint32_t k, uint32_t len,
                                         double *out);

/**
 * `beam_mixture_log_inner_product` at every length, and their total.
 *
 * `out_by_length` receives `max_length + 1` values, -INFINITY at any length no
 * member produces; pass NULL if only the total is wanted. `out_total` receives
 * `ln sum_s prod_f P_f(s)` over ALL lengths, accumulated in the log domain so
 * a long tail of tiny lengths cannot underflow it.
 *
 * The lengths run concurrently, dividing the memory budget between them the
 * way every other by-length driver in the library does.
 */
BeamError beam_mixture_log_inner_product_by_length(
        const BeamModel *const *models, uint32_t k,
        double *out_by_length, double *out_total);

/**
 * `ln sum_s prod_f P_f(s)^a_f` over every sequence of length `len`, exactly.
 *
 * The inner product with a PER-MEMBER EXPONENT. `expo` holds `k` of them;
 * NULL is all ones and is exactly `beam_mixture_log_inner_product`, bit for
 * bit, since the scaling is a multiply by 1.0.
 *
 * This is the whole Chernoff family. `expo = (1/2, 1/2)` is the Bhattacharyya
 * coefficient `BC(P_0, P_1) = sum_s sqrt(P_0 P_1)`, whose complement
 * `1 - BC` is the squared Hellinger distance -- a true metric between two
 * models over the WHOLE sequence space, owing nothing to which sequences
 * either donor happened to be sequenced deeply enough to show.
 *
 * Why it is the same sweep: a power of a product is the product of the powers,
 * so `a_f` scales member `f`'s edge weights and its per-length constant and
 * nothing else moves. One multiply per edge.
 *
 * REFUSED: a negative exponent, since `sum_s P(s)^-a` diverges over a support
 * this large and would come back as `inf`; a non-finite one; and an all-zero
 * vector, which asks for `sum_s 1` over the whole space -- a support count,
 * which `beam_mixture_support_union` answers properly.
 *
 * An exponent of exactly 0 DROPS that member rather than zeroing its weights.
 * `P^0 = 1` at every sequence including the ones that member cannot spell, so
 * it must not go on constraining the support; `<P_A^1, P_B^0>` is
 * `sum_s P_A(s) = 1` whatever `B` is.
 */
BeamError beam_mixture_log_power_product(const BeamModel *const *models,
                                         const double *expo, uint32_t k,
                                         uint32_t len, double *out);

/**
 * `beam_mixture_log_power_product` at every length, and their total.
 *
 * Same contract as `beam_mixture_log_inner_product_by_length`, to which this
 * delegates when `expo` is NULL. The per-length breakdown is what separates
 * the two halves of a Bhattacharyya coefficient: the part that is merely the
 * two donors' CDR3 LENGTH distributions overlapping, and the part that is the
 * sequences within a length. The first has a model-free analogue; the second
 * does not.
 */
BeamError beam_mixture_log_power_product_by_length(
        const BeamModel *const *models, const double *expo, uint32_t k,
        double *out_by_length, double *out_total);

/**
 * `beam_mixture_log_power_product` at one length, split BY POSITION.
 *
 * The sweep already walks the coupled trellis position by position; this
 * reports what each position contributed instead of discarding it. The
 * identity, which `tests/native/test_mixture.c` asserts, is
 *
 *     *out_total == *out_const + sum_i out_by_position[i] + *out_sink
 *
 * with `out_by_position` receiving exactly `len` doubles. `*out_total` is
 * `beam_mixture_log_power_product`'s answer BIT FOR BIT: this function adds
 * accumulators and changes no existing arithmetic, so a caller may read the
 * decomposition without wondering whether the number moved.
 *
 * WHAT THE TERMS ARE, and why they are not the sweep's rescaling increments.
 * Writing `m_i` for the total forward mass over the frontier after `i`
 * residues, position `i`'s term is
 *
 *     out_by_position[i] = log(m_{i+1} / m_i)
 *
 * a ratio of SUMS. The sweep's own per-layer increments are `wmax + log nmax`,
 * maxima chosen for numerical range, so splitting the total along them is
 * GAUGE-DEPENDENT: it would sum correctly while carrying no invariant meaning,
 * and three sweeps rescaled by three different maxima could not be compared
 * term by term. That is the defect recorded against the score-share
 * decomposition in `.private/divergence_geometry/04_refuted.md`, which summed
 * correctly and told a different mechanistic story from the chain rule on the
 * same models. A ratio of sums depends on nothing but the model.
 *
 * `*out_sink` is the final edge into the sink, which closes the last residue's
 * backward term and is NOT commensurable with the positions: report it apart.
 * `*out_const` is `sum_f a_f (log_pi[len] - log_z[len])`, the length prior and
 * partition function -- the part that is not positional at all, and which
 * `05_next.md` item 4 asks to be computable on its own, since a positional
 * signature reproduced by it alone would be the length prior in disguise.
 *
 * Any of `out_by_position`, `out_sink` and `out_const` may be NULL. A length
 * no member can produce, or one whose supports never meet, sets every term to
 * `-INFINITY`, matching the total.
 */
BeamError beam_mixture_log_power_product_by_position(
        const BeamModel *const *models, const double *expo, uint32_t k,
        uint32_t len, double *out_by_position, double *out_sink,
        double *out_const, double *out_total);

/**
 * Every pairwise log inner product, computed one pair per thread.
 *
 * `out` receives `k * k` doubles, row-major; `out[i * k + j]` is
 * `ln <P_i, P_j>` and the matrix is exactly symmetric. This is the object the
 * q = 2 quantities are built from, and the reason it exists rather than being
 * a loop in the caller is that the parallel axis matters: a single inner
 * product stops scaling at about four threads, while pairs scale with the
 * machine. See the definition for the measurements.
 */
BeamError beam_mixture_gram(const BeamModel *const *models, uint32_t k,
                            double *out);

/**
 * Every pairwise `ln sum_s (P_i(s) P_j(s))^a`, one pair per thread.
 *
 * `beam_mixture_gram` with one exponent applied to BOTH members of each pair,
 * which is the only form a symmetric matrix admits. `a = 1/2` is the pairwise
 * Bhattacharyya coefficient of a cohort and is what this is for; the diagonal
 * is then exactly `ln 1 = 0`.
 *
 * `a` must be finite and strictly positive: 0 would drop both members of every
 * pair and leave the size of the whole space on every entry.
 */
BeamError beam_mixture_power_gram(const BeamModel *const *models, uint32_t k,
                                  double a, double *out);

/**
 * `ln sum_s (P_i(s) P_j(s))^a` for a NAMED LIST of pairs, one pair per thread.
 *
 * `pairs` holds `2 * n_pairs` member indices -- `i` then `j` -- and `out`
 * receives `n_pairs` values in the same order. Out-of-range indices are
 * refused before any sweep opens.
 *
 * This exists because the Gram matrix is the wrong shape for a cohort: its
 * cost is `k(k+1)/2` sweeps, so doubling the cohort quadruples the bill, while
 * a study that wants a thinning null plus a sample of cross-donor pairs needs
 * a few hundred NAMED pairs out of the tens of thousands available. A pair may
 * name one member twice, which at `a = 1/2` must return exactly `ln 1 = 0` and
 * is therefore a check the caller can put anywhere in the list.
 */
BeamError beam_mixture_power_pairs(const BeamModel *const *models, uint32_t k,
                                   double a, const uint32_t *pairs,
                                   uint64_t n_pairs, double *out);

/**
 * Sequences of length `len` that AT LEAST ONE member can produce, exactly.
 *
 * A forward sweep over `(trellis state, member bitmask)` pairs, where the
 * bitmask holds the members that have spelled every edge of the path so far.
 * Every sequence lands on exactly one such pair at the sink, so the counts
 * never double-count and never cancel: the sweep only ever ADDS positive
 * numbers.
 *
 * That is why it is not inclusion-exclusion over the 2^M intersections, which
 * the same product machinery could compute. Those terms alternate in sign and
 * each is astronomically large, so the true union -- often many orders of
 * magnitude smaller than the leading term -- is lost to cancellation. The
 * bitmask sweep costs more code and returns the answer.
 *
 * `cap` is the largest number of live (state, mask) pairs the sweep may hold
 * in one layer; 0 means "as many as the memory budget allows". Exceeding it
 * returns BEAM_ERR_BUDGET rather than a degraded answer, and the message names
 * the length and the count reached. `out_peak` receives the largest number of
 * live pairs any layer held, which is the figure that tells a caller what M
 * this data actually supports -- the structural limit of 64 members is almost
 * never what binds.
 */
BeamError beam_mixture_support_union(const BeamModel *const *models,
                                     uint32_t n_models, uint32_t len,
                                     uint64_t cap, double *out,
                                     uint64_t *out_peak);

/**
 * `beam_mixture_support_union` at every length, and their total.
 *
 * `out_by_length` receives `max_length + 1` counts, 0.0 at any length no
 * member produces; pass NULL if only the total is wanted.
 */
BeamError beam_mixture_support_union_by_length(
        const BeamModel *const *models, uint32_t n_models, uint64_t cap,
        double *out_by_length, double *out_total, uint64_t *out_peak);

/**
 * Refuse a set of models that cannot share a trellis.
 *
 * Checks the same four fields `beam_model_add` checks -- alphabet, order,
 * and max_length -- plus that every member is finalized. Exposed
 * because a caller assembling a mixture wants to fail at construction rather
 * than at the first analytic.
 */
BeamError beam_mixture_check_shape(const BeamModel *const *models,
                                   uint32_t n_models);

#ifdef __cplusplus
}
#endif

#endif /* BEAM_MIXTURE_H */
