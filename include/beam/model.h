/**
 * @file model.h
 * @brief BEAM - Bidirectional Expert-Anchored Model.
 *
 * A probability model over CDR3 amino-acid sequences in which **every residue
 * is scored twice**: once by an N-anchored expert that indexes it by its
 * distance from the front of the sequence, and once by a C-anchored expert
 * that indexes it by its distance from the back. The two readings are combined
 * geometrically, and the result is normalised per length by an exact forward
 * dynamic program.
 *
 * The model consumes **only** the amino-acid sequence and its observed
 * abundance. It never sees a V or J gene call, a donor identity, or a sharing
 * statistic.
 *
 * ## Why two anchors
 *
 * A single left-to-right chain learns the V-germline start well and forgets by
 * the time it reaches the J motif, so it emits terminations that no J gene
 * produces. Anchoring one expert at each terminus recovers germline structure
 * without ever being told which genes exist: the V germline *is* the
 * front-anchored statistic and the J motif *is* the back-anchored statistic.
 *
 * ## The DAG
 *
 * For a fixed length `L` the model is a layered directed acyclic graph:
 *
 *   - a **node** is a position `i` plus the last `K` residues emitted,
 *     sentinel-padded at the start, so there are `BEAM_ALPHABET^K` nodes
 *     per layer. `K = max(order, 3)`, not `order`: the mid expert is always
 *     order 3, so a narrower state could not supply its context;
 *   - an **edge** emits the next residue and carries a weight fixed at build
 *     time, depending only on `(node, residue, i, L)`;
 *   - a **path** from the start node through all `L` layers is in bijection
 *     with a length-`L` sequence.
 *
 * The C-anchored expert does not break acyclicity. It reads residues to the
 * right of position `i`, but its index `r = L-1-i` is a constant once `L` is
 * fixed, so the dependence is absorbed into the edge weight at build time.
 * Concretely the backward score is paid with a lag: emitting position `i`
 * settles the backward score of position `i - K`, which only now has its full
 * right-hand context. That lag is exactly why a node carries `K` residues.
 *
 * Because the score factorises over edges of a topologically ordered DAG,
 * every analytical quantity is an exact dynamic program with no sampling:
 * the partition function, the sequence-probability spectrum, path counts
 * (support size), Hill numbers, and generation.
 *
 * ## Naming
 *
 * The C symbols are `BeamModel` / `beam_model_*` rather than
 * `beam_*`, because `beam_v_prob`, `beam_has_v` and friends in
 * gene_data.h already refer to *graph* edges. The user-facing Python class is
 * `BEAM`.
 *
 * ## Relationship to FlashBack
 *
 * BEAM is a separate model, not a FlashBack variant. It keeps the geometric
 * insight the FlashBack peel exposed on amino acids (distance from each
 * terminus is the informative coordinate) and drops the token fusion, which
 * measurement showed costs ~900x the parameters for ~1% of the information.
 * Nothing in the FlashBack implementation is changed or shared.
 *
 * Reference implementation and the measurements behind every default:
 * `.private/edge/METHOD.md`.
 */
#ifndef BEAM_MODEL_H
#define BEAM_MODEL_H

#include "beam/common.h"
#include "beam/rng.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Fixed dimensions ────────────────────────────────────────── */

/** Number of amino acids the model emits. */
#define BEAM_RESIDUES        20u

/** Emitted residues plus one sentinel used to pad contexts at the termini. */
#define BEAM_ALPHABET        21u

/** Sentinel index. Never emitted; only ever appears in a context slot. */
#define BEAM_NONE            20u

/** Largest sequence length the model can represent. */
#define BEAM_MAX_LENGTH      27u

/** Largest supported context order for the wide expert. */
#define BEAM_MAX_ORDER       4u

/* ── Configuration ───────────────────────────────────────────── */

/**
 * Build-time configuration. `beam_model_config_default()` fills this with
 * the values measured in `.private/edge/METHOD.md`; change a field only with a
 * measurement to justify it.
 */
typedef struct {
    /** Context order of the wide expert, 2..BEAM_MAX_ORDER. Default 4. */
    uint8_t order;

    /**
     * Depth cap of the wide expert. Depths at or beyond this share one bucket,
     * which is what keeps the wide table affordable. Default 15.
     *
     * Clamped to `max_length - 1` at construction: a cap beyond the longest
     * sequence resolves every depth exactly, so clamping changes nothing.
     */
    uint8_t depth_cap;

    /** Longest sequence the model will accept. Default BEAM_MAX_LENGTH. */
    uint8_t max_length;

    /**
     * Log-linear mixing coefficients for the three experts per direction:
     * wide (order `order`, fine depth), mid (order 3, full depth), and
     * length-conditioned (order 2, indexed by depth and total length).
     *
     * These MUST sum to 1. The sum is not a free parameter: it is a pure
     * cross-length calibration knob, and moving it wrecks global ranking while
     * leaving within-length ranking untouched. At sum 1.15 average precision
     * falls from 0.847 to 0.830; at 1.45 it falls to 0.739.
     * Defaults 0.60 / 0.25 / 0.15.
     */
    double coef_wide, coef_mid, coef_len;

    /**
     * Hierarchical backoff strengths (pseudo-counts) for orders 1, 2 and 3 of
     * the mid expert, the wide expert, and the length-conditioned expert.
     * Defaults 20 / 20 / 20, 20, 200.
     */
    double kappa[3];
    double kappa_wide;
    double kappa_len;

    /** Weight on the N-anchored and C-anchored readings. Defaults 0.5, 0.5. */
    double weight_forward, weight_backward;

    /**
     * Closed-world mode: a sequence the graph cannot spell scores exactly zero.
     * Default TRUE.
     *
     * The default is closed-world because that is what the model is for: it
     * should hold what the sequences it was given actually spell, not a full
     * dictionary over every state the alphabet permits. Clearing this field
     * selects the open world instead, and that is now the deliberate choice.
     *
     * Two things make zero unreachable in the open world, and this turns off
     * both. The order-0 base level adds one pseudo-count to every
     * residue, so nothing is ever unseen there; and probabilities are floored
     * at `BEAM_PROB_FLOOR` before the log, so a genuine zero arrives as a
     * large finite negative rather than -inf. With `strict` set, the base
     * level is the raw MLE and no floor is applied, so an unobserved
     * transition is -inf and any path through it is impossible.
     *
     * WHAT IT COSTS. Clearing this field is why BEAM can rank unseen CDR3s:
     * a sequence absent from the corpus still receives a graded probability
     * rather than a flat zero. Under the closed-world default it receives
     * zero, and a held-out sequence using one unobserved context is
     * unscoreable rather than merely improbable. This is the FlashBack trade,
     * made explicitly.
     *
     * WHAT IT BUYS. D0 stops being `sum_L 20**L` and becomes a real path
     * count over the transitions the corpus actually produced, so support and
     * diversity say something about the data instead of about the smoother.
     *
     * Orthogonal to `kappa` by design. `strict` alone kills only the add-one
     * and the floor, so backoff still carries mass from shorter contexts and
     * a sequence is live if SOME level supports every step. Setting `strict`
     * together with all-zero kappa is the full closed-world graph: only the
     * exact observed transitions carry mass. Callers that want FlashBack
     * semantics want both.
     */
    bool strict;

    /**
     * Store closed-world count and probability tables as occupied 20-residue
     * rows instead of dense cell arrays. Default false at the C API level.
     *
     * Sparse storage is exact only when `strict` is true and every backoff
     * strength (`kappa[0..2]`, `kappa_wide`, and `kappa_len`) is zero. A
     * configuration that requests sparse storage outside that regime is
     * rejected: positive backoff can give an unobserved row non-zero mass, so
     * omitting it would change the model.
     *
     * The Python API selects this mode automatically for the usual
     * `strict=True` configuration and exposes `sparse=False` for dense/sparse
     * equivalence checks and reproducibility.
     */
    bool sparse;
} BeamModelConfig;

/** Fill `cfg` with the measured defaults. */
void beam_model_config_default(BeamModelConfig *cfg);

/* ── Opaque model ────────────────────────────────────────────── */

typedef struct BeamModel_ BeamModel;

/* ── Lifecycle ───────────────────────────────────────────────── */

/**
 * Allocate an empty model with the given configuration.
 * Pass NULL for `cfg` to use the defaults.
 * Returns NULL on allocation failure or invalid configuration.
 */
BeamModel *beam_model_new(const BeamModelConfig *cfg);

/** Free a model and everything it owns. NULL-safe. */
void beam_model_destroy(BeamModel *m);

/* ── Training ────────────────────────────────────────────────── */

/**
 * Accumulate counts from a batch of sequences. May be called repeatedly;
 * counts are additive, which is what makes leave-one-cohort-out evaluation a
 * subtraction rather than a rebuild.
 *
 * @param seqs        `n` NUL-terminated amino-acid strings.
 * @param abundances  Per-sequence weight, or NULL for 1 each.
 *
 * Sequences longer than `max_length`, or containing a character outside the
 * 20 amino acids, are skipped and counted in `n_skipped`.
 *
 * Clears `finalized`: call `beam_model_finalize` before scoring again.
 * Not safe to call concurrently with anything else on the same model.
 */
BeamError beam_model_observe(BeamModel *m,
                                const char *const *seqs,
                                const double *abundances,
                                uint32_t n);

/**
 * Subtract another model's counts from this one, flooring at zero.
 * Both models must share the same configuration. Used to hold a cohort out of
 * a foundation model without recounting the corpus.
 *
 * The exact inverse of `beam_model_add`: `total_weight`, `n_observed` and
 * `n_skipped` all move with the cells, so a corpus size survives an add
 * followed by a subtract.
 *
 * Refused with `BEAM_ERR_EMPTY_INPUT`, leaving the model untouched, if the
 * operand accounts for every sequence in the target. The counts are the only
 * copy there is, so a subtraction that empties them cannot be undone and would
 * leave a model whose `finalize` can only fail.
 *
 * Clears `finalized`: call `beam_model_finalize` before scoring again.
 */
BeamError beam_model_subtract(BeamModel *m, const BeamModel *other);

/**
 * Add another model's counts into this one. The inverse of
 * `beam_model_subtract`, and the reason both exist: counts are additive,
 * so a corpus can be counted in shards and combined without re-reading
 * anything. Both models must share the same configuration.
 *
 * `total_weight`, `n_observed` and `n_skipped` all move with the cells.
 * Clears `finalized`: call `beam_model_finalize` before scoring again.
 */
BeamError beam_model_add(BeamModel *m, const BeamModel *other);

/**
 * Dirichlet-multinomial posterior update: fold a prior model's belief into
 * this model's counts as `kappa` pseudo-observations.
 *
 * Every row of every count table is one categorical distribution — over the
 * 20 residues for the three expert tables, over lengths for the length
 * histogram. This adds `kappa * p_prior[row]` to each of this model's rows,
 * where `p_prior` is that row of the prior NORMALISED. After the finalize the
 * caller must then perform, a row reads
 *
 *     p_post = (c_individual + kappa * p_prior) / (n_individual + kappa)
 *
 * before BEAM's own hierarchical backoff, which is the same convention
 * `beam_flashback_graph_posterior` uses for graph edges.
 *
 * `kappa = 0` is a no-op on the counts. A row the prior never observed is
 * left alone: an empty row carries no belief, and spreading `kappa` uniformly
 * over it would invent one. Both models must share the configuration, and
 * neither may be compacted.
 *
 * The sequence counters do not move. `total_weight` and `n_observed` report
 * what was observed and a pseudo-count is not an observation, so a
 * leave-one-out corpus size does not shift when the smoothing strength does.
 *
 * Clears `finalized`, exactly as `add` and `subtract` do: call
 * `beam_model_finalize` before scoring.
 */
BeamError beam_model_posterior(BeamModel *m,
                                  const BeamModel *prior,
                                  double kappa);

/**
 * Turn accumulated counts into log-probability tables and compute the
 * per-length partition functions. Must be called before any scoring or
 * analytics entry point. Calling it again after more `observe` calls is
 * legal and recomputes from the current counts.
 *
 * Atomic with respect to failure: the replacement tables are built in full
 * before the existing ones are released, so a finalize that fails leaves the
 * model exactly as it was rather than stripped of its tables. That costs a
 * second table set at peak, which for a large order-4 model is the figure
 * `beam_model_memory_bytes()` reports for the tables alone.
 *
 * NOT atomic with respect to CONCURRENCY, and neither are `observe`, `add` or
 * `subtract`. Any of them may run while no other thread is touching the model.
 * The commit at the end of a finalize is a run of ordinary pointer stores
 * followed by a free, so a reader racing it can still see a half-installed
 * table set or a pointer that has just been released. A caller that releases
 * the GIL (or otherwise reads from another thread) must serialise every
 * mutation against every reader itself. Concurrent READS are safe.
 */
BeamError beam_model_finalize(BeamModel *m);

/** True once `beam_model_finalize` has succeeded. */
bool beam_model_is_finalized(const BeamModel *m);

/**
 * Release the raw count tables, keeping only the finalized log-probability
 * tables. Roughly halves resident memory, at the cost of making `observe`,
 * `subtract`, `save` and re-`finalize` unavailable on this model.
 *
 * Worth doing for a large order-4 foundation model that is only going to be
 * scored against: see `beam_model_memory_bytes()` for the numbers.
 */
BeamError beam_model_compact(BeamModel *m);

/**
 * Approximate resident bytes of the model's tables. Dense models report their
 * flat count/probability payloads; sparse models report directories, allocated
 * row capacity, reverse maps, and their small metadata arrays.
 */
uint64_t beam_model_memory_bytes(const BeamModel *m);

/* ── Scoring ─────────────────────────────────────────────────── */

/**
 * Natural log of the sequence probability, `ln P(s)`.
 *
 * In the default open-world configuration, returns `-INFINITY` only for a
 * sequence the model cannot read: one longer than `max_length`, one containing
 * a character outside the 20 amino acids, or one whose length was never
 * observed and so has no length prior. Hierarchical backoff keeps every
 * readable sequence of an observed length finite in that regime. With
 * `strict` enabled, a path containing an unsupported transition also returns
 * `-INFINITY`.
 */
double beam_model_log_pseq(const BeamModel *m, const char *seq);

/**
 * Batch form of `beam_model_log_pseq`. Parallelised when the build has
 * OpenMP. `out` must have room for `n` doubles.
 */
BeamError beam_model_log_pseq_batch(const BeamModel *m,
                                       const char *const *seqs,
                                       uint32_t n,
                                       double *out);

/**
 * Per-residue decomposition of a score, for inspection and for figures.
 *
 * Writes `L` entries into each non-NULL output array, where `L` is the
 * sequence length: the N-anchored and C-anchored log-probabilities of each
 * residue, before mixing weights are applied. Returns the length in `*out_len`.
 *
 * This is the table that makes the model legible: the N-anchored expert is
 * confident at the start, the C-anchored expert at the end, and both are
 * surprised across the junction.
 */
BeamError beam_model_explain(const BeamModel *m,
                                const char *seq,
                                double *out_forward,
                                double *out_backward,
                                uint32_t *out_len);

/* ── Per-length quantities ───────────────────────────────────── */

/** Natural log of the partition function `Z(L)`. */
double beam_model_log_partition(const BeamModel *m, uint32_t length);

/** Natural log of the length prior `pi(L)`, the observed length distribution. */
double beam_model_log_length_prior(const BeamModel *m, uint32_t length);

/* ── Analytics (exact, no sampling) ──────────────────────────── */

/**
 * Number of distinct sequences of length `L` the model can produce, by exact
 * path counting over the trellis.
 *
 * In the default open-world configuration it will be
 * `BEAM_RESIDUES^L`, and that is not a bug. Hierarchical backoff leaves
 * every table cell strictly positive, so the number describes the smoother,
 * not the grammar. With `strict` and zero backoff, dead transitions remain
 * dead and this instead reports the exact support of the observed-transition
 * trellis. Dense and sparse storage return the same count.
 *
 * The function is kept because it is the exact q=0 corner of the Hill family
 * and because it is what the counting spectrum integrates to, so the two have
 * to agree.
 */
double beam_model_path_count(const BeamModel *m, uint32_t length);

/** Total support size, summed over every length the model generates. */
double beam_model_path_count_total(const BeamModel *m);

/**
 * Exact surprisal spectrum by forward grid transport.
 *
 * Fills `weights[0..bins-1]`, where bin `b` covers surprisal
 * `[b*spacing, (b+1)*spacing)` in nats and surprisal is `-ln P(s)`.
 *
 * @param counting  true weights every supported sequence 1, so the spectrum
 *                  integrates to the support size (see the warning on
 *                  `beam_model_path_count`); false weights each sequence
 *                  by its own probability, so it integrates to 1.
 * @param length    a specific length, or 0 to pool over all lengths.
 *
 * Mass is split linearly between adjacent bins, so the total is conserved
 * exactly and the mean surprisal is unbiased.
 *
 * Pass `spacing <= 0` to have a spacing chosen automatically from
 * `beam_model_accumulator_max()`. The chosen value is returned through
 * `out_spacing` when that is non-NULL.
 */
BeamError beam_model_spectrum(const BeamModel *m,
                                 bool counting,
                                 uint32_t length,
                                 uint32_t bins,
                                 double spacing,
                                 double *weights,
                                 double *out_spacing);

/**
 * Largest value a spectrum grid has to represent, in nats.
 *
 * This bounds TWO different things, and they are not the same:
 *
 *   - the DP's running accumulator, which grows monotonically along the path;
 *   - the final surprisal, which is that accumulator plus the per-length
 *     constant `ln Z(L) - ln pi(L) - tail`.
 *
 * Which of the two is larger is not fixed, so this returns the larger. On the
 * models this library actually builds the surprisal is the binding one: the
 * sink term `-tail` is positive, and `ln Z(L) - ln pi(L)` measures a fraction
 * of a nat either way (from -1.6 to +0.005 across lengths on a small corpus),
 * so bounding only the accumulator would size the grid too small and let mass
 * fall off the end. The reverse can happen when the per-length constant is
 * strongly negative, in which case a grid sized to the surprisal alone would
 * index past its own top. Size grids with this, which covers both.
 */
double beam_model_accumulator_max(const BeamModel *m);

/**
 * Both spectrum measures through ONE transport.
 *
 * `counting_weights` receives the measure that gives every producible
 * sequence weight 1, `generated_weights` the one that gives it its own
 * probability. Identical to calling `beam_model_spectrum` twice, at half
 * the cost — and, unlike calling it twice, guaranteed to put both on the same
 * grid with the same splitting fractions.
 *
 * That guarantee is the reason this exists rather than being a convenience.
 * Linear splitting spreads each sequence across adjacent bins, and only if
 * both measures are split by the SAME fractions is
 * `generated[b] / counting[b]` the exact mass-weighted mean probability of
 * the sequences in bin `b`. Two independent calls agree today because they
 * recompute the same arithmetic from the same inputs; nothing holds them
 * together, and everything downstream of that ratio assumes they are.
 *
 * `length = 0` pools over every length the model generates, exactly as
 * `beam_model_spectrum` does. Both buffers must hold `bins` doubles and
 * are overwritten. `spacing <= 0` sizes the grid automatically and reports it
 * through `out_spacing`.
 */
BeamError beam_model_spectrum_pair(const BeamModel *m,
                                      uint32_t length,
                                      uint32_t bins,
                                      double spacing,
                                      double *counting_weights,
                                      double *generated_weights,
                                      double *out_spacing);

/**
 * Every length's spectrum in one call, on ONE shared grid.
 *
 * `weights` must hold `(max_length + 1) * bins` doubles; row `L` receives the
 * spectrum of length `L` and rows for lengths the model does not generate are
 * left at zero. `max_length = 0` means the model's own maximum.
 *
 * Two things distinguish this from a loop over `beam_model_spectrum`. The
 * grid is sized once and shared, so the rows are directly comparable and can
 * be summed into the pooled spectrum; and the lengths run concurrently, capped
 * by `beam_model_set_memory_budget` rather than by core count, because
 * each length's scratch is measured in gigabytes at order 4.
 *
 * `spacing <= 0` sizes the grid automatically and reports it through
 * `out_spacing`. Summing the rows reproduces the pooled spectrum at the same
 * spacing to floating-point round-off.
 */
BeamError beam_model_spectra_by_length(const BeamModel *m,
                                          bool counting,
                                          uint32_t max_length,
                                          uint32_t bins,
                                          double spacing,
                                          double *weights,
                                          double *out_spacing);

/** Shannon entropy of the sequence distribution, in nats. Exact. */
BeamError beam_model_entropy(const BeamModel *m, double *out);

/**
 * Hill number of order `alpha`: the effective number of sequences.
 * `alpha = 0` gives the support size, `alpha = 1` the exponential of the
 * entropy, `alpha = 2` the inverse collision probability.
 */
BeamError beam_model_hill_number(const BeamModel *m,
                                    double alpha, double *out);

/* ── Mellin transform and surprisal moments ──────────────────── */

/**
 * `ln sum_s P(s)^q`, the log of the Mellin transform of the sequence
 * probability distribution, exactly.
 *
 * This is the generator every diversity quantity here is built from:
 * `q = 0` gives the log support size, `q = 1` gives 0, and the derivatives
 * with respect to `q` are the cumulants of `ln P`.
 */
BeamError beam_model_log_mellin(const BeamModel *m, double q, double *out);

/**
 * Raw moments of the surprisal `-ln P(s)` under the `q`-tilted distribution
 * `P_q(s) proportional to P(s)^q`.
 *
 * Writes `order + 1` values, `out[k] = E_q[(-ln P)^k]`, so `out[0]` is 1 and
 * `out[1]` is the mean surprisal. `q = 1` gives the ordinary distribution, in
 * which `out[1]` is the entropy.
 *
 * Exact, by a jet dynamic program that carries the first `order` moments of
 * the accumulated path weight alongside the mass. No sampling.
 */
BeamError beam_model_surprisal_moments(const BeamModel *m, double q,
                                          uint32_t order, double *out);

/**
 * The three quantities a saddlepoint step needs, from ONE sweep.
 *
 * `*out_log_mellin` is `ln sum_s P(s)^q`; `*out_mean` and `*out_var` are the
 * mean and variance of the surprisal `-ln P` under the `q`-tilted law. In CGF
 * terms, with `Y = -ln P` and `K_Y(s) = ln E[exp(sY)] = log_mellin(1 - s)`,
 * these give `K_Y`, `K_Y'` and `K_Y''` at `s = 1 - q` directly.
 *
 * Provided separately from `surprisal_moments` because a Newton solve calls it
 * once per iteration per evaluation point, and doing three passes over the
 * trellis where one suffices would triple the cost of every saddlepoint.
 */
BeamError beam_model_surprisal_tilt(const BeamModel *m, double q,
                                       double *out_log_mellin,
                                       double *out_mean, double *out_var);

/**
 * Expected contribution of each layer to the total surprisal, under the
 * `q`-tilted law. `out` must hold `max_length` doubles.
 *
 * The exact analogue of asking which edges of a graph carry the surprisal: it
 * shows the conserved termini costing almost nothing and the junction carrying
 * the information.
 *
 * `out` sums to the mean surprisal MINUS `*out_length_constant`, which is the
 * per-length normalisation `E[ln Z(L) - ln pi(L)]`. That constant belongs to
 * the choice of length rather than to any position, so it is reported
 * separately instead of smeared across depths; add the two to recover the
 * mean surprisal exactly. Pass NULL if you do not need it.
 */
BeamError beam_model_surprisal_by_depth(const BeamModel *m, double q,
                                           double *out,
                                           double *out_length_constant);

/**
 * Live nodes and edges of the trellis at one length, after backward trimming.
 *
 * `out_nodes` and `out_edges` receive totals; `out_layer_nodes`, if non-NULL,
 * receives `length + 1` per-layer live-state counts.
 *
 * Expect these to come back saturated in the default open-world model:
 * hierarchical backoff leaves every cell positive, so every reachable state
 * is live. Under strict zero-backoff scoring, the same trimming pass reports
 * only the structure supported by non-zero transitions.
 */
BeamError beam_model_topology(const BeamModel *m, uint32_t length,
                                 double *out_nodes, double *out_edges,
                                 uint32_t *out_layer_nodes);

/**
 * Cumulants of the surprisal under the `q`-tilted distribution, written as
 * `order` values `out[0..order-1]` for cumulants 1..order. The first is the
 * mean and the second the variance.
 */
/**
 * The q-tilted surprisal moments of EVERY length, unmixed.
 *
 * `out` must hold `(max_length + 1) * (order + 1)` doubles. Row `L` is
 *
 *     out[L * (order+1) + 0]  the length's share of the q-tilted law
 *     out[L * (order+1) + k]  E_q[ (-ln P)^k | length L ],  k >= 1
 *
 * and rows for lengths the model does not generate are zero. The shares sum
 * to 1, and at `q = 1` they are the length prior itself.
 *
 * This is `beam_model_surprisal_moments` with the cross-length mixture
 * left undone, and the two are related by exactly that mixture:
 * `mu[k] = sum_L out[L][0] * out[L][k]`. Slot 0 carries the share rather than
 * the constant 1 a raw jet would put there precisely so that a caller can
 * check the identity instead of taking it on trust.
 */
BeamError beam_model_length_derivatives(const BeamModel *m,
                                           double q,
                                           uint32_t order,
                                           double *out);

BeamError beam_model_surprisal_cumulants(const BeamModel *m, double q,
                                            uint32_t order, double *out);

/**
 * Smallest and largest surprisal any producible sequence attains, in nats.
 *
 * Exact, by min-plus and max-plus sweeps over the DAG. Their difference is the
 * dynamic range: how many orders of magnitude of probability the grammar
 * spans.
 */
BeamError beam_model_surprisal_range(const BeamModel *m,
                                        double *out_min, double *out_max);

/* ── Per-length marginals ────────────────────────────────────── */

/**
 * Decomposition of the two spectra by sequence length. Each output array must
 * hold `max_length + 1` doubles, and any may be NULL.
 *
 * @param counting        distinct producible sequences at each length
 * @param generated       probability mass at each length, i.e. pi(L)
 * @param mean_surprisal  E[-ln P | L], the mean surprisal within each length
 * @param var_surprisal   Var[-ln P | L], the spread within each length
 *
 * `counting` sums to the support size and `generated` sums to 1, which is what
 * makes this a genuine decomposition rather than a summary.
 */
BeamError beam_model_length_marginals(const BeamModel *m,
                                         double *counting,
                                         double *generated,
                                         double *mean_surprisal,
                                         double *var_surprisal);

/* ── Most probable sequences ─────────────────────────────────── */

/**
 * The `k` most probable sequences the model can produce, in descending order
 * of probability.
 *
 * Exact, not a beam: a best-first search over the DAG guided by the backward
 * values, which are an exact completion cost rather than a heuristic, so the
 * `n`-th sequence popped is genuinely the `n`-th most probable.
 *
 * @param length  a single length, or 0 to search across all lengths.
 * @param out         receives `k * (max_length + 1)` bytes of NUL-terminated
 *                    strings; may not be NULL.
 * @param out_log_p   receives `k` log probabilities; may be NULL.
 * @param out_n       receives how many were actually found, which is fewer
 *                    than `k` only if the model cannot produce that many.
 */
BeamError beam_model_top_k(const BeamModel *m, uint32_t k,
                              uint32_t length, char *out,
                              double *out_log_p, uint32_t *out_n);

/* ── Generation ──────────────────────────────────────────────── */

/**
 * Draw `n` sequences from the model, exactly in proportion to their
 * probability, by walking the trellis against its backward values.
 *
 * Writes NUL-terminated strings into `out`, which must have room for
 * `n * (max_length + 1)` bytes. `out_lengths` may be NULL.
 *
 * @param length  a specific length, or 0 to draw the length from `pi(L)`.
 */
BeamError beam_model_sample(const BeamModel *m,
                               uint32_t n,
                               uint32_t length,
                               BeamRng *rng,
                               char *out,
                               uint32_t *out_lengths);

/* ── Introspection ───────────────────────────────────────────── */

/** The configuration this model was built with. */
const BeamModelConfig *beam_model_config(const BeamModel *m);

/** Total abundance observed during training. */
double beam_model_total_weight(const BeamModel *m);

/** Number of sequences accepted during training. */
uint64_t beam_model_n_observed(const BeamModel *m);

/** Number of sequences rejected as too long or non-amino-acid. */
uint64_t beam_model_n_skipped(const BeamModel *m);

/** Nodes per layer of the DAG, `BEAM_ALPHABET^order`. */
uint32_t beam_model_n_states(const BeamModel *m);

/** Total table cells, and how many carry a non-zero count. */
uint64_t beam_model_n_cells(const BeamModel *m);
uint64_t beam_model_n_nonzero_cells(const BeamModel *m);

/* ── Persistence ─────────────────────────────────────────────── */

/**
 * Write the model's counts and configuration to `path`. Counts rather than
 * derived tables, so a loaded model can still be subtracted from and
 * re-finalized under different smoothing. Sparse counts are persisted as
 * sorted occupied-row records; the reader also accepts legacy dense format-1
 * files.
 */
BeamError beam_model_save(const BeamModel *m, const char *path);

/** Read a model written by `beam_model_save`. Returns NULL on failure. */
BeamModel *beam_model_load(const char *path);

/**
 * @brief True if `path` is a model file written before the BEAM split.
 *
 * Those files carry the old EDGE magic and cannot be loaded: the format is
 * unchanged but the identifier is not, and the foundation models are being
 * rebuilt rather than migrated. This exists so a caller can say "this is a
 * pre-BEAM file, rebuild it" instead of the generic load failure, which is
 * otherwise indistinguishable from a truncated or corrupt file.
 *
 * Returns false for a missing file, a short file, or any other magic.
 */
bool beam_model_is_legacy_file(const char *path);

/* ── Parallelism ─────────────────────────────────────────────── */

/**
 * Cap the number of threads BEAM uses, process-wide.
 *
 * A library that silently grabs every core is unusable inside a job scheduler
 * or alongside another parallel stage, so this is a hard cap rather than a
 * hint: every parallel region in the BEAM implementation runs with at most
 * `n` threads.
 *
 * `n = 0` restores the default, which is whatever the OpenMP runtime would
 * choose on its own (`OMP_NUM_THREADS`, else one thread per core). Negative
 * values are treated as 0. A build without OpenMP ignores the setting and
 * always reports 1.
 *
 * The setting is global and not synchronised: set it before starting work,
 * not from one thread while another is inside a computation.
 *
 * Results do NOT depend on it. Every reduction in the implementation is
 * either order-independent (a maximum) or summed in a fixed index order, so a
 * one-thread run and a sixteen-thread run agree bit for bit.
 */
void beam_model_set_threads(int n);

/**
 * The cap set by `beam_model_set_threads`, or 0 if none is in force.
 * This is the configured value, not the number of threads actually available.
 */
int beam_model_threads(void);

/**
 * Threads a parallel region would actually start right now: the cap if one is
 * set, otherwise the OpenMP runtime's own maximum, and always 1 in a build
 * without OpenMP.
 */
int beam_model_effective_threads(void);

/**
 * Cap the scratch memory concurrent per-length dynamic programs may hold.
 *
 * Only the spectrum is large enough to care: its per-state histogram grids
 * run to 1.6 GB at order 4 with 512 bins, so `spectra_by_length` would ask
 * for twenty-five gigabytes if it ran one length per core. It divides this
 * budget by what one length needs and runs that many at a time.
 *
 * The default is 4 GB. `bytes = 0` restores it. This caps CONCURRENCY only:
 * a single length always runs whatever its size, so lowering this can never
 * turn a working call into a failing one, only a slower one. Results do not
 * depend on it.
 */
void beam_model_set_memory_budget(uint64_t bytes);

/** The budget in force, in bytes. */
uint64_t beam_model_memory_budget(void);

#ifdef __cplusplus
}
#endif

#endif /* BEAM_MODEL_H */
