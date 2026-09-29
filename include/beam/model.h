/**
 * @file model.h
 * @brief BEAM - Bidirectional Expert-Anchored Model.
 *
 * A probability model over CDR3 sequences in which **every residue is scored
 * twice**: once by an N-anchored expert that indexes it by its distance from
 * the front of the sequence, and once by a C-anchored expert that indexes it
 * by its distance from the back. The two readings are combined geometrically,
 * and the result is normalised per length by an exact forward dynamic
 * program.
 *
 * The model consumes **only** the sequence and its observed abundance. It
 * never sees a V or J gene call, a donor identity, or a sharing statistic.
 *
 * ## Alphabets
 *
 * The model is the same over either alphabet -- one anchored table per
 * direction, combined geometrically, exact dynamic programs. Only the
 * symbols change, and with them the size of every table: twenty amino acids
 * at radix 21, or the five nucleotide residues ACGTN at radix 6. See
 * `BeamAlphabet` below, and `src/native/core/alphabet_*.c` for the
 * descriptors that carry each one's defaults and ceilings.
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
 *     sentinel-padded at the start, so there are `radix^K` nodes
 *     per layer, with `K = order`: a state holds exactly the residues one
 *     context needs. It was `max(order, 3)` while a second expert of fixed
 *     order 3 existed, because a narrower state could not supply its
 *     context;
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
 * `.private/edge/METHOD.md`, a working-tree document that is not shipped
 * with the package. Every number quoted in these comments comes from it.
 */
#ifndef BEAM_MODEL_H
#define BEAM_MODEL_H

#include "beam/common.h"
#include "beam/rng.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Alphabets ───────────────────────────────────────────────── */

/**
 * Which symbol alphabet a model is built over.
 *
 * The model is the same either way: one anchored table per direction,
 * combined geometrically, exact dynamic programs. Only the symbols change.
 *
 * `BEAM_ALPHABET_AA` is zero deliberately. `beam_model_config_default()` opens
 * with a `memset`, and a caller who zero-initialises a config has always meant
 * "amino acid", so that keeps meaning what it meant.
 */
typedef enum {
    BEAM_ALPHABET_AA = 0,   /**< 20 amino acids, radix 21. */
    BEAM_ALPHABET_NT = 1    /**< A C G T N, radix 6.       */
} BeamAlphabet;

/** Number of alphabets, for sizing a table indexed by BeamAlphabet. */
#define BEAM_ALPHABET_COUNT   2u

/* Per-alphabet dimensions. `RADIX` is the emitted residues plus one sentinel
 * used to pad contexts at the termini; `NONE` is that sentinel's index, never
 * emitted and only ever appearing in a context slot. */

#define BEAM_AA_RESIDUES     20u
#define BEAM_AA_RADIX        21u
#define BEAM_AA_NONE         20u

/* N is an emitted residue, not a wildcard: a sequence containing it is
 * scoreable, and the model learns a distribution over it. */
#define BEAM_NT_RESIDUES      5u
#define BEAM_NT_RADIX         6u
#define BEAM_NT_NONE          5u

/**
 * Maxima over all alphabets. These size the few genuinely fixed-length arrays
 * that a runtime alphabet cannot: per-row scratch buffers and the I/O record
 * staging block. Sizing those by the largest alphabet costs bytes on the
 * stack and keeps the code alphabet-agnostic.
 */
#define BEAM_MAX_RESIDUES    20u
#define BEAM_MAX_RADIX       21u

/** `char_to_idx` entry for a byte that is not a residue of its alphabet. */
#define BEAM_RESIDUE_INVALID 0xFFu

/* ── Fixed dimensions ────────────────────────────────────────── */

/**
 * The amino-acid default sequence length.
 *
 * This was the ceiling as well as the default while there was one alphabet.
 * The ceiling is now BEAM_MAX_LENGTH_CEILING and the default comes from the
 * alphabet descriptor, since nucleotides need three times the length to say
 * the same thing.
 */
#define BEAM_MAX_LENGTH      27u

/**
 * The structural ceiling on `max_length`, as opposed to today's validation
 * limit.
 *
 * `max_length` is a `uint8_t`, so 255 is the most any alphabet could ever ask
 * for, and it is what the length-indexed stack buffers must be sized by once
 * `max_length` becomes a genuinely free parameter. Nucleotide sequences are
 * three times the length of the amino acids they encode, so 27 cannot remain
 * the limit for both.
 */
#define BEAM_MAX_LENGTH_CEILING 255u

/**
 * Largest order any alphabet permits, and therefore the size of the
 * order-indexed scratch arrays: `slot[]` in the context packers, and
 * `slot[]` in the context packers.
 *
 * It sized `level_counts[]` in the backoff hierarchy too, until that went with
 * the three-expert mixture.
 *
 * It is a sizing bound, not the validation limit. What a given config may ask
 * for comes from its alphabet's own ceiling, because the real constraint is
 * on `radix^order` and the radices are far apart.
 */
#define BEAM_MAX_ORDER       9u

/**
 * Starting strengths of the hierarchical prior, in pseudo-observations: every
 * context level and the length distribution. A starting point, not a
 * measurement; fit them to a corpus by held-out likelihood.
 */
#define BEAM_DEFAULT_ALPHA        10.0
#define BEAM_DEFAULT_ALPHA_LENGTH 10.0

/* ── Configuration ───────────────────────────────────────────── */

/**
 * Build-time configuration. `beam_model_config_default()` fills this with
 * the values measured in `.private/edge/METHOD.md` (working tree, not
 * shipped); change a field only with a
 * measurement to justify it.
 */
typedef struct {
    /**
     * Which alphabet the sequences are written in. Default BEAM_ALPHABET_AA.
     *
     * It comes first because it decides what the rest of this struct means:
     * the legal range of `order`, the default `max_length`, and the size of
     * every table those imply. Zero is the amino-acid alphabet, so a config
     * that was zeroed rather than defaulted still asks for what it always
     * asked for.
     */
    BeamAlphabet alphabet;

    /**
     * Context order, from 2 up to the alphabet's own ceiling: 4 for amino
     * acids, 9 for nucleotides. Defaults to 4 and 6 respectively.
     *
     * The ceiling is per-alphabet because it bounds `radix^order`, and the
     * two radices are far apart: 21^9 does not fit in the `uint32_t` the
     * context count is held in, while 6^9 fits comfortably.
     */
    uint8_t order;

    /**
     * Longest sequence the model will accept, from 1 up to
     * BEAM_MAX_LENGTH_CEILING. Defaults to the alphabet's own typical
     * length: 27 for amino acids, 81 for the nucleotides encoding them.
     *
     * Free rather than fixed, and worth choosing deliberately: every table is
     * sized by it, so a length far beyond the corpus buys nothing and costs
     * proportionally.
     */
    uint8_t max_length;

    /**
     * Weight on the N-anchored and C-anchored readings. Defaults 0.5, 0.5.
     *
     * These SHOULD sum to 1, and the sum is not a free parameter: it is a
     * pure cross-length calibration knob. Off 1, `log W` scales with the
     * length, which leaves within-length ranking untouched and wrecks
     * cross-length comparison. It is not refused, because `(1, 0)` and
     * `(0, 1)` are the only way to measure each anchor's chain on its own,
     * which is what shows where a partition function below 1 comes from:
     * the two chains disagreeing about a sequence.
     */
    double weight_forward, weight_backward;

    /**
     * Prior strengths of the hierarchical Bayesian smoothing, one per context
     * level. Every sequence up to `max_length` has positive probability; the
     * observed ones pull the mass toward themselves.
     *
     * A row of either table is (depth, the `order` nearest residues). Its
     * level-k parent is the same depth with only the k nearest residues, and
     * level 0 is the depth alone. Each level is the posterior mean of a
     * Dirichlet centred on its parent:
     *
     *     P_k(x) = (n_k(x) + alpha[k] * P_{k-1}(x)) / (n_k + alpha[k])
     *
     * with P_{-1} the uniform 1/R. `n_k` are the counts the model already
     * keeps, summed over the residues level k drops. A row nothing was ever
     * observed at is therefore exactly its parent, so an unseen context
     * predicts what its shorter context predicts rather than being
     * impossible (the old closed world) or uniform (the old open world).
     *
     * `alpha[k]` for k = 0..order is how many pseudo-observations the parent
     * is worth at level k; entries past `order` are ignored. Every one must be
     * finite and positive: a zero would let an unobserved residue reach
     * probability zero and bring the closed world back through one level.
     */
    double alpha[BEAM_MAX_ORDER + 1];

    /**
     * Prior strength of the length distribution:
     *
     *     P(L) = (n_L + alpha_length / max_length) / (N + alpha_length)
     *
     * over every length 1..max_length, so a length the corpus never produced
     * is improbable rather than impossible. Finite and positive.
     */
    double alpha_length;

    /**
     * Store only the rows something was observed at, and resolve every other
     * row through its parent at lookup, instead of materialising all
     * `max_length * radix^order` rows. Default TRUE.
     *
     * Both layouts represent the same model bit for bit; dense exists for
     * equivalence checks and costs gigabytes at order 4.
     */
    bool sparse;
} BeamModelConfig;

/** Fill `cfg` with the measured defaults, for the amino-acid alphabet. */
void beam_model_config_default(BeamModelConfig *cfg);

/**
 * Fill `cfg` with the measured defaults for `alphabet`.
 *
 * The anchor weights are the same
 * whichever alphabet is asked for: they were measured on amino acids, and
 * re-tuning them for nucleotides needs its own measurement rather than a
 * guess. What the alphabet decides is `order` and `max_length`, which follow
 * from how much context it can afford and how long its sequences are.
 *
 * An unknown alphabet leaves `cfg` untouched.
 */
void beam_model_config_default_for(BeamAlphabet alphabet,
                                   BeamModelConfig *cfg);

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
 * Raise a finalized model to a power: `P_t(s) proportional to P(s)^t`.
 *
 * In place, and only on a FINALIZED model, because it is the log tables rather
 * than the counts that are scaled. `t` must be finite and strictly positive:
 * `t = 0` would send every finite log weight to zero and every -inf to NaN,
 * and a negative `t` would turn an unspellable edge into the likeliest one.
 *
 * `t > 1` concentrates the distribution on its likeliest sequences and `t < 1`
 * flattens it toward the rare ones. The SUPPORT is unchanged either way, so
 * `beam_model_path_count` does not move.
 *
 * Exact: the result satisfies `ln P_t(s) = t ln P(s) - ln sum_s P(s)^t`
 * identically, the second term being `beam_model_log_mellin` at `t`.
 *
 * RELEASES THE COUNTS, as `beam_model_compact` does, and for a reason that is
 * not about memory: a tempered model's distribution is not a function of its
 * counts, so add, subtract, posterior and copy would each silently rebuild the
 * untempered tables. They refuse instead.
 */
BeamError beam_model_temper(BeamModel *m, double t);

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
 * 20 residues for the two anchored tables, over lengths for the length
 * histogram. This adds `kappa * p_prior[row]` to each of this model's rows,
 * where `p_prior` is that row of the prior NORMALISED. After the finalize the
 * caller must then perform, a row reads
 *
 *     p_post = (c_individual + kappa * p_prior) / (n_individual + kappa)
 *
 * before BEAM's own smoothing, which is the same convention
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
 * Change the hierarchical prior's strengths and re-finalize from the counts.
 *
 * `alpha` holds `n_alpha` = order + 1 strengths, one per level 0..order, and
 * `alpha_length` the length prior's. Validated exactly as a build config is;
 * on any refusal the model is left as it was. Nothing is re-observed: the
 * counts are the only state the prior reads, which is what makes fitting the
 * strengths by held-out likelihood a sequence of re-finalizes of one model.
 */
BeamError beam_model_set_prior(BeamModel *m, const double *alpha,
                               uint32_t n_alpha, double alpha_length);

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
 * Approximate resident bytes of the model's TABLES. Dense models report their
 * flat count/probability payloads; sparse models report directories, allocated
 * row capacity, reverse maps, and their small metadata arrays.
 *
 * Measured against what the build actually allocates, this is exact to within
 * 0.2 MB. It excludes the per-length live cache, which appears after the first
 * analytic and can exceed this figure -- see `beam_model_live_cache_bytes`.
 */
uint64_t beam_model_memory_bytes(const BeamModel *m);

/**
 * Bytes held by the per-length live cache, which `beam_model_memory_bytes`
 * does NOT include.
 *
 * The cache is a memoisation of the trellis structure each analytic needs.
 * Without it every analytic rebuilds that structure, which is 78% of a
 * converted dynamic program on a small model and 52% on a large one -- so it
 * is not an optimisation a caller would want silently disabled. But it is
 * real memory, it appears only after the first analytic runs, and it is
 * bounded by a quarter of `beam_model_memory_budget()` -- one gigabyte at the
 * default -- so a caller holding several models can be holding far more than
 * their reported sizes suggest:
 *
 *              memory_bytes   this   actually allocated
 *   repertoire      13.8 MB   15.1 MB        29.2 MB
 *   foundation     806.7 MB  382.1 MB     1,190.4 MB
 *
 * Zero before the first analytic, and after `beam_model_drop_caches`.
 */
uint64_t beam_model_live_cache_bytes(const BeamModel *m);

/**
 * Release everything the model is holding that it can rebuild.
 *
 * Only the live cache today. Changes no result: the next analytic rebuilds
 * what it needs and pays the build cost again. For a caller that has finished
 * with a model but must keep it, or that is holding more models than fit.
 *
 * Not safe to call while another thread is inside an analytic on the same
 * model, which is the rule for every other mutation in this library.
 */
void beam_model_drop_caches(BeamModel *m);

/* ── Scoring ─────────────────────────────────────────────────── */

/**
 * Natural log of the sequence probability, `ln P(s)`.
 *
 * Finite for every sequence the model can represent: under the hierarchical
 * prior every string over the alphabet of length 1..max_length has positive
 * probability. `-INFINITY` only for a sequence it cannot read at all -- empty,
 * longer than `max_length`, or containing a character outside the alphabet.
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

/**
 * `log Z(L)` split across the positions it is a product over.
 *
 * `out` receives `length` entries: `log M_i`, the log of the trellis mass
 * once residue `i` has been emitted. Differencing them gives the per-layer
 * factors `Z(L)` is a product of, so a caller that must place `-log Z(L)` on
 * individual positions takes the trellis's own split rather than choosing
 * one. The last entry is `log Z(L)` itself and the differences telescope to
 * it exactly.
 *
 * The sink is attributed to the last layer, as `beam_model_explain` does.
 * Costs one forward sweep, where `beam_model_log_partition` reads a cached
 * value.
 */
BeamError beam_model_log_partition_by_layer(BeamModel *m, uint32_t length,
                                            double *out);

/** Natural log of the length prior `pi(L)`, the observed length distribution. */
double beam_model_log_length_prior(const BeamModel *m, uint32_t length);

/* ── Analytics (exact, no sampling) ──────────────────────────── */

/**
 * Number of distinct sequences of length `L` the model can produce, by exact
 * path counting over the trellis.
 *
 * Under the hierarchical prior this is exactly `R^L` for R residues: every
 * table cell is positive, so every string of the length is possible. The
 * number describes the alphabet, not the corpus. Dense and sparse storage
 * return the same count.
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
 * `beam_model_log_mellin` at every order in `qs`, from one traversal of each
 * length's trellis where calling it per order makes `nq`.
 *
 * `out[i]` is bit-identical to `beam_model_log_mellin(m, qs[i], ...)` for
 * every `i` -- the same bits, whatever the thread count, the memory budget,
 * or which other orders share the call. The layer of edge weights and the
 * frontier are built once per layer; each order keeps its own per-layer
 * rescaling, because each has its own dynamic range.
 *
 * A concurrent sweep holds one mass vector per order where the single-order
 * form holds one in all, and the memory budget is divided accordingly: orders
 * are fused as widely as leaves room for one sweep per thread, and split into
 * blocks when they are not. Refuses with BEAM_ERR_BUDGET only when a sweep
 * carrying a single order does not fit.
 *
 * `nq = 0` succeeds and writes nothing; `qs` and `out` may then be NULL.
 */
BeamError beam_model_log_mellin_many(const BeamModel *m, const double *qs,
                                     size_t nq, double *out);

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
 * Expect these to come back saturated: the hierarchical prior leaves every
 * cell positive, so every reachable state is live.
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

/* ── Positional marginals and expected cell usage ────────────── */

/**
 * `P(x_i = a | length)` for every position and residue, exactly.
 *
 * `out` must hold `length * n_residues` doubles and receives row-major
 * `out[i * n_residues + a]`. Every row sums to 1 to within floating point;
 * nothing is sampled and nothing is binned, so unlike
 * `beam_model_spectrum` there is no grid to be approximate ON. The
 * answer is the per-layer edge posterior of the trellis, summed over states.
 *
 * @param clamp  NULL, or `length` bytes pinning positions to residues.
 *               `BEAM_RESIDUE_INVALID` leaves a position free, any residue
 *               index pins it, and the result is then the exact conditional
 *               `P(x_i = a | length, clamp)`. A pinned row comes back one-hot,
 *               which is the correct conditional and not a placeholder.
 * @param out_log_z  optional; receives `ln Z` of the CLAMPED trellis, so that
 *               `out_log_z - beam_model_log_partition(m, length)` is the exact
 *               log probability of the clamped configuration given the length.
 *               Without a clamp it is `ln Z(length)` reached by the backward
 *               sweep instead of the forward one.
 *
 * No floor is applied. Under the hierarchical prior no cell is zero, so a
 * residue the corpus never put at a position comes back small, not absent.
 *
 * @return BEAM_ERR_PARAM_OUT_OF_RANGE if the model does not generate this
 *         length or a clamp names a residue outside the alphabet, and
 *         BEAM_ERR_NO_LIVE_PATHS if the clamp asks for a configuration of
 *         probability zero -- which is a real question with no answer rather
 *         than a division by zero to be papered over.
 */
/**
 * When the model was first built, and what building it cost.
 *
 * `created_unix` is seconds since the epoch. The two durations are
 * nanoseconds of WALL time, aggregated across every call: a model built by
 * streaming a hundred chunks reports what the whole build cost, and one the
 * caller re-finalizes reports every such finalize. They survive save and
 * load, because a loaded model was built when it was built and not when it
 * was loaded.
 *
 * The finalize every LOAD performs is excluded. A file stores counts rather
 * than finalized tables, so loading has to rebuild them; that is the format's
 * cost, not this model's, and charging it would make the build time rise each
 * time the model was opened and stay risen if it were saved again.
 *
 * Wall rather than CPU, because the counting pass is parallel and CPU time
 * would report a figure that depends on the thread count. Any argument may be
 * NULL.
 */
BeamError beam_model_provenance(const BeamModel *m, int64_t *created_unix,
                                int64_t *observe_nanos,
                                int64_t *finalize_nanos);

/**
 * Where a saved model keeps the bytes that are not a function of its corpus.
 *
 * Everything else in a `.beam` file is determined by the sequences, the
 * weights and the config: build the same model twice and the two files agree
 * byte for byte, which is what the saved-bytes baseline and
 * `test_saving_a_frozen_model_writes_the_same_bytes` check. Provenance is the
 * one exception by construction -- it reads a clock -- so two builds of
 * identical data now differ in this window, and in the trailing CRC that
 * covers it.
 *
 * Rather than let that quietly retire a byte-level check, the format says
 * where the window is and the checks skip exactly it. `*offset` and `*bytes`
 * describe a contiguous run inside the fixed header; the checksum is the last
 * four bytes of the file and is the only other place a build time can reach.
 * Comparing a file outside those two regions is still comparing every byte
 * the corpus decides.
 *
 * Either argument may be NULL.
 */
void beam_model_file_provenance_span(size_t *offset, size_t *bytes);

/**
 * Occupied rows and cells per (direction, depth), for provenance.
 *
 * `rows` and `cells` each receive `2 * max_length` entries, indexed
 * `direction * max_length + depth`, with direction 0 N-anchored.
 *
 * A sequence of length `L` reads forward depths `0..L-1` and backward depths
 * `0..L-1`, so what it can be scored through is exactly the rows below `L`,
 * out of `L * radix^order` per direction. That is the sense in which a model
 * has discovered part of the dictionary a length admits, and this is the one
 * sweep every length's answer comes from.
 *
 * @return BEAM_ERR_NOT_BUILT before finalize.
 */
BeamError beam_model_depth_occupancy(const BeamModel *m,
                                     uint64_t *rows, uint64_t *cells);

BeamError beam_model_positional_marginals(const BeamModel *m, uint32_t length,
                                          const uint8_t *clamp,
                                          double *out, double *out_log_z);

/**
 * `P(x_i = a | length, band)` for each band of a sequence's own surprisal.
 *
 * One residue distribution per position PER BAND OF PROBABILITY: what a
 * typical sequence of this length looks like given that it is also this rare.
 * `beam_model_positional_marginals` answers the same question pooled over all
 * rarities, and equals the single band that covers the whole grid.
 *
 * `edges` holds `n_edges` surprisal thresholds in nats, STRICTLY ASCENDING,
 * defining `n_edges - 1` half-open bands `[edges[j], edges[j+1])`. Surprisal
 * is `-ln P(s)`, so bands ascend in rarity and a caller thinking in
 * probabilities passes `-ln p`. Mass outside the outermost edges belongs to
 * no band; pass `0` and `bins * spacing` to cover everything.
 *
 * @param counting  true weights every supported sequence 1, false weights it
 *                  by its own probability -- the same two measures
 *                  `beam_model_spectrum` offers, and both matter here.
 * @param bins,spacing  the surprisal grid, sized exactly as the spectrum's is.
 *                  `spacing <= 0` chooses it automatically and reports the
 *                  choice through `out_spacing`. Band masses are comparable
 *                  with `beam_model_spectrum` only on the same grid.
 * @param out_marginals   `(n_edges-1) * length * n_residues` doubles,
 *                  `out[(j * length + i) * n_residues + a]`. Each row sums to
 *                  1, or is all zeros where the band holds no mass.
 * @param out_band_mass   `n_edges-1` doubles, the mass in each band. This is
 *                  not a convenience: it is how much of the model each row of
 *                  logos represents, and it is what says a band's marginals
 *                  describe nothing when it is zero.
 * @param out_position_mass  optional; `(n_edges-1) * length` doubles, the row
 *                  totals before normalising. Every position of a band is an
 *                  independent estimate of that band's mass, so these agree
 *                  across positions to grid precision, and their disagreement
 *                  is the sharpest available measure of the grid's error.
 *
 * UNLIKE `beam_model_positional_marginals`, THIS IS NOT EXACT. The residue
 * axis still is, but conditioning on a band needs the distribution of a
 * real-valued path functional, so surprisal is binned and mass is split
 * linearly between adjacent bins as in `beam_model_spectrum`. A sequence near
 * an edge can fall on either side of it. Totals are conserved exactly; band
 * membership is accurate to O(spacing), so a band only a few bins wide
 * measures the grid rather than the model.
 *
 * Choosing the spacing automatically is imperfect for a smoothed model, whose
 * accumulator maximum is set by its least likely cell rather than by anything
 * it generates often. `out_position_mass` is how that is
 * detected: its positions agree to the last bits when the grid resolves the
 * bands and visibly disagree when it does not.
 *
 * Cost is `2 + n_edges` spectrum passes. Scratch is one grid per live state
 * per layer, held at once, and is refused against the memory budget before
 * anything is allocated.
 */
BeamError beam_model_positional_banded(const BeamModel *m, uint32_t length,
                                       bool counting, uint32_t bins,
                                       double spacing,
                                       const double *edges, uint32_t n_edges,
                                       double *out_marginals,
                                       double *out_band_mass,
                                       double *out_position_mass,
                                       double *out_spacing);

/**
 * One expert's expected counts: the rows that carry any expected use at all.
 *
 * Sparse by row because the answer is: an order-4 model has 210.0 million
 * cells and the support of this expectation is typically tens of thousands of
 * rows, so a dense block would be gigabytes of zeros. A row absent from
 * `rows` has expected count exactly zero.
 *
 * `rows` holds dense row indices ASCENDING, and `counts` holds `n_rows`
 * consecutive rows of `n_residues` doubles each. The row index is the same one
 * the count tables use, so `row` decomposes as
 *
 *     row = depth * A^order + context
 *
 * with `depth` EXACT and measured from the anchoring terminus.
 *
 * with contexts packed most-distant-first in BOTH anchors, matching the counts
 * rather than the finalized C-anchored wide table, which is repacked for cache
 * reasons.
 */
typedef struct {
    uint32_t  n_rows;
    uint32_t *rows;     /**< [n_rows], ascending dense row indices */
    double   *counts;   /**< [n_rows * n_residues], row-major      */
} BeamCountBlock;

/**
 * Expected usage of every table cell under the model's own distribution.
 *
 * `block[anchor][expert]`, with anchor 0 the N-anchored (forward) tables and 1
 * the C-anchored (backward) ones, and expert 0 the wide table, 1 the mid table
 * and 2 the length-conditioned table. The indices are `BEAM_FWD`/`BEAM_BWD`
 * and the wide/mid/len order used throughout the implementation.
 *
 * Every block sums to `expected_length`, because each position of a sequence
 * increments exactly one cell of each of the two tables. That identity is the
 * cheapest available check that the sweep covered the sink and the first `K`
 * layers correctly, and it is worth running rather than assuming.
 */
typedef struct {
    BeamCountBlock block[2];
    uint32_t n_residues;
    double   expected_length;
    /**
     * The most scratch the call held at once, in bytes, this struct included.
     *
     * Reported because it cannot be predicted. The accumulators are keyed on
     * the `(depth, context)` pairs the trellis actually reaches, so their size
     * is a property of the answer rather than of the model's shape, and the
     * only bound derivable in advance is the dense table -- 148 million cells
     * at order 4, which is what the sparse accumulator exists to avoid. On the
     * published foundation model this figure is over a gigabyte, larger than
     * the model. A caller sizing a job around this analytic needs the
     * measurement, and there is nowhere else to get it.
     */
    uint64_t working_set_bytes;
} BeamExpectedCounts;

/**
 * Expected counts of every cell, from the same edge posterior the marginals
 * use, aggregated into the cells `beam_model_observe` would have
 * incremented.
 *
 * This is the E step of an EM iteration written out exactly: the expected
 * sufficient statistics of the model under itself. Comparing it with the
 * observed counts says which cells the model over- and under-uses relative to
 * the corpus it was built from, cell by cell, with no sampling in between.
 *
 * @param length  a single length, or 0 to pool over every length the model
 *                generates, each weighted by its own prior `pi(L)`. Pooled,
 *                the answer is the expected usage per sequence DRAWN FROM THE
 *                MODEL, so multiplying by `beam_model_total_weight` puts it on
 *                the same scale as the observed counts.
 *
 * The caller owns the result and must release it with
 * `beam_model_expected_counts_free`.
 *
 * Cost is a full forward-backward sweep per length, so this is the most
 * expensive analytic in the library on a wide model: budget for
 * `sum_L L * n_states * n_residues` edge visits.
 */
BeamError beam_model_expected_counts(const BeamModel *m, uint32_t length,
                                     BeamExpectedCounts *out);

/** Release a `BeamExpectedCounts` and zero it. Safe on an already-freed one. */
void beam_model_expected_counts_free(BeamExpectedCounts *ec);

/**
 * Release ONE block and zero it, for a caller that consumes the six in turn.
 *
 * A block is large -- 303 MB for the wide forward block of the published
 * foundation model -- and a caller that copies each into its own
 * representation holds two copies of the whole answer if it waits for
 * `beam_model_expected_counts_free`. Releasing each block as it is consumed
 * halves that. Freeing a block does not disturb the others, and the
 * whole-struct free stays correct afterwards.
 */
void beam_count_block_free(BeamCountBlock *b);

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

/** Number of sequences rejected as too long or outside the alphabet. */
uint64_t beam_model_n_skipped(const BeamModel *m);

/** The alphabet this model reads. Amino acid for a NULL model. */
BeamAlphabet beam_model_alphabet(const BeamModel *m);

/** Emitted residues, and that plus the context sentinel. */
uint32_t beam_model_n_residues(const BeamModel *m);
uint32_t beam_model_radix(const BeamModel *m);

/** Nodes per layer of the DAG, `radix^state_order`. */
uint32_t beam_model_n_states(const BeamModel *m);

/** Total table cells, and how many carry a non-zero count. */
uint64_t beam_model_n_cells(const BeamModel *m);
uint64_t beam_model_n_nonzero_cells(const BeamModel *m);

/* ── Persistence ─────────────────────────────────────────────── */

/**
 * Write the model's counts and configuration, the prior's strengths included,
 * to `path`. Counts rather than derived tables, so a loaded model can still be
 * observed into, subtracted from and re-finalized under other strengths.
 */
BeamError beam_model_save(const BeamModel *m, const char *path);

/**
 * Read a model written by `beam_model_save` of this build. Returns NULL on
 * failure, with the reason in the error context. A file from an earlier
 * development build is refused, not converted: rebuild it from its sequences.
 */
BeamModel *beam_model_load(const char *path);

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
 * The most scratch the library will hold at once for one operation.
 *
 * Enforced two ways, both following from that one sentence. Lengths that run
 * concurrently DIVIDE it, so the thread count cannot multiply the footprint:
 * the spectrum's per-state histogram grids run to 1.6 GB at order 4 with 512
 * bins, and `spectra_by_length` would ask for twenty-five gigabytes if it ran
 * one length per core. And a single program whose working set exceeds it is
 * REFUSED before it allocates, because the alternative protects nothing -- one
 * forward-backward sweep is 6.6 GB on a nucleotide model at the alphabet's
 * ceiling order, and letting that through unconditionally is how a machine
 * dies rather than how a call fails.
 *
 * So lowering this CAN turn a working call into a failing one. That is the
 * point of it, and the refusal names the analytic, the figure it needed and
 * this setter. Raising it restores the call.
 *
 * The default is 4 GB. `bytes = 0` restores it. Results never depend on it:
 * it changes what is refused and how much runs at once, never an answer.
 */
void beam_model_set_memory_budget(uint64_t bytes);

/** The budget in force, in bytes. */
uint64_t beam_model_memory_budget(void);

#ifdef __cplusplus
}
#endif

#endif /* BEAM_MODEL_H */
