/**
 * @file query.h
 * @brief Exact probabilistic queries: constrained probabilities and marginals.
 *
 * Every exact question about a finalized model is an independent choice on two
 * axes: the EVENT that is conditioned on, and the QUANTITY accumulated while
 * walking the trellis. This header is the event axis and the first quantity.
 *
 * An event is a `BeamConstraint`, which denotes a subset of the sequences of
 * one length. It has a deliberate cost ladder. A per-layer residue mask deletes
 * edges and adds no states, so a constraint built only from masks costs exactly
 * what an unconstrained sweep costs -- the free tier, and the common one.
 * Anything that relates two positions to each other needs an automaton and pays
 * a factor of |Q| on a product graph; that tier arrives with the automaton
 * builders and does not disturb what is here.
 *
 * The quantity here is probability:
 *
 *   P(C | L) = Z(L, C) / Z(L),
 *
 * where Z(L, C) sums the weight of every length-L path the constraint admits.
 * Marginals are the per-layer edge posterior of the same constrained trellis,
 * so `beam_query_marginals` under a constraint returns the exact conditional
 * P(x_i = a | L, C) rather than an approximation of it.
 *
 * Nothing here samples, bins, or estimates. A question with no exact answer is
 * refused; a question whose answer is zero returns zero and says so.
 *
 * The mask-only path and `beam_model_positional_marginals` are separate
 * implementations of one recursion -- a residue SET here, a residue RANGE
 * there -- and they agree BIT FOR BIT, in both worlds and on every pin. That
 * equivalence is the regression test for this file; see test_query.c.
 *
 * It is not free, and it was briefly false. Both routines used to rescale each
 * layer by a single factor, and no such factor can hold a trellis whose live
 * states span more than 745 nats: at the C API's own defaults a length-60
 * nucleotide sweep spans that, and rows came back summing to 5.9e-264 instead
 * of one. Both now carry their forward values as logarithms, and the test
 * classifies any future disagreement by which side stopped returning a
 * distribution, rather than only reporting that one exists.
 */
#ifndef BEAM_QUERY_H
#define BEAM_QUERY_H

#include "beam/common.h"
#include "beam/model.h"

/* ── Residue sets ──────────────────────────────────────────── */

/**
 * A set of residues, one bit per residue index: bit `a` set means residue `a`
 * is permitted. `BEAM_MAX_RESIDUES` is 20, so a set is one word and a
 * conjunction of two constraints is a bitwise AND.
 */
typedef uint32_t BeamResidueMask;

/** Every residue of an `n_residues` alphabet permitted. */
#define BEAM_RESIDUES_ALL(n_residues) ((BeamResidueMask)((1u << (n_residues)) - 1u))

/** No residue permitted; a layer holding this admits no sequence. */
#define BEAM_RESIDUES_NONE ((BeamResidueMask)0u)

/* ── Constraints ───────────────────────────────────────────── */

/**
 * A subset of the sequences of one fixed length.
 *
 * The length is fixed at construction because every consumer needs it anyway
 * and because position-from-the-end anchoring is otherwise ambiguous. A query
 * that ranges over lengths builds one constraint per length; the masks are the
 * only length-dependent part, and they are cheap.
 */
typedef struct BeamConstraint_ BeamConstraint;

/**
 * An unconstrained constraint: every sequence of `length` is admitted.
 *
 * @param length      Sequence length, 1..BEAM_MAX_LENGTH_CEILING.
 * @param n_residues  Alphabet size the constraint is expressed over. A query
 *                    refuses a constraint whose alphabet differs from the
 *                    model's, rather than silently reinterpreting bit indices.
 * @return NULL on a bad argument or allocation failure, with the reason in
 *         `beam_error_message()`.
 */
BeamConstraint *beam_query_constraint_new(uint32_t length, uint32_t n_residues);

/** Release a constraint. Safe on NULL. */
void beam_query_constraint_free(BeamConstraint *c);

/** The length this constraint is expressed over. Zero if `c` is NULL. */
uint32_t beam_query_constraint_length(const BeamConstraint *c);

/** The alphabet size this constraint is expressed over. Zero if `c` is NULL. */
uint32_t beam_query_constraint_residues(const BeamConstraint *c);

/**
 * The residues position `i` still permits.
 *
 * `BEAM_RESIDUES_NONE` means the constraint is already unsatisfiable, which is
 * a legitimate state to be in: two builders that disagree produce it, and a
 * query then reports BEAM_ERR_NO_LIVE_PATHS naming the position.
 */
BeamResidueMask beam_query_constraint_mask(const BeamConstraint *c,
                                           uint32_t position);

/**
 * True when no builder has narrowed anything, so the constraint admits every
 * sequence of its length. Queries use this to take the unconstrained path.
 */
bool beam_query_constraint_is_free(const BeamConstraint *c);

/*
 * Every builder INTERSECTS with what is already there. Applying two builders
 * to one position gives the sequences satisfying both, never the second
 * replacing the first, so the order they are applied in does not matter.
 * Contradictory builders leave that position empty, which is reported when the
 * constraint is used rather than when it is built -- an unsatisfiable
 * constraint is a fine thing to construct and a bad thing to divide by.
 */

/** Require position `position` to hold residue `residue`. */
BeamError beam_query_constrain_position(BeamConstraint *c, uint32_t position,
                                        uint32_t residue);

/** Require position `position` to hold one of `allowed`. */
BeamError beam_query_constrain_set(BeamConstraint *c, uint32_t position,
                                   BeamResidueMask allowed);

/** Forbid residue `residue` at position `position`. */
BeamError beam_query_constrain_exclude(BeamConstraint *c, uint32_t position,
                                       uint32_t residue);

/**
 * Require the `n` residues starting at `start` to be exactly `residues`.
 *
 * One primitive rather than separate prefix and suffix builders: a prefix is
 * `start = 0` and a suffix is `start = length - n`.
 *
 * @param residues  Residue indices [n], each below the constraint's alphabet.
 */
BeamError beam_query_constrain_run(BeamConstraint *c, uint32_t start,
                                   const uint8_t *residues, uint32_t n);

/* ── Automata ──────────────────────────────────────────────── */

/**
 * A deterministic automaton over residue indices: the general tier of the
 * constraint ladder, for anything that relates two positions to each other.
 *
 * The transition function is LAYER-INDEXED, `delta[i](q, x)`, with a uniform
 * fast path when every layer agrees. Motifs and counters are uniform; a
 * disjunction of two position-specific masks is not, and without layer
 * indexing that whole class would be inexpressible. The cost is `L * Q * R`
 * words, which is nothing.
 *
 * Transitions are TOTAL: every `(q, x)` lands somewhere, and a builder that
 * needs to reject routes to a non-accepting sink of its own making. That is
 * what makes complementation a matter of flipping the accepting set rather
 * than a construction.
 */
typedef struct BeamAutomaton_ BeamAutomaton;

/**
 * Structural ceiling on `|Q|`.
 *
 * Not the real limit, which is the memory budget: the product graph is
 * `n_states * |Q|`, so a 1024-state automaton over an order-4 model is 200
 * million product states and is refused long before this. This exists so that
 * a runaway composition fails while building a table rather than while
 * allocating a sweep.
 */
#define BEAM_QUERY_MAX_AUTOMATON_STATES 1024u

/**
 * Accepts the sequences containing `motif` at any position.
 *
 * The Knuth-Morris-Pratt automaton: `n + 1` states, the last absorbing, so a
 * sequence that has already matched cannot un-match.
 *
 * @param motif  Residue indices [n], each below `n_residues`.
 */
BeamAutomaton *beam_query_automaton_motif(uint32_t length, uint32_t n_residues,
                                          const uint8_t *motif, uint32_t n);

/**
 * Accepts the sequences with between `lo` and `hi` positions drawn from `of`.
 *
 * A counter capped one above `hi`, so the cap is absorbing and non-accepting.
 * `lo = hi` asks for an exact count; `hi >= length` asks for "at least `lo`".
 */
BeamAutomaton *beam_query_automaton_count(uint32_t length, uint32_t n_residues,
                                          BeamResidueMask of,
                                          uint32_t lo, uint32_t hi);

/**
 * The same language as a constraint's masks, as an automaton.
 *
 * Two states, alive and a dead sink. Needed because a mask cannot be
 * complemented or disjoined in place: the union of two per-position
 * restrictions is not a per-position restriction.
 */
BeamAutomaton *beam_query_automaton_from_masks(const BeamConstraint *c);

/** Accepts what both accept. `|Q|` is the product of the two. */
BeamAutomaton *beam_query_automaton_and(const BeamAutomaton *a,
                                        const BeamAutomaton *b);

/** Accepts what either accepts. */
BeamAutomaton *beam_query_automaton_or(const BeamAutomaton *a,
                                       const BeamAutomaton *b);

/** Accepts exactly what `a` rejects. */
BeamAutomaton *beam_query_automaton_not(const BeamAutomaton *a);

/** Release an automaton. Safe on NULL. */
void beam_query_automaton_free(BeamAutomaton *a);

/** `|Q|`. Zero if `a` is NULL. */
uint32_t beam_query_automaton_states(const BeamAutomaton *a);

/**
 * Whether `a` accepts this residue string, by running it directly.
 *
 * The engine never calls this -- it exists so that a test can compare the
 * automaton's language against a naive matcher over an enumerated support,
 * which is how the builders below are actually gated.
 */
bool beam_query_automaton_accepts(const BeamAutomaton *a,
                                  const uint8_t *residues, uint32_t n);

/**
 * Narrow a constraint by an automaton, INTERSECTING as every builder does.
 *
 * Takes ownership of `a` on every path, success or failure, so a caller never
 * has to decide whether it still owns what it passed. A constraint that
 * already carries an automaton ends up carrying the conjunction of the two.
 */
BeamError beam_query_constrain_automaton(BeamConstraint *c, BeamAutomaton *a);

/** The automaton a constraint carries, or NULL. Borrowed, not owned. */
const BeamAutomaton *beam_query_constraint_automaton(const BeamConstraint *c);

/* ── Additive functionals and their moments ────────────────── */

/**
 * What a functional adds up.
 *
 * Only functionals that DECOMPOSE OVER POSITIONS are exactly computable, and
 * that is the whole vocabulary: `f(s) = sum_i g(i, x_i)`. An arbitrary
 * function of a whole sequence is not here and will not be, because computing
 * its expectation means enumerating the support or sampling it, and this
 * library does neither.
 */
typedef enum {
    /** `g(i, x)` read from a table. Indicators, residue counts and any
        weighted positional score are this, with a different table. */
    BEAM_FUNCTIONAL_TABLE = 0,
    /** `g` is the edge's own log weight, including the sink term. The path
        total is then `ln w(s)`, which is what makes entropy fall out of the
        same sweep rather than needing an algebra of its own. */
    BEAM_FUNCTIONAL_LOG_WEIGHT = 1
} BeamFunctionalKind;

typedef struct {
    BeamFunctionalKind kind;
    /** `[length * n_residues]`, row-major `table[i * n_residues + x]`.
        Required for BEAM_FUNCTIONAL_TABLE, ignored otherwise.
        EVERY ENTRY MUST BE FINITE: a functional taking an infinite value on a
        live path has no variance and an infinite expectation, and is refused.
        A residue that should not count wants a weight of ZERO here, not the
        logarithm of a zero probability. */
    const double *table;
} BeamFunctional;

/** Exact first two moments of a functional, and the mass they are taken over. */
typedef struct {
    double mean;      /**< `E[f | length, C]`                        */
    double variance;  /**< `Var[f | length, C]`, never negative      */
    double log_z;     /**< `ln Z` of the constrained trellis         */
} BeamQueryMoments;

/**
 * `E[f]` and `Var[f]` under `P(. | length, C)`, exactly.
 *
 * One FORWARD pass, with no backward field at all: the moments of an additive
 * functional propagate along the trellis, so unlike `beam_query_marginals`
 * this does not have to hold `length + 1` layers of anything. It is therefore
 * much the cheaper of the two, and much the more likely to fit.
 *
 * The variance is accumulated in centred form -- a running mean and a running
 * sum of squared deviations, merged pairwise -- rather than as
 * `E[f^2] - E[f]^2`. The uncentred form cancels catastrophically exactly where
 * variances are interesting, when the spread is small beside the mean.
 *
 * @return BEAM_ERR_NO_LIVE_PATHS if the constraint admits nothing, since
 *         moments of a distribution with no support do not exist.
 */
BeamError beam_query_expect(const BeamModel *m, const BeamConstraint *c,
                            const BeamFunctional *f, BeamQueryMoments *out);

/**
 * `H(X | length, C)` in nats, and optionally the variance of the surprisal.
 *
 * Entropy needs no algebra of its own. `ln P(s) = ln w(s) - ln Z` and
 * `ln w(s)` is additive over edges, so
 *
 *   H = -sum P ln P = ln Z - E[ln w],
 *
 * which is `beam_query_expect` with BEAM_FUNCTIONAL_LOG_WEIGHT and one
 * subtraction. `out_variance` receives `Var[-ln P]`, the spread of the
 * surprisal, which is the same sweep's second moment.
 */
BeamError beam_query_entropy(const BeamModel *m, const BeamConstraint *c,
                             double *out_entropy, double *out_variance);

/* ── Cross-length algebra: insertions and deletions ────────── */

/**
 * The single-edit neighbours of one sequence, with what the model makes of
 * them.
 *
 * Each entry is a distinct string, so `n` is below the number of candidate
 * edits wherever the source has a repeated residue: inserting `S` at gaps 2,
 * 3 and 4 of `CASSF` all give `CASSSF`, and that string appears once,
 * attributed to the leftmost gap that produces it.
 */
typedef struct {
    uint32_t  n;           /**< distinct neighbours                        */
    uint32_t  length;      /**< their common length                        */
    char     *sequences;   /**< [n * (length + 1)] NUL-terminated, in order */
    uint32_t *site;        /**< [n] the LEFTMOST gap or position producing
                                this string -- a canonical label, not a
                                claim about where an edit occurred          */
    uint32_t *residue;     /**< [n] residue inserted; UINT32_MAX for a
                                deletion, which removes rather than adds    */
    uint32_t *multiplicity;/**< [n] how many distinct edits produce this
                                string; see the note on `posterior`         */
    double   *log_odds;    /**< [n] `ln P(neighbour) - ln P(source)`        */
    double   *posterior;   /**< [n] `P(s' | s' in this set)`; see below     */
    double    log_source;  /**< `ln P(source)`                              */
    bool      length_producible; /**< false when the model generates NO
                                sequence of the neighbours' length, which is
                                why they are all impossible -- a fact about
                                the length prior rather than about any of
                                these strings                               */
} BeamNeighbourhood;

/** Release a neighbourhood and zero it. Safe on an already-freed one. */
void beam_query_neighbourhood_free(BeamNeighbourhood *nb);

/**
 * Every sequence one insertion away from `sequence`, scored.
 *
 * `log_odds` answers "how much more or less likely is this than what I have",
 * as a joint probability: `log_pseq` carries the length prior, so the
 * comparison across two lengths is well posed and the prior does not cancel.
 *
 * `posterior` is `P(s' | s' in N+(s))` -- the model's own law RESTRICTED to
 * this set of strings, with the length prior a common factor that cancels.
 *
 * **It is a distribution over STRINGS, not over edits**, and the difference is
 * not cosmetic. Inserting `S` into the `SS` of `CASSSQYF` yields one string
 * reachable four ways; this reports it once, with `multiplicity = 4`. Weighted
 * per EDIT instead -- `P(s') * multiplicity(s')`, renormalised -- that string
 * is the likeliest neighbour, where here it ranks ninth.
 *
 * Both are right, under different questions. Per string asks what the model
 * makes of a candidate set. Per edit asks what happens when a longer sequence
 * is drawn and a uniformly chosen position removed. A caller who wants the
 * per-edit convention has `multiplicity` and one line of arithmetic.
 *
 * **This is not an inferred edit.** The model has no alignment and no indel
 * process; nothing here says a sequence was edited. The quantity is a ratio
 * between what the model assigns to two strings that happen to differ by one
 * residue, and it describes where a REPERTOIRE tolerates insertions. The
 * per-sequence question -- given this observed sequence, where was it edited
 * -- is a different one, and this does not answer it.
 *
 * @return BEAM_ERR_INVALID_SEQUENCE if the source is not scoreable;
 *         BEAM_ERR_NO_LIVE_PATHS if the source has probability zero, since
 *         the odds are then a ratio to nothing.
 */
BeamError beam_query_insertions(const BeamModel *m, const char *sequence,
                                BeamNeighbourhood *out);

/**
 * Every sequence one deletion away from `sequence`, scored.
 *
 * The mirror of `beam_query_insertions`, with `site` naming the position
 * removed and `residue` left as `UINT32_MAX`. Deleting either `S` of `CASSF`
 * gives `CASF`, which appears once, attributed to the leftmost position.
 *
 * @return as `beam_query_insertions`, and additionally
 *         BEAM_ERR_PARAM_OUT_OF_RANGE for a source of one residue, which has
 *         nothing to delete down to. That refusal is structural and is made
 *         before the model is consulted.
 */
BeamError beam_query_deletions(const BeamModel *m, const char *sequence,
                               BeamNeighbourhood *out);

/* ── Cost ──────────────────────────────────────────────────── */

/**
 * What a query will cost, without running it.
 *
 * Exposed because the difference between a query that returns in two
 * milliseconds and one that returns in twenty seconds is four orders of
 * magnitude of model, and the caller cannot see that from the question.
 */
typedef struct {
    uint64_t states;  /**< trellis states the sweep will VISIT, summed
                           over layers -- an upper bound on the live count,
                           and on a closed-world model a very loose one   */
    uint64_t edges;   /**< edge relaxations the sweep will perform    */
    uint64_t bytes;   /**< peak working-set bytes, the budgeted figure */
} BeamQueryCost;

/**
 * Estimate a query's cost. Allocates nothing and runs no sweep.
 *
 * `bytes` is the figure checked against `beam_model_memory_budget()`, so a
 * caller can compare the two itself and warn before asking rather than catch a
 * refusal afterwards.
 */
BeamError beam_query_estimate(const BeamModel *m, const BeamConstraint *c,
                              BeamQueryCost *out);

/* ── Information ───────────────────────────────────────────── */

/**
 * `H(X_i | length, C)` in nats, for every position.
 *
 * The Shannon entropy of each position's own marginal, which is not the
 * entropy of the sequence: summing these would double-count everything the
 * positions share. `beam_query_entropy` gives the sequence entropy, and the
 * gap between the two is exactly what the information matrix measures.
 *
 * @param out  [length] nats.
 */
BeamError beam_query_position_entropy(const BeamModel *m,
                                      const BeamConstraint *c, double *out);

/**
 * `P(x_i = a, x_j = b | length, C)` for one pair of positions.
 *
 * @param out  [n_residues * n_residues], row-major `out[a * n_residues + b]`,
 *             summing to 1. With `i == j` it is diagonal, which is the
 *             correct joint of a position with itself.
 *
 * Costs one sweep per residue that position `i` can take -- free-tier if the
 * constraint is masks alone, and `|Q|` times that if it carries an automaton,
 * since conditioning inside a language means sweeping the product.
 */
BeamError beam_query_pair_joint(const BeamModel *m, const BeamConstraint *c,
                                uint32_t i, uint32_t j, double *out);

/**
 * Mutual information between every pair of positions, in nats.
 *
 * `out[i * length + j]` is `I(X_i ; X_j) = H(X_i) + H(X_j) - H(X_i, X_j)`,
 * the reduction in uncertainty about one position from learning the other.
 * The diagonal is `H(X_i)`, since a position tells you everything about
 * itself, and it falls out of the same arithmetic rather than being written
 * in afterwards.
 *
 * Conditional entropy is a subtraction away: `H(X_i | X_j)` is
 * `out[i * length + i] - out[i * length + j]`.
 *
 * Costs one sweep per `(position, residue)` pair carrying mass -- up to
 * `length * n_residues` of them, so this is much the most expensive query
 * here. `beam_query_information_cost` reports the total beforehand.
 *
 * @param out  [length * length] nats, symmetric and non-negative.
 */
BeamError beam_query_information_matrix(const BeamModel *m,
                                        const BeamConstraint *c, double *out);

/**
 * What `beam_query_information_matrix` will cost, as an upper bound.
 *
 * The bound assumes every `(position, residue)` pair carries mass; the real
 * count is lower wherever the model cannot spell a residue at a position, and
 * in a closed-world model it is dramatically lower. Reported without running
 * any sweep, so a caller can find out that a matrix is a twenty-second
 * question before waiting twenty seconds for it.
 */
BeamError beam_query_information_cost(const BeamModel *m,
                                      const BeamConstraint *c,
                                      BeamQueryCost *out);

/* ── Queries ───────────────────────────────────────────────── */

/**
 * `ln P(C | length)`, the exact mass the constraint keeps.
 *
 * The partition function of the constrained trellis over that of the full one.
 * Returns `-INFINITY` for a constraint of probability zero rather than failing:
 * an event the model cannot produce has a well-defined probability, and it is
 * zero. It is CONDITIONING on such an event that has no answer, which is where
 * BEAM_ERR_NO_LIVE_PATHS is raised instead.
 */
BeamError beam_query_log_probability(const BeamModel *m,
                                     const BeamConstraint *c,
                                     double *out_log_p);

/**
 * `P(x_i = a | length, C)` for every position and residue, exactly.
 *
 * @param out  [length * n_residues] doubles, row-major `out[i * n_residues + a]`.
 *             Every row sums to 1 to within floating point. A position the
 *             constraint pins comes back one-hot, which is the correct
 *             conditional and not a placeholder.
 * @param out_log_z  optional; receives `ln Z` of the CONSTRAINED trellis, so
 *             that subtracting `beam_model_log_partition` gives the same number
 *             as `beam_query_log_probability`.
 *
 * Zero-probability cells come back as exactly 0.0, unfloored: in a strict model
 * a residue that cannot appear at a position is information, not a number to be
 * smoothed.
 *
 * @return BEAM_ERR_NO_LIVE_PATHS if the constraint admits nothing, since the
 *         conditional it asks for does not exist; BEAM_ERR_PARAM_OUT_OF_RANGE
 *         if the model does not generate the constraint's length or the
 *         constraint is expressed over a different alphabet.
 */
BeamError beam_query_marginals(const BeamModel *m, const BeamConstraint *c,
                               double *out, double *out_log_z);

#endif /* BEAM_QUERY_H */
