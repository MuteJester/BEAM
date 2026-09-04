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
 * The mask-only path is arithmetically the same computation as
 * `beam_model_positional_marginals`, which imposes a single-residue clamp by
 * loop bounds. A singleton mask iterated in ascending residue order visits the
 * same one residue, and a full mask visits 0..R-1 in the same order, so the two
 * agree BIT FOR BIT and not merely to a tolerance. That equivalence is the
 * regression test for this file; see test_query.c.
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

/* ── Cost ──────────────────────────────────────────────────── */

/**
 * What a query will cost, without running it.
 *
 * Exposed because the difference between a query that returns in two
 * milliseconds and one that returns in twenty seconds is four orders of
 * magnitude of model, and the caller cannot see that from the question.
 */
typedef struct {
    uint64_t states;  /**< live states summed over layers            */
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
