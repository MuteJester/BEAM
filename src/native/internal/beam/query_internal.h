/**
 * @file query_internal.h
 * @brief Constraint layout and sweep sizing, shared inside src/native/query/.
 *
 * Internal to src/native/query/. Not a public header and not installed. It is
 * included directly rather than through beam/internal.h, which is the narrowed
 * style the newer subsystems use.
 */
#ifndef BEAM_INTERNAL_QUERY_H
#define BEAM_INTERNAL_QUERY_H

#include "beam/query.h"
#include "beam/model_state.h"   /* BEAM_SUM_CHUNKS */

/**
 * A constraint is its masks and nothing else while the alphabet is the only
 * axis; the automaton tier adds fields here and does not move these.
 *
 * The masks are a flexible array member so a constraint is one allocation, and
 * so freeing one cannot free half of it.
 */
struct BeamConstraint_ {
    uint32_t        length;
    uint32_t        n_residues;
    BeamAutomaton  *automaton;  /**< owned; NULL when the masks are the whole
                                     constraint, which is the free tier    */
    BeamResidueMask mask[];     /**< [length], bit `a` = residue `a` permitted */
};

/**
 * A total, deterministic automaton over residue indices.
 *
 * `delta` is `[layers][n_states][n_residues]`, with `layers` equal to 1 when
 * `uniform` and to `length` otherwise. State 0 is always the start state, so
 * no field records it and no construction may renumber it away.
 */
struct BeamAutomaton_ {
    uint32_t  n_states;
    uint32_t  n_residues;
    uint32_t  length;
    bool      uniform;    /**< one transition table shared by every layer */
    uint32_t *delta;      /**< [layers * n_states * n_residues]           */
    uint8_t  *accept;     /**< [n_states], 0 or 1                         */
};

/** `delta[i](q, x)`, reading the shared layer when the automaton is uniform. */
uint32_t beam_query_automaton_step(const BeamAutomaton *a, uint32_t i,
                                   uint32_t q, uint32_t x);

/**
 * Peak bytes one constrained sweep needs at this length.
 *
 * `plan.c` reports this and `engine.c` allocates exactly it, from this one
 * expression. They must not each carry their own copy: a budget guard computed
 * from a different formula than the allocation it guards is not a guard, and
 * the difference only ever shows up on the machine that runs out of memory.
 *
 * The backward field dominates and is held whole, because the posterior needs
 * layer `i + 1` while the forward sweep is at layer `i`.
 *
 * `want_forward` is false for a query that needs only a partition function.
 * That is not a rounding difference: on an order-4 model the forward buffers
 * are twice the backward field, so charging a probability query for them
 * would refuse questions that fit comfortably.
 *
 * `n_aut` is the automaton's state count, 1 when the constraint is masks
 * alone. It multiplies the backward field and the forward vectors but not the
 * layer weights, which do not depend on the automaton. The sink buffer exists
 * only in the product sweep, where acceptance has to be applied to the tail
 * weights rather than written straight into the last backward layer.
 */
static inline uint64_t beam_query_sweep_bytes(uint32_t n_states,
                                              uint32_t n_aut,
                                              uint32_t n_residues,
                                              uint32_t length,
                                              bool want_forward) {
    const uint64_t S = n_states;
    const uint64_t P = (uint64_t)n_states * n_aut;   /* product states */
    uint64_t n = ((uint64_t)length + 1u) * P * sizeof(double)  /* beta field */
               + S * (uint64_t)n_residues * sizeof(double);    /* one layer  */
    if (n_aut > 1u) n += S * sizeof(double);         /* sink weights   */
    if (want_forward)
        n += 2u * P * sizeof(double)                 /* cur and next   */
           + (uint64_t)BEAM_SUM_CHUNKS * n_residues * sizeof(double);
    return n;
}

/** True when `position` is inside the constraint and `residue` is an index. */
BeamError beam_query_check_site(const BeamConstraint *c, uint32_t position,
                                uint32_t residue, const char *what);

/**
 * The first position whose mask is empty, or `length` when every position
 * still admits something. An empty mask makes the constraint unsatisfiable
 * before any trellis is consulted, and naming the position is most of the
 * diagnosis.
 */
uint32_t beam_query_empty_position(const BeamConstraint *c);

/** True when no position has been narrowed away from the full residue set. */
bool beam_query_all_free(const BeamConstraint *c);

/* ── The constrained sweep ─────────────────────────────────── */

/**
 * Scratch for one constrained forward-backward sweep.
 *
 * Deliberately the same shape as `FBWork` in math/positional.c, because this
 * sweep must reproduce that one bit for bit on a mask that pins one residue.
 * Where the two differ is only that a residue range `lo..hi` becomes a residue
 * SET: a singleton set visits the one residue the range would, and a full set
 * visits 0..R-1 in the same ascending order, so the arithmetic is not merely
 * equivalent but identical.
 */
typedef struct {
    double **beta;    /**< [length + 1][n_states * n_aut], log space  */
    double  *w;       /**< [n_states * n_residues], one layer         */
    double  *cur;     /**< [n_states * n_aut], rescaled forward       */
    double  *next;    /**< [n_states * n_aut]                         */
    double  *tail;    /**< [n_states], sink weights; product path only */
    uint32_t n_beta;  /**< rows actually allocated, for the free      */
    uint32_t n_aut;   /**< 1 when the constraint is masks alone       */
} BeamQueryWork;

/** Release a work block and zero it. Safe on a partially built one. */
void beam_query_work_free(BeamQueryWork *k);

/**
 * Allocate a work block, refusing over the memory budget before allocating.
 *
 * The budget figure comes from `beam_query_sweep_bytes` so that the guard and
 * the allocation cannot drift apart.
 */
BeamError beam_query_work_alloc(const BeamModel *m, uint32_t length,
                                uint32_t n_aut, bool want_forward,
                                BeamQueryWork *k);

/**
 * `beta[i][s]`, the log total weight of every completion from state `s` at
 * layer `i` that the constraint admits, sink term included.
 *
 * `mask` is `length` residue sets. `beta` must hold `length + 1` rows of
 * `n_states` doubles, and `w` one layer of `n_states * n_residues`.
 *
 * The scratch is the caller's because the work block already holds one and a
 * second would be the largest allocation in the call -- `n_states *
 * n_residues` doubles is 31 MB on an order-4 model, invisible to the budget
 * guard because it lived inside the function.
 */
BeamError beam_query_betas(const BeamModel *m, uint32_t length,
                           const BeamResidueMask *mask, double *w,
                           double **beta);

/**
 * The rescale a constrained layer needs: the largest weight among the edges
 * the sweep will actually sum, over the states it has reached.
 *
 * Returns -inf when the layer kills every live path.
 */
double beam_query_layer_max(const double *w, const double *cur,
                            uint32_t n_states, uint32_t n_residues,
                            BeamResidueMask allowed);

/**
 * Advance the rescaled forward vector across one layer, admitting only the
 * residues in `allowed`. Returns false when the layer leaves no mass, which
 * under a constraint is an outcome rather than an error.
 */
bool beam_query_advance(const BeamModel *m, const double *w, double wmax,
                        const double *cur, double *next,
                        BeamResidueMask allowed, double *log_scale);

/**
 * The masks a constraint imposes, or NULL for a constraint that admits
 * everything. Validated against the model's length and alphabet.
 */
BeamError beam_query_resolve(const BeamModel *m, const BeamConstraint *c,
                             const BeamResidueMask **out_mask,
                             uint32_t *out_length);

/**
 * As `beam_query_resolve`, and additionally refuses a constraint carrying an
 * automaton.
 *
 * The mask engine cannot answer one: an automaton relates positions to each
 * other and needs the product graph. Refusing is the only honest thing an
 * engine that would otherwise ignore the automaton entirely can do, and
 * returning a confidently wrong probability is what the alternative looks
 * like.
 */
BeamError beam_query_resolve_masks_only(const BeamModel *m,
                                        const BeamConstraint *c,
                                        const BeamResidueMask **out_mask,
                                        uint32_t *out_length);

/* ── The product sweep ─────────────────────────────────────── */

/**
 * `beta[i][s * Q + q]` over the product of the trellis and `aut`.
 *
 * `w` is one layer of scratch and `tail` holds the sink weights; both are the
 * caller's, so the budget guard sees every byte the sweep uses.
 */
BeamError beam_query_betas_product(const BeamModel *m, uint32_t length,
                                   const BeamResidueMask *mask,
                                   const BeamAutomaton *aut,
                                   double *w, double *tail, double **beta);

/** As `beam_query_layer_max`, over states reached in any automaton state. */
double beam_query_layer_max_product(const double *w, const double *cur,
                                    uint32_t n_states, uint32_t n_residues,
                                    uint32_t n_aut, BeamResidueMask allowed);

/**
 * Advance the rescaled forward vector across one layer of the product.
 *
 * `layer` indexes the automaton's transition function, which may differ per
 * position. Returns false when the layer leaves no mass.
 */
bool beam_query_advance_product(const BeamModel *m, const BeamAutomaton *aut,
                                uint32_t layer, const double *w, double wmax,
                                const double *cur, double *next,
                                BeamResidueMask allowed, double *log_scale);

#endif /* BEAM_INTERNAL_QUERY_H */
