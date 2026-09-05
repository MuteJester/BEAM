/**
 * @file automaton.c
 * @brief Automaton constructions and their boolean algebra.
 *
 * Pure data. Nothing here consults a model, allocates a sweep, or knows what a
 * trellis is -- an automaton is a language over residue indices, and it can be
 * built, composed and run against a string with no BEAM model in existence.
 * That is deliberate: it means the builders can be gated by enumerating a
 * small language exhaustively and comparing against a naive matcher, which is
 * a test that cannot be passed by accident.
 *
 * Two invariants hold everywhere below and the rest of the subsystem depends
 * on both. Transitions are TOTAL, so a rejecting builder routes to a sink of
 * its own rather than leaving a hole; that is what makes complementation a
 * flip of the accepting set. And transitions are DETERMINISTIC, so the product
 * constructions are a plain pairing of states rather than a subset
 * construction.
 */
#include <stdlib.h>
#include <string.h>

#include "beam/query.h"
#include "beam/query_internal.h"
#include "beam/errors.h"
#include "beam/model_state.h"

/* ── Construction ────────────────────────────────────────────── */

/** Layers of transition table this automaton stores: 1 if uniform, else L. */
static inline uint32_t delta_layers(const BeamAutomaton *a) {
    return a->uniform ? 1u : a->length;
}

static BeamAutomaton *aut_new(uint32_t length, uint32_t n_residues,
                              uint32_t n_states, bool uniform,
                              const char *what) {
    if (n_states == 0u || n_states > BEAM_QUERY_MAX_AUTOMATON_STATES) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: the construction needs %u automaton states and the "
                      "ceiling is %u; the product graph is this many times the "
                      "trellis, so a larger one would be refused by the memory "
                      "budget in any case",
                      what, (unsigned)n_states,
                      (unsigned)BEAM_QUERY_MAX_AUTOMATON_STATES);
        return NULL;
    }
    if (length == 0u || length > BEAM_MAX_LENGTH_CEILING
        || n_residues == 0u || n_residues > BEAM_MAX_RESIDUES) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: length %u over %u residues is outside 1..%u and "
                      "1..%u", what, (unsigned)length, (unsigned)n_residues,
                      (unsigned)BEAM_MAX_LENGTH_CEILING,
                      (unsigned)BEAM_MAX_RESIDUES);
        return NULL;
    }

    /*
     * Bounded by the state and length ceilings, so this is nowhere near the
     * model-construction hazard -- but a maximal layer-indexed table is still
     * 21 MB, and a composition transiently holds three of them. A subsystem
     * that refuses a sweep over budget should not allocate around it here.
     */
    const uint64_t need = (uint64_t)(uniform ? 1u : length) * n_states
                        * n_residues * sizeof(uint32_t) + n_states;
    const uint64_t budget = beam_model_memory_budget();
    if (need > budget) {
        beam_set_error(BEAM_ERR_ALLOC,
                      "%s: an automaton of %u states over %u layers needs "
                      "about %.1f GB for its transition table, and the memory "
                      "budget is %.1f GB; raise it with "
                      "BEAM.set_memory_budget, or use a simpler constraint",
                      what, (unsigned)n_states,
                      (unsigned)(uniform ? 1u : length),
                      (double)need / 1e9, (double)budget / 1e9);
        return NULL;
    }

    BeamAutomaton *a = (BeamAutomaton *)calloc(1u, sizeof(*a));
    if (!a) { beam_set_error(BEAM_ERR_ALLOC, "%s: out of memory allocating "
                            "the automaton", what); return NULL; }
    a->n_states   = n_states;
    a->n_residues = n_residues;
    a->length     = length;
    a->uniform    = uniform;

    const size_t cells = (size_t)delta_layers(a) * n_states * n_residues;
    a->delta  = (uint32_t *)calloc(cells, sizeof(uint32_t));
    a->accept = (uint8_t *)calloc(n_states, sizeof(uint8_t));
    if (!a->delta || !a->accept) {
        beam_query_automaton_free(a);
        beam_set_error(BEAM_ERR_ALLOC, "%s: out of memory allocating the "
                      "transition table", what);
        return NULL;
    }
    return a;
}

void beam_query_automaton_free(BeamAutomaton *a) {
    if (!a) return;
    free(a->delta);
    free(a->accept);
    free(a);
}

uint32_t beam_query_automaton_states(const BeamAutomaton *a) {
    return a ? a->n_states : 0u;
}

/** Writable transition cell for layer `i`. */
static inline uint32_t *delta_at(BeamAutomaton *a, uint32_t i, uint32_t q,
                                 uint32_t x) {
    const uint32_t layer = a->uniform ? 0u : i;
    return a->delta + (((size_t)layer * a->n_states + q) * a->n_residues + x);
}

uint32_t beam_query_automaton_layers(const BeamAutomaton *a) {
    return a ? delta_layers(a) : 0u;
}

uint32_t beam_query_automaton_step(const BeamAutomaton *a, uint32_t i,
                                   uint32_t q, uint32_t x) {
    const uint32_t layer = a->uniform ? 0u : i;
    return a->delta[((size_t)layer * a->n_states + q) * a->n_residues + x];
}

bool beam_query_automaton_accepts(const BeamAutomaton *a,
                                  const uint8_t *residues, uint32_t n) {
    if (!a || !residues || n != a->length) return false;
    uint32_t q = 0u;                      /* state 0 is the start, always */
    for (uint32_t i = 0; i < n; ++i) {
        if (residues[i] >= a->n_residues) return false;
        q = beam_query_automaton_step(a, i, q, residues[i]);
    }
    return a->accept[q] != 0u;
}

/* ── Motif: the Knuth-Morris-Pratt automaton ─────────────────── */

BeamAutomaton *beam_query_automaton_motif(uint32_t length, uint32_t n_residues,
                                          const uint8_t *motif, uint32_t n) {
    static const char *WHAT = "beam_query_automaton_motif";
    if (!motif) { beam_set_error(BEAM_ERR_NULL_ARG, "%s: motif must not be "
                                "NULL", WHAT); return NULL; }
    if (n == 0u) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: an empty motif is contained in everything; omit "
                      "the constraint instead", WHAT);
        return NULL;
    }
    if (n > length) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: a motif of %u residues cannot occur in a sequence "
                      "of %u", WHAT, (unsigned)n, (unsigned)length);
        return NULL;
    }
    for (uint32_t k = 0; k < n; ++k) {
        if (motif[k] >= n_residues) {
            beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                          "%s: element %u of the motif is residue index %u, "
                          "which is not one of the %u residues",
                          WHAT, (unsigned)k, (unsigned)motif[k],
                          (unsigned)n_residues);
            return NULL;
        }
    }

    BeamAutomaton *a = aut_new(length, n_residues, n + 1u, true, WHAT);
    if (!a) return NULL;

    /* KMP failure function: fail[q] is the length of the longest proper
     * prefix of motif[0..q-1] that is also a suffix of it. */
    uint32_t *fail = (uint32_t *)calloc((size_t)n + 1u, sizeof(uint32_t));
    if (!fail) {
        beam_query_automaton_free(a);
        beam_set_error(BEAM_ERR_ALLOC, "%s: out of memory allocating the "
                      "failure function", WHAT);
        return NULL;
    }
    uint32_t k = 0u;
    for (uint32_t q = 1u; q < n; ++q) {
        while (k > 0u && motif[q] != motif[k]) k = fail[k];
        if (motif[q] == motif[k]) ++k;
        fail[q + 1u] = k;
    }

    /*
     * Built in increasing q so that the fallback row is already final when it
     * is read: fail[q] < q for every q >= 1.
     */
    for (uint32_t q = 0u; q < n; ++q) {
        for (uint32_t x = 0; x < n_residues; ++x) {
            if (x == motif[q]) *delta_at(a, 0u, q, x) = q + 1u;
            else if (q == 0u)  *delta_at(a, 0u, q, x) = 0u;
            else *delta_at(a, 0u, q, x) =
                     beam_query_automaton_step(a, 0u, fail[q], x);
        }
    }
    /* Matched already: absorbing, so "contains" cannot be un-said. */
    for (uint32_t x = 0; x < n_residues; ++x) *delta_at(a, 0u, n, x) = n;
    a->accept[n] = 1u;

    free(fail);
    return a;
}

/* ── Count ───────────────────────────────────────────────────── */

BeamAutomaton *beam_query_automaton_count(uint32_t length, uint32_t n_residues,
                                          BeamResidueMask of,
                                          uint32_t lo, uint32_t hi) {
    static const char *WHAT = "beam_query_automaton_count";
    if (lo > hi) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: the count range %u..%u is empty", WHAT,
                      (unsigned)lo, (unsigned)hi);
        return NULL;
    }
    if (n_residues <= BEAM_MAX_RESIDUES && (of & ~BEAM_RESIDUES_ALL(n_residues))) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: the residue set 0x%x names a residue at or above "
                      "%u", WHAT, (unsigned)of, (unsigned)n_residues);
        return NULL;
    }

    /*
     * Counts above `hi` never become acceptable again, so the counter stops at
     * `hi + 1` and sits there. When `hi` reaches the length no overflow is
     * possible and the cap is the length itself.
     */
    const uint32_t cap = (hi < length) ? hi + 1u : length;
    BeamAutomaton *a = aut_new(length, n_residues, cap + 1u, true, WHAT);
    if (!a) return NULL;

    for (uint32_t q = 0; q <= cap; ++q) {
        for (uint32_t x = 0; x < n_residues; ++x) {
            const bool counted = ((of >> x) & 1u) != 0u;
            uint32_t nq = counted ? q + 1u : q;
            if (nq > cap) nq = cap;
            *delta_at(a, 0u, q, x) = nq;
        }
        a->accept[q] = (q >= lo && q <= hi) ? 1u : 0u;
    }
    return a;
}

/* ── Lifting masks ───────────────────────────────────────────── */

BeamAutomaton *beam_query_automaton_from_masks(const BeamConstraint *c) {
    static const char *WHAT = "beam_query_automaton_from_masks";
    if (!c) { beam_set_error(BEAM_ERR_NULL_ARG, "%s: constraint must not be "
                            "NULL", WHAT); return NULL; }

    BeamAutomaton *a = aut_new(c->length, c->n_residues, 2u, false, WHAT);
    if (!a) return NULL;
    for (uint32_t i = 0; i < c->length; ++i) {
        for (uint32_t x = 0; x < c->n_residues; ++x) {
            *delta_at(a, i, 0u, x) = ((c->mask[i] >> x) & 1u) ? 0u : 1u;
            *delta_at(a, i, 1u, x) = 1u;      /* dead is absorbing */
        }
    }
    a->accept[0] = 1u;
    return a;
}

/* ── Boolean algebra ─────────────────────────────────────────── */

/** Both operands' languages, paired state by state. `both` selects AND/OR. */
static BeamAutomaton *aut_product(const BeamAutomaton *x,
                                  const BeamAutomaton *y, bool both,
                                  const char *what) {
    if (!x || !y) { beam_set_error(BEAM_ERR_NULL_ARG, "%s: both automata must "
                                  "be non-NULL", what); return NULL; }
    if (x->n_residues != y->n_residues || x->length != y->length) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: cannot combine a length-%u automaton over %u "
                      "residues with a length-%u one over %u",
                      what, (unsigned)x->length, (unsigned)x->n_residues,
                      (unsigned)y->length, (unsigned)y->n_residues);
        return NULL;
    }
    /*
     * Widened, though the ceiling already makes it safe: 1024 squared is 2^20
     * and cannot wrap a uint32. The static assert is what actually holds that
     * reasoning -- raise the ceiling past 65535 and it fires here, pointing at
     * the one line that would otherwise start wrapping silently.
     */
    _Static_assert((uint64_t)BEAM_QUERY_MAX_AUTOMATON_STATES
                   * BEAM_QUERY_MAX_AUTOMATON_STATES <= 0xFFFFFFFFu,
                   "the state ceiling must keep a product of two |Q| inside "
                   "a uint32, or aut_product must widen before multiplying");
    const uint64_t n = (uint64_t)x->n_states * y->n_states;
    if (n > BEAM_QUERY_MAX_AUTOMATON_STATES) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "%s: combining %u states with %u needs %llu and the "
                      "ceiling is %u", what, (unsigned)x->n_states,
                      (unsigned)y->n_states, (unsigned long long)n,
                      (unsigned)BEAM_QUERY_MAX_AUTOMATON_STATES);
        return NULL;
    }

    const bool uniform = x->uniform && y->uniform;
    BeamAutomaton *r = aut_new(x->length, x->n_residues, (uint32_t)n, uniform,
                               what);
    if (!r) return NULL;

    const uint32_t nb = y->n_states;
    for (uint32_t i = 0; i < delta_layers(r); ++i) {
        for (uint32_t qa = 0; qa < x->n_states; ++qa) {
            for (uint32_t qb = 0; qb < nb; ++qb) {
                for (uint32_t c = 0; c < r->n_residues; ++c) {
                    const uint32_t ta = beam_query_automaton_step(x, i, qa, c);
                    const uint32_t tb = beam_query_automaton_step(y, i, qb, c);
                    *delta_at(r, i, qa * nb + qb, c) = ta * nb + tb;
                }
            }
        }
    }
    for (uint32_t qa = 0; qa < x->n_states; ++qa)
        for (uint32_t qb = 0; qb < nb; ++qb)
            r->accept[qa * nb + qb] = both
                ? (x->accept[qa] && y->accept[qb])
                : (x->accept[qa] || y->accept[qb]);
    return r;
}

BeamAutomaton *beam_query_automaton_and(const BeamAutomaton *a,
                                        const BeamAutomaton *b) {
    return aut_product(a, b, true, "beam_query_automaton_and");
}

BeamAutomaton *beam_query_automaton_or(const BeamAutomaton *a,
                                       const BeamAutomaton *b) {
    return aut_product(a, b, false, "beam_query_automaton_or");
}

BeamAutomaton *beam_query_automaton_clone(const BeamAutomaton *a) {
    if (!a) return NULL;
    BeamAutomaton *r = aut_new(a->length, a->n_residues, a->n_states,
                               a->uniform, "beam_query_automaton_clone");
    if (!r) return NULL;
    memcpy(r->delta, a->delta,
           (size_t)delta_layers(a) * a->n_states * a->n_residues
           * sizeof(uint32_t));
    memcpy(r->accept, a->accept, a->n_states * sizeof(uint8_t));
    return r;
}

BeamAutomaton *beam_query_automaton_not(const BeamAutomaton *a) {
    static const char *WHAT = "beam_query_automaton_not";
    if (!a) { beam_set_error(BEAM_ERR_NULL_ARG, "%s: automaton must not be "
                            "NULL", WHAT); return NULL; }
    /*
     * Flipping the accepting set is complementation only because transitions
     * are total and deterministic: every string drives the run to exactly one
     * state, so it is accepted by the complement precisely when it was not
     * accepted here. A partial transition function would make this wrong for
     * every string that fell off the table.
     */
    BeamAutomaton *r = aut_new(a->length, a->n_residues, a->n_states,
                               a->uniform, WHAT);
    if (!r) return NULL;
    memcpy(r->delta, a->delta,
           (size_t)delta_layers(a) * a->n_states * a->n_residues
           * sizeof(uint32_t));
    for (uint32_t q = 0; q < a->n_states; ++q)
        r->accept[q] = a->accept[q] ? 0u : 1u;
    return r;
}

/* ── Attaching to a constraint ───────────────────────────────── */

BeamError beam_query_constrain_automaton(BeamConstraint *c, BeamAutomaton *a) {
    if (!a) return BEAM_ENULL("automaton");
    if (!c) { beam_query_automaton_free(a); return BEAM_ENULL("constraint"); }

    if (a->length != c->length || a->n_residues != c->n_residues) {
        const uint32_t al = a->length, ar = a->n_residues;
        beam_query_automaton_free(a);
        return BEAM_ERANGE("beam_query_constrain_automaton: a length-%u "
                          "automaton over %u residues does not fit a length-%u "
                          "constraint over %u",
                          (unsigned)al, (unsigned)ar,
                          (unsigned)c->length, (unsigned)c->n_residues);
    }

    if (!c->automaton) { c->automaton = a; return BEAM_OK; }

    /* Intersecting, as every builder does. */
    BeamAutomaton *both = beam_query_automaton_and(c->automaton, a);
    beam_query_automaton_free(a);
    if (!both) return beam_last_error();
    beam_query_automaton_free(c->automaton);
    c->automaton = both;
    return BEAM_OK;
}

const BeamAutomaton *beam_query_constraint_automaton(const BeamConstraint *c) {
    return c ? c->automaton : NULL;
}
