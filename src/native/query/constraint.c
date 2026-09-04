/**
 * @file constraint.c
 * @brief Building and composing the residue-mask tier of a constraint.
 *
 * A mask constraint is the free tier: it deletes edges from the trellis and
 * adds no states, so a query carrying one costs what an unconstrained query
 * costs. Everything in this file is bookkeeping over `length` words.
 *
 * Every builder intersects. That is what makes the builders commutative and
 * what makes a contradiction representable -- two builders that disagree leave
 * an empty mask, and an empty mask is a fact about the question rather than an
 * error in asking it. The refusal happens where the answer would have been
 * divided by zero, not here.
 */
#include <stdlib.h>
#include <string.h>

#include "beam/query.h"
#include "beam/query_internal.h"
#include "beam/errors.h"

/* ── Construction ────────────────────────────────────────────── */

BeamConstraint *beam_query_constraint_new(uint32_t length,
                                          uint32_t n_residues) {
    if (length == 0u || length > BEAM_MAX_LENGTH_CEILING) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "beam_query_constraint_new: length %u is outside "
                      "1..%u", (unsigned)length,
                      (unsigned)BEAM_MAX_LENGTH_CEILING);
        return NULL;
    }
    if (n_residues == 0u || n_residues > BEAM_MAX_RESIDUES) {
        beam_set_error(BEAM_ERR_PARAM_OUT_OF_RANGE,
                      "beam_query_constraint_new: %u residues is outside "
                      "1..%u, which is what a residue mask can address",
                      (unsigned)n_residues, (unsigned)BEAM_MAX_RESIDUES);
        return NULL;
    }

    BeamConstraint *c = (BeamConstraint *)malloc(
        sizeof(*c) + (size_t)length * sizeof(BeamResidueMask));
    if (!c) {
        beam_set_error(BEAM_ERR_ALLOC,
                      "beam_query_constraint_new: out of memory allocating "
                      "the constraint");
        return NULL;
    }
    c->length     = length;
    c->n_residues = n_residues;
    c->automaton  = NULL;
    const BeamResidueMask all = BEAM_RESIDUES_ALL(n_residues);
    for (uint32_t i = 0; i < length; ++i) c->mask[i] = all;
    return c;
}

void beam_query_constraint_free(BeamConstraint *c) {
    if (!c) return;
    beam_query_automaton_free(c->automaton);
    free(c);
}

BeamConstraint *beam_query_constraint_clone(const BeamConstraint *c) {
    if (!c) return NULL;
    BeamConstraint *d = beam_query_constraint_new(c->length, c->n_residues);
    if (!d) return NULL;
    for (uint32_t i = 0; i < c->length; ++i) d->mask[i] = c->mask[i];
    if (c->automaton) {
        d->automaton = beam_query_automaton_clone(c->automaton);
        if (!d->automaton) { beam_query_constraint_free(d); return NULL; }
    }
    return d;
}

/* ── Inspection ──────────────────────────────────────────────── */

uint32_t beam_query_constraint_length(const BeamConstraint *c) {
    return c ? c->length : 0u;
}

uint32_t beam_query_constraint_residues(const BeamConstraint *c) {
    return c ? c->n_residues : 0u;
}

BeamResidueMask beam_query_constraint_mask(const BeamConstraint *c,
                                           uint32_t position) {
    if (!c || position >= c->length) return BEAM_RESIDUES_NONE;
    return c->mask[position];
}

bool beam_query_constraint_is_free(const BeamConstraint *c) {
    /* An automaton narrows even when every mask is still full, so the two
     * questions are not the same one: `beam_query_all_free` asks whether the
     * MASKS restrict anything, which is what selects the engine's fast path. */
    return c && !c->automaton && beam_query_all_free(c);
}

bool beam_query_all_free(const BeamConstraint *c) {
    if (!c) return false;
    const BeamResidueMask all = BEAM_RESIDUES_ALL(c->n_residues);
    for (uint32_t i = 0; i < c->length; ++i)
        if (c->mask[i] != all) return false;
    return true;
}

uint32_t beam_query_empty_position(const BeamConstraint *c) {
    if (!c) return 0u;
    for (uint32_t i = 0; i < c->length; ++i)
        if (c->mask[i] == BEAM_RESIDUES_NONE) return i;
    return c->length;
}

/* ── Validation ──────────────────────────────────────────────── */

BeamError beam_query_check_site(const BeamConstraint *c, uint32_t position,
                                uint32_t residue, const char *what) {
    if (!c) return BEAM_ENULL("constraint");
    if (position >= c->length)
        return BEAM_ERANGE("%s: position %u is outside 0..%u, the positions "
                          "of a length-%u constraint",
                          what, (unsigned)position,
                          (unsigned)c->length - 1u, (unsigned)c->length);
    if (residue >= c->n_residues)
        return BEAM_ERANGE("%s: residue index %u is not one of the %u "
                          "residues this constraint is expressed over",
                          what, (unsigned)residue, (unsigned)c->n_residues);
    return BEAM_OK;
}

/* ── Builders ────────────────────────────────────────────────── */

BeamError beam_query_constrain_position(BeamConstraint *c, uint32_t position,
                                        uint32_t residue) {
    BeamError e = beam_query_check_site(c, position, residue,
                                        "beam_query_constrain_position");
    if (e != BEAM_OK) return e;
    c->mask[position] &= (BeamResidueMask)1u << residue;
    return BEAM_OK;
}

BeamError beam_query_constrain_set(BeamConstraint *c, uint32_t position,
                                   BeamResidueMask allowed) {
    if (!c) return BEAM_ENULL("constraint");
    if (position >= c->length)
        return BEAM_ERANGE("beam_query_constrain_set: position %u is outside "
                          "0..%u, the positions of a length-%u constraint",
                          (unsigned)position, (unsigned)c->length - 1u,
                          (unsigned)c->length);
    const BeamResidueMask all = BEAM_RESIDUES_ALL(c->n_residues);
    if (allowed & ~all)
        return BEAM_ERANGE("beam_query_constrain_set: the residue set 0x%x "
                          "names a residue at or above %u, and this "
                          "constraint has %u",
                          (unsigned)allowed, (unsigned)c->n_residues,
                          (unsigned)c->n_residues);
    c->mask[position] &= allowed;
    return BEAM_OK;
}

BeamError beam_query_constrain_exclude(BeamConstraint *c, uint32_t position,
                                       uint32_t residue) {
    BeamError e = beam_query_check_site(c, position, residue,
                                        "beam_query_constrain_exclude");
    if (e != BEAM_OK) return e;
    c->mask[position] &= ~((BeamResidueMask)1u << residue);
    return BEAM_OK;
}

BeamError beam_query_constrain_run(BeamConstraint *c, uint32_t start,
                                   const uint8_t *residues, uint32_t n) {
    if (!c) return BEAM_ENULL("constraint");
    if (!residues) return BEAM_ENULL("residues");
    if (n == 0u)
        return BEAM_ERANGE("beam_query_constrain_run: a run of zero residues "
                          "constrains nothing; omit the call instead");
    /* Widened before adding, so a start near UINT32_MAX cannot wrap into a
     * range that looks legal. */
    if ((uint64_t)start + (uint64_t)n > (uint64_t)c->length)
        return BEAM_ERANGE("beam_query_constrain_run: a run of %u from "
                          "position %u runs past the end of a length-%u "
                          "constraint", (unsigned)n, (unsigned)start,
                          (unsigned)c->length);
    for (uint32_t k = 0; k < n; ++k) {
        if (residues[k] >= c->n_residues)
            return BEAM_ERANGE("beam_query_constrain_run: element %u of the "
                              "run is residue index %u, which is not one of "
                              "the %u residues this constraint is expressed "
                              "over", (unsigned)k, (unsigned)residues[k],
                              (unsigned)c->n_residues);
    }
    /* Applied only after the whole run validates, so a refused run leaves the
     * constraint exactly as it was rather than half narrowed. */
    for (uint32_t k = 0; k < n; ++k)
        c->mask[start + k] &= (BeamResidueMask)1u << residues[k];
    return BEAM_OK;
}
