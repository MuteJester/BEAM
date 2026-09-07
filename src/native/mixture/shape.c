/**
 * @file shape.c
 * @brief The one gate every mixture entry point passes through.
 */
#include "beam/internal.h"
#include "beam/mixture.h"


/**
 * Refuse a set of models that cannot share a trellis.
 *
 * The four fields are the ones `beam_model_add` checks, and they are checked
 * here for the same reason: the alphabet belongs in the shape, not beside it,
 * because two models can agree on order, length and depth and still have rows
 * of different widths. A sweep that read one model's row at the other's stride
 * would run off the end of the buffer rather than return a wrong number, which
 * is the better of the two failures but not one to rely on.
 *
 * Everything is compared against member 0 rather than pairwise, which is the
 * same test -- agreement is transitive -- and which makes the message name a
 * single culprit instead of a pair.
 */
BeamError beam_mixture_check_shape(const BeamModel *const *models,
                                   uint32_t n_models) {
    if (!models) return BEAM_ENULL("models");
    if (n_models == 0u)
        return BEAM_FAIL(BEAM_ERR_EMPTY_INPUT,
                         "a mixture needs at least one member, got none");
    /*
     * The union's 64-member bitmask is NOT enforced here, deliberately. This
     * gate runs for every entry point, and refusing a 256-member mixture
     * because one of the five things you can ask it cannot be answered would
     * refuse the other four with it -- which is exactly what happened: a
     * cohort-scale mixture could not be constructed at all, so q >= 2, the
     * alpha diversity and the q = 1 bounds were unreachable for a reason that
     * belongs to q = 0 alone. `beam_mixture_support_union` enforces its own
     * limit, and says which sweep the limit belongs to.
     */
    if (n_models > BEAM_MIXTURE_MAX_PRODUCT_MEMBERS)
        return BEAM_FAIL(BEAM_ERR_PARAM_OUT_OF_RANGE,
                         "a mixture holds at most %u members and %u were "
                         "given",
                         (unsigned)BEAM_MIXTURE_MAX_PRODUCT_MEMBERS,
                         (unsigned)n_models);

    for (uint32_t i = 0; i < n_models; ++i) {
        if (!models[i])
            return BEAM_FAIL(BEAM_ERR_NULL_ARG,
                             "mixture member %u is NULL", (unsigned)i);
        if (!models[i]->finalized)
            return BEAM_FAIL(BEAM_ERR_NOT_BUILT,
                             "mixture member %u is not finalized",
                             (unsigned)i);
    }

    const BeamModel *a = models[0];
    for (uint32_t i = 1; i < n_models; ++i) {
        const BeamModel *b = models[i];
        if (a->cfg.alphabet != b->cfg.alphabet
            || a->order != b->order || a->max_len != b->max_len
            || a->n_depth_wide != b->n_depth_wide)
            return BEAM_FAIL(BEAM_ERR_VARIANT_MISMATCH,
                             "mixture: shape mismatch, member 0 is %s "
                             "order=%u max_length=%u depth_cap=%u and member "
                             "%u is %s order=%u max_length=%u depth_cap=%u; "
                             "a mixture reads its members at one state index, "
                             "which only holds when all four agree",
                             a->alpha->name, (unsigned)a->order,
                             (unsigned)a->max_len,
                             (unsigned)a->n_depth_wide - 1u,
                             (unsigned)i,
                             b->alpha->name, (unsigned)b->order,
                             (unsigned)b->max_len,
                             (unsigned)b->n_depth_wide - 1u);
        /*
         * Implied by the four fields above, and asserted anyway. The whole
         * design rests on one sentence -- a state index is a function of the
         * alphabet and the order, never of the data -- and if that sentence
         * ever stops being true, every number this folder produces is silently
         * wrong rather than loudly broken. This is the cheapest possible place
         * to find that out.
         */
        if (a->n_states != b->n_states || a->n_residues != b->n_residues
            || a->state_shift != b->state_shift)
            return BEAM_FAIL(BEAM_ERR_INTERNAL,
                             "mixture: members 0 and %u agree on alphabet, "
                             "order, max_length and depth_cap but index "
                             "different state spaces (%u vs %u states); the "
                             "state index is no longer a pure function of the "
                             "configuration, which every sweep here assumes",
                             (unsigned)i, (unsigned)a->n_states,
                             (unsigned)b->n_states);
    }
    return BEAM_OK;
}
