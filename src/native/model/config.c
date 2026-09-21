/**
 * @file config.c
 * @brief Configuration defaults and validation.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"


/* ── Configuration ───────────────────────────────────────────── */

void beam_model_config_default(BeamModelConfig *cfg) {
    beam_model_config_default_for(BEAM_ALPHABET_AA, cfg);
}

void beam_model_config_default_for(BeamAlphabet alphabet,
                                   BeamModelConfig *cfg) {
    if (!cfg) return;
    const BeamAlphabetInfo *a = beam_alphabet_info(alphabet);
    if (!a) return;

    memset(cfg, 0, sizeof(*cfg));
    cfg->alphabet        = alphabet;
    /* The three the alphabet decides. Everything below this point was
     * measured on amino acids and is carried across unchanged, because
     * re-tuning it for another alphabet needs a measurement of its own. */
    cfg->order           = a->default_order;
    cfg->max_length      = a->default_max_length;
    /* The closed-world default, and it has to be expressed in full here.
     *
     * `strict` alone does not give it: strict removes the order-0 add-one and
     * the probability floor, but hierarchical backoff still carries mass from
     * shorter contexts, so a sequence the corpus never spelled can still be
     * live. The closed world is strict AND every backoff strength at zero,
     * which is also the exact regime sparse storage is valid in.
     *
     * The Python layer resolves these the same way (see _smoothing_defaults),
     * and the two must agree: a caller reaching beam_model_config_default()
     * directly and a caller writing BeamModel(seqs) are asking for the same
     * model. They did not agree between 2026-08-27 and this commit, and the
     * difference was silent -- both produced a model, with different support.
     * Storage is the one thing that legitimately differs: this layer stays
     * dense while the Python layer selects sparse, because a dense and a
     * sparse strict model are bit-identical and a C caller may not want the
     * sparse path. Support and probabilities are what must agree, and those
     * are set by strict plus the backoff strengths, not by the layout. */
    cfg->strict          = true;
    cfg->sparse          = false;
    cfg->weight_forward  = 0.5;
    cfg->weight_backward = 0.5;
}

/*
 * WHY a config is invalid, not merely that it is.
 *
 * This returned a bare `bool`, and the caller built a refusal from a
 * hard-coded list of fields: order, max_length, the anchor weights
 * sum, the two weights. Every one of them printed with its value and its
 * permitted range, and none of them was necessarily the one that failed --
 * the kappas were not in the list at all, and neither was the rule that
 * sparse storage requires the closed world. Asking for `sparse` on a
 * nucleotide model produced
 *
 *   invalid BEAM config for the nt alphabet: order=2 (need 2..9),
 *   max_length=24 (need 1..255), order=4 (need 2..4)
 *   (need 1), weights=0.5/0.5 (need >= 0)
 *
 * -- five fields, all of them inside their stated ranges, and the reason
 * absent. It cost two build cycles to diagnose during one afternoon's work,
 * twice, because the message reads as though nothing is wrong.
 *
 * So each rule writes its own sentence, naming the field, the value it was
 * given and what was required. `beam_config_valid` is this with the sentence
 * thrown away, which is what its callers in the tests want.
 */
const char *beam_config_invalid_reason(const BeamModelConfig *c,
                                       char *buf, size_t cap) {
#define BEAM_CFG_WHY(...) do { snprintf(buf, cap, __VA_ARGS__); return buf; } while (0)

    /* The alphabet is checked first because the order bound below is read
     * from it. An unknown alphabet is a config that cannot be interpreted at
     * all, not merely one that is out of range. */
    const BeamAlphabetInfo *a = beam_alphabet_info(c->alphabet);
    if (!a) BEAM_CFG_WHY("alphabet id %u is not one this build knows",
                         (unsigned)c->alphabet);

    if (c->order < 2 || c->order > a->max_order)
        BEAM_CFG_WHY("order=%u, and the %s alphabet permits 2..%u",
                     (unsigned)c->order, a->name, (unsigned)a->max_order);
    /* No upper bound is tested on max_length: the field is a uint8_t and
     * BEAM_MAX_LENGTH_CEILING is 255, so the type is the bound. Every buffer
     * indexed by a length is sized by that ceiling for the same reason. */
    if (c->max_length < 1)
        BEAM_CFG_WHY("max_length=0, and a model of no length is not a model");
    if (!(c->weight_forward >= 0.0) || !(c->weight_backward >= 0.0))
        BEAM_CFG_WHY("anchor weights are %.17g/%.17g and neither may be "
                     "negative", c->weight_forward, c->weight_backward);
    /* An omitted count row is a dead probability row only in the fully
     * closed-world regime. The open world's add-one gives every row mass,
     * including rows nothing was ever observed at, so accepting sparse
     * storage there would silently change scores. */
    if (c->sparse && !c->strict)
        BEAM_CFG_WHY("sparse storage needs strict=true: an omitted count row "
                     "is a dead probability row only in the closed world, and "
                     "in the open world it would silently change scores");
    return NULL;
#undef BEAM_CFG_WHY
}

bool beam_config_valid(const BeamModelConfig *c) {
    char why[BEAM_CONFIG_REASON_MAX];
    return beam_config_invalid_reason(c, why, sizeof why) == NULL;
}
