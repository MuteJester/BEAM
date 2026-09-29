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
    /* The strengths are estimated from the counts unless a caller fixes
     * them; the values below are what a caller who turns estimation off
     * without naming strengths gets, and what an unfinalized model reports. */
    for (uint32_t k = 0; k <= BEAM_MAX_ORDER; ++k)
        cfg->alpha[k]    = BEAM_DEFAULT_ALPHA;
    cfg->alpha_length    = BEAM_DEFAULT_ALPHA_LENGTH;
    cfg->alpha_auto        = true;
    cfg->alpha_length_auto = true;
    /* Sparse is the default at every layer now. It used to be dense here and
     * sparse in Python, because a dense and a sparse closed-world model were
     * bit-identical and a C caller might not want the sparse path; a dense
     * model under the hierarchical prior materialises every row there is,
     * which is gigabytes at order 4, so it is no longer a neutral default. */
    cfg->sparse          = true;
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
 * sparse storage then required the closed world. Asking for `sparse` on a
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
    /* A zero strength lets an unobserved residue reach probability zero at
     * that level, which is the closed world returning through one row; a
     * negative one is not a Dirichlet at all. */
    for (uint32_t k = 0; !c->alpha_auto && k <= c->order && k <= BEAM_MAX_ORDER; ++k)
        if (!(c->alpha[k] > 0.0) || !isfinite(c->alpha[k]))
            BEAM_CFG_WHY("alpha[%u]=%.17g, and every prior strength up to the "
                         "order must be finite and positive", (unsigned)k,
                         c->alpha[k]);
    if (!c->alpha_length_auto
        && (!(c->alpha_length > 0.0) || !isfinite(c->alpha_length)))
        BEAM_CFG_WHY("alpha_length=%.17g, and the length prior's strength "
                     "must be finite and positive", c->alpha_length);
    return NULL;
#undef BEAM_CFG_WHY
}

bool beam_config_valid(const BeamModelConfig *c) {
    char why[BEAM_CONFIG_REASON_MAX];
    return beam_config_invalid_reason(c, why, sizeof why) == NULL;
}
