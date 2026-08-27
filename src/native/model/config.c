/**
 * @file config.c
 * @brief Configuration defaults and validation.
 */
#include <stdio.h>

#include "beam/internal.h"
#include "beam/model_build.h"


/* ── Configuration ───────────────────────────────────────────── */

void beam_model_config_default(BeamModelConfig *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->order           = 4;
    cfg->depth_cap       = 15;
    cfg->max_length      = BEAM_MAX_LENGTH;
    cfg->coef_wide       = 0.60;
    cfg->coef_mid        = 0.25;
    cfg->coef_len        = 0.15;
    cfg->strict          = true;
    cfg->sparse          = false;
    cfg->kappa[0]        = 20.0;
    cfg->kappa[1]        = 20.0;
    cfg->kappa[2]        = 20.0;
    cfg->kappa_wide      = 20.0;
    cfg->kappa_len       = 200.0;
    cfg->weight_forward  = 0.5;
    cfg->weight_backward = 0.5;
}

bool beam_config_valid(const BeamModelConfig *c) {
    if (c->order < 2 || c->order > BEAM_MAX_ORDER) return false;
    if (c->max_length < 1 || c->max_length > BEAM_MAX_LENGTH) return false;
    if (c->depth_cap < 1 || c->depth_cap >= c->max_length) return false;
    if (!(c->kappa[0] >= 0.0) || !(c->kappa[1] >= 0.0) || !(c->kappa[2] >= 0.0))
        return false;
    if (!(c->kappa_wide >= 0.0) || !(c->kappa_len >= 0.0)) return false;
    /* The coefficient SUM is a cross-length calibration knob, not a free
     * parameter: moving it off 1 leaves within-length ranking untouched while
     * wrecking global ranking. Reject rather than silently mis-calibrate. */
    double s = c->coef_wide + c->coef_mid + c->coef_len;
    if (!(fabs(s - 1.0) < 1e-9)) return false;
    if (!(c->weight_forward >= 0.0) || !(c->weight_backward >= 0.0)) return false;
    /* An omitted count row is a dead probability row only in the fully
     * closed-world regime. Positive backoff can populate an unobserved child
     * row, so accepting sparse storage there would silently change scores. */
    if (c->sparse
        && (!c->strict || c->kappa[0] != 0.0 || c->kappa[1] != 0.0
            || c->kappa[2] != 0.0 || c->kappa_wide != 0.0
            || c->kappa_len != 0.0))
        return false;
    return true;
}
