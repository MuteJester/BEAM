/**
 * @file logo.c
 * @brief Sequence logos, and the refusals that keep one honest.
 *
 * The arithmetic here is a few hundred flops over a `length x n_residues`
 * array, and none of it is the interesting part. What this file is actually
 * for is everything around that: resolving an open-ended band against a grid
 * only the library can size, routing to the exact sweep when no grid is
 * needed, and refusing the questions whose answers would look reasonable and
 * be wrong.
 *
 * The worst of those is an empty band. Zero probabilities give zero entropy,
 * and zero entropy gives the MAXIMUM information content -- so a threshold no
 * sequence satisfies yields a blank plot beside a summary saying every
 * position is perfectly conserved. `beam_model_positional_banded` tolerates
 * an empty band on purpose, because one empty band must not deny a caller the
 * others; a logo is a single conditional, and here the same tolerance is a
 * defect. `beam_model_positional_marginals` already refuses a clamp of
 * probability zero for exactly this reason, and this refuses to match.
 */
#include "beam/internal.h"
#include "beam/insights.h"
#include "beam/math_internal.h"


/** `log2(e)`, so bits are one multiply from nats rather than a second log. */
#define BEAM_LOG2E 1.4426950408889634074

/** Room above the top edge of the grid, in bins, for an unbounded band.
 *
 * Not `bins * spacing` exactly: linear splitting can push a sliver of mass one
 * bin past where the exact arithmetic puts it, so a band that means
 * "everything above" has to reach past the last bin it can reach. */
#define BEAM_LOGO_TOP_SLACK 4u


/* ── Sizing ──────────────────────────────────────────────────── */

/**
 * The grid a request lands on, with the defaults resolved.
 *
 * `bins == 0` means "choose", and it cannot be a fixed constant: automatic
 * spacing reserves `max_length + 9` bins of splitting headroom, which is 90
 * on a nucleotide model at its ceiling, so a default of 64 would refuse
 * itself. Scale with the model instead.
 */
static uint32_t logo_bins(const BeamModel *m, const BeamLogoRequest *req) {
    if (req->bins != 0u) return req->bins;
    const uint32_t reserve = m->max_len + 9u;
    uint32_t b = 1024u;
    while (b <= reserve * 4u) b <<= 1;
    return b;
}


/* ── Deriving the logo from a row of marginals ───────────────── */

/**
 * Fill one logo's derived arrays from its `probability` rows.
 *
 * Everything is computed once in nats from the probabilities, and the bits
 * are that multiplied by `log2(e)`. Computing each unit from its own
 * logarithm would make the two disagree in the last bits for no gain, and
 * would turn the relation between them from an identity into a coincidence.
 */
static void derive(BeamLogo *g, double max_entropy_nats) {
    const uint32_t L = g->length, R = g->n_residues;
    for (uint32_t i = 0; i < L; ++i) {
        const double *p = g->probability + (size_t)i * R;
        double h = 0.0, best = -1.0;
        uint32_t support = 0, arg = 0;
        for (uint32_t a = 0; a < R; ++a) {
            const double v = p[a];
            if (v > 0.0) {
                /* Zero contributes nothing, which is the limit of `p ln p`
                 * and not a special case -- but `0 * log(0)` is `0 * -inf`,
                 * which is NaN, and one NaN poisons a whole position. */
                h -= v * log(v);
                ++support;
            }
            if (v > best) { best = v; arg = a; }
        }
        const double info = max_entropy_nats - h;
        g->entropy_nats[i]     = h;
        g->entropy_bits[i]     = h * BEAM_LOG2E;
        g->information_nats[i] = info;
        g->information_bits[i] = info * BEAM_LOG2E;
        g->n_support[i]        = support;
        g->consensus[i]        = arg;
        g->consensus_probability[i] = best < 0.0 ? 0.0 : best;
        double *hb = g->height_bits + (size_t)i * R;
        double *hn = g->height_nats + (size_t)i * R;
        for (uint32_t a = 0; a < R; ++a) {
            hn[a] = p[a] * info;
            hb[a] = hn[a] * BEAM_LOG2E;
        }
    }
}


/* ── Allocation ──────────────────────────────────────────────── */

static void logo_free_one(BeamLogo *g) {
    free(g->probability); free(g->height_bits); free(g->height_nats);
    free(g->entropy_bits); free(g->entropy_nats);
    free(g->information_bits); free(g->information_nats);
    free(g->n_support); free(g->consensus); free(g->consensus_probability);
    free(g->position_mass);
    memset(g, 0, sizeof(*g));
}

void beam_insights_logo_set_free(BeamLogoSet *set) {
    if (!set) return;
    for (uint32_t j = 0; j < set->n_logos; ++j) logo_free_one(&set->logo[j]);
    free(set->logo);
    memset(set, 0, sizeof(*set));
}

/** All of one logo's arrays, or false having freed whatever it did get. */
static bool logo_alloc(BeamLogo *g, uint32_t L, uint32_t R, bool with_mass) {
    memset(g, 0, sizeof(*g));
    g->length = L; g->n_residues = R;
    const size_t cell = (size_t)L * R;
    g->probability      = (double *)calloc(cell, sizeof(double));
    g->height_bits      = (double *)calloc(cell, sizeof(double));
    g->height_nats      = (double *)calloc(cell, sizeof(double));
    g->entropy_bits     = (double *)calloc(L, sizeof(double));
    g->entropy_nats     = (double *)calloc(L, sizeof(double));
    g->information_bits = (double *)calloc(L, sizeof(double));
    g->information_nats = (double *)calloc(L, sizeof(double));
    g->n_support        = (uint32_t *)calloc(L, sizeof(uint32_t));
    g->consensus        = (uint32_t *)calloc(L, sizeof(uint32_t));
    g->consensus_probability = (double *)calloc(L, sizeof(double));
    g->position_mass    = with_mass ? (double *)calloc(L, sizeof(double)) : NULL;
    if (!g->probability || !g->height_bits || !g->height_nats
        || !g->entropy_bits || !g->entropy_nats || !g->information_bits
        || !g->information_nats || !g->n_support || !g->consensus
        || !g->consensus_probability || (with_mass && !g->position_mass)) {
        logo_free_one(g);
        return false;
    }
    return true;
}


/* ── Validation ──────────────────────────────────────────────── */

void beam_logo_request_default(BeamLogoRequest *r) {
    if (!r) return;
    memset(r, 0, sizeof(*r));
    /* Everything this struct means by "default" is zero: the unconditional
     * logo of the generated measure, the model's own alphabet as the
     * reference, and a grid chosen from the model. Only the length has to be
     * set, which is the one thing that has no sensible default. */
}

/**
 * Everything checkable without a sweep, in the order a caller gets it wrong.
 *
 * `out_lo` / `out_hi` receive the surprisal span the request asks about, so
 * the caller can compare it against what the length attains before allocating
 * anything.
 */
static BeamError validate(const BeamModel *m, const BeamLogoRequest *req,
                          uint32_t *out_bins, uint32_t *out_ref) {
    if (!m)   return BEAM_ENULL("model");
    if (!req) return BEAM_ENULL("logo request");
    if (!m->finalized) return BEAM_ENOTBUILT();
    if (req->length == 0u)
        return BEAM_ERANGE("%s: a logo needs a specific length of at least 1. "
                           "Logos are per length and there is no pooled form: "
                           "position 5 of a length-12 sequence and position 5 "
                           "of a length-18 one are different distances from "
                           "the C terminus", __func__);
    if (!beam_length_live(m, req->length)) return BEAM_EBADLEN(m, req->length);

    const uint32_t R = m->n_residues;
    const uint32_t ref = req->reference_residues ? req->reference_residues : R;
    if (ref < 2u || ref > R)
        return BEAM_ERANGE("%s: reference_residues is %u, which is not "
                           "between 2 and this model's %u residues; it sets "
                           "the maximum entropy a position is measured "
                           "against, so a reference wider than the alphabet "
                           "has no meaning", __func__,
                           (unsigned)req->reference_residues, (unsigned)R);

    const uint32_t bins = logo_bins(m, req);
    if (bins < 2u)
        return BEAM_ERANGE("%s: a surprisal grid needs at least 2 bins, "
                           "got %u", __func__, (unsigned)bins);

    if (req->n_edges > 1u) {
        if (!req->edges) return BEAM_ENULL("band edges");
        for (uint32_t k = 0; k < req->n_edges; ++k) {
            /*
             * NaN before ascent, and by its own test rather than by falling
             * out of a comparison. Every comparison against NaN is false, so
             * an ascent check alone lets it through -- and downstream
             * `ceil(NaN)` is clamped to the empty branch, which returns an
             * empty band as though the question had been answered.
             */
            if (isnan(req->edges[k]))
                return BEAM_ERANGE("%s: band edge %u is NaN", __func__,
                                   (unsigned)k);
            if (req->edges[k] < 0.0 && !isinf(req->edges[k]))
                return BEAM_ERANGE("%s: band edge %u is %.17g, and surprisal "
                                   "is never negative. A negative edge is "
                                   "almost always ln(p) where -ln(p) was "
                                   "meant", __func__, (unsigned)k,
                                   req->edges[k]);
            if (isinf(req->edges[k]) && k != 0u && k != req->n_edges - 1u)
                return BEAM_ERANGE("%s: band edge %u is infinite; only the "
                                   "outermost edges may be, where they mean "
                                   "the ends of the grid", __func__,
                                   (unsigned)k);
        }
        for (uint32_t k = 1; k < req->n_edges; ++k)
            if (!(req->edges[k] > req->edges[k - 1u]))
                return BEAM_ERANGE("%s: band edges must ascend strictly in "
                                   "surprisal, but edge %u is %.17g and edge "
                                   "%u is %.17g. Thinking in probabilities, "
                                   "-ln p reverses the order", __func__,
                                   (unsigned)(k - 1u), req->edges[k - 1u],
                                   (unsigned)k, req->edges[k]);
    }
    *out_bins = bins;
    *out_ref = ref;
    return BEAM_OK;
}


/* ── The exact route ─────────────────────────────────────────── */

/**
 * The unconditional logo of the generated measure, with no grid at all.
 *
 * `beam_model_positional_marginals` answers exactly this question exactly, so
 * routing here is not an optimisation -- it is the difference between an
 * answer and an approximation of one, and it is what lets `exact` be true.
 */
static BeamError run_exact(const BeamModel *m, const BeamLogoRequest *req,
                           uint32_t ref, BeamLogoSet *out) {
    const uint32_t L = req->length, R = m->n_residues;

    BeamLogo *g = (BeamLogo *)calloc(1u, sizeof(BeamLogo));
    if (!g) return BEAM_EALLOC("the logo");
    if (!logo_alloc(g, L, R, false)) {
        free(g);
        return BEAM_EALLOC("the logo arrays");
    }

    const BeamError e = beam_model_positional_marginals(m, L, NULL,
                                                        g->probability, NULL);
    if (e != BEAM_OK) { logo_free_one(g); free(g); return e; }

    const double lp = beam_model_log_length_prior(m, L);
    g->lo_nats = 0.0;
    g->hi_nats = INFINITY;
    g->log_mass = lp;
    g->mass = exp(lp);
    g->fraction = 1.0;
    g->resolution_error = NAN;
    derive(g, log((double)ref));

    out->n_logos = 1u;
    out->logo = g;
    out->on_grid_total = g->mass;
    out->length_total = g->mass;
    out->spacing = NAN;
    out->exact = true;
    return BEAM_OK;
}


/* ── The banded route ────────────────────────────────────────── */

/**
 * Resolve the request's edges into the concrete ascending array the DP takes,
 * padded so the whole grid is covered.
 *
 * Two jobs. `INFINITY` becomes the top of the grid, which only the library
 * can know once the spacing is chosen -- the reason this layer is in C.
 * And the span outside the caller's outermost edges is added as extra bands,
 * so the mass the grid actually held is measured rather than assumed: the
 * caller's bands summed against `on_grid_total` says how much of the length
 * they cover, and `on_grid_total` against `length_total` says whether the
 * grid was long enough to hold the length at all. Those are different faults
 * and neither is visible without the padding.
 */
static BeamError resolve_edges(const BeamLogoRequest *req, uint32_t bins,
                               double ds, double **out_edges,
                               uint32_t *out_n, uint32_t *out_first) {
    const double top = (double)(bins + BEAM_LOGO_TOP_SLACK) * ds;
    const uint32_t nu = req->n_edges > 1u ? req->n_edges : 0u;

    double *e = (double *)malloc((size_t)(nu + 2u) * sizeof(double));
    if (!e) return BEAM_EALLOC("the resolved band edges");

    /*
     * A finite edge is kept exactly as given, even above the top of the grid.
     * Clamping it there would be the silent kind of wrong: a band asking
     * about surprisal 900 would come back describing the whole grid instead
     * of being refused as the empty question it is. Only the infinities are
     * resolved, because only they were asking for "wherever the grid ends".
     */
    uint32_t n = 0, first = 0;
    if (nu == 0u) {
        e[n++] = 0.0;
        e[n++] = top;
    } else {
        double lo = req->edges[0];
        double hi = req->edges[nu - 1u];
        if (lo == -INFINITY) lo = 0.0;
        if (hi ==  INFINITY) hi = top;
        if (lo > 0.0) { e[n++] = 0.0; first = 1u; }
        e[n++] = lo;
        for (uint32_t k = 1u; k + 1u < nu; ++k) e[n++] = req->edges[k];
        if (nu > 1u) e[n++] = hi;
        if (hi < top) e[n++] = top;
    }

    for (uint32_t k = 1; k < n; ++k)
        if (!(e[k] > e[k - 1u])) {
            /* Reachable only when an infinity resolved onto a neighbour --
             * `+INFINITY` as the top edge of a band whose lower edge is
             * already past the grid, say. The user's own edges were checked
             * for ascent before this ran. */
            const double a = e[k - 1u], b = e[k];
            free(e);
            return BEAM_ERANGE("%s: resolving the band edges against a grid "
                               "that ends at %.6g nats left %.6g and %.6g "
                               "out of order; the band lies past the grid, so "
                               "widen it with more bins or a larger spacing",
                               __func__, top, a, b);
        }

    *out_edges = e; *out_n = n; *out_first = first;
    return BEAM_OK;
}


/**
 * Refuse a band that lies outside what this length can attain, before the
 * multi-gigabyte sweep rather than after it.
 *
 * `beam_surprisal_range_at` is two sweeps over the state space; what it
 * saves is the sweep's backward rows, gigabytes on an order-4 model. The
 * margin is the splitting width,
 * because a band just outside the exact extremes can still catch mass the
 * grid smeared past them -- so this refuses only what is CLEARLY empty and
 * leaves the borderline to the zero-mass check after the sweep.
 */
static BeamError check_reachable(const BeamModel *m, uint32_t len,
                                 const double *edges, uint32_t first,
                                 uint32_t n_user, double ds) {
    double lo = 0.0, hi = 0.0;
    const BeamError e = beam_surprisal_range_at(m, len, &lo, &hi);
    if (e != BEAM_OK) return e;
    if (!isfinite(lo) || !isfinite(hi)) return BEAM_OK;   /* nothing to say */

    const double margin = (double)(len + 2u) * ds;
    for (uint32_t j = 0; j < n_user; ++j) {
        const double a = edges[first + j], b = edges[first + j + 1u];
        if (b < lo - margin || a > hi + margin)
            return BEAM_ERANGE("%s: the band [%.6g, %.6g) nats holds no "
                               "sequence of length %u, whose surprisal runs "
                               "from %.6g to %.6g nats (Pseq %.3g down to "
                               "%.3g)", __func__, a, b, (unsigned)len,
                               lo, hi, exp(-lo), exp(-hi));
    }
    return BEAM_OK;
}

static BeamError run_banded(const BeamModel *m, const BeamLogoRequest *req,
                            uint32_t bins, uint32_t ref, BeamLogoSet *out) {
    const uint32_t L = req->length, R = m->n_residues;

    double ds = 0.0;
    BeamError err = beam_spectrum_spacing(m, bins, req->spacing, &ds);
    if (err != BEAM_OK) return err;

    double *edges = NULL;
    uint32_t n_edges = 0, first = 0;
    err = resolve_edges(req, bins, ds, &edges, &n_edges, &first);
    if (err != BEAM_OK) return err;
    const uint32_t n_all  = n_edges - 1u;
    const uint32_t n_user = req->n_edges > 1u ? req->n_edges - 1u : n_all;
    const uint32_t user0  = first;

    err = check_reachable(m, L, edges, user0, n_user, ds);
    if (err != BEAM_OK) { free(edges); return err; }

    double *marg = (double *)calloc((size_t)n_all * L * R, sizeof(double));
    double *bm   = (double *)calloc(n_all, sizeof(double));
    double *pm   = (double *)calloc((size_t)n_all * L, sizeof(double));
    if (!marg || !bm || !pm) {
        free(edges); free(marg); free(bm); free(pm);
        return BEAM_EALLOC("the banded marginals");
    }
    err = beam_model_positional_banded(m, L, req->counting, bins, ds,
                                       edges, n_edges, marg, bm, pm, &ds);
    if (err != BEAM_OK) {
        free(edges); free(marg); free(bm); free(pm);
        return err;
    }

    double on_grid = 0.0;
    for (uint32_t j = 0; j < n_all; ++j) on_grid += bm[j];

    BeamLogo *gs = (BeamLogo *)calloc(n_user, sizeof(BeamLogo));
    if (!gs) {
        free(edges); free(marg); free(bm); free(pm);
        return BEAM_EALLOC("the logos");
    }
    const double max_nats = log((double)ref);
    for (uint32_t j = 0; j < n_user; ++j) {
        const uint32_t src = user0 + j;
        if (!logo_alloc(&gs[j], L, R, true)) {
            for (uint32_t k = 0; k < j; ++k) logo_free_one(&gs[k]);
            free(gs); free(edges); free(marg); free(bm); free(pm);
            return BEAM_EALLOC("the logo arrays");
        }
        BeamLogo *g = &gs[j];
        memcpy(g->probability, marg + (size_t)src * L * R,
               (size_t)L * R * sizeof(double));
        memcpy(g->position_mass, pm + (size_t)src * L, (size_t)L * sizeof(double));
        g->lo_nats = edges[src];
        g->hi_nats = edges[src + 1u];
        g->mass = bm[src];
        g->log_mass = bm[src] > 0.0 ? log(bm[src]) : -INFINITY;
        g->fraction = on_grid > 0.0 ? bm[src] / on_grid : 0.0;

        double worst = 0.0;
        if (bm[src] > 0.0)
            for (uint32_t i = 0; i < L; ++i) {
                const double r = fabs(g->position_mass[i] - bm[src]) / bm[src];
                if (r > worst) worst = r;
            }
        g->resolution_error = worst;
        derive(g, max_nats);
    }

    /*
     * An empty band is refused, not returned. See the file header: zero
     * probabilities give zero entropy and therefore MAXIMUM information, so
     * the alternative is a blank plot beside a summary claiming perfect
     * conservation. This is the same judgement `positional_marginals` makes
     * about a clamp of probability zero.
     */
    for (uint32_t j = 0; j < n_user; ++j) {
        if (gs[j].mass > 0.0) continue;
        const double a = gs[j].lo_nats, b = gs[j].hi_nats;
        for (uint32_t k = 0; k < n_user; ++k) logo_free_one(&gs[k]);
        free(gs); free(edges); free(marg); free(bm); free(pm);
        return BEAM_FAIL(BEAM_ERR_NO_LIVE_PATHS,
                         "the band [%.6g, %.6g) nats (Pseq %.3g down to %.3g) "
                         "holds no mass at length %u, so it has no residue "
                         "distribution to report; a logo of nothing would come "
                         "back as maximum information content at every "
                         "position", a, b, exp(-a), exp(-b), (unsigned)L);
    }

    if (req->max_resolution_error > 0.0) {
        for (uint32_t j = 0; j < n_user; ++j) {
            if (gs[j].resolution_error <= req->max_resolution_error) continue;
            /* Everything the message needs, read BEFORE the free. */
            const double got = gs[j].resolution_error;
            const double width = gs[j].hi_nats - gs[j].lo_nats;
            for (uint32_t k = 0; k < n_user; ++k) logo_free_one(&gs[k]);
            free(gs); free(edges); free(marg); free(bm); free(pm);
            return BEAM_ERANGE("%s: band %u resolves to %.3g, past the %.3g "
                               "asked for. The grid is too coarse for a band "
                               "this narrow: the splitting kernel is "
                               "(length+1)*spacing = %.4g nats against a band "
                               "of %.4g, so raise bins or widen the band",
                               __func__, (unsigned)j, got,
                               req->max_resolution_error,
                               (double)(L + 1u) * ds, width);
        }
    }

    out->n_logos = n_user;
    out->logo = gs;
    out->on_grid_total = on_grid;
    out->spacing = ds;
    /* One band over the whole grid cannot misclassify anything: splitting
     * moves mass between bins and never across an edge that is not there. */
    out->exact = (n_user == 1u && n_all == 1u);
    free(edges); free(marg); free(bm); free(pm);
    return BEAM_OK;
}


/* ── Entry points ────────────────────────────────────────────── */

BeamError beam_insights_logos(const BeamModel *m, const BeamLogoRequest *req,
                              BeamLogoSet *out) {
    if (!out) return BEAM_ENULL("logo set");
    uint32_t bins = 0, ref = 0;
    const BeamError v = validate(m, req, &bins, &ref);
    if (v != BEAM_OK) return v;

    BeamLogoSet set;
    memset(&set, 0, sizeof(set));
    set.alphabet = beam_model_alphabet(m);
    set.reference_residues = ref;
    set.max_entropy_nats = log((double)ref);
    set.max_entropy_bits = set.max_entropy_nats * BEAM_LOG2E;
    set.counting = req->counting;

    const bool unconditional = req->n_edges <= 1u;
    const BeamError e = (unconditional && !req->counting)
        ? run_exact(m, req, ref, &set)
        : run_banded(m, req, bins, ref, &set);
    if (e != BEAM_OK) return e;

    set.length_total = req->counting ? beam_model_path_count(m, req->length)
                                     : exp(beam_model_log_length_prior(
                                               m, req->length));
    *out = set;
    return BEAM_OK;
}

BeamError beam_insights_logo_cost(const BeamModel *m,
                                  const BeamLogoRequest *req,
                                  BeamQueryCost *out) {
    if (!out) return BEAM_ENULL("cost");
    uint32_t bins = 0, ref = 0;
    const BeamError v = validate(m, req, &bins, &ref);
    if (v != BEAM_OK) return v;
    (void)ref;
    memset(out, 0, sizeof(*out));

    if (req->n_edges <= 1u && !req->counting) {
        /* The exact route: one forward-backward workspace over the state
         * space, and no grid at all. */
        const uint64_t S = m->n_states, R = m->n_residues;
        out->states = S * ((uint64_t)req->length + 1u);
        out->edges  = out->states * R;
        out->bytes  = S * R * sizeof(double)
                    + ((uint64_t)req->length + 3u) * S * sizeof(double);
        return BEAM_OK;
    }

    LiveHandle lh;
    const BeamError e = beam_live_acquire(m, req->length, &lh);
    if (e != BEAM_OK) return e;
    uint64_t slots = 0;
    for (uint32_t i = 0; i <= req->length; ++i) slots += lh.L->n_live[i];
    out->states = slots;
    out->edges  = slots * m->n_residues;
    beam_live_release(&lh);

    /*
     * The spacing and the edges exactly as `run_banded` resolves them, so the
     * figure below is the one the sweep's guard will check: the backward rows
     * are stored at their own widths, and the plan that measures them is what
     * the estimate runs.
     */
    double ds = 0.0;
    BeamError err = beam_spectrum_spacing(m, bins, req->spacing, &ds);
    if (err != BEAM_OK) return err;
    double *edges = NULL;
    uint32_t n_edges = 0, first = 0;
    err = resolve_edges(req, bins, ds, &edges, &n_edges, &first);
    if (err != BEAM_OK) return err;
    free(edges);
    (void)first;
    return beam_positional_banded_cost(m, req->length, bins, ds, n_edges,
                                       &out->bytes);
}
