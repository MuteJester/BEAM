/**
 * @file test_insights_logo.c
 * @brief What a sequence logo must satisfy, and what it must refuse.
 *
 * The arithmetic a logo does is small; almost everything that can go wrong
 * here is a question answered when it should have been refused. So the tests
 * split in two, and the second half is the larger one.
 *
 * The correctness half is anchored on ENUMERATION. Under the hierarchical
 * prior every string over the alphabet has mass, so the support of a
 * nucleotide model at length 5 is all 3,125 strings over ACGTN, and the
 * reference for a logo is a direct sum over them with no dynamic program in
 * it -- which is the only kind of check that can catch a quantity that has
 * been wrong since the day it was written. Every other check here relates the
 * answer to another answer from the same machinery.
 *
 * A banded logo can only be compared EXACTLY where no sequence sits within
 * the grid's splitting reach of an edge. With every string in the support
 * there is mass near almost any threshold, so every exact comparison of a
 * band places its edges in a gap of the enumerated surprisals, and the
 * checks that cannot enumerate use only edges that nothing can cross.
 *
 * Two things are deliberately NOT tested, and saying why is part of the
 * point. `height_bits == height_nats * log2(e)` is an identity of the
 * implementation -- one array is computed from the other -- so asserting it
 * gates the multiply and nothing else. And "two bands partition the mass" is
 * vacuous against the banded sweep, whose bands are differences of cumulative
 * totals and telescope by construction; the recombination test below compares
 * against the EXACT route instead, which shares none of that arithmetic.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/insights.h"

#include "../../src/native/internal/beam/internal.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

static char msg[512];

/* ── Fixtures ────────────────────────────────────────────────── */

#define N_AA 24
static const char *TRAIN_AA[N_AA] = {
    "CASSF",    "CASSY",    "CAGSF",    "CGSSF",    "CASSL",    "CTSSF",
    "CASSQEV",  "CASRWMT",  "CAWSFGN",  "CAYSLKQ",  "CASSFGH",  "CASSYWM",
    "CAGSFKPD", "CGSSFPDE", "CASSLNRT", "CTSSFRWY", "CASSQEVI", "CASRWMTL",
    "CAWSFGNE", "CAYSLKQD", "CASSLGGYE", "CASSLGGTE", "CASRLGGYE", "CASSPGGYE",
};
static const double W_AA[N_AA] = {
    40, 25, 18, 15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 12, 9, 7, 5,
};

#define N_NT 25
static const char *TRAIN_NT[N_NT] = {
    "ACGTA", "ACGTC", "ACGGA", "AGGTA", "ACTTA",
    "CGTAC", "CGGAC", "GGTAC", "CTTAG", "CGTAG",
    "TACGT", "TAGGC", "GCATG", "GCTAG", "TTACG",
    "ATCGA", "AGCTT", "CATGC", "TGCAT", "GATCA",
    "ACGTAC", "AGGTCA", "TTACGG", "CATGCA", "GCTAGT",
};
static const double W_NT[N_NT] = {
    30, 22, 17, 13, 11, 9, 8, 7, 6, 5, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 9, 8, 7, 6, 5,
};

/*
 * Order 2 or 3 at most. Every banded sweep carries a surprisal grid per live
 * state, and under the prior every state is live: order 3 holds 8,000 states
 * a layer where order 2 holds 400, which is gigabytes against megabytes at
 * the grids used here.
 */
static BeamModel *build_aa(uint32_t order) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.order = order; c.max_length = 10u;
    BeamModel *m = beam_model_new(&c);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN_AA, W_AA, N_AA) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

/* Max length 6, so the whole model is enumerable, not only the length the
 * tests read. */
static BeamModel *build_nt(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.alphabet = BEAM_ALPHABET_NT;
    c.order = 2u; c.max_length = 6u;
    /* Fixed strengths. The banded tests need gaps in the spectrum where the
     * band edges fall, and chose the grid for this spectrum; strengths
     * re-estimated from the counts would move the atoms under them. */
    c.alpha_auto = false;
    c.alpha_length_auto = false;
    BeamModel *m = beam_model_new(&c);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN_NT, W_NT, N_NT) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

/** The whole support of one length, as (surprisal, probability, residues). */
typedef struct {
    uint32_t n, length, n_residues;
    double  *surprisal, *probability;
    uint8_t *letters;
} Support;

/*
 * Every string of length `L` over the model's whole alphabet. Under the
 * prior each one has mass, so a string that fails to score is itself a
 * failure, reported by returning false with `s->n` short of R^L.
 */
static bool enumerate_support(const BeamModel *m, uint32_t L, Support *s) {
    const uint32_t R = beam_model_n_residues(m);
    uint64_t total = 1;
    for (uint32_t i = 0; i < L; ++i) total *= R;
    s->n = 0; s->length = L; s->n_residues = R;
    s->surprisal   = (double *)malloc((size_t)total * sizeof(double));
    s->probability = (double *)malloc((size_t)total * sizeof(double));
    s->letters     = (uint8_t *)malloc((size_t)total * L);
    if (!s->surprisal || !s->probability || !s->letters) return false;
    char seq[16];
    uint32_t idx[16];
    for (uint64_t code = 0; code < total; ++code) {
        uint64_t c = code;
        for (uint32_t i = 0; i < L; ++i) { idx[i] = (uint32_t)(c % R); c /= R; }
        for (uint32_t i = 0; i < L; ++i) seq[i] = m->alpha->idx_to_char[idx[i]];
        seq[L] = '\0';
        const double lp = beam_model_log_pseq(m, seq);
        if (!isfinite(lp)) return false;
        s->surprisal[s->n] = -lp;
        s->probability[s->n] = exp(lp);
        for (uint32_t i = 0; i < L; ++i) s->letters[(size_t)s->n * L + i] = (uint8_t)idx[i];
        ++s->n;
    }
    return true;
}

static void support_free(Support *s) {
    free(s->surprisal); free(s->probability); free(s->letters);
    memset(s, 0, sizeof(*s));
}

/** Marginals and mass over the sequences of the support inside `[lo, hi)`. */
static void direct(const Support *s, double lo, double hi, bool counting,
                   double *out_p, double *out_mass) {
    const uint32_t L = s->length, R = s->n_residues;
    for (size_t k = 0; k < (size_t)L * R; ++k) out_p[k] = 0.0;
    double z = 0.0;
    for (uint32_t k = 0; k < s->n; ++k) {
        if (!(s->surprisal[k] >= lo && s->surprisal[k] < hi)) continue;
        const double w = counting ? 1.0 : s->probability[k];
        z += w;
        for (uint32_t i = 0; i < L; ++i)
            out_p[(size_t)i * R + s->letters[(size_t)k * L + i]] += w;
    }
    if (z > 0.0) for (size_t k = 0; k < (size_t)L * R; ++k) out_p[k] /= z;
    *out_mass = z;
}

/** The spacing the spectrum, and therefore an automatic logo grid, uses. */
static double grid_spacing(const BeamModel *m, uint32_t L, uint32_t bins) {
    double ds = 0.0;
    double *g = (double *)calloc(bins, sizeof(double));
    if (!g) return 0.0;
    if (beam_model_spectrum(m, true, L, bins, 0.0, g, &ds) != BEAM_OK) ds = 0.0;
    free(g);
    return ds;
}

typedef struct { double s, p; } SurpProb;

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int cmp_surp(const void *a, const void *b) {
    const double x = ((const SurpProb *)a)->s, y = ((const SurpProb *)b)->s;
    return (x > y) - (x < y);
}

/*
 * Linear splitting spreads one sequence's mass over up to `L+1` adjacent
 * bins; twice that is the distance from every sequence that a threshold has
 * to keep for the classification to be exact rather than approximate.
 */
static double splitting_reach(uint32_t L, double ds) {
    return (double)(2u * (L + 2u)) * ds;
}

/*
 * A threshold in a gap of the support's surprisals wider than twice the
 * splitting reach, chosen so the weight below it is as near `frac` of the
 * whole as any such gap allows -- in the generated measure, or by count.
 * Returns -1 when no gap is wide enough.
 */
static double gap_cut(const Support *s, double reach, bool counting,
                      double frac) {
    SurpProb *v = (SurpProb *)malloc((size_t)s->n * sizeof(SurpProb));
    if (!v) return -1.0;
    double total = 0.0;
    for (uint32_t k = 0; k < s->n; ++k) {
        v[k].s = s->surprisal[k];
        v[k].p = counting ? 1.0 : s->probability[k];
        total += v[k].p;
    }
    qsort(v, s->n, sizeof(SurpProb), cmp_surp);
    double cut = -1.0, best = INFINITY, below = 0.0;
    for (uint32_t k = 1; k < s->n; ++k) {
        below += v[k - 1u].p;
        if (v[k].s - v[k - 1u].s <= 2.0 * reach || v[k - 1u].s <= reach)
            continue;
        const double d = fabs(below / total - frac);
        if (d < best) { best = d; cut = 0.5 * (v[k - 1u].s + v[k].s); }
    }
    free(v);
    return cut;
}

/*
 * The median surprisal of a length's generated mass, read off the spectrum
 * at a bin boundary. For the checks whose claim does not depend on where a
 * sequence falls, only on there being real mass either side.
 */
static double median_surprisal(const BeamModel *m, uint32_t L) {
    const uint32_t B = 4096u;
    double ds = 0.0;
    double *g = (double *)calloc(B, sizeof(double));
    if (!g) return -1.0;
    if (beam_model_spectrum(m, false, L, B, 0.0, g, &ds) != BEAM_OK) {
        free(g); return -1.0;
    }
    double total = 0.0, cum = 0.0;
    for (uint32_t b = 0; b < B; ++b) total += g[b];
    uint32_t b = 0;
    for (; b < B; ++b) { cum += g[b]; if (cum >= 0.5 * total) break; }
    free(g);
    return (double)(b + 1u) * ds;
}


/* ── 1. Against enumeration ──────────────────────────────────── */

/**
 * The unconditional logo, against a direct sum over every sequence of that
 * length.
 *
 * The reference shares nothing with the code under test: no trellis, no grid,
 * no dynamic program -- just `beam_model_log_pseq` over all R^L strings and a
 * running total. Everything else in this file compares one answer from this
 * machinery against another from the same machinery, and would pass in
 * concert if the machinery were wrong.
 */
static void test_the_unconditional_logo_is_a_direct_sum_over_the_support(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t L = 5u, R = beam_model_n_residues(m);
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not reach length 5");

    Support s;
    ASSERT_MSG(enumerate_support(m, L, &s),
               "a string of length 5 did not score; every readable string "
               "has mass under the prior");

    double *ref = (double *)calloc((size_t)L * R, sizeof(double));
    double zref = 0.0;
    ASSERT_MSG(ref, "allocation");
    direct(&s, 0.0, INFINITY, false, ref, &zref);

    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = L;
    BeamLogoSet set;
    const BeamError e = beam_insights_logos(m, &req, &set);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());
    ASSERT_MSG(set.exact, "the unconditional generated logo should be exact");

    double worst = 0.0;
    for (uint32_t i = 0; i < L; ++i)
        for (uint32_t a = 0; a < R; ++a) {
            const double d = fabs(set.logo[0].probability[(size_t)i * R + a]
                                  - ref[(size_t)i * R + a]);
            if (d > worst) worst = d;
        }
    const double mass = set.logo[0].mass;
    beam_insights_logo_set_free(&set);
    free(ref); support_free(&s);
    beam_model_destroy(m);

    if (worst > 1e-12) {
        snprintf(msg, sizeof msg,
                 "the logo departs from a direct sum over the support by %.3g",
                 worst);
        ASSERT_MSG(false, msg);
    }
    if (fabs(mass - zref) > 1e-12 * zref) {
        snprintf(msg, sizeof msg, "mass %.17g against the enumerated %.17g",
                 mass, zref);
        ASSERT_MSG(false, msg);
    }
    PASS();
}

/**
 * Conditioned logos, against the same enumeration filtered by surprisal.
 *
 * Two thresholds, each in a GAP of the surprisal distribution wider than the
 * `length + 1` bins linear splitting can smear a sequence across, so every
 * sequence falls wholly one side and the comparison is exact rather than
 * approximate. One gap is chosen to halve the generated mass, which sits
 * among the few strings the corpus made likely; the other to halve the
 * COUNT, which sits deep in the tail where the strings the corpus never saw
 * are. Each measure therefore has one band split where its own weight is.
 * Both measures are checked; the counting measure has no other reference in
 * this file.
 */
static void test_a_banded_logo_is_a_filtered_sum_over_the_support(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    /* Fine enough that gaps exist near both medians: at 4,096 bins the
     * count-balanced gap sits at 94% of the strings instead of half. */
    const uint32_t L = 5u, R = beam_model_n_residues(m), B = 16384u;

    Support s;
    ASSERT_MSG(enumerate_support(m, L, &s), "enumeration");

    const double ds = grid_spacing(m, L, B);
    ASSERT_MSG(ds > 0.0, beam_error_message());
    const double reach = splitting_reach(L, ds);
    const double by_mass  = gap_cut(&s, reach, false, 0.5);
    const double by_count = gap_cut(&s, reach, true, 0.5);
    if (!(by_mass > 0.0 && by_count > by_mass)) {
        snprintf(msg, sizeof msg, "no pair of usable gaps: the mass cut is "
                 "%.6g and the count cut %.6g, so this fixture no longer "
                 "splits both measures exactly", by_mass, by_count);
        support_free(&s); beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }

    double use[4] = { -INFINITY, by_mass, by_count, INFINITY };
    const uint32_t NB = 3u;

    for (int counting = 0; counting <= 1; ++counting) {
        BeamLogoRequest req;
        beam_logo_request_default(&req);
        req.length = L; req.edges = use; req.n_edges = NB + 1u;
        req.counting = (bool)counting; req.bins = B; req.spacing = ds;
        BeamLogoSet set;
        const BeamError e = beam_insights_logos(m, &req, &set);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
        ASSERT_MSG(set.n_logos == NB, "expected three logos");

        double *ref = (double *)calloc((size_t)L * R, sizeof(double));
        ASSERT_MSG(ref, "allocation");
        for (uint32_t j = 0; j < NB; ++j) {
            double zref = 0.0;
            direct(&s, j == 0u ? 0.0 : use[j], use[j + 1u], (bool)counting,
                   ref, &zref);
            if (!(zref > 0.0)) {
                snprintf(msg, sizeof msg, "band %u is empty in the reference, "
                         "so the cut proves nothing", (unsigned)j);
                goto fail;
            }
            if (fabs(set.logo[j].mass - zref) > 1e-9 * zref) {
                snprintf(msg, sizeof msg,
                         "counting %d band %u: mass %.17g, enumeration %.17g",
                         counting, (unsigned)j, set.logo[j].mass, zref);
                goto fail;
            }
            for (uint32_t i = 0; i < L; ++i)
                for (uint32_t a = 0; a < R; ++a) {
                    const double got = set.logo[j].probability[(size_t)i * R + a];
                    const double want = ref[(size_t)i * R + a];
                    if (fabs(got - want) > 1e-9) {
                        snprintf(msg, sizeof msg,
                                 "counting %d band %u position %u residue %u: "
                                 "%.17g against %.17g", counting, (unsigned)j,
                                 (unsigned)i, (unsigned)a, got, want);
                        goto fail;
                    }
                }
        }
        free(ref);
        beam_insights_logo_set_free(&set);
        continue;
    fail:
        free(ref);
        beam_insights_logo_set_free(&set);
        support_free(&s);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }
    support_free(&s);
    beam_model_destroy(m);
    PASS();
}


/* ── 2. The logo arithmetic ──────────────────────────────────── */

/* Every stack of the exact logos of lengths `L_lo..L_hi`, or false with
 * `msg` naming the first that fails. */
static bool check_stacks(BeamModel *m, uint32_t L_lo, uint32_t L_hi,
                         const char *name, uint32_t *checked) {
    const uint32_t R = beam_model_n_residues(m);
    for (uint32_t L = L_lo; L <= L_hi; ++L) {
        if (!beam_length_live(m, L)) continue;
        BeamLogoRequest req;
        beam_logo_request_default(&req);
        req.length = L;
        BeamLogoSet set;
        if (beam_insights_logos(m, &req, &set) != BEAM_OK) {
            snprintf(msg, sizeof msg, "%s L %u: %s", name, (unsigned)L,
                     beam_error_message());
            return false;
        }
        const BeamLogo *g = &set.logo[0];
        bool ok = true;
        for (uint32_t i = 0; i < L && ok; ++i) {
            double p = 0.0, hb = 0.0;
            for (uint32_t a = 0; a < R; ++a) {
                p  += g->probability[(size_t)i * R + a];
                hb += g->height_bits[(size_t)i * R + a];
            }
            ++*checked;
            const double info = g->information_bits[i];
            if (fabs(p - 1.0) > 1e-12) {
                snprintf(msg, sizeof msg, "%s L %u pos %u: probabilities sum "
                         "to %.17g", name, (unsigned)L, (unsigned)i, p);
                ok = false;
            } else if (fabs(hb - info) > 1e-12) {
                snprintf(msg, sizeof msg, "%s L %u pos %u: heights sum to "
                         "%.17g but information is %.17g", name, (unsigned)L,
                         (unsigned)i, hb, info);
                ok = false;
            } else if (fabs(g->information_bits[i]
                            - (set.max_entropy_bits - g->entropy_bits[i])) > 1e-12) {
                snprintf(msg, sizeof msg, "%s L %u pos %u: information is not "
                         "max_entropy - entropy", name, (unsigned)L, (unsigned)i);
                ok = false;
            } else if (!(g->entropy_bits[i] >= -1e-12
                         && g->entropy_bits[i] <= set.max_entropy_bits + 1e-12)) {
                snprintf(msg, sizeof msg, "%s L %u pos %u: entropy %.17g is "
                         "outside [0, %.17g]", name, (unsigned)L, (unsigned)i,
                         g->entropy_bits[i], set.max_entropy_bits);
                ok = false;
            } else if (g->n_support[i] != R) {
                snprintf(msg, sizeof msg, "%s L %u pos %u: %u residues carry "
                         "probability, where the prior gives every one of the "
                         "%u some", name, (unsigned)L, (unsigned)i,
                         (unsigned)g->n_support[i], (unsigned)R);
                ok = false;
            }
        }
        beam_insights_logo_set_free(&set);
        if (!ok) return false;
    }
    return true;
}

/**
 * The identities every stack must satisfy, over several models and lengths.
 *
 * `sum_a height[i][a] == information[i]` is the one that matters: it is what
 * makes a logo's total height READ as information, which is the entire point
 * of the Schneider-Stephens construction. The rest pin the definition, and
 * one pins the prior: no residue is impossible anywhere, so every position
 * reports the whole alphabet as its support.
 */
static void test_the_stack_identities_hold(void) {
    uint32_t checked = 0;
    for (uint32_t order = 2; order <= 3; ++order) {
        BeamModel *m = build_aa(order);
        ASSERT_MSG(m, "model did not build");
        const bool ok = check_stacks(m, 4u, 10u, order == 2u ? "aa order 2"
                                                            : "aa order 3",
                                     &checked);
        beam_model_destroy(m);
        ASSERT_MSG(ok, msg);
    }
    /* And a radix of 6, where N is a residue, every length of the model. */
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const bool ok = check_stacks(m, 1u, 6u, "nt", &checked);
    beam_model_destroy(m);
    ASSERT_MSG(ok, msg);
    ASSERT_MSG(checked > 100u, "too few positions were checked");
    PASS();
}

/**
 * A position the corpus always spelled one way is dominated by that residue,
 * but it is not one-hot, and its logo must say both.
 *
 * Every training sequence of the amino-acid fixture begins with C. Under the
 * prior that makes C the consensus at position 0 by a wide margin, while
 * every other residue keeps some probability: the support is the whole
 * alphabet, the entropy is above zero, and the information falls short of
 * the maximum. A logo reporting one residue of support, or the maximum
 * information, would be reporting the closed world the prior replaced.
 *
 * The consensus itself is checked against the probability row at every
 * position, which is the only definition it has.
 */
static void test_a_conserved_position_is_dominated_but_not_one_hot(void) {
    BeamModel *m = build_aa(3u);
    ASSERT_MSG(m, "model did not build");
    const uint32_t L = 9u, R = beam_model_n_residues(m);
    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = L;
    BeamLogoSet set;
    ASSERT_MSG(beam_insights_logos(m, &req, &set) == BEAM_OK, beam_error_message());
    const BeamLogo *g = &set.logo[0];

    bool consensus_ok = true;
    for (uint32_t i = 0; i < L; ++i) {
        uint32_t arg = 0;
        for (uint32_t a = 1; a < R; ++a)
            if (g->probability[(size_t)i * R + a] > g->probability[(size_t)i * R + arg])
                arg = a;
        if (g->consensus[i] != arg
            || g->consensus_probability[i] != g->probability[(size_t)i * R + arg])
            consensus_ok = false;
    }
    const double h = g->entropy_bits[0], info = g->information_bits[0];
    const uint32_t sup = g->n_support[0];
    const uint32_t cons = g->consensus[0];
    const double cp = g->consensus_probability[0];
    const double maxe = set.max_entropy_bits;
    const uint32_t c_index = m->alpha->char_to_idx[(unsigned char)'C'];
    beam_insights_logo_set_free(&set);
    beam_model_destroy(m);

    ASSERT_MSG(consensus_ok,
               "a consensus is not the argmax of its probability row");
    ASSERT_MSG(cons == c_index, "position 0 is not C, which every sequence "
               "of the corpus begins with");
    if (!(cp > 0.5 && cp < 1.0)) {
        snprintf(msg, sizeof msg, "position 0 is C with probability %.17g; it "
                 "should dominate without being certain", cp);
        ASSERT_MSG(false, msg);
    }
    if (sup != R) {
        snprintf(msg, sizeof msg, "position 0 spells %u residues, not all %u",
                 (unsigned)sup, (unsigned)R);
        ASSERT_MSG(false, msg);
    }
    ASSERT_MSG(h > 0.0 && info < maxe,
               "a position every residue can take cannot carry the maximum "
               "information");
    ASSERT_MSG(fabs(maxe - log2(20.0)) < 1e-12,
               "the amino-acid maximum is not log2(20) bits");
    PASS();
}

/**
 * `reference_residues` moves the ceiling, and nucleotides are why it exists.
 *
 * `BEAM_NT_RESIDUES` is 5 because N is an emitted residue here, so the honest
 * maximum is `log2(5)`. Every published DNA logo tops out at 2 bits, and a
 * caller who wants that axis has to be able to ask for it -- otherwise ours
 * is 16% tall and silently incomparable.
 */
static void test_the_reference_alphabet_sets_the_ceiling(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    ASSERT_MSG(beam_model_n_residues(m) == 5u,
               "this test exists because nucleotides have five residues here");

    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = 5u;
    BeamLogoSet def, four;
    ASSERT_MSG(beam_insights_logos(m, &req, &def) == BEAM_OK, beam_error_message());
    req.reference_residues = 4u;
    ASSERT_MSG(beam_insights_logos(m, &req, &four) == BEAM_OK, beam_error_message());

    const double d = def.max_entropy_bits, f = four.max_entropy_bits;
    const double id = def.logo[0].information_bits[0];
    const double if4 = four.logo[0].information_bits[0];
    const double hd = def.logo[0].entropy_bits[0], hf = four.logo[0].entropy_bits[0];
    beam_insights_logo_set_free(&def);
    beam_insights_logo_set_free(&four);
    beam_model_destroy(m);

    ASSERT_MSG(fabs(d - log2(5.0)) < 1e-12, "the default ceiling is not log2(5)");
    ASSERT_MSG(fabs(f - 2.0) < 1e-12, "asking for 4 residues did not give 2 bits");
    ASSERT_MSG(fabs(hd - hf) < 1e-15,
               "the reference changed the entropy, which it must not: it moves "
               "the ceiling, not the distribution");
    ASSERT_MSG(fabs((id - if4) - (d - f)) < 1e-12,
               "information did not move with the ceiling");
    PASS();
}


/* ── 3. Bands against the exact route ────────────────────────── */

/**
 * Two bands recombine into the unconditional logo, compared against the EXACT
 * route rather than against each other.
 *
 * The comparison has to leave the banded machinery to mean anything. Band
 * masses are differences of consecutive cumulative totals, so any two of them
 * sum to the whole by construction and a test that only checks that is
 * asserting an identity of the implementation. `positional_marginals` shares
 * none of it: no grid, no cumulative, no threshold.
 *
 * Recombining is exact only if every position of a band agrees on its mass,
 * which the grid guarantees only where no sequence sits near the edge -- so
 * the edge is placed in a gap of the enumerated surprisals, halving the mass.
 */
static void test_two_bands_recombine_into_the_exact_logo(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t L = 5u, R = beam_model_n_residues(m), B = 4096u;

    Support s;
    ASSERT_MSG(enumerate_support(m, L, &s), "enumeration");
    const double ds = grid_spacing(m, L, B);
    const double cut = gap_cut(&s, splitting_reach(L, ds), false, 0.5);
    support_free(&s);
    ASSERT_MSG(ds > 0.0 && cut > 0.0,
               "no gap wide enough to split the support exactly");

    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = L;
    BeamLogoSet exact;
    ASSERT_MSG(beam_insights_logos(m, &req, &exact) == BEAM_OK, beam_error_message());
    ASSERT_MSG(exact.exact, "the reference route is not the exact one");

    const double edges[3] = { -INFINITY, cut, INFINITY };
    req.edges = edges; req.n_edges = 3u; req.bins = B; req.spacing = ds;
    BeamLogoSet band;
    ASSERT_MSG(beam_insights_logos(m, &req, &band) == BEAM_OK, beam_error_message());
    ASSERT_MSG(band.n_logos == 2u, "expected two bands");
    ASSERT_MSG(!band.exact, "a real split must not claim to be exact");

    double spread = 0.0, worst = 0.0;
    const double tot = band.logo[0].mass + band.logo[1].mass;
    for (uint32_t i = 0; i < L; ++i)
        for (uint32_t a = 0; a < R; ++a) {
            const size_t k = (size_t)i * R + a;
            const double mix = (band.logo[0].mass * band.logo[0].probability[k]
                              + band.logo[1].mass * band.logo[1].probability[k]) / tot;
            const double d = fabs(mix - exact.logo[0].probability[k]);
            if (d > worst) worst = d;
            for (uint32_t j = 0; j < 2u; ++j) {
                const double sp = fabs(band.logo[j].probability[k]
                                       - exact.logo[0].probability[k]);
                if (sp > spread) spread = sp;
            }
        }
    const double on_grid = band.on_grid_total, whole = band.length_total;
    beam_insights_logo_set_free(&exact);
    beam_insights_logo_set_free(&band);
    beam_model_destroy(m);

    if (spread < 1e-3) {
        snprintf(msg, sizeof msg,
                 "the bands differ from the pooled logo by at most %.3g, so "
                 "recombining them would pass with banding doing nothing",
                 spread);
        ASSERT_MSG(false, msg);
    }
    if (worst > 1e-9) {
        snprintf(msg, sizeof msg, "recombined logo departs from the exact one "
                 "by %.3g", worst);
        ASSERT_MSG(false, msg);
    }
    if (fabs(on_grid - whole) > 1e-9 * whole) {
        snprintf(msg, sizeof msg, "the grid held %.17g of the length's %.17g, "
                 "so it is too short and the bands are missing mass",
                 on_grid, whole);
        ASSERT_MSG(false, msg);
    }
    PASS();
}

/**
 * A single band over the whole grid is reported as exact, and matches the
 * exact route.
 *
 * `exact` describes the answer, not the route. Nothing can be misclassified
 * when there is no interior edge to misclassify it across, and a flag that
 * said otherwise would be a route label wearing the name of a property.
 */
static void test_one_band_over_the_grid_is_exact(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");
    const uint32_t L = 9u, R = beam_model_n_residues(m);

    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = L;
    BeamLogoSet ex;
    ASSERT_MSG(beam_insights_logos(m, &req, &ex) == BEAM_OK, beam_error_message());

    const double edges[2] = { -INFINITY, INFINITY };
    req.edges = edges; req.n_edges = 2u; req.bins = 1024u;
    BeamLogoSet one;
    ASSERT_MSG(beam_insights_logos(m, &req, &one) == BEAM_OK, beam_error_message());

    const bool flagged = one.exact;
    double worst = 0.0;
    for (size_t k = 0; k < (size_t)L * R; ++k) {
        const double d = fabs(one.logo[0].probability[k] - ex.logo[0].probability[k]);
        if (d > worst) worst = d;
    }
    const double sp = one.spacing, re = one.logo[0].resolution_error;
    const double exsp = ex.spacing, exre = ex.logo[0].resolution_error;
    beam_insights_logo_set_free(&ex);
    beam_insights_logo_set_free(&one);
    beam_model_destroy(m);

    ASSERT_MSG(flagged, "an all-covering band is not flagged exact");
    if (worst > 1e-12) {
        snprintf(msg, sizeof msg, "an all-covering band departs from the exact "
                 "route by %.3g", worst);
        ASSERT_MSG(false, msg);
    }
    ASSERT_MSG(isnan(exsp) && isnan(exre),
               "the no-grid route must report spacing and resolution as NaN, "
               "not 0.0, which would claim a grid error was measured");
    ASSERT_MSG(isfinite(sp) && isfinite(re),
               "a band that did use a grid must report what it was");
    PASS();
}


/* ── 4. What it refuses ──────────────────────────────────────── */

/**
 * An empty band is refused rather than answered.
 *
 * This is the refusal the file exists for. Zero probabilities give zero
 * entropy and therefore the MAXIMUM information content, so an answered empty
 * band is a blank plot beside a summary claiming every position is perfectly
 * conserved. `beam_model_positional_banded` returns an empty band happily and
 * is right to -- one empty band must not deny a caller the others -- so this
 * is a judgement the logo layer has to make for itself.
 */
static void test_an_empty_band_is_refused_not_answered(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t L = 5u, B = 4096u;

    /*
     * The band has to sit INSIDE the surprisal this length attains, or it is
     * refused earlier and cheaper by the reachability check, which is a
     * different refusal testing different code. A gap between two sequences
     * of the support is inside the range and holds nothing -- and with every
     * string in the support, the gaps wide enough are among the few likely
     * strings at the bottom of the range.
     */
    Support s;
    ASSERT_MSG(enumerate_support(m, L, &s), "enumeration");
    double *sorted = (double *)malloc((size_t)s.n * sizeof(double));
    ASSERT_MSG(sorted, "allocation");
    memcpy(sorted, s.surprisal, (size_t)s.n * sizeof(double));
    qsort(sorted, s.n, sizeof(double), cmp_double);
    const double ds = grid_spacing(m, L, B);
    ASSERT_MSG(ds > 0.0, beam_error_message());
    const double margin = splitting_reach(L, ds);
    double lo = -1.0, hi = -1.0;
    for (uint32_t k = 1; k < s.n; ++k) {
        if (sorted[k] - sorted[k - 1u] <= 4.0 * margin) continue;
        lo = sorted[k - 1u] + 1.5 * margin;
        hi = sorted[k]      - 1.5 * margin;
        break;
    }
    free(sorted);
    support_free(&s);
    ASSERT_MSG(lo > 0.0 && hi > lo,
               "no gap wide enough to hold an empty band; this fixture no "
               "longer covers the case");

    const double edges[2] = { lo, hi };
    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = L; req.edges = edges; req.n_edges = 2u;
    req.bins = B; req.spacing = ds;
    BeamLogoSet set;
    const BeamError e = beam_insights_logos(m, &req, &set);
    if (e == BEAM_OK) beam_insights_logo_set_free(&set);
    beam_model_destroy(m);

    if (e != BEAM_ERR_NO_LIVE_PATHS) {
        snprintf(msg, sizeof msg,
                 "a band in a gap of the support came back with %d rather "
                 "than BEAM_ERR_NO_LIVE_PATHS; it would have reported maximum "
                 "information at every position", (int)e);
        ASSERT_MSG(false, msg);
    }
    PASS();
}

/**
 * Every malformed request, each with the code it must give -- and a control
 * that must SUCCEED.
 *
 * The control is not ceremony. During the banded work an entire
 * malformed-input test passed while covering nothing, because the length it
 * used was invalid and every case refused for that reason; it would have
 * passed with the validation deleted. A positive control is what makes the
 * negatives attributable.
 */
static void test_what_a_malformed_request_gets(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");
    const uint32_t L = 9u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not generate length 9");

    const double descending[2] = { 5.0, 3.0 };
    const double repeated[2]   = { 3.0, 3.0 };
    const double nan_edge[2]   = { 0.0, NAN };
    const double negative[2]   = { -8.0, -3.0 };
    /* Far past the most improbable string of length 9, which is some tens of
     * nats even though every string has mass. */
    const double far[2]        = { 900.0, 1000.0 };
    const double fine[2]       = { -INFINITY, INFINITY };

    /*
     * Each case pins the MESSAGE as well as the code, and that is the whole
     * value of the table. Every refusal here is the same code, and several of
     * these inputs are caught by more than one check -- a NaN edge fails the
     * ascent test too, since every comparison against NaN is false. Asserting
     * only the code let a mutation that deleted the NaN check survive, because
     * the ascent check refused it anyway with a far less useful message.
     */
    struct { const char *what; const double *e; uint32_t n, ref, len;
             BeamError want; const char *says; }
    cases[] = {
        { "descending edges",  descending, 2u, 0u,  L,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "ascend strictly" },
        { "a repeated edge",   repeated,   2u, 0u,  L,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "ascend strictly" },
        { "a NaN edge",        nan_edge,   2u, 0u,  L,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "is NaN" },
        { "a negative edge",   negative,   2u, 0u,  L,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "never negative" },
        { "a band past the support", far,  2u, 0u,  L,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "holds no sequence of length" },
        { "a reference wider than the alphabet", NULL, 0u, 99u, L,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "reference_residues" },
        { "a length the model lacks", NULL, 0u, 0u, 200u,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "not one this model generates" },
        { "length zero",       NULL,       0u, 0u,  0u,
          BEAM_ERR_PARAM_OUT_OF_RANGE, "per length" },
    };
    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; ++k) {
        BeamLogoRequest req;
        beam_logo_request_default(&req);
        req.length = cases[k].len;
        req.reference_residues = cases[k].ref;
        req.bins = 1024u;
        if (cases[k].n) { req.edges = cases[k].e; req.n_edges = cases[k].n; }
        BeamLogoSet set;
        const BeamError e = beam_insights_logos(m, &req, &set);
        if (e == BEAM_OK) beam_insights_logo_set_free(&set);
        if (e != cases[k].want) {
            snprintf(msg, sizeof msg, "%s gave %d, wanted %d",
                     cases[k].what, (int)e, (int)cases[k].want);
            beam_model_destroy(m);
            ASSERT_MSG(false, msg);
        }
        if (!strstr(beam_error_message(), cases[k].says)) {
            snprintf(msg, sizeof msg,
                     "%s was refused, but for the wrong reason: the message "
                     "does not mention \"%s\". It said: %.200s",
                     cases[k].what, cases[k].says, beam_error_message());
            beam_model_destroy(m);
            ASSERT_MSG(false, msg);
        }
    }

    /* The control. */
    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = L; req.edges = fine; req.n_edges = 2u; req.bins = 1024u;
    BeamLogoSet set;
    const BeamError e = beam_insights_logos(m, &req, &set);
    if (e != BEAM_OK) {
        snprintf(msg, sizeof msg, "the well-formed control was refused with %d "
                 "(%s), so the refusals above prove nothing about the band",
                 (int)e, beam_error_message());
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }
    beam_insights_logo_set_free(&set);
    beam_model_destroy(m);
    PASS();
}

/**
 * A band with a FINITE top edge still knows how much the grid held.
 *
 * Every other band here is bounded by `INFINITY`, which resolves to the top
 * of the grid and leaves nothing above it -- so the padding that measures the
 * rest of the grid is never exercised, and a mutation deleting it survives.
 * It matters because two different faults are read off these totals: the
 * caller's bands against `on_grid_total` say what fraction of the length they
 * cover, and `on_grid_total` against `length_total` says whether the grid was
 * long enough to hold the length at all. Without the padding the first is
 * always 1 and the second always looks short.
 */
static void test_a_finite_top_edge_still_measures_the_whole_grid(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");
    const uint32_t L = 9u;

    /* One band from the bottom of the grid to the middle of the mass, so
     * real mass is left above it. */
    const double mid = median_surprisal(m, L);
    ASSERT_MSG(mid > 0.0, "could not place the median");
    const double edges[2] = { -INFINITY, mid };
    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = L; req.edges = edges; req.n_edges = 2u; req.bins = 2048u;
    BeamLogoSet set;
    const BeamError e = beam_insights_logos(m, &req, &set);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());
    ASSERT_MSG(set.n_logos == 1u, "expected one logo");

    const double mass = set.logo[0].mass, frac = set.logo[0].fraction;
    const double on_grid = set.on_grid_total, whole = set.length_total;
    beam_insights_logo_set_free(&set);
    beam_model_destroy(m);

    if (!(frac > 0.05 && frac < 0.95)) {
        snprintf(msg, sizeof msg,
                 "the band is %.4f of the grid; a fraction of 1 means the "
                 "mass above the top edge was never counted", frac);
        ASSERT_MSG(false, msg);
    }
    if (fabs(on_grid - whole) > 1e-9 * whole) {
        snprintf(msg, sizeof msg,
                 "the grid held %.17g of the length's %.17g, which would say "
                 "it is too short -- but the band simply stops early and the "
                 "rest of the grid has to be measured anyway",
                 on_grid, whole);
        ASSERT_MSG(false, msg);
    }
    ASSERT_MSG(fabs(mass / on_grid - frac) < 1e-15,
               "fraction is not mass over what the grid held");
    PASS();
}

/**
 * A grid too coarse for the band asked for is refused when the caller says to.
 *
 * The library's own habit is to refuse a question it cannot answer exactly
 * rather than to return an answer with a footnote. `resolution_error` is the
 * footnote; this is the refusal, for callers who would rather not have to
 * remember to read it.
 */
static void test_a_grid_too_coarse_is_refused_on_request(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");
    const double mid = median_surprisal(m, 9u);
    ASSERT_MSG(mid > 0.0, "could not place the median");
    const double edges[3] = { -INFINITY, mid, INFINITY };
    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = 9u; req.edges = edges; req.n_edges = 3u; req.bins = 1024u;

    /* Impossible to meet: no grid resolves a band to one part in 1e-300. */
    req.max_resolution_error = 1e-300;
    BeamLogoSet set;
    BeamError e = beam_insights_logos(m, &req, &set);
    if (e == BEAM_OK) beam_insights_logo_set_free(&set);
    if (e != BEAM_ERR_PARAM_OUT_OF_RANGE) {
        snprintf(msg, sizeof msg, "an unmeetable resolution gave %d", (int)e);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }

    /* And the same request without the demand must succeed, or the refusal
     * above was about something else. */
    req.max_resolution_error = 0.0;
    e = beam_insights_logos(m, &req, &set);
    if (e != BEAM_OK) {
        snprintf(msg, sizeof msg, "the control was refused with %d (%s)",
                 (int)e, beam_error_message());
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }
    beam_insights_logo_set_free(&set);
    beam_model_destroy(m);
    PASS();
}

/**
 * The cost estimate is reported before the sweep, and it is the figure the
 * sweep's guard checks -- not a bound on it.
 */
static void test_the_cost_estimate_precedes_the_sweep(void) {
    BeamModel *m = build_aa(3u);
    ASSERT_MSG(m, "model did not build");
    const double edges[2] = { -INFINITY, INFINITY };
    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = 9u;

    BeamQueryCost c0;
    BeamError e = beam_insights_logo_cost(m, &req, &c0);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());
    ASSERT_MSG(c0.bytes > 0u, "the exact route costs nothing");

    /* The estimate must SEE the grid: a finer grid costs more. */
    uint64_t b[3];
    const uint32_t bins[3] = { 1024u, 2048u, 4096u };
    req.edges = edges; req.n_edges = 2u;
    for (int k = 0; k < 3; ++k) {
        BeamQueryCost c;
        req.bins = bins[k];
        e = beam_insights_logo_cost(m, &req, &c);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
        b[k] = c.bytes;
    }
    if (!(b[2] > b[1] && b[1] > b[0])) {
        beam_model_destroy(m);
        snprintf(msg, sizeof msg, "the estimate does not grow with bins: "
                 "%llu, %llu, %llu", (unsigned long long)b[0],
                 (unsigned long long)b[1], (unsigned long long)b[2]);
        ASSERT_MSG(false, msg);
    }

    /*
     * And it is EXACTLY the guard's figure. Growth alone would pass an
     * estimate charging every backward row the full grid, which is what this
     * one did before the rows were stored at their own widths -- three to
     * five times what the sweep holds. A budget of the reported bytes must
     * run the sweep, and one byte less must refuse it: nothing looser pins
     * that the two come from one expression.
     */
    req.bins = bins[1];
    const uint64_t was = beam_model_memory_budget();
    BeamLogoSet set;
    memset(&set, 0, sizeof set);
    beam_model_set_memory_budget(b[1]);
    const BeamError at = beam_insights_logos(m, &req, &set);
    beam_insights_logo_set_free(&set);
    beam_model_set_memory_budget(b[1] - 1u);
    const BeamError under = beam_insights_logos(m, &req, &set);
    beam_insights_logo_set_free(&set);
    beam_model_set_memory_budget(was);
    beam_model_destroy(m);

    if (at != BEAM_OK) {
        snprintf(msg, sizeof msg, "a budget of exactly the estimate (%llu "
                 "bytes) was refused with %d", (unsigned long long)b[1],
                 (int)at);
        ASSERT_MSG(false, msg);
    }
    if (under != BEAM_ERR_BUDGET) {
        snprintf(msg, sizeof msg, "a budget one byte under the estimate was "
                 "answered with %d rather than refused, so the estimate "
                 "overstates what the sweep holds", (int)under);
        ASSERT_MSG(false, msg);
    }
    PASS();
}

int main(void) {
    printf("insights: sequence logos\n");
    RUN_TEST(test_the_unconditional_logo_is_a_direct_sum_over_the_support);
    RUN_TEST(test_a_banded_logo_is_a_filtered_sum_over_the_support);
    RUN_TEST(test_the_stack_identities_hold);
    RUN_TEST(test_a_conserved_position_is_dominated_but_not_one_hot);
    RUN_TEST(test_the_reference_alphabet_sets_the_ceiling);
    RUN_TEST(test_two_bands_recombine_into_the_exact_logo);
    RUN_TEST(test_one_band_over_the_grid_is_exact);
    RUN_TEST(test_an_empty_band_is_refused_not_answered);
    RUN_TEST(test_what_a_malformed_request_gets);
    RUN_TEST(test_a_finite_top_edge_still_measures_the_whole_grid);
    RUN_TEST(test_a_grid_too_coarse_is_refused_on_request);
    RUN_TEST(test_the_cost_estimate_precedes_the_sweep);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
