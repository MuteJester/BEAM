/**
 * @file test_positional_banded.c
 * @brief What a banded positional marginal must satisfy, hardest check last.
 *
 * `beam_model_positional_banded` conditions a per-position residue
 * distribution on a band of the sequence's own surprisal. It is the first
 * quantity here whose CONDITIONING EVENT is approximate: the residue axis is
 * exact, but a band is a cut through a real-valued path functional, so the
 * surprisal axis is binned and a sequence near an edge can land either side.
 *
 * That shapes what can be pinned. Four of these checks are self-consistency
 * -- they relate the new answer to exact answers this library already
 * computes, and each would survive a bug that biased the whole band the same
 * way. The fifth does not: it enumerates every sequence of a short length,
 * splits them by their EXACT surprisal, and compares. It is the only check
 * here that tests the banding rather than the plumbing, and it is written to
 * be exact rather than approximate by placing every threshold in a gap of the
 * surprisal distribution wide enough that no sequence's binned mass can cross
 * it.
 *
 * Under the hierarchical prior every string over the alphabet has mass, so
 * there is mass near almost any threshold and a band whose edge is not in
 * such a gap is only as exact as its grid. Every check below that compares a
 * band with an interior edge exactly therefore runs at a length that can be
 * enumerated in full -- the 3,125 nucleotide strings of length 5, or the
 * 8,000 amino-acid strings of length 3 -- with its edges placed in gaps of
 * that enumeration. The checks with no interior edge are exact whatever the
 * support, and run on amino acids at ordinary lengths.
 *
 * The order below is the order they are worth having. The first is the one
 * that catches almost everything; the last is the one that catches what the
 * others cannot.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"

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

/* Nucleotides, so a whole length can be enumerated: 5^5 is 3,125 strings,
 * N included, and every one of them has mass. The set is deliberately
 * varied, which leaves the few strings it makes likely spread over a couple
 * of nats with gaps between them far wider than the grid -- the gaps every
 * exact band comparison below places its edges in. */
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
 * Order 2 or 3 at most. The sweep holds a surprisal grid per live state per
 * layer, and under the prior every state is live: order 3 holds 8,000
 * states a layer where order 2 holds 400, which is the difference between
 * gigabytes and megabytes at a few thousand bins. Lengths 5, 7, 8 and 9 are
 * in the corpus; 6 and 10 are not.
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

static BeamModel *build_nt(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.alphabet = BEAM_ALPHABET_NT;
    c.order = 2u; c.max_length = 6u;
    /* Fixed strengths: the band edges the brute-force test uses are placed
     * in gaps of THIS spectrum, and strengths re-estimated from the counts
     * would move its atoms onto them. */
    c.alpha_auto = false;
    c.alpha_length_auto = false;
    BeamModel *m = beam_model_new(&c);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN_NT, W_NT, N_NT) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

/** Every band edge above the grid, so exactly one band holds everything. */
static const double ALL[2] = { 0.0, 1e12 };

/** The nucleotide length every enumeration below uses. */
#define L_NT 5u


/* ── Enumeration and the gaps in it ──────────────────────────── */

/** Every string of one length: surprisal, probability, residues. */
typedef struct {
    uint32_t n, length, n_residues;
    double  *surprisal, *probability;
    uint8_t *letters;
} Support;

/*
 * All R^L strings over the model's WHOLE alphabet. Under the prior each has
 * mass, so a string that fails to score is itself a failure, reported by
 * returning false.
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

/** The spacing an automatic grid of `bins` gets, which the spectrum reports. */
static double grid_spacing(const BeamModel *m, uint32_t L, uint32_t bins) {
    double ds = 0.0;
    double *g = (double *)calloc(bins, sizeof(double));
    if (!g) return 0.0;
    if (beam_model_spectrum(m, true, L, bins, 0.0, g, &ds) != BEAM_OK) ds = 0.0;
    free(g);
    return ds;
}

/*
 * Linear splitting spreads one sequence's mass over up to `L+1` adjacent
 * bins; twice that is how far every sequence a threshold has to stay for the
 * classification to be exact rather than approximate.
 */
static double splitting_reach(uint32_t L, double ds) {
    return (double)(2u * (L + 2u)) * ds;
}

typedef struct { double s, p; } SurpProb;

static int cmp_surp(const void *a, const void *b) {
    const double x = ((const SurpProb *)a)->s, y = ((const SurpProb *)b)->s;
    return (x > y) - (x < y);
}

/** A threshold nothing can cross, the gap it sits in, and what lies below. */
typedef struct { double cut, lo, hi, mass_below, count_below; } Gap;

/*
 * Every gap of the enumerated surprisals wider than twice the splitting
 * reach, as a threshold at its midpoint with the fraction of the generated
 * mass and of the strings below it. Returns how many were found, or 0.
 */
static uint32_t usable_gaps(const Support *s, double reach, Gap *out,
                            uint32_t cap) {
    SurpProb *v = (SurpProb *)malloc((size_t)s->n * sizeof(SurpProb));
    if (!v) return 0;
    double total = 0.0;
    for (uint32_t k = 0; k < s->n; ++k) {
        v[k].s = s->surprisal[k];
        v[k].p = s->probability[k];
        total += v[k].p;
    }
    qsort(v, s->n, sizeof(SurpProb), cmp_surp);
    uint32_t n = 0;
    double below = 0.0;
    for (uint32_t k = 1; k < s->n && n < cap; ++k) {
        below += v[k - 1u].p;
        if (v[k].s - v[k - 1u].s <= 2.0 * reach || v[k - 1u].s <= reach)
            continue;
        out[n].cut = 0.5 * (v[k - 1u].s + v[k].s);
        out[n].lo = v[k - 1u].s;
        out[n].hi = v[k].s;
        out[n].mass_below = below / total;
        out[n].count_below = (double)k / (double)s->n;
        ++n;
    }
    free(v);
    return n;
}

/*
 * Up to three usable gaps, the ones nearest a quarter, a half and three
 * quarters of the generated mass, each a different gap, returned ascending.
 * None is taken so far into the tail that less than 1% of the mass is left
 * above it: a band there is a difference of two cumulative totals that
 * agree to eight digits, and its relative error measures the cancellation
 * rather than the sweep.
 */
static uint32_t pick_gaps(const Gap *g, uint32_t n, Gap out[3]) {
    const double target[3] = { 0.25, 0.5, 0.75 };
    bool used[512] = { false };
    uint32_t got = 0;
    for (uint32_t t = 0; t < 3u; ++t) {
        int best = -1;
        for (uint32_t k = 0; k < n && k < 512u; ++k) {
            if (used[k] || g[k].mass_below > 0.99) continue;
            if (best < 0 || fabs(g[k].mass_below - target[t])
                            < fabs(g[best].mass_below - target[t]))
                best = (int)k;
        }
        if (best < 0) break;
        used[best] = true;
        out[got++] = g[best];
    }
    for (uint32_t i = 1; i < got; ++i)          /* ascending */
        for (uint32_t j = i; j > 0 && out[j].cut < out[j - 1u].cut; --j) {
            const Gap t = out[j]; out[j] = out[j - 1u]; out[j - 1u] = t;
        }
    return got;
}

/*
 * Five edges -- 0, three thresholds in gaps, and past the grid -- for the
 * checks that need several bands inside the mass. Valid at every grid at
 * least as fine as `coarsest_bins`, since a finer grid only shortens the
 * reach.
 */
static bool inside_edges(const BeamModel *m, uint32_t coarsest_bins,
                         double edges[5]) {
    Support s;
    if (!enumerate_support(m, L_NT, &s)) { support_free(&s); return false; }
    const double ds = grid_spacing(m, L_NT, coarsest_bins);
    Gap g[512], pick[3];
    const uint32_t n = ds > 0.0
                     ? usable_gaps(&s, splitting_reach(L_NT, ds), g, 512u) : 0u;
    support_free(&s);
    if (pick_gaps(g, n, pick) != 3u) return false;
    edges[0] = 0.0;
    for (uint32_t k = 0; k < 3u; ++k) edges[k + 1u] = pick[k].cut;
    edges[4] = 1e12;
    return true;
}
#define N_INSIDE 5u
#define N_INSIDE_BANDS 4u


/* ── 1. The whole grid as one band is the exact marginal ─────── */

/**
 * A single band covering every bin conditions on nothing, so it must return
 * what `beam_model_positional_marginals` returns.
 *
 * This is the check worth writing first. It ties the new code to exact code
 * along its entire length -- both sweeps, the sink seed, the cumulative, the
 * cutoff, the normalisation -- and almost any error in any of them moves a
 * number here. It cannot be bit-exact: `positional.c` carries its forward
 * values as logarithms and this carries linear mass on a grid, so the two
 * differ in the last bits of a double and no more. With no interior edge
 * nothing can be misclassified, so this holds on a support of every string.
 *
 * It pins the GENERATED measure only. `positional_marginals` is the edge
 * posterior, which weights each sequence by its own probability; the counting
 * measure has no exact counterpart here and is pinned by the brute force
 * below instead.
 */
static void test_one_band_over_everything_is_the_exact_marginal(void) {
    uint32_t compared = 0, skipped = 0;
    for (uint32_t order = 2; order <= 3; ++order) {
        BeamModel *m = build_aa(order);
        ASSERT_MSG(m, "model did not build");
        const uint32_t R = beam_model_n_residues(m);

        for (uint32_t L = 2; L <= 10u; ++L) {
            if (!beam_length_live(m, L)) continue;
            double *exact = (double *)calloc((size_t)L * R, sizeof(double));
            double *band  = (double *)calloc((size_t)L * R, sizeof(double));
            double mass = 0.0;
            ASSERT_MSG(exact && band, "allocation");
            if (beam_model_positional_marginals(m, L, NULL, exact, NULL)
                != BEAM_OK) { free(exact); free(band); continue; }
            /* 128 bins, not because it is enough to resolve anything but
             * because one band covering the grid does not depend on the
             * binning at all -- and a fine grid over every state of an
             * order-3 trellis is memory this check has no use for. */
            const BeamError e = beam_model_positional_banded(
                m, L, false, 128u, 0.0, ALL, 2u, band, &mass, NULL, NULL);
            if (e == BEAM_ERR_BUDGET) { ++skipped; free(exact); free(band); continue; }
            if (e != BEAM_OK) {
                snprintf(msg, sizeof msg, "order %u length %u: %s",
                         (unsigned)order, (unsigned)L, beam_error_message());
                free(exact); free(band); beam_model_destroy(m);
                ASSERT_MSG(false, msg);
            }
            for (uint32_t i = 0; i < L; ++i) {
                for (uint32_t x = 0; x < R; ++x) {
                    const double a = band[(size_t)i * R + x];
                    const double b = exact[(size_t)i * R + x];
                    ++compared;
                    if (fabs(a - b) > 1e-12) {
                        snprintf(msg, sizeof msg,
                                 "order %u length %u position %u residue %u: "
                                 "the all-covering band gives %.17g where the "
                                 "exact marginal is %.17g", (unsigned)order,
                                 (unsigned)L, (unsigned)i, (unsigned)x, a, b);
                        free(exact); free(band); beam_model_destroy(m);
                        ASSERT_MSG(false, msg);
                    }
                }
            }
            free(exact); free(band);
        }
        beam_model_destroy(m);
    }
    ASSERT_MSG(compared > 1000u, "too few cells were compared to mean anything");
    if (skipped > 0u) printf("(%u over budget) ", (unsigned)skipped);
    PASS();
}


/* ── 2. The band masses account for the whole spectrum ───────── */

/**
 * The mass of the all-covering band is the total mass on the grid, so it must
 * equal what the spectrum integrates to -- and in the COUNTING measure that
 * total has an independent exact name, `beam_model_path_count`.
 *
 * Tying to `path_count` is the point. It is computed by a different file from
 * a different recursion with no grid in it at all, so agreement says the
 * banded sweep visits exactly the paths the model has, neither dropping a
 * live state nor double-counting one. The spectrum comparison alone could not
 * say that: both would drop the same state. Under the prior every path is
 * one, so the count is also 20^L, and that is checked too: it is the one
 * number here that owes nothing to any sweep.
 */
static void test_the_band_masses_account_for_the_whole_spectrum(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t B = 512u;
    uint32_t checked = 0;

    for (uint32_t L = 2; L <= 10u; ++L) {
        if (!beam_length_live(m, L)) continue;
        for (int counting = 0; counting <= 1; ++counting) {
            double *band = (double *)calloc((size_t)L * R, sizeof(double));
            double *spec = (double *)calloc(B, sizeof(double));
            double mass = 0.0, ds = 0.0;
            ASSERT_MSG(band && spec, "allocation");
            BeamError e = beam_model_positional_banded(
                m, L, (bool)counting, B, 0.0, ALL, 2u, band, &mass, NULL, &ds);
            if (e != BEAM_OK) { free(band); free(spec); continue; }
            e = beam_model_spectrum(m, (bool)counting, L, B, ds, spec, NULL);
            if (e != BEAM_OK) { free(band); free(spec); continue; }

            double total = 0.0;
            for (uint32_t b = 0; b < B; ++b) total += spec[b];
            ++checked;
            if (fabs(mass - total) > 1e-9 * (total > 1.0 ? total : 1.0)) {
                snprintf(msg, sizeof msg,
                         "length %u counting %d: the band holds %.17g and the "
                         "spectrum integrates to %.17g",
                         (unsigned)L, counting, mass, total);
                free(band); free(spec); beam_model_destroy(m);
                ASSERT_MSG(false, msg);
            }
            if (counting) {
                const double paths = beam_model_path_count(m, L);
                const double strings = pow((double)R, (double)L);
                if (fabs(mass - paths) > 1e-6 * paths || paths != strings) {
                    snprintf(msg, sizeof msg,
                             "length %u: the counting band holds %.17g paths, "
                             "beam_model_path_count says %.17g, and there are "
                             "%.17g strings", (unsigned)L, mass, paths, strings);
                    free(band); free(spec); beam_model_destroy(m);
                    ASSERT_MSG(false, msg);
                }
            }
            free(band); free(spec);
        }
    }
    beam_model_destroy(m);
    ASSERT_MSG(checked >= 4u, "too few lengths were checked");
    PASS();
}


/* ── 3. Every position agrees on the band's mass ─────────────── */

/**
 * Each sequence crosses exactly one edge at every layer, so summing a band's
 * unnormalised mass over residues gives that band's total AT EVERY POSITION
 * -- the same number, len times over, computed from a different split of the
 * trellis each time.
 *
 * This replaces the obvious check, "each band's rows sum to 1", which is
 * vacuous: the rows are normalised by their own totals, so they sum to 1 by
 * construction and a bug that scaled the whole accumulation would survive.
 * The cross-position invariant is what that check was reaching for, and it is
 * the sharpest statement available of whether the forward and backward halves
 * were split and recombined consistently -- position 0 puts the entire
 * trellis in the backward grid and position len-1 puts almost all of it in
 * the forward grid, so a bug in either half moves the two apart.
 *
 * The positions agree only to grid precision where a sequence sits near an
 * edge, which the edges here are placed to avoid; the grids start at 4,096
 * bins because that is the coarsest at which this fixture has gaps wide
 * enough to hold three of them.
 */
static void test_every_position_agrees_on_the_band_mass(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = L_NT;
    double inside[N_INSIDE];
    if (!inside_edges(m, 4096u, inside)) {
        beam_model_destroy(m);
        ASSERT_MSG(false, "three usable gaps inside the mass were not found; "
                   "this fixture no longer covers the check");
    }

    uint32_t grids = 0;
    for (uint32_t bins = 4096u; bins <= 32768u; bins *= 2u) {
        double *band = (double *)calloc((size_t)N_INSIDE_BANDS * L * R, sizeof(double));
        double *pm   = (double *)calloc((size_t)N_INSIDE_BANDS * L, sizeof(double));
        double *bm   = (double *)calloc(N_INSIDE_BANDS, sizeof(double));
        ASSERT_MSG(band && pm && bm, "allocation");

        const BeamError e = beam_model_positional_banded(
            m, L, false, bins, 0.0, inside, N_INSIDE, band, bm, pm, NULL);
        if (e == BEAM_ERR_BUDGET) { free(band); free(pm); free(bm); continue; }
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
        ++grids;

        uint32_t nonempty = 0;
        for (uint32_t b = 0; b < N_INSIDE_BANDS; ++b) {
            if (!(bm[b] > 0.0)) continue;
            ++nonempty;
            for (uint32_t i = 0; i < L; ++i) {
                const double rel = fabs(pm[(size_t)b * L + i] - bm[b]) / bm[b];
                /*
                 * Floating point, not grid error: with no sequence within
                 * the splitting reach of an edge, every position classifies
                 * every sequence the same way. The bound is still loose by
                 * the standards of this file, and eight orders tighter than
                 * the disagreement an edge among the sequences produces.
                 */
                if (rel > 1e-9) {
                    snprintf(msg, sizeof msg,
                             "%u bins, band %u: position %u puts %.17g in the "
                             "band and position 0 puts %.17g (relative %.3g)",
                             (unsigned)bins, (unsigned)b, (unsigned)i,
                             pm[(size_t)b * L + i], bm[b], rel);
                    goto fail;
                }
                double sum = 0.0;
                for (uint32_t x = 0; x < R; ++x)
                    sum += band[((size_t)b * L + i) * R + x];
                if (fabs(sum - 1.0) > 1e-12) {
                    snprintf(msg, sizeof msg,
                             "%u bins, band %u position %u sums to %.17g, not 1",
                             (unsigned)bins, (unsigned)b, (unsigned)i, sum);
                    goto fail;
                }
            }
        }
        if (nonempty < N_INSIDE_BANDS) {
            snprintf(msg, sizeof msg,
                     "%u bins: only %u band(s) held mass, so the split was not "
                     "exercised", (unsigned)bins, (unsigned)nonempty);
            goto fail;
        }
        free(band); free(pm); free(bm);
        continue;
    fail:
        free(band); free(pm); free(bm);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }
    beam_model_destroy(m);
    ASSERT_MSG(grids >= 4u, "too few grids were checked");
    PASS();
}


/* ── 3b. Refining the grid does not move a band ──────────────── */

/**
 * The same bands, measured on grids from 4,096 to 32,768 bins, hold the same
 * mass.
 *
 * This is what pins the cutoff arithmetic. Every other check here is taken on
 * ONE grid, and an error in turning a threshold into a bin index -- a
 * ceiling that should be a floor, an inclusive total read as exclusive, a
 * missing edge-surprisal term -- is a shift of a fixed number of BINS, so it
 * changes the answer by an amount proportional to the spacing while remaining
 * perfectly self-consistent on any single grid. Halving the spacing halves
 * such an error, and it shows up here as a band that moves -- once the shift
 * reaches past the gap the edge sits in, which a one-bin shift does not; the
 * mean-surprisal check below is the one that sees that.
 */
static void test_refining_the_grid_does_not_move_a_band(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = L_NT;
    double inside[N_INSIDE];
    if (!inside_edges(m, 4096u, inside)) {
        beam_model_destroy(m);
        ASSERT_MSG(false, "three usable gaps inside the mass were not found");
    }
    double ref[N_INSIDE_BANDS];
    bool have = false;
    uint32_t grids = 0;

    for (uint32_t bins = 4096u; bins <= 32768u; bins *= 2u) {
        double *band = (double *)calloc((size_t)N_INSIDE_BANDS * L * R, sizeof(double));
        double *bm   = (double *)calloc(N_INSIDE_BANDS, sizeof(double));
        ASSERT_MSG(band && bm, "allocation");
        const BeamError e = beam_model_positional_banded(
            m, L, false, bins, 0.0, inside, N_INSIDE, band, bm, NULL, NULL);
        if (e == BEAM_ERR_BUDGET) { free(band); free(bm); continue; }
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
        if (!have) {
            for (uint32_t b = 0; b < N_INSIDE_BANDS; ++b) ref[b] = bm[b];
            have = true;
        } else {
            for (uint32_t b = 0; b < N_INSIDE_BANDS; ++b) {
                if (!(ref[b] > 0.0)) continue;
                const double rel = fabs(bm[b] - ref[b]) / ref[b];
                if (rel > 1e-9) {
                    snprintf(msg, sizeof msg,
                             "band %u holds %.17g at 4096 bins and %.17g at %u "
                             "bins (relative %.3g), so it moves with the grid",
                             (unsigned)b, ref[b], bm[b], (unsigned)bins, rel);
                    free(band); free(bm); beam_model_destroy(m);
                    ASSERT_MSG(false, msg);
                }
            }
        }
        ++grids;
        free(band); free(bm);
    }
    beam_model_destroy(m);
    ASSERT_MSG(grids >= 4u, "too few grids were compared");
    PASS();
}


/* ── 4. The bands recombine into the unconditional marginal ──── */

/**
 * Averaging the band marginals weighted by their masses undoes the
 * conditioning, so the result is the unconditional marginal again.
 *
 * Distinct from check 1, which asks one band to be the whole answer. This
 * asks several bands, each individually different from the unconditional
 * marginal, to reconstruct it when recombined -- which is false if the masses
 * are inconsistent with the marginals they weight, the one relationship the
 * other checks leave open. The rows are normalised by their own position's
 * mass and weighted here by the band's, so the recombination is exact only
 * where the positions agree, which is why the edges sit in gaps.
 */
static void test_the_bands_recombine_into_the_unconditional_marginal(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = L_NT;
    double edges[N_INSIDE];
    ASSERT_MSG(inside_edges(m, 4096u, edges),
               "three usable gaps inside the mass were not found");
    const uint32_t NE = N_INSIDE, NB = N_INSIDE_BANDS;

    double *band  = (double *)calloc((size_t)NB * L * R, sizeof(double));
    double *bm    = (double *)calloc(NB, sizeof(double));
    double *exact = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(band && bm && exact, "allocation");

    BeamError e = beam_model_positional_banded(m, L, false, 4096u, 0.0,
                                               edges, NE, band, bm, NULL, NULL);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());
    e = beam_model_positional_marginals(m, L, NULL, exact, NULL);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());

    double total = 0.0;
    for (uint32_t b = 0; b < NB; ++b) total += bm[b];
    ASSERT_MSG(total > 0.0, "every band was empty");

    /*
     * A band must genuinely differ from the pooled answer, or recombining
     * them proves nothing. Measured over EVERY position rather than one: a
     * position the corpus always spelled the same way can look alike in
     * every band and report a spread of nearly zero on its own.
     */
    double spread = 0.0;
    for (uint32_t b = 0; b < NB; ++b)
        if (bm[b] > 0.0)
            for (uint32_t i = 0; i < L; ++i)
                for (uint32_t x = 0; x < R; ++x) {
                    const double d = fabs(band[((size_t)b * L + i) * R + x]
                                          - exact[(size_t)i * R + x]);
                    if (d > spread) spread = d;
                }

    double worst = 0.0;
    for (uint32_t i = 0; i < L; ++i)
        for (uint32_t x = 0; x < R; ++x) {
            double mix = 0.0;
            for (uint32_t b = 0; b < NB; ++b)
                mix += bm[b] * band[((size_t)b * L + i) * R + x];
            const double d = fabs(mix / total - exact[(size_t)i * R + x]);
            if (d > worst) worst = d;
        }

    free(band); free(bm); free(exact); beam_model_destroy(m);
    if (spread < 1e-3) {
        snprintf(msg, sizeof msg,
                 "the bands differ from the pooled marginal by at most %.3g, "
                 "so recombining them would pass even if banding did nothing",
                 spread);
        ASSERT_MSG(false, msg);
    }
    if (worst > 1e-9) {
        snprintf(msg, sizeof msg,
                 "the mass-weighted average of the bands departs from the "
                 "unconditional marginal by %.3g", worst);
        ASSERT_MSG(false, msg);
    }
    PASS();
}


/* ── 5. Brute force against every sequence of a short length ─── */

/**
 * Enumerate every string of one length, split them by their EXACT surprisal,
 * and require the banded sweep to have found the same thing.
 *
 * The other four checks relate this code to itself or to its own pooled
 * answer. This one steps outside: the reference is `beam_model_log_pseq`
 * over all 5^L strings, summed by hand, with no trellis, no grid and no
 * dynamic program in it. It is the only check that can catch the banding
 * being systematically wrong -- a threshold applied to the prefix instead of
 * the total, a sink term counted twice, a cutoff off by one bin -- because
 * every one of those is self-consistent.
 *
 * It is written to be EXACT rather than approximate, which takes some care.
 * Linear splitting spreads one sequence's mass over up to `L+1` adjacent
 * bins, so a threshold within that of a real sequence classifies part of it
 * the wrong side and no tight tolerance is available. So each threshold is
 * placed in a GAP of the enumerated surprisal distribution, wider than the
 * splitting can reach; then every sequence is wholly one side and the two
 * answers must agree to floating point. The test asserts it found such gaps
 * rather than quietly settling for worse ones.
 *
 * Two thresholds: the usable gap nearest half the generated mass, which lies
 * among the few strings the corpus made likely, and the one nearest half the
 * strings, which lies deep in the tail of those it never saw. Each measure
 * therefore has a split where its own weight is. Both measures are checked;
 * the counting measure has no exact counterpart elsewhere in this file, so
 * this is the only thing pinning it.
 */
static void test_the_banding_matches_brute_force_on_a_tiny_model(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = L_NT;
    /* Fine enough to have a usable gap near half the strings; at 4,096 bins
     * the nearest one is past nine tenths of them. */
    const uint32_t B = 16384u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not reach length 5");

    Support s;
    ASSERT_MSG(enumerate_support(m, L, &s),
               "a string did not score; every readable string has mass");
    ASSERT_MSG(s.n == 3125u, "the enumeration is not the whole length");

    /* The grid this will be binned on, so the gaps can be measured in bins. */
    const double ds = grid_spacing(m, L, B);
    ASSERT_MSG(ds > 0.0, beam_error_message());
    Gap g[512];
    const uint32_t ng = usable_gaps(&s, splitting_reach(L, ds), g, 512u);
    int by_mass = -1, by_count = -1;
    for (uint32_t k = 0; k < ng; ++k) {
        if (by_mass < 0 || fabs(g[k].mass_below - 0.5)
                           < fabs(g[by_mass].mass_below - 0.5)) by_mass = (int)k;
        if (by_count < 0 || fabs(g[k].count_below - 0.5)
                            < fabs(g[by_count].count_below - 0.5)) by_count = (int)k;
    }
    if (!(by_mass >= 0 && by_count >= 0 && g[by_count].cut > g[by_mass].cut)) {
        snprintf(msg, sizeof msg,
                 "%u usable gaps among %u surprisals, and no pair that splits "
                 "the mass and the count separately, so no threshold here can "
                 "be checked exactly", (unsigned)ng, (unsigned)s.n);
        support_free(&s);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }

    const double edges[4] = { 0.0, g[by_mass].cut, g[by_count].cut, 1e12 };
    const uint32_t NE = 4u, NB = 3u;

    for (int counting = 0; counting <= 1; ++counting) {
        /* The reference, summed by hand over the enumeration. */
        double *ref  = (double *)calloc((size_t)NB * L * R, sizeof(double));
        double *rmass = (double *)calloc(NB, sizeof(double));
        ASSERT_MSG(ref && rmass, "allocation");
        for (uint32_t k = 0; k < s.n; ++k) {
            const uint32_t b = s.surprisal[k] < edges[1] ? 0u
                             : s.surprisal[k] < edges[2] ? 1u : 2u;
            const double wgt = counting ? 1.0 : s.probability[k];
            rmass[b] += wgt;
            for (uint32_t i = 0; i < L; ++i)
                ref[((size_t)b * L + i) * R + s.letters[(size_t)k * L + i]] += wgt;
        }

        double *band = (double *)calloc((size_t)NB * L * R, sizeof(double));
        double *bm   = (double *)calloc(NB, sizeof(double));
        ASSERT_MSG(band && bm, "allocation");
        const BeamError e = beam_model_positional_banded(
            m, L, (bool)counting, B, ds, edges, NE, band, bm, NULL, NULL);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());

        for (uint32_t b = 0; b < NB; ++b) {
            if (!(rmass[b] > 0.0)) {
                snprintf(msg, sizeof msg,
                         "counting %d band %u is empty, so the split put every "
                         "sequence on one side and proves nothing",
                         counting, (unsigned)b);
                goto fail;
            }
            if (fabs(bm[b] - rmass[b]) > 1e-9 * rmass[b]) {
                snprintf(msg, sizeof msg,
                         "counting %d band %u: the sweep puts %.17g in the "
                         "band and enumerating every sequence puts %.17g",
                         counting, (unsigned)b, bm[b], rmass[b]);
                goto fail;
            }
            for (uint32_t i = 0; i < L; ++i)
                for (uint32_t x = 0; x < R; ++x) {
                    const double want = ref[((size_t)b * L + i) * R + x] / rmass[b];
                    const double got  = band[((size_t)b * L + i) * R + x];
                    if (fabs(got - want) > 1e-9) {
                        snprintf(msg, sizeof msg,
                                 "counting %d band %u position %u residue %u: "
                                 "the sweep says %.17g, enumeration says %.17g",
                                 counting, (unsigned)b, (unsigned)i,
                                 (unsigned)x, got, want);
                        goto fail;
                    }
                }
        }
        free(ref); free(rmass); free(band); free(bm);
        continue;
    fail:
        free(ref); free(rmass); free(band); free(bm);
        support_free(&s);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }

    support_free(&s);
    beam_model_destroy(m);
    PASS();
}


/* ── 6. The threshold lands where it should, to a fraction of a bin ── */

/**
 * Recover the mean surprisal from a fine sweep of thresholds and require it
 * to be the mean the enumeration gives.
 *
 * This exists because of a mutation that survived everything above. Replacing
 * the cumulative read `cum[j-1]` by `cum[j]` shifts every band edge by
 * exactly one bin, and NOT ONE of the five checks noticed: band masses are
 * still conserved, positions still agree with each other, the bands still
 * recombine, and the brute-force check places its thresholds in gaps wide
 * enough that a one-bin shift moves no sequence across them -- which is the
 * very property that makes that check exact. A shifted cutoff is
 * self-consistent, so only something that knows where the edge SHOULD be can
 * see it.
 *
 * The construction. A triple (prefix bin `p`, edge surprisal `e`, suffix bin
 * `q`) enters a band exactly when the threshold passes `p*ds + e + q*ds`,
 * which is the surprisal that triple represents; so the centroid of the band
 * masses is the mean represented surprisal, and linear splitting is unbiased
 * in the mean. The centroid must therefore be the true mean, and a cutoff
 * wrong by one bin moves it by exactly one spacing.
 *
 * Sampling the thresholds every `ds/4` puts the centroid's own discretisation
 * error at a few hundredths of a spacing, where the one-bin mutation lands at
 * about one. The bound below sits between them with room either side.
 */
static void test_the_recovered_mean_surprisal_is_unbiased(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = L_NT, B = 256u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not reach length 5");

    uint64_t strings = 1;
    for (uint32_t i = 0; i < L; ++i) strings *= R;

    /* The reference: every sequence, weighted both ways. */
    char seq[16];
    double mass[2] = { 0.0, 0.0 }, moment[2] = { 0.0, 0.0 }, smax = 0.0;
    uint32_t n = 0;
    for (uint64_t code = 0; code < strings; ++code) {
        uint64_t c = code;
        for (uint32_t i = 0; i < L; ++i) {
            seq[i] = m->alpha->idx_to_char[c % R];
            c /= R;
        }
        seq[L] = '\0';
        const double lp = beam_model_log_pseq(m, seq);
        if (!isfinite(lp)) continue;
        const double su = -lp, pr = exp(lp);
        ++n;
        if (su > smax) smax = su;
        mass[0] += pr;  moment[0] += pr * su;    /* generated */
        mass[1] += 1.0; moment[1] += su;         /* counting  */
    }
    ASSERT_MSG(n == strings, "every string of the length must have mass");

    double ds = 0.0;
    {
        double *g = (double *)calloc(B, sizeof(double));
        ASSERT_MSG(g, "allocation");
        const BeamError e = beam_model_spectrum(m, true, L, B, 0.0, g, &ds);
        free(g);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
    }

    const double h = ds / 4.0;
    const uint32_t K = (uint32_t)((smax + 8.0 * ds) / h) + 2u;
    double *edges = (double *)malloc((size_t)K * sizeof(double));
    ASSERT_MSG(edges, "allocation");
    for (uint32_t k = 0; k < K; ++k) edges[k] = (double)k * h;

    for (int counting = 0; counting <= 1; ++counting) {
        double *band = (double *)calloc((size_t)(K - 1u) * L * R, sizeof(double));
        double *bm   = (double *)calloc(K - 1u, sizeof(double));
        ASSERT_MSG(band && bm, "allocation");
        const BeamError e = beam_model_positional_banded(
            m, L, (bool)counting, B, ds, edges, K, band, bm, NULL, NULL);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());

        double tot = 0.0, mom = 0.0;
        for (uint32_t j = 0; j + 1u < K; ++j) {
            tot += bm[j];
            mom += bm[j] * 0.5 * (edges[j] + edges[j + 1u]);
        }
        const uint32_t ix = counting ? 1u : 0u;
        const double want_mass = mass[ix];
        const double want_mean = moment[ix] / mass[ix];

        if (fabs(tot - want_mass) > 1e-9 * want_mass) {
            snprintf(msg, sizeof msg,
                     "counting %d: the bands hold %.17g and enumeration gives "
                     "%.17g", counting, tot, want_mass);
            goto fail;
        }
        const double bias = (mom / tot - want_mean) / ds;
        if (fabs(bias) > 0.25) {
            snprintf(msg, sizeof msg,
                     "counting %d: the mean surprisal recovered from the bands "
                     "is %.9f where enumeration gives %.9f -- a bias of %.4f "
                     "of a bin, which is a threshold landing in the wrong bin "
                     "rather than grid error",
                     counting, mom / tot, want_mean, bias);
            goto fail;
        }
        free(band); free(bm);
        continue;
    fail:
        free(band); free(bm); free(edges);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }
    free(edges);
    beam_model_destroy(m);
    PASS();
}


/* ── 7. A backward grid whose origin is below zero ───────────── */

/*
 * The lowest backward seed of a length, in nats, computed exactly as the
 * sweep computes it: `-tail(s) + ln Z(L) - ln pi(L)` over the live sink
 * states. Negative exactly when the backward grid has to carry an origin
 * below zero, which is the precondition the next two checks need and which
 * `sink < 0` alone does not imply -- the tail term can outweigh it.
 */
static bool lowest_seed(const BeamModel *m, uint32_t L, double *out) {
    double *tail = (double *)malloc((size_t)m->n_states * sizeof(double));
    if (!tail) return false;
    beam_tail_weights(m, L, tail);
    LiveHandle h;
    if (beam_live_acquire(m, L, &h) != BEAM_OK) { free(tail); return false; }
    const double sink = beam_model_log_partition(m, L)
                      - beam_model_log_length_prior(m, L);
    double lowest = INFINITY;
    for (uint32_t k = 0; k < h.L->n_live[L]; ++k) {
        const uint32_t st = h.L->live[L][k];
        if (tail[st] == -INFINITY) continue;
        const double u = -tail[st] + sink;
        if (u < lowest) lowest = u;
    }
    beam_live_release(&h);
    free(tail);
    *out = lowest;
    return isfinite(lowest);
}

/**
 * A length whose sink term is NEGATIVE still returns its mass.
 *
 * The backward sweep is seeded with `-tail(s) + ln Z(L) - ln pi(L)`. When
 * that is negative the seed sits below the grid origin, and a grid that
 * begins at zero loses every suffix off the bottom -- the sweep then returns
 * zero mass, confidently, for a model whose spectrum is perfectly healthy.
 *
 * That is not hypothetical: it is what this code did. `spectrum.c` cannot
 * meet it, because it applies the sink LAST onto an accumulator already clear
 * of zero, so no existing test covered the case. It was found from the
 * Python layer, on a model built for something else.
 *
 * Under the hierarchical prior it no longer takes a contrived corpus. At a
 * length the corpus never produced, the N-anchored and C-anchored chains
 * read depth rows that saw different evidence and disagree strongly, so
 * `ln Z(L)` sits nats further below zero than `ln pi(L)` does and the seed
 * lands well below the origin. Length 6 is such a length for the amino-acid
 * corpus: its lowest seed is about -4.7 nats.
 *
 * The test asserts its own precondition: a fixture whose seeds drifted above
 * the origin would still pass every line below while covering nothing.
 */
static void test_a_sink_below_the_grid_origin_still_reports_its_mass(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");

    const uint32_t L = 6u, R = beam_model_n_residues(m), B = 2048u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not reach length 6");

    double seed = 0.0;
    ASSERT_MSG(lowest_seed(m, L, &seed), "could not compute the sink seeds");
    if (!(seed < 0.0)) {
        snprintf(msg, sizeof msg,
                 "this fixture's lowest backward seed is %.6g nats, not "
                 "negative, so it no longer covers a backward grid that "
                 "starts below zero", seed);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }

    double *band = (double *)calloc((size_t)L * R, sizeof(double));
    double *spec = (double *)calloc(B, sizeof(double));
    double *exact = (double *)calloc((size_t)L * R, sizeof(double));
    double mass = 0.0, ds = 0.0;
    ASSERT_MSG(band && spec && exact, "allocation");

    BeamError e = beam_model_positional_banded(m, L, false, B, 0.0, ALL, 2u,
                                               band, &mass, NULL, &ds);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());
    e = beam_model_spectrum(m, false, L, B, ds, spec, NULL);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());
    e = beam_model_positional_marginals(m, L, NULL, exact, NULL);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());

    double total = 0.0;
    for (uint32_t b = 0; b < B; ++b) total += spec[b];
    const double whole = exp(beam_model_log_length_prior(m, L));

    double worst = 0.0;
    for (uint32_t i = 0; i < L; ++i)
        for (uint32_t x = 0; x < R; ++x) {
            const double d = fabs(band[(size_t)i * R + x]
                                  - exact[(size_t)i * R + x]);
            if (d > worst) worst = d;
        }

    free(band); free(spec); free(exact);
    beam_model_destroy(m);

    if (!(total > 0.0)) { ASSERT_MSG(false, "the spectrum itself is empty"); }
    if (fabs(mass - total) > 1e-9 * total) {
        snprintf(msg, sizeof msg,
                 "seed %.6g: the band holds %.17g where the spectrum "
                 "integrates to %.17g -- the backward grid lost its mass off "
                 "the bottom", seed, mass, total);
        ASSERT_MSG(false, msg);
    }
    /* And the spectrum is not merely agreeing with a shared loss: the
     * generated mass of a length is its length prior, which no grid set. */
    if (fabs(mass - whole) > 1e-9 * whole) {
        snprintf(msg, sizeof msg,
                 "seed %.6g: the band holds %.17g of the length's %.17g",
                 seed, mass, whole);
        ASSERT_MSG(false, msg);
    }
    if (worst > 1e-12) {
        snprintf(msg, sizeof msg,
                 "seed %.6g: the all-covering band departs from the exact "
                 "marginal by %.3g", seed, worst);
        ASSERT_MSG(false, msg);
    }
    PASS();
}


/* ── 8. A bin-aligned band is the spectrum summed over those bins ── */

/*
 * One length's worth of test 8: thresholds on bin boundaries inside gaps of
 * the enumeration, and the bands below each against both the spectrum and
 * the enumeration. Returns the number of thresholds checked, or 0 with `msg`
 * set.
 */
static uint32_t check_bin_aligned(const BeamModel *m, uint32_t L,
                                  uint32_t B, const char *name) {
    const uint32_t R = beam_model_n_residues(m);
    Support s;
    if (!enumerate_support(m, L, &s)) {
        support_free(&s);
        snprintf(msg, sizeof msg, "%s: a string did not score", name);
        return 0;
    }
    double *spec = (double *)calloc(B, sizeof(double));
    double *band = (double *)calloc(4ull * L * R, sizeof(double));
    double ds = 0.0, bm[4] = { 0.0, 0.0, 0.0, 0.0 };
    uint32_t checked = 0;
    if (!spec || !band) {
        snprintf(msg, sizeof msg, "allocation");
        goto done;
    }
    if (beam_model_spectrum(m, false, L, B, 0.0, spec, &ds) != BEAM_OK) {
        snprintf(msg, sizeof msg, "%s: %s", name, beam_error_message());
        goto done;
    }
    double total = 0.0;
    for (uint32_t b = 0; b < B; ++b) total += spec[b];

    const double reach = splitting_reach(L, ds);
    Gap g[512], pick[3];
    const uint32_t ng = usable_gaps(&s, reach, g, 512u);
    const uint32_t np = pick_gaps(g, ng, pick);

    /* Each threshold moved onto the nearest bin boundary, which has to
     * leave it clear of every sequence by the reach on both sides. */
    uint32_t qb[3];
    uint32_t nq = 0;
    for (uint32_t k = 0; k < np; ++k) {
        const uint32_t b = (uint32_t)lround(pick[k].cut / ds);
        const double t = (double)b * ds;
        if (t - pick[k].lo > reach && pick[k].hi - t > reach) qb[nq++] = b;
    }
    if (nq < 2u) {
        snprintf(msg, sizeof msg, "%s: only %u bin-aligned threshold(s) fit in "
                 "a gap, so this length no longer covers the check",
                 name, (unsigned)nq);
        goto done;
    }

    double edges[5];
    edges[0] = 0.0;
    for (uint32_t k = 0; k < nq; ++k) edges[k + 1u] = (double)qb[k] * ds;
    edges[nq + 1u] = 1e12;
    if (beam_model_positional_banded(m, L, false, B, ds, edges, nq + 2u,
                                     band, bm, NULL, NULL) != BEAM_OK) {
        snprintf(msg, sizeof msg, "%s: %s", name, beam_error_message());
        goto done;
    }

    double running = 0.0;
    for (uint32_t q = 0; q < nq; ++q) {
        running += bm[q];
        double ref = 0.0, enumerated = 0.0;
        for (uint32_t b = 0; b < qb[q]; ++b) ref += spec[b];
        for (uint32_t k = 0; k < s.n; ++k)
            if (s.surprisal[k] < edges[q + 1u]) enumerated += s.probability[k];
        if (!(ref > 0.0 && ref < total)) {
            snprintf(msg, sizeof msg, "%s threshold %u: it does not divide "
                     "the mass (%.17g of %.17g)", name, q, ref, total);
            goto done;
        }
        if (fabs(running - ref) > 1e-12 * total) {
            snprintf(msg, sizeof msg,
                     "%s threshold %u at bin %u: the bands hold %.17g below it "
                     "and the spectrum holds %.17g (%.3g of the total apart)",
                     name, q, qb[q], running, ref,
                     fabs(running - ref) / total);
            goto done;
        }
        if (fabs(running - enumerated) > 1e-9 * enumerated) {
            snprintf(msg, sizeof msg,
                     "%s threshold %u: the bands hold %.17g below it and the "
                     "enumeration %.17g", name, q, running, enumerated);
            goto done;
        }
        ++checked;
    }
done:
    free(spec); free(band);
    support_free(&s);
    return checked;
}

/**
 * With a threshold ON a bin boundary, a band's mass equals
 * `beam_model_spectrum` summed over the same bins -- exactly, not to grid
 * precision -- wherever no sequence lies within the splitting reach of it.
 *
 * The qualification is not caution. The two get there by different routes:
 * the spectrum splits every edge's surprisal between two bins as it
 * transports a total through `L+1` steps, while this splits a prefix and a
 * suffix separately and takes the edge between them EXACTLY at the cut. So a
 * sequence within a bin or two of the threshold is divided differently by
 * the two, and on a support of every string that is always some mass: at
 * length 9 of the order-2 amino-acid fixture on 2,048 bins the cumulatives
 * differ by 1e-4 of the total at a quartile boundary, where the closed-world
 * model this test was first written on agreed to 1e-17 only because its
 * few sequences happened to sit clear of the boundaries. The thresholds are
 * therefore placed on bin boundaries inside gaps of an enumeration, where
 * both must count every sequence below and nothing else -- which also makes
 * the enumeration a third, independent reference.
 *
 * It is also the only check that pins the backward grid's ORIGIN into the
 * cutoff. `pad` shifts where every suffix sits, and the accumulation has to
 * shift its threshold by the same amount; drop that and the sweep answers a
 * different question perfectly self-consistently -- same total, same
 * agreement between positions, same recombination. Only a reference that
 * knows which threshold was asked for can tell, and only at a length where
 * `pad` is not zero. So this runs at two: length 5 of the nucleotide
 * fixture, whose seeds are all above the origin, and length 3 of the
 * amino-acid one, a length its corpus never produced, whose seeds are about
 * two nats below it -- a shift far wider than the gaps the thresholds sit
 * in, so dropping it would move sequences across every one of them.
 */
static void test_a_bin_aligned_band_is_the_spectrum_over_those_bins(void) {
    const uint32_t B = 4096u;
    uint32_t checked = 0;

    BeamModel *nt = build_nt();
    ASSERT_MSG(nt, "model did not build");
    uint32_t got = check_bin_aligned(nt, L_NT, B, "nt length 5");
    beam_model_destroy(nt);
    ASSERT_MSG(got > 0u, msg);
    checked += got;

    BeamModel *aa = build_aa(2u);
    ASSERT_MSG(aa, "model did not build");
    double seed = 0.0;
    if (!(lowest_seed(aa, 3u, &seed) && seed < 0.0)) {
        snprintf(msg, sizeof msg, "length 3's lowest seed is %.6g nats, not "
                 "below the origin, so nothing here pins the grid's origin "
                 "into the cutoff", seed);
        beam_model_destroy(aa);
        ASSERT_MSG(false, msg);
    }
    got = check_bin_aligned(aa, 3u, B, "aa length 3");
    beam_model_destroy(aa);
    ASSERT_MSG(got > 0u, msg);
    checked += got;

    ASSERT_MSG(checked >= 4u, "too few thresholds were checked");
    PASS();
}


/* ── What it refuses ─────────────────────────────────────────── */

/**
 * A malformed band is refused with a message naming the edge that is wrong,
 * and refused BEFORE anything is allocated.
 *
 * Each of these is a way a caller gets band edges wrong in practice --
 * descending them because they were thinking in probabilities, passing one
 * edge and expecting one band, leaving an infinity in from a division.
 */
static void test_what_a_malformed_band_gets(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");
    /* A length this model GENERATES. The length check runs before the edge
     * checks, so a length it refused would make every case below refuse for
     * the wrong reason -- a version of this test once passed in full with no
     * edge validation at all that way. The positive control at the end is
     * what stops that happening again. */
    const uint32_t R = beam_model_n_residues(m), L = 7u;
    double out[64 * 20], mass[8];
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not generate length 7");
    ASSERT_MSG(L * R <= 64u * 20u, "scratch too small");

    const double descending[3] = { 20.0, 14.0, 8.0 };
    const double repeated[3]   = { 8.0, 14.0, 14.0 };
    const double one_edge[1]   = { 8.0 };
    const double infinite[2]   = { 0.0, INFINITY };
    const double fine[2]       = { 0.0, 1e12 };

    struct { const double *e; uint32_t n; uint32_t bins; const char *what; }
    cases[] = {
        { descending, 3u, 512u, "edges in descending order" },
        { repeated,   3u, 512u, "a repeated edge" },
        { one_edge,   1u, 512u, "a single edge" },
        { infinite,   2u, 512u, "an infinite edge" },
        { fine,       2u, 1u,   "a one-bin grid" },
    };
    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; ++k) {
        const BeamError e = beam_model_positional_banded(
            m, L, false, cases[k].bins, 0.0, cases[k].e, cases[k].n,
            out, mass, NULL, NULL);
        if (e != BEAM_ERR_PARAM_OUT_OF_RANGE) {
            snprintf(msg, sizeof msg,
                     "%s was answered with %d rather than refused",
                     cases[k].what, (int)e);
            beam_model_destroy(m);
            ASSERT_MSG(false, msg);
        }
    }
    /* A length past max_length is refused too. */
    BeamError e = beam_model_positional_banded(
        m, 200u, false, 512u, 0.0, fine, 2u, out, mass, NULL, NULL);
    ASSERT_MSG(e != BEAM_OK, "an impossible length was answered");

    /* The control: the same call with everything valid must SUCCEED, or the
     * refusals above prove only that something was wrong, not what. */
    e = beam_model_positional_banded(m, L, false, 512u, 0.0, fine, 2u,
                                     out, mass, NULL, NULL);
    if (e != BEAM_OK) {
        snprintf(msg, sizeof msg,
                 "the well-formed control call was refused with %d (%s), so "
                 "the refusals above cannot be attributed to the band",
                 (int)e, beam_error_message());
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }

    beam_model_destroy(m);
    PASS();
}

/**
 * A grid this sweep cannot fit is refused rather than allocated.
 *
 * The budget doctrine is refuse-before-allocating, and this is the only
 * statement of it for this file. The figure it refuses on comes from the same
 * expression the allocations use, so a grid that fits the check fits memory.
 */
static void test_a_grid_over_budget_is_refused(void) {
    BeamModel *m = build_aa(2u);
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m), L = 7u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not generate length 7");
    double *out = (double *)calloc((size_t)L * R, sizeof(double));
    double mass = 0.0;
    ASSERT_MSG(out, "allocation");

    /*
     * An EXPLICIT spacing. Sizing the grid automatically reads
     * `beam_model_accumulator_max`, which is itself a sweep and refuses first
     * under a budget this small -- so the automatic form tests that
     * refusal rather than this one.
     */
    const uint64_t was = beam_model_memory_budget();
    beam_model_set_memory_budget(4096u);
    const BeamError e = beam_model_positional_banded(
        m, L, false, 4096u, 0.5, ALL, 2u, out, &mass, NULL, NULL);
    beam_model_set_memory_budget(was);

    free(out);
    beam_model_destroy(m);
    if (e != BEAM_ERR_BUDGET) {
        snprintf(msg, sizeof msg,
                 "a 4 kB budget answered a 4096-bin sweep with %d", (int)e);
        ASSERT_MSG(false, msg);
    }
    PASS();
}


int main(void) {
    printf("banded positional marginals\n");
    RUN_TEST(test_one_band_over_everything_is_the_exact_marginal);
    RUN_TEST(test_the_band_masses_account_for_the_whole_spectrum);
    RUN_TEST(test_every_position_agrees_on_the_band_mass);
    RUN_TEST(test_refining_the_grid_does_not_move_a_band);
    RUN_TEST(test_the_bands_recombine_into_the_unconditional_marginal);
    RUN_TEST(test_the_banding_matches_brute_force_on_a_tiny_model);
    RUN_TEST(test_the_recovered_mean_surprisal_is_unbiased);
    RUN_TEST(test_a_sink_below_the_grid_origin_still_reports_its_mass);
    RUN_TEST(test_a_bin_aligned_band_is_the_spectrum_over_those_bins);
    RUN_TEST(test_what_a_malformed_band_gets);
    RUN_TEST(test_a_grid_over_budget_is_refused);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
