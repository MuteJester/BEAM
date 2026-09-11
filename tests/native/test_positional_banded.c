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

/* Nucleotides, so a whole length can be enumerated: 4^5 is 1024 strings. The
 * set is deliberately varied and the model STRICT, which is what makes the
 * brute-force check possible at all. A strict model's support here is 21
 * sequences with surprisals spread over three nats, leaving gaps far wider
 * than the grid -- where an open-world model would put 1024 sequences on the
 * grid with the unobserved ones sitting near `ln(1e-300)`, stretching the
 * automatic spacing to thousands of nats and leaving no resolution at all
 * where the mass actually is. */
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

static BeamModel *build_aa(uint32_t order, bool strict) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.order = order; c.max_length = 12u; c.strict = strict;
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
    c.order = 2u; c.max_length = 8u; c.strict = true;
    BeamModel *m = beam_model_new(&c);
    if (!m) return NULL;
    if (beam_model_observe(m, TRAIN_NT, W_NT, N_NT) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

/** Every band edge above the grid, so exactly one band holds everything. */
static const double ALL[2] = { 0.0, 1e12 };

/*
 * Band edges INSIDE the mass of the strict order-3 fixture, whose length-9
 * surprisals run from about 2.7 to 4.8 nats. Choosing them from the
 * distribution is not a convenience: `beam_model_accumulator_max` sizes the
 * automatic grid, and on an OPEN-world model that bound is set by the
 * backoff floor near `ln(1e-300)`, so a 512-bin grid there has bins 13 nats
 * wide and every band narrower than that is sub-bin. Bands must be many bins
 * wide for any of this to mean anything, which the strict fixture gives and
 * the open one does not.
 */
static const double INSIDE[5] = { 0.0, 3.0, 3.6, 4.2, 1e12 };
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
 * differ in the last bits of a double and no more.
 *
 * It pins the GENERATED measure only. `positional_marginals` is the edge
 * posterior, which weights each sequence by its own probability; the counting
 * measure has no exact counterpart here and is pinned by the brute force
 * below instead.
 */
static void test_one_band_over_everything_is_the_exact_marginal(void) {
    uint32_t compared = 0, skipped = 0;
    for (uint32_t order = 2; order <= 4; ++order) {
        for (int strict = 0; strict <= 1; ++strict) {
            BeamModel *m = build_aa(order, (bool)strict);
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
                /* 128 bins, not because it is enough to resolve anything
                 * but because one band covering the grid does not depend on
                 * the binning at all -- and a fine grid at order 4 is
                 * gigabytes this check has no use for. */
                const BeamError e = beam_model_positional_banded(
                    m, L, false, 128u, 0.0, ALL, 2u, band, &mass, NULL, NULL);
                if (e == BEAM_ERR_BUDGET) { ++skipped; free(exact); free(band); continue; }
                if (e != BEAM_OK) {
                    snprintf(msg, sizeof msg, "order %u strict %d length %u: %s",
                             (unsigned)order, strict, (unsigned)L,
                             beam_error_message());
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
                                     "order %u strict %d length %u position %u "
                                     "residue %u: the all-covering band gives "
                                     "%.17g where the exact marginal is %.17g",
                                     (unsigned)order, strict, (unsigned)L,
                                     (unsigned)i, (unsigned)x, a, b);
                            free(exact); free(band); beam_model_destroy(m);
                            ASSERT_MSG(false, msg);
                        }
                    }
                }
                free(exact); free(band);
            }
            beam_model_destroy(m);
        }
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
 * say that: both would drop the same state.
 */
static void test_the_band_masses_account_for_the_whole_spectrum(void) {
    BeamModel *m = build_aa(3u, false);
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
                if (fabs(mass - paths) > 1e-6 * paths) {
                    snprintf(msg, sizeof msg,
                             "length %u: the counting band holds %.17g paths "
                             "and beam_model_path_count says %.17g",
                             (unsigned)L, mass, paths);
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
 */
static void test_every_position_agrees_on_the_band_mass(void) {
    BeamModel *m = build_aa(3u, true);
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = 9u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not reach length 9");

    uint32_t grids = 0;
    for (uint32_t bins = 256u; bins <= 4096u; bins *= 2u) {
        double *band = (double *)calloc((size_t)N_INSIDE_BANDS * L * R, sizeof(double));
        double *pm   = (double *)calloc((size_t)N_INSIDE_BANDS * L, sizeof(double));
        double *bm   = (double *)calloc(N_INSIDE_BANDS, sizeof(double));
        ASSERT_MSG(band && pm && bm, "allocation");

        const BeamError e = beam_model_positional_banded(
            m, L, false, bins, 0.0, INSIDE, N_INSIDE, band, bm, pm, NULL);
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
                 * Grid error, not floating point, so the bound is loose by
                 * the standards of this file -- and it is still eight orders
                 * tighter than any real defect. Measured, this sits at 1e-15
                 * on every grid but one, where a sequence lands on a bin
                 * boundary and it reaches 2e-8.
                 */
                if (rel > 1e-6) {
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
        if (nonempty < 3u) {
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
 * The same bands, measured on grids from 256 to 4096 bins, hold the same mass.
 *
 * This is what pins the cutoff arithmetic. Every other check here is taken on
 * ONE grid, and an error in turning a threshold into a bin index -- a
 * ceiling that should be a floor, an inclusive total read as exclusive, a
 * missing edge-surprisal term -- is a shift of a fixed number of BINS, so it
 * changes the answer by an amount proportional to the spacing while remaining
 * perfectly self-consistent on any single grid. Halving the spacing halves
 * such an error, and it shows up here as a band that moves.
 */
static void test_refining_the_grid_does_not_move_a_band(void) {
    BeamModel *m = build_aa(3u, true);
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = 9u;
    double ref[N_INSIDE_BANDS];
    bool have = false;
    uint32_t grids = 0;

    for (uint32_t bins = 256u; bins <= 4096u; bins *= 2u) {
        double *band = (double *)calloc((size_t)N_INSIDE_BANDS * L * R, sizeof(double));
        double *bm   = (double *)calloc(N_INSIDE_BANDS, sizeof(double));
        ASSERT_MSG(band && bm, "allocation");
        const BeamError e = beam_model_positional_banded(
            m, L, false, bins, 0.0, INSIDE, N_INSIDE, band, bm, NULL, NULL);
        if (e == BEAM_ERR_BUDGET) { free(band); free(bm); continue; }
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
        if (!have) {
            for (uint32_t b = 0; b < N_INSIDE_BANDS; ++b) ref[b] = bm[b];
            have = true;
        } else {
            for (uint32_t b = 0; b < N_INSIDE_BANDS; ++b) {
                if (!(ref[b] > 0.0)) continue;
                const double rel = fabs(bm[b] - ref[b]) / ref[b];
                if (rel > 1e-6) {
                    snprintf(msg, sizeof msg,
                             "band %u holds %.17g at 256 bins and %.17g at %u "
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
 * other checks leave open.
 */
static void test_the_bands_recombine_into_the_unconditional_marginal(void) {
    BeamModel *m = build_aa(3u, true);
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = 9u;
    const double *edges = INSIDE;
    const uint32_t NE = N_INSIDE, NB = N_INSIDE_BANDS;

    double *band  = (double *)calloc((size_t)NB * L * R, sizeof(double));
    double *bm    = (double *)calloc(NB, sizeof(double));
    double *exact = (double *)calloc((size_t)L * R, sizeof(double));
    ASSERT_MSG(band && bm && exact, "allocation");

    BeamError e = beam_model_positional_banded(m, L, false, 1024u, 0.0,
                                               edges, NE, band, bm, NULL, NULL);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());
    e = beam_model_positional_marginals(m, L, NULL, exact, NULL);
    ASSERT_MSG(e == BEAM_OK, beam_error_message());

    double total = 0.0;
    for (uint32_t b = 0; b < NB; ++b) total += bm[b];
    ASSERT_MSG(total > 0.0, "every band was empty");

    /*
     * A band must genuinely differ from the pooled answer, or recombining
     * them proves nothing. Measured over EVERY position, not position 0:
     * every training sequence of this fixture starts with C, so position 0 is
     * one-hot in every band and looking only there reports a spread of
     * exactly zero -- which is how this guard first earned its place.
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

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/**
 * Enumerate every string of one length, split them by their EXACT surprisal,
 * and require the banded sweep to have found the same thing.
 *
 * The other four checks relate this code to itself or to its own pooled
 * answer. This one steps outside: the reference is
 * `beam_model_log_pseq` over all 4^L strings, summed by hand, with no
 * trellis, no grid and no dynamic program in it. It is the only check that
 * can catch the banding being systematically wrong -- a threshold applied to
 * the prefix instead of the total, a sink term counted twice, a cutoff off by
 * one bin -- because every one of those is self-consistent.
 *
 * It is written to be EXACT rather than approximate, which takes some care.
 * Linear splitting spreads one sequence's mass over up to `L+1` adjacent
 * bins, so a threshold within that of a real sequence classifies part of it
 * the wrong side and no tight tolerance is available. So the threshold is
 * placed in a GAP of the observed surprisal distribution, wider than the
 * splitting can reach; then every sequence is wholly one side and the two
 * answers must agree to floating point. The test asserts it found such a gap
 * rather than quietly settling for a worse one.
 *
 * Both measures are checked here. The counting measure has no exact
 * counterpart elsewhere in this file, so this is the only thing pinning it.
 */
static void test_the_banding_matches_brute_force_on_a_tiny_model(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = 5u;
    const uint32_t B = 4096u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not reach length 5");

    uint64_t total_strings = 1;
    for (uint32_t i = 0; i < L; ++i) total_strings *= R;

    char seq[16];
    uint32_t idx[16];
    double  *surp = (double *)malloc((size_t)total_strings * sizeof(double));
    double  *prob = (double *)malloc((size_t)total_strings * sizeof(double));
    uint8_t *lets = (uint8_t *)malloc((size_t)total_strings * L);
    double  *sorted = (double *)malloc((size_t)total_strings * sizeof(double));
    ASSERT_MSG(surp && prob && lets && sorted, "allocation");

    uint32_t n = 0;
    for (uint64_t code = 0; code < total_strings; ++code) {
        uint64_t c = code;
        for (uint32_t i = 0; i < L; ++i) { idx[i] = (uint32_t)(c % R); c /= R; }
        for (uint32_t i = 0; i < L; ++i) seq[i] = m->alpha->idx_to_char[idx[i]];
        seq[L] = '\0';
        const double lp = beam_model_log_pseq(m, seq);
        if (!isfinite(lp)) continue;           /* not producible: no mass */
        surp[n] = -lp;
        prob[n] = exp(lp);
        for (uint32_t i = 0; i < L; ++i) lets[(size_t)n * L + i] = (uint8_t)idx[i];
        sorted[n] = -lp;
        ++n;
    }
    ASSERT_MSG(n >= 8u, "the fixture produces too few sequences to split");
    qsort(sorted, n, sizeof(double), cmp_double);

    /* The grid this will be binned on, so the gap can be measured in bins. */
    double ds = 0.0;
    {
        double *tmp = (double *)calloc(B, sizeof(double));
        ASSERT_MSG(tmp, "allocation");
        const BeamError e = beam_model_spectrum(m, true, L, B, 0.0, tmp, &ds);
        free(tmp);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
    }
    /* Splitting can move a sequence's mass `L+1` bins; take twice that. */
    const double margin = (double)(2u * (L + 2u)) * ds;

    /* The most balanced usable gap, not the first: a cut that leaves one
     * sequence on one side is a weaker test than one that halves the set. */
    double cut = -1.0, widest = 0.0;
    uint32_t best = 0;
    for (uint32_t k = 1; k < n; ++k) {
        const double gap = sorted[k] - sorted[k - 1u];
        if (gap > widest) widest = gap;
        if (gap <= 2.0 * margin || sorted[k - 1u] <= margin) continue;
        const uint32_t bal = (k > n / 2u) ? k - n / 2u : n / 2u - k;
        if (cut < 0.0 || bal < best) {
            best = bal;
            cut = 0.5 * (sorted[k - 1u] + sorted[k]);
        }
    }
    if (!(cut > 0.0)) {
        snprintf(msg, sizeof msg,
                 "no gap wider than %.3g nats (%u bins) was found among %u "
                 "surprisals; the widest was %.3g, so no threshold here can "
                 "be checked exactly", 2.0 * margin,
                 (unsigned)(2u * 2u * (L + 2u)), (unsigned)n, widest);
        free(surp); free(prob); free(lets); free(sorted);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }
    free(sorted);

    const double edges[3] = { 0.0, cut, 1e12 };
    const uint32_t NE = 3u, NB = 2u;

    for (int counting = 0; counting <= 1; ++counting) {
        /* The reference, summed by hand over the enumeration. */
        double *ref  = (double *)calloc((size_t)NB * L * R, sizeof(double));
        double *rmass = (double *)calloc(NB, sizeof(double));
        ASSERT_MSG(ref && rmass, "allocation");
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t b = (surp[k] < cut) ? 0u : 1u;
            const double wgt = counting ? 1.0 : prob[k];
            rmass[b] += wgt;
            for (uint32_t i = 0; i < L; ++i)
                ref[((size_t)b * L + i) * R + lets[(size_t)k * L + i]] += wgt;
        }

        double *band = (double *)calloc((size_t)NB * L * R, sizeof(double));
        double *bm   = (double *)calloc(NB, sizeof(double));
        ASSERT_MSG(band && bm, "allocation");
        const BeamError e = beam_model_positional_banded(
            m, L, (bool)counting, B, ds, edges, NE, band, bm, NULL, NULL);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());

        for (uint32_t b = 0; b < NB; ++b) {
            const double tol = 1e-9 * (rmass[b] > 1.0 ? rmass[b] : 1.0);
            if (fabs(bm[b] - rmass[b]) > tol) {
                snprintf(msg, sizeof msg,
                         "counting %d band %u: the sweep puts %.17g in the "
                         "band and enumerating every sequence puts %.17g",
                         counting, (unsigned)b, bm[b], rmass[b]);
                goto fail;
            }
            if (!(rmass[b] > 0.0)) {
                snprintf(msg, sizeof msg,
                         "counting %d band %u is empty, so the split put every "
                         "sequence on one side and proves nothing",
                         counting, (unsigned)b);
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
        free(surp); free(prob); free(lets);
        beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }

    free(surp); free(prob); free(lets);
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
 * recombine, and the brute-force check places its threshold in a gap wide
 * enough that a one-bin shift moves no sequence across it -- which is the
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
 * error at a few hundredths of a spacing: measured, this lands within 0.033
 * of a bin, where the one-bin mutation lands at 1.03. The bound below sits
 * between them with an order of magnitude either side.
 */
static void test_the_recovered_mean_surprisal_is_unbiased(void) {
    BeamModel *m = build_nt();
    ASSERT_MSG(m, "model did not build");
    const uint32_t R = beam_model_n_residues(m);
    const uint32_t L = 5u, B = 256u;
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
    ASSERT_MSG(n >= 8u, "the fixture produces too few sequences");

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

#define N_NEG 10
/* The exact set that exposed the defect, kept verbatim. A conserved head and
 * tail with a two-letter middle: what matters is that the sink states carry a
 * tail weight of zero, so the negative per-length constant is the WHOLE of
 * the seed and drives it below the origin unopposed. Fixtures whose tails are
 * strongly negative absorb the constant and cover nothing -- the first
 * version of this test was one, and passed against the bug. */
static const char *TRAIN_NEG[N_NEG] = {
    "CASSGGGGGGGEQYF", "CASSAAAAGGAEQYF", "CASSAGAGAAGEQYF", "CASSAAAGAGAEQYF",
    "CASSAGGGGGGEQYF", "CASSGGGAGAAEQYF", "CASSGAGGAAAEQYF", "CASSAAGGAGGEQYF",
    "CASSGGAGAAAEQYF", "CASSAGAAAGGEQYF",
};

/**
 * A model whose sink term is NEGATIVE still returns its mass.
 *
 * The backward sweep is seeded with `-tail(s) + ln Z(L) - ln pi(L)`, and that
 * constant is a fraction of a nat of either sign. When it is negative the
 * seed sits below the grid origin, and a grid that begins at zero loses every
 * suffix off the bottom -- the sweep then returns zero mass, confidently,
 * for a model whose spectrum is perfectly healthy.
 *
 * That is not hypothetical: it is what this code did. `spectrum.c` cannot
 * meet it, because it applies the sink LAST onto an accumulator already clear
 * of zero, so no existing test covered the case and none of the six checks
 * above did either -- their fixtures all happen to have non-negative sinks.
 * It was found from the Python layer, on a model built for something else.
 *
 * The test asserts its own precondition. A fixture whose sink drifted
 * non-negative would still pass every line below while covering nothing, so
 * the sink is computed from the public partition function and the length
 * prior and required to be negative before anything else is checked.
 */
static BeamModel *build_neg(void) {
    BeamModelConfig cfg;
    beam_model_config_default(&cfg);
    cfg.order = 3u; cfg.max_length = 18u; cfg.strict = true;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    double w[N_NEG];
    for (uint32_t k = 0; k < N_NEG; ++k) w[k] = 1.0;
    if (beam_model_observe(m, TRAIN_NEG, w, N_NEG) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

static void test_a_sink_below_the_grid_origin_still_reports_its_mass(void) {
    BeamModel *m = build_neg();
    ASSERT_MSG(m, "model did not build");

    const uint32_t L = 15u, R = beam_model_n_residues(m), B = 2048u;
    ASSERT_MSG(beam_length_live(m, L), "the fixture does not reach length 15");

    const double sink = beam_model_log_partition(m, L)
                      - beam_model_log_length_prior(m, L);
    if (!(sink < 0.0)) {
        snprintf(msg, sizeof msg,
                 "this fixture's sink term is %.6g, not negative, so it no "
                 "longer covers a backward grid that starts below zero",
                 sink);
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
                 "sink %.6g: the band holds %.17g where the spectrum "
                 "integrates to %.17g -- the backward grid lost its mass off "
                 "the bottom", sink, mass, total);
        ASSERT_MSG(false, msg);
    }
    if (worst > 1e-12) {
        snprintf(msg, sizeof msg,
                 "sink %.6g: the all-covering band departs from the exact "
                 "marginal by %.3g", sink, worst);
        ASSERT_MSG(false, msg);
    }
    PASS();
}


/* ── 8. A bin-aligned band is the spectrum summed over those bins ── */

/**
 * With a threshold ON a bin boundary, a band's mass equals
 * `beam_model_spectrum` summed over the same bins -- exactly, not to grid
 * precision.
 *
 * That is worth stating because it is stronger than it looks. The two get
 * there by different routes: the spectrum transports a total through `L+1`
 * sequential splits, while this splits a prefix and a suffix separately and
 * then cuts. They need not agree bin by bin, and at a bin boundary their
 * CUMULATIVES do -- measured across nine deciles of two models, to 1.4e-17.
 *
 * It is also the only check that pins the backward grid's ORIGIN into the
 * cutoff. `pad` shifts where every suffix sits, and the accumulation has to
 * shift its threshold by the same amount; drop that and the sweep answers a
 * different question perfectly self-consistently -- same total, same
 * agreement between positions, same recombination. Only a reference that
 * knows which threshold was asked for can tell, and only on a model where
 * `pad` is not zero, which is why this runs on the negative-sink fixture as
 * well as an ordinary one.
 */
static void test_a_bin_aligned_band_is_the_spectrum_over_those_bins(void) {
    const uint32_t B = 2048u;
    uint32_t checked = 0;

    for (int which = 0; which <= 1; ++which) {
        BeamModel *m = which ? build_neg() : build_aa(3u, true);
        ASSERT_MSG(m, "model did not build");
        const uint32_t R = beam_model_n_residues(m);
        const uint32_t L = which ? 15u : 9u;
        ASSERT_MSG(beam_length_live(m, L), "fixture does not reach its length");

        double *spec = (double *)calloc(B, sizeof(double));
        ASSERT_MSG(spec, "allocation");
        double ds = 0.0;
        BeamError e = beam_model_spectrum(m, false, L, B, 0.0, spec, &ds);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());
        double total = 0.0;
        for (uint32_t b = 0; b < B; ++b) total += spec[b];
        ASSERT_MSG(total > 0.0, "the spectrum is empty");

        /* Quartile bin boundaries: thresholds that actually divide the mass. */
        uint32_t qb[3] = { 0, 0, 0 };
        {
            double cum = 0.0;
            uint32_t got = 0;
            for (uint32_t b = 0; b < B && got < 3u; ++b) {
                cum += spec[b];
                while (got < 3u && cum >= 0.25 * (double)(got + 1u) * total)
                    qb[got++] = b;
            }
            ASSERT_MSG(got == 3u, "could not place three quartile thresholds");
        }

        const double edges[5] = { 0.0, (double)qb[0] * ds, (double)qb[1] * ds,
                                  (double)qb[2] * ds, 1e12 };
        double *band = (double *)calloc(4ull * L * R, sizeof(double));
        double bm[4];
        ASSERT_MSG(band, "allocation");
        e = beam_model_positional_banded(m, L, false, B, ds, edges, 5u,
                                         band, bm, NULL, NULL);
        ASSERT_MSG(e == BEAM_OK, beam_error_message());

        double running = 0.0;
        for (uint32_t q = 0; q < 3u; ++q) {
            running += bm[q];
            double ref = 0.0;
            for (uint32_t b = 0; b < qb[q]; ++b) ref += spec[b];
            ++checked;
            if (!(ref > 0.0 && ref < total)) {
                snprintf(msg, sizeof msg,
                         "fixture %d quartile %u: the threshold does not "
                         "divide the mass (%.17g of %.17g)",
                         which, q, ref, total);
                goto fail;
            }
            if (fabs(running - ref) > 1e-12 * total) {
                snprintf(msg, sizeof msg,
                         "fixture %d quartile %u at bin %u: the bands hold "
                         "%.17g below it and the spectrum holds %.17g "
                         "(%.3g of the total apart)",
                         which, q, qb[q], running, ref,
                         fabs(running - ref) / total);
                goto fail;
            }
        }
        free(spec); free(band); beam_model_destroy(m);
        continue;
    fail:
        free(spec); free(band); beam_model_destroy(m);
        ASSERT_MSG(false, msg);
    }
    ASSERT_MSG(checked == 6u, "not every threshold was checked");
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
    BeamModel *m = build_aa(3u, false);
    ASSERT_MSG(m, "model did not build");
    /* A length this fixture GENERATES. Length 6 is not one of them, and the
     * length check runs before the edge checks, so using it made every case
     * below refuse for the wrong reason -- this test passed in full with no
     * edge validation at all. The positive control at the end is what stops
     * that happening again. */
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
    /* A length the model does not generate is refused too. */
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
    BeamModel *m = build_aa(3u, false);
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
