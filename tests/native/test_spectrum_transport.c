/**
 * @file test_spectrum_transport.c
 * @brief The spectrum DP, bit for bit against the transport it was written as.
 *
 * The per-layer transport of `spectrum_one` (math/spectrum.c) is the loop the
 * figure pipeline spends most of its binned time in, and it is free to be
 * rearranged for speed only because every bin's sum is taken in a fixed
 * order. This file keeps the straightforward form -- each target gathers its
 * sources in ascending order, each source's bins moved by
 * `beam_grid_transport` one at a time -- as a sequential reference, and holds
 * every public route to it BIT FOR BIT: one measure and both, pooled and per
 * length, on grids that hold everything and on one so fine that most mass
 * runs off its top, at one thread and several.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"

#include "../../src/native/internal/beam/internal.h"
#include "../../src/native/internal/beam/grid.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

static char msg[512];

/* ── Fixtures ────────────────────────────────────────────────── */

#define N_AA 12
static const char *TRAIN_AA[N_AA] = {
    "CASSF", "CASSY", "CAGSF", "CGSSF", "CASSQEV", "CASRWMT",
    "CAWSFGN", "CASSLNRT", "CTSSFRWY", "CASSLGGYE", "CASRLGGYE", "CASSPGGYE",
};
static const double W_AA[N_AA] = { 40, 25, 18, 15, 9, 8, 7, 9, 8, 12, 7, 5 };

#define N_NT 8
static const char *TRAIN_NT[N_NT] = {
    "ACGTA", "ACGTC", "CGTAC", "TACGT", "GCATG", "ACGTAC", "TTACGG", "GATCA",
};
static const double W_NT[N_NT] = { 30, 22, 9, 14, 10, 9, 6, 3 };

static BeamModel *build(BeamAlphabet alphabet, uint32_t order, uint32_t max_len) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.alphabet = alphabet;
    c.order = order;
    c.max_length = max_len;
    BeamModel *m = beam_model_new(&c);
    if (!m) return NULL;
    const bool aa = alphabet == BEAM_ALPHABET_AA;
    if (beam_model_observe(m, aa ? TRAIN_AA : TRAIN_NT, aa ? W_AA : W_NT,
                           aa ? N_AA : N_NT) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

/* ── The reference: spectrum_one as it was first written ─────── */

/*
 * One length, sequential, every source of a target in ascending order and
 * every bin moved on its own. `out_b` NULL carries one measure (`counting`
 * picks which); otherwise counting into `out_a` and generated into `out_b`.
 */
static int ref_one(const BeamModel *m, bool counting, uint32_t len,
                   uint32_t bins, double ds, double *out_a, double *out_b) {
    const uint32_t A = m->n_alphabet, R = m->n_residues, S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t nm = out_b ? 2u : 1u;
    LiveHandle lh;
    if (beam_live_acquire(m, len, &lh) != BEAM_OK) return 0;
    const BeamLive *L = lh.L;
    uint32_t maxlive = 0;
    for (uint32_t i = 0; i <= len; ++i)
        if (L->n_live[i] > maxlive) maxlive = L->n_live[i];
    const size_t hist = (size_t)nm * bins;
    double *w = malloc((size_t)S * R * sizeof(double));
    double *tail = malloc((size_t)S * sizeof(double));
    double *cur = calloc((size_t)maxlive * hist, sizeof(double));
    double *next = calloc((size_t)maxlive * hist, sizeof(double));
    BeamBinRange *rc = malloc((size_t)maxlive * sizeof(BeamBinRange));
    BeamBinRange *rn = malloc((size_t)maxlive * sizeof(BeamBinRange));
    uint32_t *sc = malloc((size_t)S * sizeof(uint32_t));
    uint32_t *sn = malloc((size_t)S * sizeof(uint32_t));
    uint8_t *av = malloc((size_t)S);
    int ok = w && tail && cur && next && rc && rn && sc && sn && av;
    uint32_t rk = 0;
    bool have = false;
    if (ok) {
        for (uint32_t k = 0; k < maxlive; ++k) { rc[k].lo = bins; rc[k].hi = 0; }
        for (uint32_t k = 0; k < L->n_live[0]; ++k) {
            sc[L->live[0][k]] = k;
            if (L->live[0][k] == S - 1u) { rk = k; have = true; }
        }
    }
    if (ok && have) {
        for (uint32_t mi = 0; mi < nm; ++mi) cur[(size_t)rk * hist + (size_t)mi * bins] = 1.0;
        rc[rk].lo = 0; rc[rk].hi = 1;
        for (uint32_t i = 0; i < len; ++i) {
            memset(av, 0, S);
            for (uint32_t k = 0; k < L->n_live[i]; ++k) av[L->live[i][k]] = 1;
            beam_layer_weights_live(m, i, len, w, NULL, av);
            const uint32_t *lv = L->live[i + 1];
            const uint32_t n_next = L->n_live[i + 1];
            for (uint32_t k = 0; k < n_next; ++k) sn[lv[k]] = k;
            for (uint32_t kk = 0; kk < n_next; ++kk) {
                rn[kk].lo = bins; rn[kk].hi = 0;
                const uint32_t tt = lv[kk], x = tt % A;
                if (x >= R) continue;
                const uint32_t rest = tt / A;
                double *dst = next + (size_t)kk * hist;
                for (uint32_t a = 0; a < A; ++a) {
                    const uint32_t s = a * SK + rest;
                    if (!live_bit_get(L->mask[i], s)) continue;
                    const uint32_t ks = sc[s];
                    if (rc[ks].lo >= rc[ks].hi) continue;
                    const double wv = w[(size_t)s * R + x];
                    if (wv == -INFINITY) continue;
                    const double *src = cur + (size_t)ks * hist;
                    if (nm == 1u)
                        beam_grid_transport(src, dst, bins, -wv / ds,
                                            counting ? 1.0 : exp(wv), rc[ks], &rn[kk]);
                    else
                        beam_grid_transport2(src, src + bins, dst, dst + bins, bins,
                                             -wv / ds, 1.0, exp(wv), rc[ks], &rn[kk]);
                }
            }
            for (uint32_t k = 0; k < L->n_live[i]; ++k) {
                if (rc[k].lo >= rc[k].hi) continue;
                for (uint32_t mi = 0; mi < nm; ++mi)
                    memset(cur + (size_t)k * hist + (size_t)mi * bins + rc[k].lo, 0,
                           (size_t)(rc[k].hi - rc[k].lo) * sizeof(double));
            }
            { double *t = cur; cur = next; next = t; }
            { BeamBinRange *t = rc; rc = rn; rn = t; }
            { uint32_t *t = sc; sc = sn; sn = t; }
        }
        beam_tail_weights(m, len, tail);
        const double off = m->log_z[len] - m->log_pi[len];
        BeamBinRange sink = { bins, 0 };
        for (uint32_t k = 0; k < L->n_live[len]; ++k) {
            const uint32_t s = L->live[len][k];
            if (tail[s] == -INFINITY || rc[k].lo >= rc[k].hi) continue;
            const double *src = cur + (size_t)k * hist;
            const double u = (-tail[s] + off) / ds;
            if (nm == 1u)
                beam_grid_transport(src, out_a, bins, u,
                                    counting ? 1.0 : exp(tail[s] - off), rc[k], &sink);
            else
                beam_grid_transport2(src, src + bins, out_a, out_b, bins, u,
                                     1.0, exp(tail[s] - off), rc[k], &sink);
        }
    }
    free(w); free(tail); free(cur); free(next); free(rc); free(rn);
    free(sc); free(sn); free(av);
    beam_live_release(&lh);
    return ok;
}

/* `length` 0 pools every length the model generates, in ascending order. */
static int ref_spectrum(const BeamModel *m, bool counting, uint32_t length,
                        uint32_t bins, double ds, double *a, double *b) {
    memset(a, 0, bins * sizeof(double));
    if (b) memset(b, 0, bins * sizeof(double));
    if (length) return ref_one(m, counting, length, bins, ds, a, b);
    for (uint32_t L = 1; L < m->n_len_slots; ++L)
        if (beam_length_live(m, L) && !ref_one(m, counting, L, bins, ds, a, b)) return 0;
    return 1;
}

/* ── The checks ──────────────────────────────────────────────── */

typedef struct { const char *name; BeamAlphabet alphabet; uint32_t order, max_len; } Fixture;
static const Fixture FIXTURES[] = {
    { "aa order 2", BEAM_ALPHABET_AA, 2u, 10u },
    { "aa order 3", BEAM_ALPHABET_AA, 3u, 9u },
    { "nt order 2", BEAM_ALPHABET_NT, 2u, 6u },
};

/* Bins, and the spacing as a fraction of the automatic one: 1 holds
 * everything, 1/12 runs most of the mass off the grid's top. */
typedef struct { uint32_t bins; double shrink; } Grid;
static const Grid GRIDS[] = { { 40u, 1.0 }, { 300u, 1.0 }, { 1500u, 1.0 }, { 256u, 1.0 / 12.0 } };

static int same(const double *a, const double *b, uint32_t n, const char *what,
                const char *fx, uint32_t bins, uint32_t L, int th) {
    if (memcmp(a, b, n * sizeof(double)) == 0) return 1;
    uint32_t i = 0;
    while (i < n && memcmp(a + i, b + i, sizeof(double)) == 0) ++i;
    snprintf(msg, sizeof msg, "%s, %s, %u bins, length %u, %d threads: bin %u is "
             "%.17g, the reference has %.17g", fx, what, bins, L, th, i, a[i], b[i]);
    return 0;
}

static void check_fixture(const Fixture *fx, int *bad) {
    BeamModel *m = build(fx->alphabet, fx->order, fx->max_len);
    if (!m) { snprintf(msg, sizeof msg, "%s: build failed", fx->name); *bad = 1; return; }
    const uint32_t lengths[] = { 0u, 3u, 5u, fx->max_len };
    for (size_t g = 0; g < sizeof GRIDS / sizeof GRIDS[0] && !*bad; ++g) {
        const uint32_t bins = GRIDS[g].bins;
        if (fx->order == 3u && bins > 300u) continue;      /* scratch, not coverage */
        double ds = 0.0;
        double *a = malloc(bins * sizeof(double)), *b = malloc(bins * sizeof(double));
        double *ra = malloc(bins * sizeof(double)), *rb = malloc(bins * sizeof(double));
        const uint32_t rows = fx->max_len + 1u;
        double *byl = malloc((size_t)rows * bins * sizeof(double));
        if (beam_model_spectrum(m, true, 0, bins, 0.0, a, &ds) != BEAM_OK) {
            snprintf(msg, sizeof msg, "%s: %u bins refused", fx->name, bins);
            *bad = 1;
        }
        ds *= GRIDS[g].shrink;
        for (int th = 1; th <= 6 && !*bad; th += 5) {
            beam_model_set_threads(th);
            for (size_t li = 0; li < 4 && !*bad; ++li) {
                const uint32_t L = lengths[li];
                for (int counting = 0; counting <= 1 && !*bad; ++counting) {
                    beam_model_spectrum(m, counting, L, bins, ds, a, NULL);
                    ref_spectrum(m, counting, L, bins, ds, ra, NULL);
                    if (!same(a, ra, bins, counting ? "counting" : "generated",
                              fx->name, bins, L, th)) *bad = 1;
                }
                if (*bad) break;
                beam_model_spectrum_pair(m, L, bins, ds, a, b, NULL);
                ref_spectrum(m, false, L, bins, ds, ra, rb);
                if (!same(a, ra, bins, "pair counting", fx->name, bins, L, th)
                    || !same(b, rb, bins, "pair generated", fx->name, bins, L, th)) *bad = 1;
            }
            for (int counting = 0; counting <= 1 && !*bad; ++counting) {
                beam_model_spectra_by_length(m, counting, 0, bins, ds, byl, NULL);
                for (uint32_t L = 1; L < rows && !*bad; ++L) {
                    if (!beam_length_live(m, L)) continue;
                    ref_spectrum(m, counting, L, bins, ds, ra, NULL);
                    if (!same(byl + (size_t)L * bins, ra, bins, "by length",
                              fx->name, bins, L, th)) *bad = 1;
                }
            }
        }
        beam_model_set_threads(0);
        free(a); free(b); free(ra); free(rb); free(byl);
    }
    beam_model_destroy(m);
}

static void test_every_route_matches_the_reference_bit_for_bit(void) {
    int bad = 0;
    for (size_t f = 0; f < sizeof FIXTURES / sizeof FIXTURES[0] && !bad; ++f)
        check_fixture(&FIXTURES[f], &bad);
    ASSERT_MSG(!bad, msg);
    PASS();
}

/* The fine grid really does run mass off its top, so the dropping at the
 * grid's edge is exercised rather than assumed. */
static void test_the_fine_grid_loses_mass_off_its_top(void) {
    BeamModel *m = build(BEAM_ALPHABET_AA, 2u, 10u);
    ASSERT_MSG(m, "build failed");
    double *a = malloc(256 * sizeof(double)), ds = 0.0, full = 0.0, fine = 0.0;
    beam_model_spectrum(m, true, 0, 256, 0.0, a, &ds);
    for (int i = 0; i < 256; ++i) full += a[i];
    beam_model_spectrum(m, true, 0, 256, ds / 12.0, a, NULL);
    for (int i = 0; i < 256; ++i) fine += a[i];
    free(a);
    beam_model_destroy(m);
    snprintf(msg, sizeof msg, "sequences on the fine grid %.6g of %.6g, expected under half", fine, full);
    ASSERT_MSG(fine < 0.5 * full, msg);
    PASS();
}

int main(void) {
    printf("spectrum transport\n");
    RUN_TEST(test_every_route_matches_the_reference_bit_for_bit);
    RUN_TEST(test_the_fine_grid_loses_mass_off_its_top);
    printf("  %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
