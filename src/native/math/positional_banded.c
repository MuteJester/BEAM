/**
 * @file positional_banded.c
 * @brief Per-position residue marginals conditioned on a band of surprisal.
 *
 * ## The quantity
 *
 *     P(x_i = a | L, T0 <= -ln P(s) < T1)
 *
 * the residue distribution at each position of a length-`L` sequence, given
 * that the whole sequence falls in a chosen band of probability. Bands are
 * given as surprisal thresholds in nats because that is the axis the grid is
 * on; a caller thinking in probabilities passes `-ln p`, descending in `p`.
 *
 * One logo per band is the point. `beam_model_positional_marginals` answers
 * what a typical sequence of this length looks like; this answers what a
 * typical sequence of this length AND THIS RARITY looks like, which is a
 * different question whenever the model's rare sequences are not simply its
 * common ones reweighted.
 *
 * ## Where this is approximate, and where it is not
 *
 * `positional.c` is exact: a residue is one of twenty, the axis is already
 * discrete, and nothing is binned. This file gives that up, and the docstring
 * there explains why the distinction matters. Conditioning on a surprisal band
 * needs the DISTRIBUTION of a real-valued path functional, so the surprisal
 * axis must be discretised, and mass is split linearly between adjacent bins
 * exactly as `beam_model_spectrum` splits it.
 *
 * So: the residue axis is exact, the conditioning event is approximate, and
 * the approximation is entirely in which side of a threshold a sequence near
 * that threshold lands on. Two consequences a caller must know:
 *
 *   - A band's marginals are exact for the set of sequences the grid assigns
 *     to that band, which differs from the set the exact arithmetic would
 *     assign by O(spacing) around each edge. Narrow bands are therefore worse
 *     conditioned than wide ones, and a band narrower than a few bins is
 *     measuring the grid rather than the model.
 *   - Totals are conserved regardless. Splitting moves mass between bins and
 *     never destroys it, so the band masses sum to the total mass on the grid
 *     whatever the thresholds are.
 *
 * ## The algorithm
 *
 * The trellis state is augmented with accumulated surprisal, as `spectrum.c`
 * does, but swept both ways.
 *
 *   - `F[i][u][b]`: mass of prefixes reaching state `u` at layer `i` with
 *     accumulated surprisal `b * spacing`. This is the spectrum's existing
 *     transport, kept per layer instead of discarded.
 *   - `B[i][v][b]`: mass of suffixes from `v` to the sink, the same sweep
 *     mirrored, seeded with the sink tail and the per-length constant so that
 *     forward plus edge plus backward is the full surprisal.
 *
 * THE BACKWARD GRID DOES NOT START AT ZERO, and that is not a detail. Its
 * seed is `-tail(s) + ln Z(L) - ln pi(L)`, and the per-length constant there
 * is a fraction of a nat of EITHER sign -- `beam_model_accumulator_max` says
 * so, from -1.6 to +0.005 across the lengths of a small corpus. A negative
 * seed placed at the origin falls off the bottom of the grid, taking every
 * suffix with it and returning a confident zero.
 *
 * `spectrum.c` never meets this because it applies the sink LAST, onto a
 * forward accumulator that has already climbed well clear of zero; only the
 * mirror starts there. So the backward grid carries an origin: bin `b` holds
 * suffix surprisal `(b - pad) * spacing`, with `pad` whole bins chosen from
 * the seeds themselves so the lowest of them lands at bin 0. The suffix
 * accumulator only rises as the sweep runs backward -- every edge surprisal
 * is non-negative, which is the same monotonicity the spectrum's own grid
 * sizing assumes -- so the seeds are its minimum and nothing else can
 * underflow.
 *
 * Every complete sequence through an edge at layer `i` from `u` to `v`
 * emitting `a` with weight `w` has surprisal `prefix + (-ln w) + suffix`, so
 * the mass in a band is a double sum over prefix and suffix bins -- edges
 * times bins squared, which is not affordable.
 *
 * It collapses because a band needs only a THRESHOLD, not the joint. Replace
 * `B` by its running total `C[j] = sum_{q<=j} B[q]`; then the suffixes that
 * keep a given prefix bin under a threshold are a single prefix of the
 * backward grid, and the whole inner loop is one multiply per prefix bin:
 *
 *     mass(total < T) = sum_p F[p] * C[ceil((T + ln w)/spacing) - p - 1]
 *
 * The cutoff index falls by exactly one as `p` rises by one, because the
 * ceiling of a real shifted by an integer is that integer shift of the
 * ceiling -- so it is a decrement, not a recomputation. Bands are then
 * differences of consecutive thresholds.
 *
 * ## Cost
 *
 * Two grid sweeps plus one accumulation PER THRESHOLD, so `2 + K` spectrum
 * passes for `K` thresholds rather than the three a single band would suggest.
 * The accumulation is the same order as the transport it reuses.
 *
 * Memory is the real constraint, because one side's grids must be held for
 * every layer at once while the other streams. It is
 *
 *     sum_i n_live[i] * bins * 8 bytes
 *
 * and the sum matters: charging the WIDEST layer to every layer, which is the
 * obvious bound, is about three times too much. On the length-15 trellis
 * `spectrum.c` measures, the widest layer holds 11,099 live states and the
 * mean layer 3,502, so per-layer sizing turns 727 MB into 229 MB at 512 bins
 * and 2.9 GB into 916 MB at 2,048. The frontier already stores each layer's
 * live list, so this costs nothing to exploit.
 *
 * It goes through the memory budget and is REFUSED BEFORE ALLOCATING, like
 * every other sweep here.
 */
#include "beam/internal.h"
#include "beam/grid.h"
#include "beam/math_internal.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* ── Scratch sizing ──────────────────────────────────────────── */

/**
 * Scratch this sweep holds at once, in bytes, from the SAME expressions the
 * allocations below use. The backward pool dominates everything else.
 */
static uint64_t banded_bytes(const BeamModel *m, uint32_t bins,
                             uint32_t bbins, uint64_t slots, uint32_t maxlive,
                             uint32_t len, uint32_t n_thr) {
    const uint64_t S = m->n_states;
    const uint64_t R = m->n_residues;
    return slots * bbins * sizeof(double)              /* backward pool    */
         + slots * sizeof(BeamBinRange)                /* its bin ranges   */
         + 2ull * maxlive * bins * sizeof(double)      /* forward pair     */
         + 2ull * maxlive * sizeof(BeamBinRange)
         + S * R * sizeof(double)                      /* edge weights     */
         + S * sizeof(double)                          /* tail             */
         + 2ull * S * sizeof(uint32_t)                 /* slot maps        */
         + S                                           /* byte mask        */
         + (uint64_t)n_thr * len * R * sizeof(double)  /* threshold totals */
         + (uint64_t)BEAM_SUM_CHUNKS * n_thr * R * sizeof(double);
}


/* ── The threshold cutoff ────────────────────────────────────── */

/**
 * Mass of complete sequences through one edge whose total surprisal is below
 * a threshold, given the prefix grid, the backward RUNNING TOTAL, and the
 * cutoff index `jc` the threshold and the edge weight imply at prefix bin 0.
 *
 * `jc - p` is the number of backward bins admissible at prefix bin `p`, and it
 * only falls, so the loop stops at the first prefix bin that admits none.
 */
static inline double below_threshold(const double *src, const double *cum,
                                     uint32_t bbins, BeamBinRange sr,
                                     int64_t jc) {
    const double all = cum[bbins - 1u];
    double sum = 0.0;
    for (uint32_t p = sr.lo; p < sr.hi; ++p) {
        const int64_t j = jc - (int64_t)p;
        if (j <= 0) break;               /* and none of the later bins do */
        const double v = src[p];
        if (v == 0.0) continue;
        sum += v * ((j >= (int64_t)bbins) ? all : cum[j - 1]);
    }
    return sum;
}


/* ── Backward grids ──────────────────────────────────────────── */

/**
 * Fill `pool` with every layer's backward grid, converting each to a running
 * total once the layer below it has been built from the raw values.
 *
 * Layer `len` is seeded by transporting a unit spike by the sink term, so the
 * surprisal a suffix carries is on the same footing as the one a prefix does:
 * `spectrum.c` applies `(-tail + ln Z - ln pi) / spacing` at the sink, and
 * this applies it at the start of the mirror.
 *
 * A target absent from layer `i+1`'s live set is skipped, and that is exact
 * rather than a pruning. A live source at layer `i` with a finite edge to `t`
 * makes `t` forward-reachable, so a missing `t` can only be one the sink
 * cannot be reached from -- and its suffix mass is zero by definition.
 */
static void backward_grids(const BeamModel *m, uint32_t len, bool counting,
                           uint32_t bbins, uint32_t pad, double ds,
                           const BeamLive *L,
                           const size_t *off, double *pool, BeamBinRange *rng,
                           double *w, const double *tail, uint8_t *av,
                           uint32_t *slot_a, uint32_t *slot_b) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;

    for (uint64_t k = 0; k < off[len + 1u] / bbins; ++k)
        rng[k] = (BeamBinRange){ bbins, 0 };

    {
        const double sink = m->log_z[len] - m->log_pi[len];
        const double one = 1.0;
        const BeamBinRange spike = { 0u, 1u };
        uint32_t *sn = slot_b;
        for (uint32_t k = 0; k < L->n_live[len]; ++k) {
            const uint32_t s = L->live[len][k];
            sn[s] = k;
            if (tail[s] == -INFINITY) continue;
            beam_grid_transport(&one, pool + off[len] + (size_t)k * bbins,
                                bbins, (-tail[s] + sink) / ds + (double)pad,
                                counting ? 1.0 : exp(tail[s] - sink),
                                spike, &rng[off[len] / bbins + k]);
        }
    }

    uint32_t *sc = slot_a, *sn = slot_b;
    for (uint32_t i = len; i-- > 0; ) {
        memset(av, 0, S);
        for (uint32_t k = 0; k < L->n_live[i]; ++k) av[L->live[i][k]] = 1;
        beam_layer_weights_live(m, i, len, w, NULL, av);

        const uint64_t *mk_next = L->mask[i + 1];
        const size_t base_i = off[i] / bbins, base_n = off[i + 1] / bbins;
        const uint32_t n_i = L->n_live[i];
        /* Each source owns its own grid, so writing by source cannot race and
         * the sum over `x` is taken in one fixed order however many threads
         * run it. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(beam_threads())
#endif
        for (int64_t kk = 0; kk < (int64_t)n_i; ++kk) {
            const uint32_t s = L->live[i][kk];
            sc[s] = (uint32_t)kk;
            double *dst = pool + off[i] + (size_t)kk * bbins;
            BeamBinRange dr = { bbins, 0 };
            const double *row = w + (size_t)s * R;
            const uint32_t rest = (s % SK) * A;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] == -INFINITY) continue;
                const uint32_t t = rest + x;
                if (!live_bit_get(mk_next, t)) continue;
                const uint32_t kt = sn[t];
                const BeamBinRange sr = rng[base_n + kt];
                if (sr.lo >= sr.hi) continue;
                beam_grid_transport(pool + off[i + 1] + (size_t)kt * bbins,
                                    dst, bbins, -row[x] / ds,
                                    counting ? 1.0 : exp(row[x]), sr, &dr);
            }
            rng[base_i + kk] = dr;
        }

        /* Layer `i+1` has been consumed raw; turn it into the running total
         * the accumulation reads. Bins below `lo` are zero and stay zero. */
        const uint32_t n_n = L->n_live[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t kk = 0; kk < (int64_t)n_n; ++kk) {
            double *g = pool + off[i + 1] + (size_t)kk * bbins;
            const BeamBinRange r = rng[base_n + kk];
            double acc = 0.0;
            for (uint32_t b = (r.lo < bbins ? r.lo : bbins); b < bbins; ++b) {
                acc += g[b];
                g[b] = acc;
            }
        }
        { uint32_t *t = sc; sc = sn; sn = t; }
    }
}


/* ── Forward sweep and accumulation ──────────────────────────── */

/**
 * Sweep the prefix grids forward, accumulating each threshold's mass per
 * position and residue on the way through.
 *
 * `acc[(t * len + i) * R + x]` receives the mass of complete sequences with
 * `x_i = x` whose total surprisal is below threshold `t`. Bands are the
 * differences of consecutive thresholds, taken by the caller.
 *
 * The sum over sources is split into `BEAM_SUM_CHUNKS` fixed chunks combined
 * in index order, so the answer does not depend on how many threads ran --
 * the property `verification/check.py` pins as thread independence.
 */
static BeamError forward_accumulate(const BeamModel *m, uint32_t len,
                                    bool counting, uint32_t bins,
                                    uint32_t bbins, uint32_t pad, double ds,
                                    const BeamLive *L, const size_t *off,
                                    const double *pool, const BeamBinRange *rng,
                                    const double *thr, uint32_t n_thr,
                                    double *acc, double *cur, double *next,
                                    BeamBinRange *rc, BeamBinRange *rn,
                                    double *w, uint8_t *av,
                                    uint32_t *sc, uint32_t *sn, double *chunk) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const size_t part_sz = (size_t)n_thr * R;

    for (uint32_t k = 0; k < L->n_live[0]; ++k) {
        rc[k] = (BeamBinRange){ bins, 0 };
        sc[L->live[0][k]] = k;
    }
    {   /* The root alone occupies layer 0, at surprisal zero. */
        const uint32_t root = S - 1u;
        bool have = false;
        for (uint32_t k = 0; k < L->n_live[0]; ++k)
            if (L->live[0][k] == root) {
                cur[(size_t)k * bins] = 1.0;
                rc[k] = (BeamBinRange){ 0u, 1u };
                have = true;
                break;
            }
        if (!have) return BEAM_OK;   /* no path: every band stays empty */
    }

    for (uint32_t i = 0; i < len; ++i) {
        memset(av, 0, S);
        for (uint32_t k = 0; k < L->n_live[i]; ++k) av[L->live[i][k]] = 1;
        beam_layer_weights_live(m, i, len, w, NULL, av);

        const uint64_t *mk_cur  = L->mask[i];
        const uint64_t *mk_next = L->mask[i + 1];
        const uint32_t *lv_next = L->live[i + 1];
        const uint32_t  n_next  = L->n_live[i + 1];
        const uint32_t  n_i     = L->n_live[i];
        const size_t    base_n  = off[i + 1] / bbins;
        for (uint32_t k = 0; k < n_next; ++k) sn[lv_next[k]] = k;

        /* Accumulate every threshold's mass through the edges of this layer. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(beam_threads())
#endif
        for (int64_t c = 0; c < (int64_t)BEAM_SUM_CHUNKS; ++c) {
            double *part = chunk + (size_t)c * part_sz;
            for (size_t j = 0; j < part_sz; ++j) part[j] = 0.0;
            const uint32_t klo = (uint32_t)(((uint64_t)n_i * c) / BEAM_SUM_CHUNKS);
            const uint32_t khi =
                (uint32_t)(((uint64_t)n_i * (c + 1)) / BEAM_SUM_CHUNKS);
            for (uint32_t kk = klo; kk < khi; ++kk) {
                const BeamBinRange sr = rc[kk];
                if (sr.lo >= sr.hi) continue;
                const uint32_t s = L->live[i][kk];
                const double *src = cur + (size_t)kk * bins;
                const double *row = w + (size_t)s * R;
                const uint32_t rest = (s % SK) * A;
                for (uint32_t x = 0; x < R; ++x) {
                    if (row[x] == -INFINITY) continue;
                    const uint32_t t = rest + x;
                    if (!live_bit_get(mk_next, t)) continue;
                    const uint32_t kt = sn[t];
                    if (rng[base_n + kt].lo >= rng[base_n + kt].hi) continue;
                    const double *cum = pool + off[i + 1] + (size_t)kt * bbins;
                    const double fac = counting ? 1.0 : exp(row[x]);
                    const double e = -row[x];      /* the edge's surprisal */
                    for (uint32_t tt = 0; tt < n_thr; ++tt) {
                        /*
                         * Clamped before the cast, not after. A threshold far
                         * above the grid -- which is how a caller says "and
                         * everything rarer" -- makes this quotient enormous,
                         * and converting an out-of-range double to int64 is
                         * undefined rather than saturating. The ceiling is
                         * only ever compared against `jc - p` for `p < bins`,
                         * so `2 * bins` admits every backward bin at every
                         * prefix bin, which is what an unreachable threshold
                         * means.
                         */
                        const double jd =
                            ceil((thr[tt] - e) / ds) + (double)pad;
                        const int64_t jc =
                            jd >= (double)(2u * (uint64_t)bbins)
                                ? (int64_t)(2u * (uint64_t)bbins)
                                : (jd <= 0.0 ? 0 : (int64_t)jd);
                        part[(size_t)tt * R + x] +=
                            fac * below_threshold(src, cum, bbins, sr, jc);
                    }
                }
            }
        }
        for (uint32_t c = 0; c < BEAM_SUM_CHUNKS; ++c) {
            const double *part = chunk + (size_t)c * part_sz;
            for (uint32_t tt = 0; tt < n_thr; ++tt)
                for (uint32_t x = 0; x < R; ++x)
                    acc[((size_t)tt * len + i) * R + x] += part[(size_t)tt * R + x];
        }

        /* Advance the prefix grids, gathering by target exactly as
         * `spectrum.c` does and for the same reason: 21 sources land on every
         * target, so a source-major scatter would race. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(beam_threads())
#endif
        for (int64_t kk = 0; kk < (int64_t)n_next; ++kk) {
            rn[kk] = (BeamBinRange){ bins, 0 };
            const uint32_t tt = lv_next[kk];
            const uint32_t x = tt % A;
            if (x >= R) continue;
            const uint32_t rest = tt / A;
            double *dst = next + (size_t)kk * bins;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t s = a * SK + rest;
                if (!live_bit_get(mk_cur, s)) continue;
                const uint32_t ks = sc[s];
                if (rc[ks].lo >= rc[ks].hi) continue;
                const double wv = w[(size_t)s * R + x];
                if (wv == -INFINITY) continue;
                beam_grid_transport(cur + (size_t)ks * bins, dst, bins,
                                    -wv / ds, counting ? 1.0 : exp(wv),
                                    rc[ks], &rn[kk]);
            }
        }
        /* Clear only what was written, so a layer costs its occupied bins. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)n_i; ++k) {
            if (rc[k].lo >= rc[k].hi) continue;
            memset(cur + (size_t)k * bins + rc[k].lo, 0,
                   (size_t)(rc[k].hi - rc[k].lo) * sizeof(double));
        }
        { double *t = cur; cur = next; next = t; }
        { BeamBinRange *t = rc; rc = rn; rn = t; }
        { uint32_t *t = sc; sc = sn; sn = t; }
    }
    return BEAM_OK;
}


/* ── Entry point ─────────────────────────────────────────────── */

BeamError beam_model_positional_banded(const BeamModel *m, uint32_t length,
                                       bool counting, uint32_t bins,
                                       double spacing,
                                       const double *edges, uint32_t n_edges,
                                       double *out_marginals,
                                       double *out_band_mass,
                                       double *out_position_mass,
                                       double *out_spacing) {
    if (!m)              return BEAM_ENULL("model");
    if (!edges)          return BEAM_ENULL("band edges");
    if (!out_marginals)  return BEAM_ENULL("marginal buffer");
    if (!out_band_mass)  return BEAM_ENULL("band-mass buffer");
    if (!m->finalized)   return BEAM_ENOTBUILT();
    if (!beam_length_live(m, length)) return BEAM_EBADLEN(m, length);
    if (bins < 2)
        return BEAM_ERANGE("%s: a surprisal grid needs at least 2 bins, "
                           "got %u", __func__, (unsigned)bins);
    if (n_edges < 2)
        return BEAM_ERANGE("%s: %u band edge(s) define no band; two ascending "
                           "edges define one", __func__, (unsigned)n_edges);
    for (uint32_t k = 0; k < n_edges; ++k)
        if (!isfinite(edges[k]))
            return BEAM_ERANGE("%s: band edge %u is not finite; a band is "
                               "bounded in nats at both ends, so pass 0 and "
                               "bins*spacing to cover the whole grid",
                               __func__, (unsigned)k);
    for (uint32_t k = 1; k < n_edges; ++k)
        if (!(edges[k] > edges[k - 1u]))
            return BEAM_ERANGE("%s: band edges must ascend strictly, but "
                               "edge %u is %.17g and edge %u is %.17g",
                               __func__, (unsigned)(k - 1u), edges[k - 1u],
                               (unsigned)k, edges[k]);

    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t n_bands = n_edges - 1u;

    memset(out_marginals, 0, (size_t)n_bands * length * R * sizeof(double));
    memset(out_band_mass, 0, (size_t)n_bands * sizeof(double));
    if (out_position_mass)
        memset(out_position_mass, 0, (size_t)n_bands * length * sizeof(double));

    double ds = 0.0;
    BeamError err = beam_spectrum_spacing(m, bins, spacing, &ds);
    if (err != BEAM_OK) return err;
    if (out_spacing) *out_spacing = ds;

    LiveHandle lh;
    err = beam_live_acquire(m, length, &lh);
    if (err != BEAM_OK) return err;
    const BeamLive *L = lh.L;

    uint64_t slots = 0;
    uint32_t maxlive = 0;
    for (uint32_t i = 0; i <= length; ++i) {
        slots += L->n_live[i];
        if (L->n_live[i] > maxlive) maxlive = L->n_live[i];
    }
    if (maxlive == 0u) { beam_live_release(&lh); return BEAM_OK; }

    /*
     * The sink terms decide how far BELOW zero the backward grid has to
     * reach, and the pool cannot be sized until that is known -- so they are
     * built first, under their own refusal, and charged again in the figure
     * below. `beam_tail_weights` is one double per state and the pool is
     * gigabytes; this keeps "refuse before allocating" true of the thing that
     * can actually exhaust the machine.
     */
    const uint64_t tail_bytes = (uint64_t)S * sizeof(double);
    if (!beam_budget_fits(tail_bytes)) {
        beam_live_release(&lh);
        return beam_budget_refuse("the sink weights of a banded sweep",
                                  length, tail_bytes);
    }
    double *tail = (double *)malloc((size_t)tail_bytes);
    if (!tail) {
        beam_live_release(&lh);
        return BEAM_EALLOC("the sink weights");
    }
    beam_tail_weights(m, length, tail);

    /*
     * One whole bin past the lowest seed, so the lowest lands at or above bin
     * 0 rather than exactly on it -- linear splitting writes to `floor(u)` as
     * well as `floor(u)+1`, and a seed sitting precisely at 0 would still
     * spill one bin below.
     */
    uint32_t pad = 0;
    {
        const double sink = m->log_z[length] - m->log_pi[length];
        double lowest = 0.0;
        for (uint32_t k = 0; k < L->n_live[length]; ++k) {
            const uint32_t st = L->live[length][k];
            if (tail[st] == -INFINITY) continue;
            const double u = (-tail[st] + sink) / ds;
            if (u < lowest) lowest = u;
        }
        if (lowest < 0.0) {
            const double p = ceil(-lowest) + 1.0;
            if (p >= (double)UINT32_MAX) {
                free(tail);
                beam_live_release(&lh);
                return BEAM_ERANGE("%s: the sink terms sit %.3g nats below "
                                   "the grid origin, which no grid of this "
                                   "spacing can hold", __func__, -lowest * ds);
            }
            pad = (uint32_t)p;
        }
    }
    const uint32_t bbins = bins + pad;

    const uint64_t need =
        banded_bytes(m, bins, bbins, slots, maxlive, length, n_edges);
    if (!beam_budget_fits(need)) {
        free(tail);
        beam_live_release(&lh);
        return beam_budget_refuse("a banded positional-marginal sweep",
                                  length, need);
    }

    size_t        *off   = (size_t *)malloc((size_t)(length + 2u) * sizeof(size_t));
    double        *pool  = (double *)calloc((size_t)slots * bbins, sizeof(double));
    BeamBinRange  *rng   = (BeamBinRange *)malloc((size_t)slots * sizeof(BeamBinRange));
    double        *cur   = (double *)calloc((size_t)maxlive * bins, sizeof(double));
    double        *next  = (double *)calloc((size_t)maxlive * bins, sizeof(double));
    BeamBinRange  *rc    = (BeamBinRange *)malloc((size_t)maxlive * sizeof(BeamBinRange));
    BeamBinRange  *rn    = (BeamBinRange *)malloc((size_t)maxlive * sizeof(BeamBinRange));
    double        *w     = (double *)malloc((size_t)S * R * sizeof(double));
    uint32_t      *sc    = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint32_t      *sn    = (uint32_t *)malloc((size_t)S * sizeof(uint32_t));
    uint8_t       *av    = (uint8_t *)malloc((size_t)S);
    double        *acc   = (double *)calloc((size_t)n_edges * length * R, sizeof(double));
    double        *chunk = (double *)malloc((size_t)BEAM_SUM_CHUNKS * n_edges * R
                                            * sizeof(double));
    if (!off || !pool || !rng || !cur || !next || !rc || !rn || !w
        || !sc || !sn || !av || !acc || !chunk) {
        err = BEAM_EALLOC("the banded surprisal grids");
        goto done;
    }

    off[0] = 0;
    for (uint32_t i = 0; i <= length; ++i)
        off[i + 1u] = off[i] + (size_t)L->n_live[i] * bbins;

    backward_grids(m, length, counting, bbins, pad, ds, L, off, pool, rng,
                   w, tail, av, sc, sn);
    err = forward_accumulate(m, length, counting, bins, bbins, pad, ds, L, off,
                             pool, rng, edges, n_edges, acc, cur, next, rc, rn,
                             w, av, sc, sn, chunk);
    if (err != BEAM_OK) goto done;

    /*
     * Bands are differences of consecutive running totals. In exact
     * arithmetic that difference cannot be negative, the totals being
     * monotone in the threshold; in floating point it can be negative at the
     * last bits, and a negative probability is worse than a zero one.
     */
    for (uint32_t b = 0; b < n_bands; ++b) {
        for (uint32_t i = 0; i < length; ++i) {
            const double *lo = acc + ((size_t)b * length + i) * R;
            const double *hi = acc + ((size_t)(b + 1u) * length + i) * R;
            double *row = out_marginals + ((size_t)b * length + i) * R;
            double total = 0.0;
            for (uint32_t x = 0; x < R; ++x) {
                const double v = hi[x] - lo[x];
                row[x] = v > 0.0 ? v : 0.0;
                total += row[x];
            }
            if (out_position_mass)
                out_position_mass[(size_t)b * length + i] = total;
            /*
             * A band with no mass leaves its rows at zero rather than
             * refusing. One empty band must not deny the caller the others,
             * and `out_band_mass[b] == 0` already says the marginals there
             * describe nothing.
             */
            if (total > 0.0)
                for (uint32_t x = 0; x < R; ++x) row[x] /= total;
            /*
             * Position 0, and any other position would do: each is an
             * independent estimate of the same total and they agree to the
             * last bits, which `test_every_position_agrees_on_the_band_mass`
             * is what pins. Taking one rather than averaging keeps the
             * reported figure a number the sweep actually computed.
             */
            if (i == 0u) out_band_mass[b] = total;
        }
    }

done:
    free(off); free(pool); free(rng); free(cur); free(next); free(rc);
    free(rn); free(w); free(tail); free(sc); free(sn); free(av);
    free(acc); free(chunk);
    beam_live_release(&lh);
    return err;
}
