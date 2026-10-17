/**
 * @file jet.c
 * @brief The moment-jet forward sweep: one pass carrying r+1 moments.
 */
#include "beam/internal.h"
#include "beam/vmath.h"

#ifdef _WIN32
#include <malloc.h>   /* _aligned_malloc, _aligned_free */
#endif


/* ── Shared scratch for a forward sweep ──────────────────────── */

typedef struct {
    double *w;      /* [n_states * R] edge weights of the current layer  */
    double *tail;   /* [n_states]      sink weights                      */
    double *cur;    /* [(order+1) * n_states] jet, moment-major          */
    double *next;
} Sweep;

static void sweep_free(Sweep *s) {
    free(s->w); free(s->tail); free(s->cur); free(s->next);
}

static bool sweep_alloc(Sweep *s, uint32_t n_states, uint32_t n_residues,
                        uint32_t n_moments) {
    const size_t jet = (size_t)n_moments * n_states;
    s->w    = (double *)malloc((size_t)n_states * n_residues * sizeof(double));
    s->tail = (double *)malloc((size_t)n_states * sizeof(double));
    s->cur  = (double *)calloc(jet, sizeof(double));
    s->next = (double *)calloc(jet, sizeof(double));
    if (!s->w || !s->tail || !s->cur || !s->next) { sweep_free(s); return false; }
    return true;
}

/** Binomial coefficients up to order 16, enough for any sane moment order. */
void beam_binomials(double c[][BEAM_MAX_MOMENT + 1], uint32_t n) {
    for (uint32_t i = 0; i <= n; ++i) {
        c[i][0] = 1.0;
        for (uint32_t j = 1; j <= i; ++j)
            c[i][j] = (j == i) ? 1.0 : c[i - 1][j - 1] + c[i - 1][j];
        for (uint32_t j = i + 1; j <= n; ++j) c[i][j] = 0.0;
    }
}

/**
 * Forward jet sweep over one length.
 *
 * On return `out_moments[j]` is `sum over length-L paths of P(s)^q * raw(s)^j`
 * divided by the same sum at `j = 0`, i.e. `E_q[raw^j | L]`, and `*out_log_z`
 * is `ln sum over paths of exp(q * raw)`.
 *
 * `raw` is the accumulated edge weight, so `ln P(s) = c_L + raw(s)` with
 * `c_L = ln pi(L) - ln Z(L)`; the caller shifts by that constant.
 *
 * Numerics: the whole jet is rescaled by the same factor each layer, so the
 * moment ratios are untouched while the mass stays in range. That is why the
 * scale never has to be tracked for the moments, only for the partition
 * function.
 */
BeamError jet_sweep(const BeamModel *m, uint32_t len, double q,
                          uint32_t order, double *out_moments,
                          double *out_log_z) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t NM = order + 1u;

    double bc[BEAM_MAX_MOMENT + 1][BEAM_MAX_MOMENT + 1];
    beam_binomials(bc, order);

    /* Refused before allocating, from the sizing function beside the
     * allocation so the two cannot drift apart. */
    BEAM_BUDGET_OR_REFUSE("a jet sweep", len,
                          beam_jet_sweep_bytes(S, R, NM));

    Sweep sw = {0};
    if (!sweep_alloc(&sw, S, R, NM))
        return BEAM_EALLOC("the jet-sweep scratch buffers");

    /* Sweep the reached set, not the state space. This asks for no layer
     * maximum -- it computes its own below, over states carrying mass, which
     * is a subset of reached -- so the live weight builder is sound here.
     * Ascending order keeps every sum's terms in their original sequence. */
    uint8_t *alive = (uint8_t *)malloc(S);
    LiveHandle h;
    if (!alive || beam_live_acquire(m, len, &h) != BEAM_OK) {
        free(alive); sweep_free(&sw);
        return BEAM_EALLOC("the forward frontier");
    }
    const BeamLive *L = h.L;

    sw.cur[(size_t)0 * S + (S - 1)] = 1.0;    /* moment 0 of the start state */
    double log_scale = 0.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_reached_mask(L, i, S, alive);
        beam_layer_weights_live(m, i, len, sw.w, NULL, alive);

        /*
         * Rescale by the best edge THAT CARRIES MASS, not the best edge
         * anywhere in the layer. At a strong tilt the two can be far apart,
         * and scaling by an unreachable edge underflows every reachable one to
         * zero, which looks exactly like a length with no paths. That is what
         * made the saddlepoint fail near the ends of the support.
         */
        double wmax = -INFINITY;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:wmax) \
                num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)L->n_reached[i]; ++k) {
            const uint32_t s = L->reached[i][k];
            if (sw.cur[s] == 0.0) continue;
            const double *row = sw.w + (size_t)s * R;
            for (uint32_t x = 0; x < R; ++x)
                if (row[x] > wmax) wmax = row[x];
        }
        if (wmax == -INFINITY) { free(alive); beam_live_release(&h); sweep_free(&sw);
          *out_log_z = -INFINITY; return BEAM_OK; }
        const double qmax = beam_tilted_layer_max(sw.w, sw.cur, S, R, q,
                                                  wmax);
        if (qmax == -INFINITY) { free(alive); beam_live_release(&h);
          sweep_free(&sw); *out_log_z = -INFINITY; return BEAM_OK; }

        memset(sw.next, 0, (size_t)NM * S * sizeof(double));
        double nmax = 0.0;
        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax) \
                num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            const uint32_t t = tgt[k];
            const uint32_t x = t % A;
            if (x >= R) continue;
            const uint32_t rest = t / A;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                if (sw.cur[src] == 0.0) continue;          /* no mass here */
                const double wv = sw.w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                const double e = exp(q * wv - qmax);
                /* (raw + w)^j expands binomially over the carried moments. */
                for (uint32_t j = 0; j < NM; ++j) {
                    double acc = 0.0;
                    double wpow = 1.0;
                    for (uint32_t u = 0; u <= j; ++u) {
                        acc += bc[j][j - u] * wpow
                             * sw.cur[(size_t)(j - u) * S + src];
                        wpow *= wv;
                    }
                    sw.next[(size_t)j * S + t] += e * acc;
                }
            }
            if (sw.next[t] > nmax) nmax = sw.next[t];
        }
        log_scale += qmax;

        if (!(nmax > 0.0)) { free(alive); beam_live_release(&h); sweep_free(&sw);
          *out_log_z = -INFINITY; return BEAM_OK; }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            /* Moment 0 is the mass and carries the invariant; the higher
             * moments ride along and may legitimately be zero. */
            sw.next[tgt[k]] = beam_rescale(sw.next[tgt[k]], nmax);
            for (uint32_t j = 1; j < NM; ++j)
                sw.next[(size_t)j * S + tgt[k]] /= nmax;
        }
        log_scale += log(nmax);

        double *tmp = sw.cur; sw.cur = sw.next; sw.next = tmp;
    }

    /* Sink: one more edge whose weight is the tail. */
    beam_tail_weights(m, len, sw.tail);
    double tmax = -INFINITY;
    for (uint32_t k = 0; k < L->n_reached[len]; ++k) {
        const uint32_t s = L->reached[len][k];
        if (sw.cur[s] > 0.0 && sw.tail[s] != -INFINITY && q * sw.tail[s] > tmax)
            tmax = q * sw.tail[s];
    }
    if (tmax == -INFINITY) { free(alive); beam_live_release(&h); sweep_free(&sw);
          *out_log_z = -INFINITY; return BEAM_OK; }

    double *fin = (double *)calloc(NM, sizeof(double));
    if (!fin) { free(alive); beam_live_release(&h); sweep_free(&sw);
                return BEAM_EALLOC("the sink jet"); }
    for (uint32_t k = 0; k < L->n_reached[len]; ++k) {
        const uint32_t s = L->reached[len][k];
        if (sw.cur[s] == 0.0 || sw.tail[s] == -INFINITY) continue;
        const double tv = sw.tail[s];
        const double e = exp(q * tv - tmax);
        for (uint32_t j = 0; j < NM; ++j) {
            double acc = 0.0, wpow = 1.0;
            for (uint32_t u = 0; u <= j; ++u) {
                acc += bc[j][j - u] * wpow * sw.cur[(size_t)(j - u) * S + s];
                wpow *= tv;
            }
            fin[j] += e * acc;
        }
    }
    if (!(fin[0] > 0.0)) {
        free(fin); free(alive); beam_live_release(&h); sweep_free(&sw);
        *out_log_z = -INFINITY;
        return BEAM_OK;
    }
    for (uint32_t j = 0; j < NM; ++j) out_moments[j] = fin[j] / fin[0];
    *out_log_z = log_scale + tmax + log(fin[0]);

    free(fin); free(alive);
    beam_live_release(&h);
    sweep_free(&sw);
    return BEAM_OK;
}


/* ── Many tilts in one pass ──────────────────────────────────── */

/**
 * `jet_sweep` at `order = 0`, for `nq` tilts in one traversal of the
 * length-`len` trellis where there used to be `nq` traversals.
 *
 * `out_log_z[j * out_stride]` receives exactly what
 * `jet_sweep(m, len, qs[j], 0u, ..., &lz)` puts in `lz`: the same bits, not
 * the same number to rounding. That is the contract, and the Mellin
 * transform's many-order form is tested against it with `==`.
 *
 * WHAT IS SHARED is everything that does not depend on the tilt: the
 * frontier, the layer of edge weights, the reached mask and the sink weights.
 * Building the layer is the part of a single-q sweep that is not an `exp`, and
 * a 31-order Hill profile used to build every layer of every length 31 times
 * over one unchanging graph.
 *
 * WHAT IS NOT SHARED is anything the rescaling touches. Each tilt has its own
 * dynamic range -- at q = 3 a layer's mass spans three times the nats it does
 * at q = 1 -- so each carries its own mass plane, its own layer maximum, its
 * own `nmax` and its own running log-scale. One shared rescale would be
 * cheaper, and it would move the last bits of every q but one.
 *
 * WHY THE BITS SURVIVE. At order 0 the single-q jet's inner sum is
 * `acc = 0.0 + (1.0 * 1.0) * cur[src]`, which is `cur[src]` exactly, so its
 * update is `next[t] += exp(q * w - qmax) * cur[src]` over sources in
 * ascending `a`. Every tilt here performs that same sequence of operations on
 * its own plane, in the same order. The mass test (`cur == 0.0`), the layer
 * maximum over the states carrying THIS tilt's mass, the negative-q minimum
 * `beam_tilted_layer_max` takes, and the order in which the log-scale receives
 * `qmax` and then `log(nmax)` are all taken per tilt. Maxima and minima are
 * exact in whatever order they are reduced; sums are the one thing that is
 * not, and every sum here is taken by one thread in index order.
 *
 * A tilt that dies -- no mass left, which `jet_sweep` reports by returning
 * `-inf` at that layer -- is dropped from the active list there and reported
 * `-inf`, and the others carry on. Dropping it is not only a saving: a tilt
 * whose `qmax` overflowed to `-inf` would otherwise feed `exp(-inf - -inf)`,
 * a NaN, into its own plane.
 *
 * LAYOUT is state-major, `cur[state * nq + j]`. The tilts of one state share
 * a cache line, so the loop over tilts reads and writes contiguously where a
 * tilt-major layout would take one miss per tilt per edge.
 *
 * `next` is cleared by the states the previous layer reached, as
 * `tilted_log_z` in math/diversity.c does, rather than by `jet_sweep`'s
 * memset of the whole plane. That memset would be `nq * S * 8` bytes per
 * layer -- 48 MB at 31 tilts on an order-4 amino-acid model -- and on a
 * per-repertoire model nearly all of it is already zero. Both leave the same
 * contents: only reached targets are ever written.
 */
BeamError jet_sweep_fan(const BeamModel *m, uint32_t len, const double *qs,
                        uint32_t nq, double *out_log_z, size_t out_stride) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t W = nq;

    for (uint32_t j = 0; j < W; ++j) out_log_z[(size_t)j * out_stride] = -INFINITY;
    if (W == 0u) return BEAM_OK;

    /* Refused before allocating, from the sizing function beside the
     * allocation so the two cannot drift apart. */
    BEAM_BUDGET_OR_REFUSE("a fused tilted sweep", len,
                          beam_jet_fan_bytes(S, R, W));

    double  *w     = (double *)malloc((size_t)S * R * sizeof(double));
    double  *tail  = (double *)malloc((size_t)S * sizeof(double));
    double  *cur   = (double *)calloc((size_t)S * W, sizeof(double));
    double  *next  = (double *)calloc((size_t)S * W, sizeof(double));
    uint8_t *alive = (uint8_t *)malloc(S);
    /* Seven scalars per tilt, and the list of tilts still alive. */
    double   *pt  = (double *)malloc((size_t)W * 7u * sizeof(double));
    uint32_t *act = (uint32_t *)malloc((size_t)W * sizeof(uint32_t));
    LiveHandle h;
    if (!w || !tail || !cur || !next || !alive || !pt || !act
        || beam_live_acquire(m, len, &h) != BEAM_OK) {
        free(w); free(tail); free(cur); free(next); free(alive);
        free(pt); free(act);
        return BEAM_EALLOC("the fused tilted-sweep buffers");
    }
    const BeamLive *L = h.L;
    double *scale = pt;             /* running log-scale, jet_sweep's log_scale */
    double *qmax  = pt + 1u * W;
    double *hi    = pt + 2u * W;    /* layer max over this tilt's mass          */
    double *lo    = pt + 3u * W;    /* layer min, which only q < 0 reads        */
    double *nmax  = pt + 4u * W;
    double *tmax  = pt + 5u * W;
    double *fin   = pt + 6u * W;

    uint32_t nact = W;
    for (uint32_t j = 0; j < W; ++j) {
        act[j] = j;
        scale[j] = 0.0;
        cur[(size_t)(S - 1) * W + j] = 1.0;     /* the start state, every tilt */
    }

    for (uint32_t i = 0; i < len; ++i) {
        beam_reached_mask(L, i, S, alive);
        beam_layer_weights_live(m, i, len, w, NULL, alive);

        /* Each tilt's layer extremes over the states carrying ITS mass. A
         * row's own max and min do not depend on the tilt, so they are found
         * once and offered to every tilt with mass there; the max of maxima
         * is the max, bit for bit. `lo` is `beam_tilted_layer_max`'s scan for
         * q < 0 -- finite entries only -- restricted to reached states, which
         * are the only ones whose mass is nonzero. */
        for (uint32_t b = 0; b < nact; ++b) {
            hi[act[b]] = -INFINITY;
            lo[act[b]] = INFINITY;
        }
        const uint32_t  nr = L->n_reached[i];
        const uint32_t *rv = L->reached[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:hi[:W]) \
                reduction(min:lo[:W]) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nr; ++k) {
            const uint32_t s = rv[k];
            const double *row = w + (size_t)s * R;
            const double *c = cur + (size_t)s * W;
            double rmax = -INFINITY, rmin = INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] > rmax) rmax = row[x];
                if (row[x] != -INFINITY && row[x] < rmin) rmin = row[x];
            }
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                if (c[j] == 0.0) continue;
                if (rmax > hi[j]) hi[j] = rmax;
                if (rmin < lo[j]) lo[j] = rmin;
            }
        }
        /* `jet_sweep` returns -inf at exactly these two points: no edge that
         * carries mass, and a tilted maximum that is -inf. */
        uint32_t keep = 0;
        for (uint32_t b = 0; b < nact; ++b) {
            const uint32_t j = act[b];
            if (hi[j] == -INFINITY) continue;
            const double qm = qs[j] >= 0.0 ? qs[j] * hi[j]
                            : (lo[j] == INFINITY ? -INFINITY : qs[j] * lo[j]);
            if (qm == -INFINITY) continue;
            qmax[j] = qm;
            act[keep++] = j;
        }
        nact = keep;
        if (nact == 0u) goto done;

        /* By TARGET, as in `jet_sweep`: each target row is the private
         * property of one thread, and receives its sources in ascending `a`.
         * `alive` is the reached set of this layer, which is exactly where
         * `cur` can be nonzero, so it skips a source for every tilt at once
         * without reading a row the live builder did not build. */
        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
        for (uint32_t b = 0; b < nact; ++b) nmax[act[b]] = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax[:W]) \
                num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            const uint32_t t = tgt[k];
            const uint32_t x = t % A;
            if (x >= R) continue;                  /* its row stays zero */
            const uint32_t rest = t / A;
            double *nx = next + (size_t)t * W;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                if (!alive[src]) continue;         /* no mass, any tilt */
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                const double *c = cur + (size_t)src * W;
                for (uint32_t b = 0; b < nact; ++b) {
                    const uint32_t j = act[b];
                    if (c[j] == 0.0) continue;     /* no mass, this tilt */
                    nx[j] += exp(qs[j] * wv - qmax[j]) * c[j];
                }
            }
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                if (nx[j] > nmax[j]) nmax[j] = nx[j];
            }
        }
        /* `qmax` enters the log-scale before the empty-layer test, as in
         * `jet_sweep`; for a tilt that dies here it no longer matters. */
        keep = 0;
        for (uint32_t b = 0; b < nact; ++b) {
            const uint32_t j = act[b];
            scale[j] += qmax[j];
            if (!(nmax[j] > 0.0)) continue;
            act[keep++] = j;
        }
        nact = keep;
        if (nact == 0u) goto done;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            double *nx = next + (size_t)tgt[k] * W;
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                nx[j] = beam_rescale(nx[j], nmax[j]);
            }
        }
        for (uint32_t b = 0; b < nact; ++b)
            scale[act[b]] += log(nmax[act[b]]);

        /* `next` becomes `cur`, and the old `cur` is reused as the next
         * layer's `next`. Its only nonzero rows are this layer's reached
         * states -- every tilt's, dead ones included -- so clearing those
         * restores an all-zero plane. */
        for (uint32_t k = 0; k < nr; ++k)
            memset(cur + (size_t)rv[k] * W, 0, (size_t)W * sizeof(double));
        double *tmp = cur; cur = next; next = tmp;
    }

    /* Sink: one more edge whose weight is the tail, per tilt exactly as
     * `jet_sweep` takes it. */
    beam_tail_weights(m, len, tail);
    {
        const uint32_t  ns = L->n_reached[len];
        const uint32_t *sv = L->reached[len];
        for (uint32_t b = 0; b < nact; ++b) {
            tmax[act[b]] = -INFINITY;
            fin[act[b]] = 0.0;
        }
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            const double tv = tail[s];
            if (tv == -INFINITY) continue;
            const double *c = cur + (size_t)s * W;
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                if (c[j] > 0.0 && qs[j] * tv > tmax[j]) tmax[j] = qs[j] * tv;
            }
        }
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            const double tv = tail[s];
            if (tv == -INFINITY) continue;
            const double *c = cur + (size_t)s * W;
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                if (c[j] == 0.0 || tmax[j] == -INFINITY) continue;
                fin[j] += exp(qs[j] * tv - tmax[j]) * c[j];
            }
        }
        for (uint32_t b = 0; b < nact; ++b) {
            const uint32_t j = act[b];
            if (tmax[j] == -INFINITY || !(fin[j] > 0.0)) continue;
            out_log_z[(size_t)j * out_stride] = scale[j] + tmax[j] + log(fin[j]);
        }
    }

done:
    free(w); free(tail); free(cur); free(next); free(alive);
    free(pt); free(act);
    beam_live_release(&h);
    return BEAM_OK;
}


/* ── Complex tilts ───────────────────────────────────────────── */

/**
 * `jet_sweep_fan` at COMPLEX tilts `q = q_re + i q_im`: the per-length
 * `ln sum over paths of exp(q * raw)`, returned as its modulus in log form
 * and its phase, `out_log_mod[j * out_stride]` and `out_phase[j * out_stride]`.
 *
 * WHY. `P(s)^q` factors along the path at ANY q, so the power sum
 * `M(q) = sum_s P(s)^q` is an exact DP everywhere in the complex plane, and on
 * a vertical line it is what a Mellin-Barnes integral needs: expected
 * richness, occupancy counts and pairwise sharing are 1-D contour integrals of
 * M, with no spectrum and no grid.
 *
 * THE SAME SWEEP AS `jet_sweep_fan`, with the mass planes complex. Each edge
 * multiplies by `exp(q_re * w - qmax) * exp(i q_im * w)`: the modulus is the
 * real tilt's, so `qmax` -- the per-layer shift that keeps it in range -- is
 * taken from `q_re` exactly as there, and the phase rides in the complex
 * accumulator, which needs no range control. Each layer is rescaled by its
 * largest component magnitude, `max(|re|, |im|)`, which is within a factor
 * sqrt(2) of the largest modulus and costs no square root.
 *
 * WHAT THE PHASES COST. Paths with different `raw` arrive with different
 * phases and cancel, so `|M(q)|` can sit far below `M(q_re)`, the sum of the
 * moduli. The arithmetic is exact to rounding in the moduli; relative to
 * `|M(q)|` it loses what the cancellation takes. A caller integrating along a
 * line is exposed to it only where the integrand is small.
 *
 * DETERMINISM is kept as in `jet_sweep_fan`: every sum is taken by one thread
 * in ascending source order, maxima are exact in any order, and each tilt's
 * sequence of operations does not depend on the others, so a tilt computed
 * alone and in company gives the same bits.
 */
BeamError jet_sweep_cfan(const BeamModel *m, uint32_t len, const double *q_re,
                         const double *q_im, uint32_t nq, double *out_log_mod,
                         double *out_phase, size_t out_stride) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t W = nq;

    for (uint32_t j = 0; j < W; ++j) {
        out_log_mod[(size_t)j * out_stride] = -INFINITY;
        out_phase[(size_t)j * out_stride] = 0.0;
    }
    if (W == 0u) return BEAM_OK;

    BEAM_BUDGET_OR_REFUSE("a complex tilted sweep", len,
                          beam_jet_cfan_bytes(S, R, W));

    double  *w     = (double *)malloc((size_t)S * R * sizeof(double));
    double  *tail  = (double *)malloc((size_t)S * sizeof(double));
    /* re and im interleaved per tilt: [state][tilt][2] */
    double  *cur   = (double *)calloc((size_t)S * W * 2u, sizeof(double));
    double  *next  = (double *)calloc((size_t)S * W * 2u, sizeof(double));
    uint8_t *alive = (uint8_t *)malloc(S);
    double   *pt  = (double *)malloc((size_t)W * 8u * sizeof(double));
    uint32_t *act = (uint32_t *)malloc((size_t)W * sizeof(uint32_t));
    LiveHandle h;
    if (!w || !tail || !cur || !next || !alive || !pt || !act
        || beam_live_acquire(m, len, &h) != BEAM_OK) {
        free(w); free(tail); free(cur); free(next); free(alive);
        free(pt); free(act);
        return BEAM_EALLOC("the complex tilted-sweep buffers");
    }
    const BeamLive *L = h.L;
    double *scale = pt;
    double *qmax  = pt + 1u * W;
    double *hi    = pt + 2u * W;
    double *lo    = pt + 3u * W;
    double *nmax  = pt + 4u * W;
    double *tmax  = pt + 5u * W;
    double *fre   = pt + 6u * W;
    double *fim   = pt + 7u * W;
    const size_t W2 = (size_t)W * 2u;

    uint32_t nact = W;
    for (uint32_t j = 0; j < W; ++j) {
        act[j] = j;
        scale[j] = 0.0;
        cur[(size_t)(S - 1) * W2 + 2u * j] = 1.0;   /* start state, real 1 */
    }

    for (uint32_t i = 0; i < len; ++i) {
        beam_reached_mask(L, i, S, alive);
        beam_layer_weights_live(m, i, len, w, NULL, alive);

        for (uint32_t b = 0; b < nact; ++b) {
            hi[act[b]] = -INFINITY;
            lo[act[b]] = INFINITY;
        }
        const uint32_t  nr = L->n_reached[i];
        const uint32_t *rv = L->reached[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:hi[:W]) \
                reduction(min:lo[:W]) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nr; ++k) {
            const uint32_t s = rv[k];
            const double *row = w + (size_t)s * R;
            const double *c = cur + (size_t)s * W2;
            double rmax = -INFINITY, rmin = INFINITY;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] > rmax) rmax = row[x];
                if (row[x] != -INFINITY && row[x] < rmin) rmin = row[x];
            }
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                if (c[2u * j] == 0.0 && c[2u * j + 1u] == 0.0) continue;
                if (rmax > hi[j]) hi[j] = rmax;
                if (rmin < lo[j]) lo[j] = rmin;
            }
        }
        uint32_t keep = 0;
        for (uint32_t b = 0; b < nact; ++b) {
            const uint32_t j = act[b];
            if (hi[j] == -INFINITY) continue;
            const double qm = q_re[j] >= 0.0 ? q_re[j] * hi[j]
                            : (lo[j] == INFINITY ? -INFINITY : q_re[j] * lo[j]);
            if (qm == -INFINITY) continue;
            qmax[j] = qm;
            act[keep++] = j;
        }
        nact = keep;
        if (nact == 0u) goto done;

        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
        for (uint32_t b = 0; b < nact; ++b) nmax[act[b]] = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:nmax[:W]) \
                num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            const uint32_t t = tgt[k];
            const uint32_t x = t % A;
            if (x >= R) continue;
            const uint32_t rest = t / A;
            double *nx = next + (size_t)t * W2;
            for (uint32_t a = 0; a < A; ++a) {
                const uint32_t src = a * SK + rest;
                if (!alive[src]) continue;
                const double wv = w[(size_t)src * R + x];
                if (wv == -INFINITY) continue;
                const double *c = cur + (size_t)src * W2;
                /* Orders that share a real part share their layer shift, and
                 * then `exp(q_re * w - qmax)` is the same number computed from
                 * the same inputs: it is taken once per edge and reused. The
                 * bits are exactly those of computing it per order. */
                double last_re = NAN, last_qm = NAN, mag_c = 0.0;
                for (uint32_t b = 0; b < nact; ++b) {
                    const uint32_t j = act[b];
                    const double cr = c[2u * j], ci = c[2u * j + 1u];
                    if (cr == 0.0 && ci == 0.0) continue;
                    if (!(q_re[j] == last_re && qmax[j] == last_qm)) {
                        last_re = q_re[j]; last_qm = qmax[j];
                        mag_c = exp(q_re[j] * wv - qmax[j]);
                    }
                    const double mag = mag_c;
                    const double ang = q_im[j] * wv;
                    const double er = mag * cos(ang), ei = mag * sin(ang);
                    nx[2u * j]      += er * cr - ei * ci;
                    nx[2u * j + 1u] += er * ci + ei * cr;
                }
            }
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                const double v = fmax(fabs(nx[2u * j]), fabs(nx[2u * j + 1u]));
                if (v > nmax[j]) nmax[j] = v;
            }
        }
        keep = 0;
        for (uint32_t b = 0; b < nact; ++b) {
            const uint32_t j = act[b];
            scale[j] += qmax[j];
            if (!(nmax[j] > 0.0)) continue;
            act[keep++] = j;
        }
        nact = keep;
        if (nact == 0u) goto done;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            double *nx = next + (size_t)tgt[k] * W2;
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                nx[2u * j]      = beam_rescale(nx[2u * j], nmax[j]);
                nx[2u * j + 1u] = beam_rescale(nx[2u * j + 1u], nmax[j]);
            }
        }
        for (uint32_t b = 0; b < nact; ++b)
            scale[act[b]] += log(nmax[act[b]]);

        for (uint32_t k = 0; k < nr; ++k)
            memset(cur + (size_t)rv[k] * W2, 0, W2 * sizeof(double));
        double *tmp = cur; cur = next; next = tmp;
    }

    /* Sink: one more edge whose weight is the tail. */
    beam_tail_weights(m, len, tail);
    {
        const uint32_t  ns = L->n_reached[len];
        const uint32_t *sv = L->reached[len];
        for (uint32_t b = 0; b < nact; ++b) {
            tmax[act[b]] = -INFINITY;
            fre[act[b]] = 0.0;
            fim[act[b]] = 0.0;
        }
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            const double tv = tail[s];
            if (tv == -INFINITY) continue;
            const double *c = cur + (size_t)s * W2;
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                if (c[2u * j] == 0.0 && c[2u * j + 1u] == 0.0) continue;
                if (q_re[j] * tv > tmax[j]) tmax[j] = q_re[j] * tv;
            }
        }
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            const double tv = tail[s];
            if (tv == -INFINITY) continue;
            const double *c = cur + (size_t)s * W2;
            double last_re = NAN, last_tm = NAN, mag_c = 0.0;
            for (uint32_t b = 0; b < nact; ++b) {
                const uint32_t j = act[b];
                const double cr = c[2u * j], ci = c[2u * j + 1u];
                if ((cr == 0.0 && ci == 0.0) || tmax[j] == -INFINITY) continue;
                if (!(q_re[j] == last_re && tmax[j] == last_tm)) {
                    last_re = q_re[j]; last_tm = tmax[j];
                    mag_c = exp(q_re[j] * tv - tmax[j]);
                }
                const double mag = mag_c;
                const double ang = q_im[j] * tv;
                const double er = mag * cos(ang), ei = mag * sin(ang);
                fre[j] += er * cr - ei * ci;
                fim[j] += er * ci + ei * cr;
            }
        }
        for (uint32_t b = 0; b < nact; ++b) {
            const uint32_t j = act[b];
            if (tmax[j] == -INFINITY) continue;
            const double mod = hypot(fre[j], fim[j]);
            if (!(mod > 0.0)) continue;
            out_log_mod[(size_t)j * out_stride] = scale[j] + tmax[j] + log(mod);
            out_phase[(size_t)j * out_stride] = atan2(fim[j], fre[j]);
        }
    }

done:
    free(w); free(tail); free(cur); free(next); free(alive);
    free(pt); free(act);
    beam_live_release(&h);
    return BEAM_OK;
}


/* ── A vertical line of complex tilts ────────────────────────── */

/**
 * `jet_sweep_cfan` for the orders `re + i (im0 + g dim)`, `g = k0 .. k0+nq-1`:
 * a stretch of one vertical line, which is the contour a Mellin-Barnes
 * integral runs along. Output as `jet_sweep_cfan`'s, one entry per order.
 * `k0` must be a multiple of BEAM_LINE_LANES; `beam_line_plan` cuts stretches
 * that way. The call uses every thread itself: they share out each layer's
 * targets, so one stretch of one length keeps them all busy, and a stretch
 * can be as wide as the budget allows.
 *
 * WHAT THE LINE SHARES. Every order has the same real part, so the layer shift
 * and each edge's modulus `exp(re * w - qmax)` are one number for all of them,
 * taken once per edge. The shift comes from the REACHED states' weights,
 * which do not depend on any order, rather than from each order's mass as in
 * `jet_sweep_cfan`; on a reached state the two coincide.
 *
 * WHAT THE LINE RECURS. The rotation `exp(i (im0 + g dim) w)`, times the
 * edge's modulus, is taken DIRECTLY at every g that is a multiple of
 * BEAM_LINE_BLOCK (the anchor). Each edge carries a sixteen-lane pattern
 * `exp(i r dim w)`, r = 0 .. 15 -- lanes 1-3 each the one before times
 * `exp(i dim w)`, lanes 4-15 the one four before times `exp(4 i dim w)` --
 * so a block's first sixteen orders are its anchor times the pattern, and
 * every later order in the block is the one sixteen before times
 * `exp(16 i dim w)`, all three factors taken directly. No rotation is more
 * than 14 multiplies from direct values. The cosines and sines are taken for
 * many edges in one vector call (`beam_sincos_many`), the patterns once per
 * edge per layer.
 *
 * WHY SIXTEEN CHAINS. A multiply waits for the one before it in its chain, so
 * with only four chains the edge spent its time waiting (41% of a line's
 * time, measured on the foundation, against 25% on the multiply-add it
 * feeds). Sixteen chains keep both arithmetic units busy, and the rotations
 * go straight from registers into the multiply-add rather than through a
 * buffer.
 *
 * GROUPS AND TILES. The targets that differ only in their newest residue have
 * the same A sources, so a thread takes such a group whole and walks the
 * stretch a tile of LINE_TILE orders at a time: the group's source tiles stay
 * in cache while each of its targets reads them, and a target's tile of edges
 * accumulates in a cached row that is then scaled and written once. So the
 * planes are neither cleared between layers nor swept again to rescale, and a
 * group's moduli and fixed rotations are taken once for the whole stretch.
 *
 * THE SCALE IS A POWER OF TWO, CHOSEN A LAYER AHEAD. What keeps an order's
 * mass in range is dividing each layer by its largest entry. Here each layer
 * is instead multiplied, as it is written, by the power of two nearest the
 * reciprocal of the PREVIOUS layer's largest entry: a layer changes the scale
 * by at most the sum of its incoming moduli, so the mass stays within a few
 * binary orders of one, and a power of two changes no digit of it. The
 * divisions this replaces rounded every entry of every layer.
 *
 * LAYOUT. A state's row holds the real parts of all its orders and then the
 * imaginary parts, each padded to a multiple of sixteen orders so the inner
 * loop runs whole chains with no branch, and each followed by one cache line
 * so that no two rows, and no row's two halves, sit a power of two apart
 * (`beam_line_row`). A padded order starts at zero and stays zero; it is
 * never read out. The next layer's rows are written with non-temporal stores
 * where the CPU has them: nothing reads a row until the layer is done.
 *
 * ONE WORKSPACE A LINE. The planes and every thread's scratch are made once
 * per line (`beam_line_work_new`) and reused by each (length, stretch).
 *
 * WHY THAT MATTERS: an order's value depends on the line and on g alone --
 * not on k0, nq, the thread count or the budget, all of which only decide
 * which stretch, tile or thread it is computed in. Each target's edges are
 * summed in a fixed order by one thread, the largest entry is a maximum, and
 * the sink's sum is taken over fixed chunks of end states and then over the
 * chunks in order. It agrees with `jet_sweep_cfan` to rounding, not to the
 * bit: the rotation is the same number reached by a different sequence of
 * operations.
 */
#define BEAM_LN2        0.693147180559945309417232121458176568
#define BEAM_LINE_VEC   4u
#define BEAM_LINE_NV    (BEAM_LINE_LANES / BEAM_LINE_VEC)
#define BEAM_LINE_WIDE  8u                       /* the multiply-add's vectors */
#define BEAM_LINE_NW    (BEAM_LINE_LANES / BEAM_LINE_WIDE)
_Static_assert(BEAM_LINE_NW == 2, "line_edge_body holds two eight-lane chains by name");
#define LINE_ROT_FIXED  3u     /* exp(i dim w), exp(4 i dim w), exp(16 i dim w) */

typedef double beam_v4d __attribute__((vector_size(32)));
typedef long long beam_v4i __attribute__((vector_size(32)));
/* Eight lanes: one AVX-512 register, two AVX2 ones, four SSE2 ones -- each
 * lane the same arithmetic whichever the CPU runs. */
typedef double beam_v8d __attribute__((vector_size(64)));
#define V8_LOAD(dst, p)  memcpy(&(dst), (p), sizeof(beam_v8d))
#define V8_STORE(p, src) memcpy((p), &(src), sizeof(beam_v8d))

/* Unaligned whole-vector access, as macros: a function passing a vector by
 * value changes the ABI between builds with and without AVX. */
#define V4_LOAD(dst, p)  memcpy(&(dst), (p), sizeof(beam_v4d))
#define V4_STORE(p, src) memcpy((p), &(src), sizeof(beam_v4d))

/* The fixed rotations of `ne` edges with weights `w`: angles [3e .. 3e+2] are
 * dim w, 4 dim w and 16 dim w, formed exactly as they always were. */
static void line_fixed(const double *w, uint32_t ne, double dim, double *ang,
                       double *sn, double *cs) {
    for (uint32_t e = 0; e < ne; ++e) {
        ang[3u * e + 0u] = dim * w[e];
        ang[3u * e + 1u] = 4.0 * dim * w[e];
        ang[3u * e + 2u] = (double)BEAM_LINE_LANES * dim * w[e];
    }
    beam_sincos_many(ang, sn, cs, (size_t)ne * LINE_ROT_FIXED);
}

/* The anchors of `ne` edges for `nb` blocks from the one at order a0:
 * [e nb + b] = (im0 + (a0 + b BLOCK) dim) w. */
static void line_anchors(const double *w, uint32_t ne, double im0, double dim,
                         uint64_t a0, uint32_t nb, double *ang, double *sn,
                         double *cs) {
    for (uint32_t e = 0; e < ne; ++e)
        for (uint32_t b = 0; b < nb; ++b)
            ang[(size_t)e * nb + b] =
                (im0 + (double)(a0 + (uint64_t)b * BEAM_LINE_BLOCK) * dim) * w[e];
    beam_sincos_many(ang, sn, cs, (size_t)ne * nb);
}

/* Each edge's sixteen-lane pattern exp(i r dim w), r = 0 .. 15, from its
 * fixed rotations: lanes 1-3 are the one before times exp(i dim w), lanes
 * 4-15 the one four before times exp(4 i dim w). Taken once per edge per
 * layer, so a block's sixteen seeds are one vector product with its anchor
 * rather than a chain of them per edge per block. */
static void line_patterns(uint32_t ne, const double *fc, const double *fs,
                          double *pr, double *pi) {
    for (uint32_t e = 0; e < ne; ++e) {
        const double d1r = fc[LINE_ROT_FIXED * e], d1i = fs[LINE_ROT_FIXED * e];
        const double d4r = fc[LINE_ROT_FIXED * e + 1u], d4i = fs[LINE_ROT_FIXED * e + 1u];
        double *R = pr + (size_t)BEAM_LINE_LANES * e, *I = pi + (size_t)BEAM_LINE_LANES * e;
        R[0] = 1.0; I[0] = 0.0;
        for (uint32_t r = 1; r < 4u; ++r) {
            R[r] = R[r - 1] * d1r - I[r - 1] * d1i;
            I[r] = R[r - 1] * d1i + I[r - 1] * d1r;
        }
        for (uint32_t r = 4; r < BEAM_LINE_LANES; ++r) {
            R[r] = R[r - 4] * d4r - I[r - 4] * d4i;
            I[r] = R[r - 4] * d4i + I[r - 4] * d4r;
        }
    }
}

/* acc (= if first, else +=) mag exp(i (im0 + g dim) w) c over the tile of T
 * orders from g0 (both multiples of BEAM_LINE_LANES): each block's sixteen
 * seeds are its anchor, times the modulus, times the edge's pattern; then
 * every step multiplies each lane by exp(16 i dim w) = (dLr, dLi). `ac`/`as`
 * start at the block that holds g0; `cre`, `cim`, `are`, `aim` at the tile's
 * first order. Each lane performs exactly this recurrence whichever tile,
 * stretch or thread it falls in. */
__attribute__((always_inline))
static inline void line_edge_body(double mag, double dLr, double dLi,
                                  const double *restrict pr,
                                  const double *restrict pi,
                                  const double *restrict ac,
                                  const double *restrict as, uint64_t g0,
                                  uint32_t T, const double *restrict cre,
                                  const double *restrict cim,
                                  double *restrict are, double *restrict aim,
                                  int first) {
    beam_v8d p0r, p1r, p0i, p1i;
    V8_LOAD(p0r, pr); V8_LOAD(p1r, pr + BEAM_LINE_WIDE);
    V8_LOAD(p0i, pi); V8_LOAD(p1i, pi + BEAM_LINE_WIDE);
    const uint64_t end = g0 + T;
    uint32_t b = 0;
    for (uint64_t a = g0 - g0 % BEAM_LINE_BLOCK; a < end; a += BEAM_LINE_BLOCK, ++b) {
        const double Ar = mag * ac[b], Ai = mag * as[b];
        /* Two eight-lane chains, held by name so they stay in registers. */
        beam_v8d r0 = p0r * Ar - p0i * Ai, i0 = p0r * Ai + p0i * Ar;
        beam_v8d r1 = p1r * Ar - p1i * Ai, i1 = p1r * Ai + p1i * Ar;
        uint64_t g = a;
        const uint64_t stop = a + BEAM_LINE_BLOCK < end ? a + BEAM_LINE_BLOCK : end;
        for (; g < g0; g += BEAM_LINE_LANES) {            /* a tile mid-block */
            const beam_v8d t0 = r0 * dLr - i0 * dLi, t1 = r1 * dLr - i1 * dLi;
            i0 = r0 * dLi + i0 * dLr;
            i1 = r1 * dLi + i1 * dLr;
            r0 = t0; r1 = t1;
        }
        for (; g < stop; g += BEAM_LINE_LANES) {
            const size_t o = (size_t)(g - g0);
            beam_v8d c0r, c0i, c1r, c1i, a0r, a0i, a1r, a1i;
            V8_LOAD(c0r, cre + o); V8_LOAD(c0i, cim + o);
            V8_LOAD(c1r, cre + o + BEAM_LINE_WIDE); V8_LOAD(c1i, cim + o + BEAM_LINE_WIDE);
            const beam_v8d u0r = r0 * c0r - i0 * c0i, u0i = r0 * c0i + i0 * c0r;
            const beam_v8d u1r = r1 * c1r - i1 * c1i, u1i = r1 * c1i + i1 * c1r;
            if (first) {
                a0r = u0r; a0i = u0i; a1r = u1r; a1i = u1i;
            } else {
                V8_LOAD(a0r, are + o); V8_LOAD(a0i, aim + o);
                V8_LOAD(a1r, are + o + BEAM_LINE_WIDE); V8_LOAD(a1i, aim + o + BEAM_LINE_WIDE);
                a0r += u0r; a0i += u0i; a1r += u1r; a1i += u1i;
            }
            V8_STORE(are + o, a0r); V8_STORE(aim + o, a0i);
            V8_STORE(are + o + BEAM_LINE_WIDE, a1r); V8_STORE(aim + o + BEAM_LINE_WIDE, a1i);
            const beam_v8d t0 = r0 * dLr - i0 * dLi, t1 = r1 * dLr - i1 * dLi;
            i0 = r0 * dLi + i0 * dLr;
            i1 = r1 * dLi + i1 * dLr;
            r0 = t0; r1 = t1;
        }
    }
}

/* line_edge_body eight lanes at a time, as two four-lane chains, for CPUs
 * with sixteen 256-bit registers. There, each eight-lane chain above is two
 * registers, and the two chains with their rotation, sources and accumulators
 * want about twice the sixteen that exist: the compiler keeps them on the
 * stack and reloads them every step -- on an AVX2 desktop 57% of the edge
 * loop's samples were those spills, against 23% on arithmetic. Half the
 * lanes at a time fits; one four-lane chain alone would, but then the core
 * waits on its multiply-subtract recurrence, and two hide each other's
 * latency (on the foundation, 384 orders: 64 s as above, 49 s with one
 * chain, 43 s with two).
 *
 * Every lane performs the same operations in the same order as there --
 * the same products, the same differences and sums, the accumulator added
 * in the same edge order -- only which lanes are in flight together
 * differs, so the values are the same to the bit. */
__attribute__((always_inline))
static inline void line_edge_body4(double mag, double dLr, double dLi,
                                   const double *restrict pr,
                                   const double *restrict pi,
                                   const double *restrict ac,
                                   const double *restrict as, uint64_t g0,
                                   uint32_t T, const double *restrict cre,
                                   const double *restrict cim,
                                   double *restrict are, double *restrict aim,
                                   int first) {
    const uint64_t end = g0 + T;
    for (uint32_t l = 0; l < BEAM_LINE_LANES; l += 2u * BEAM_LINE_VEC) {
        beam_v4d par, pai, pbr, pbi;
        V4_LOAD(par, pr + l); V4_LOAD(pai, pi + l);
        V4_LOAD(pbr, pr + l + BEAM_LINE_VEC); V4_LOAD(pbi, pi + l + BEAM_LINE_VEC);
        uint32_t b = 0;
        for (uint64_t a = g0 - g0 % BEAM_LINE_BLOCK; a < end; a += BEAM_LINE_BLOCK, ++b) {
            const double Ar = mag * ac[b], Ai = mag * as[b];
            /* Two four-lane chains, held by name: two recurrences in flight
             * hide each other's latency, and still fit the registers. */
            beam_v4d ra = par * Ar - pai * Ai, ia = par * Ai + pai * Ar;
            beam_v4d rb = pbr * Ar - pbi * Ai, ib = pbr * Ai + pbi * Ar;
            uint64_t g = a;
            const uint64_t stop = a + BEAM_LINE_BLOCK < end ? a + BEAM_LINE_BLOCK : end;
            for (; g < g0; g += BEAM_LINE_LANES) {        /* a tile mid-block */
                const beam_v4d ta = ra * dLr - ia * dLi, tb = rb * dLr - ib * dLi;
                ia = ra * dLi + ia * dLr;
                ib = rb * dLi + ib * dLr;
                ra = ta; rb = tb;
            }
            for (; g < stop; g += BEAM_LINE_LANES) {
                const size_t o = (size_t)(g - g0) + l, ob = o + BEAM_LINE_VEC;
                beam_v4d car, cai, cbr, cbi, xar, xai, xbr, xbi;
                V4_LOAD(car, cre + o); V4_LOAD(cai, cim + o);
                V4_LOAD(cbr, cre + ob); V4_LOAD(cbi, cim + ob);
                const beam_v4d uar = ra * car - ia * cai, uai = ra * cai + ia * car;
                const beam_v4d ubr = rb * cbr - ib * cbi, ubi = rb * cbi + ib * cbr;
                if (first) {
                    xar = uar; xai = uai; xbr = ubr; xbi = ubi;
                } else {
                    V4_LOAD(xar, are + o); V4_LOAD(xai, aim + o);
                    V4_LOAD(xbr, are + ob); V4_LOAD(xbi, aim + ob);
                    xar += uar; xai += uai; xbr += ubr; xbi += ubi;
                }
                V4_STORE(are + o, xar); V4_STORE(aim + o, xai);
                V4_STORE(are + ob, xbr); V4_STORE(aim + ob, xbi);
                const beam_v4d ta = ra * dLr - ia * dLi, tb = rb * dLr - ib * dLi;
                ia = ra * dLi + ia * dLr;
                ib = rb * dLi + ib * dLr;
                ra = ta; rb = tb;
            }
        }
    }
}

/* nx = acc * s, and nmax = max(nmax, |re|, |im|), over T orders. */
__attribute__((always_inline))
static inline void line_store_body(uint32_t T, const double *restrict are,
                                   const double *restrict aim,
                                   const double *restrict s,
                                   double *restrict nre, double *restrict nim,
                                   double *restrict nmax) {
    const beam_v4i absmask = {0x7fffffffffffffffLL, 0x7fffffffffffffffLL,
                              0x7fffffffffffffffLL, 0x7fffffffffffffffLL};
    for (uint32_t j = 0; j < T; j += BEAM_LINE_VEC) {
        beam_v4d ar, ai, sv, mv;
        V4_LOAD(ar, are + j); V4_LOAD(ai, aim + j);
        V4_LOAD(sv, s + j); V4_LOAD(mv, nmax + j);
        ar *= sv; ai *= sv;
        V4_STORE(nre + j, ar); V4_STORE(nim + j, ai);
        const beam_v4d xr = (beam_v4d)((beam_v4i)ar & absmask);
        const beam_v4d xi = (beam_v4d)((beam_v4i)ai & absmask);
        const beam_v4i g1 = xi > xr;
        const beam_v4d xm = (beam_v4d)((g1 & (beam_v4i)xi) | (~g1 & (beam_v4i)xr));
        const beam_v4i g2 = xm > mv;
        mv = (beam_v4d)((g2 & (beam_v4i)xm) | (~g2 & (beam_v4i)mv));
        V4_STORE(nmax + j, mv);
    }
}

/* The CPU picks, at load time, a copy compiled for AVX-512, for AVX2 or for
 * the baseline. All perform the same operations per order in the same order
 * with no fused multiply-add -- contraction is off under -std=c11 -- so they
 * agree to the bit. On AVX2 without AVX-512 the tile runs line_edge_body4,
 * the same arithmetic per lane with fewer lanes in flight
 * (`line_narrow_pick`). */
#if defined(__x86_64__) && defined(__ELF__) && defined(__GNUC__)
#define BEAM_EDGE_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define BEAM_EDGE_CLONES
#endif

/* What a tile of a group needs: where each target's edges start in the
 * edge arrays, and each edge's modulus, source row, exp(16 i dim w),
 * pattern and anchors. */
typedef struct {
    const uint32_t *tgt;      /* the group's targets                    */
    const uint32_t *first;    /* [ntg + 1] a target's first edge        */
    uint32_t        ntg;
    const double   *mag;
    const uint32_t *src;
    const double   *fc, *fs;  /* [3 E] fixed rotations                  */
    const double   *pr, *pi;  /* [16 E] patterns                        */
    const double   *ac, *as;  /* [nb E] anchors                         */
    uint32_t        nb;
    int             narrow;   /* line_edge_body4 rather than line_edge_body */
} LineTile;

/* Writing a target's tile into the next layer's plane. The row is not read
 * again until the whole layer is done, gigabytes later, so on x86 it is
 * written with non-temporal stores: an ordinary store first reads the line
 * it lands in, and with the sweep bound by memory bandwidth that read was a
 * third of the traffic. The values stored are the same either way. */
typedef void (*LineStoreFn)(uint32_t T, const double *are, const double *aim,
                            const double *s, double *nre, double *nim,
                            double *nmax);

static void line_store_plain(uint32_t T, const double *are, const double *aim,
                             const double *s, double *nre, double *nim,
                             double *nmax) {
    line_store_body(T, are, aim, s, nre, nim, nmax);
}

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>

/* As line_store_body, lane for lane, but streamed; `nre`, `nim` 64-byte
 * aligned (rows are whole lines, tiles whole lines apart). */
#define LINE_STORE_STREAM(NAME, TARGET, VT, VI, W, STREAM)                     \
__attribute__((target(TARGET)))                                                \
static void NAME(uint32_t T, const double *are, const double *aim,             \
                 const double *s, double *nre, double *nim, double *nmax) {    \
    typedef double VT __attribute__((vector_size(8 * (W))));                   \
    typedef long long VI __attribute__((vector_size(8 * (W))));                \
    VI absmask;                                                                \
    for (uint32_t l = 0; l < (W); ++l) absmask[l] = 0x7fffffffffffffffLL;      \
    for (uint32_t j = 0; j < T; j += (W)) {                                    \
        VT ar, ai, sv, mv;                                                     \
        memcpy(&ar, are + j, sizeof ar); memcpy(&ai, aim + j, sizeof ai);      \
        memcpy(&sv, s + j, sizeof sv); memcpy(&mv, nmax + j, sizeof mv);       \
        ar *= sv; ai *= sv;                                                    \
        STREAM(nre + j, ar); STREAM(nim + j, ai);                              \
        const VT xr = (VT)((VI)ar & absmask), xi = (VT)((VI)ai & absmask);     \
        const VI g1 = xi > xr;                                                 \
        const VT xm = (VT)((g1 & (VI)xi) | (~g1 & (VI)xr));                    \
        const VI g2 = xm > mv;                                                 \
        mv = (VT)((g2 & (VI)xm) | (~g2 & (VI)mv));                             \
        memcpy(nmax + j, &mv, sizeof mv);                                      \
    }                                                                          \
}
#define LINE_STREAM512(p, v) _mm512_stream_pd((p), (__m512d)(v))
#define LINE_STREAM256(p, v) _mm256_stream_pd((p), (__m256d)(v))
LINE_STORE_STREAM(line_store_nt512, "avx512f", ls_v8d, ls_v8i, 8u, LINE_STREAM512)
LINE_STORE_STREAM(line_store_nt256, "avx2", ls_v4d, ls_v4i, 4u, LINE_STREAM256)

static LineStoreFn line_store_pick(void) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return line_store_nt512;
    if (__builtin_cpu_supports("avx2")) return line_store_nt256;
    return line_store_plain;
}

/* line_edge_body4 where the registers are 256 bits and sixteen of them:
 * AVX2 without AVX-512. Elsewhere the eight-lane chains fit, or this has not
 * been measured, and they stay. */
static int line_narrow_pick(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && !__builtin_cpu_supports("avx512f");
}
#define LINE_FENCE() _mm_sfence()
#else
static LineStoreFn line_store_pick(void) { return line_store_plain; }
static int line_narrow_pick(void) { return 0; }
#define LINE_FENCE() ((void)0)
#endif

/* One tile of one group: every target's edges into `acc`, then the row
 * scaled into `next` and its largest entries into `nmax`. One call for the
 * whole tile -- not one per edge, whose dispatch and prologue cost more
 * than the edge's first block. */
BEAM_EDGE_CLONES
static void line_tile(const LineTile *lt, uint64_t g0, uint32_t ts, uint32_t T,
                      uint32_t Wp, const double *cur, double *next,
                      const double *sv, double *acc, double *nmax,
                      LineStoreFn store) {
    const size_t RS = beam_line_row(Wp), IM = beam_line_im(Wp);
    for (uint32_t k = 0; k < lt->ntg; ++k) {
        const uint32_t q0 = lt->first[k], q1 = lt->first[k + 1u];
        for (uint32_t q = q0; q < q1; ++q) {
            const double *c = cur + (size_t)lt->src[q] * RS;
            if (lt->narrow)
                line_edge_body4(lt->mag[q], lt->fc[LINE_ROT_FIXED * q + 2u],
                                lt->fs[LINE_ROT_FIXED * q + 2u],
                                lt->pr + (size_t)BEAM_LINE_LANES * q,
                                lt->pi + (size_t)BEAM_LINE_LANES * q,
                                lt->ac + (size_t)q * lt->nb, lt->as + (size_t)q * lt->nb,
                                g0, T, c + ts, c + IM + ts, acc, acc + BEAM_LINE_TILE,
                                q == q0);
            else
                line_edge_body(lt->mag[q], lt->fc[LINE_ROT_FIXED * q + 2u],
                               lt->fs[LINE_ROT_FIXED * q + 2u],
                               lt->pr + (size_t)BEAM_LINE_LANES * q,
                               lt->pi + (size_t)BEAM_LINE_LANES * q,
                               lt->ac + (size_t)q * lt->nb, lt->as + (size_t)q * lt->nb,
                               g0, T, c + ts, c + IM + ts, acc, acc + BEAM_LINE_TILE,
                               q == q0);
        }
        if (q0 == q1) memset(acc, 0, 2u * BEAM_LINE_TILE * sizeof(double));
        double *nx = next + (size_t)lt->tgt[k] * RS;
        store(T, acc, acc + BEAM_LINE_TILE, sv + ts, nx + ts, nx + IM + ts, nmax + ts);
    }
}

/* The sink's tile: every edge in `lt` (one target) into `part`. */
BEAM_EDGE_CLONES
static void line_sink_tile(const LineTile *lt, uint64_t g0, uint32_t ts,
                           uint32_t T, uint32_t Wp, const double *cur,
                           double *part) {
    const size_t RS = beam_line_row(Wp), IM = beam_line_im(Wp);
    const uint32_t q1 = lt->first[1];
    for (uint32_t q = 0; q < q1; ++q) {
        const double *c = cur + (size_t)lt->src[q] * RS;
        if (lt->narrow)
            line_edge_body4(lt->mag[q], lt->fc[LINE_ROT_FIXED * q + 2u],
                            lt->fs[LINE_ROT_FIXED * q + 2u],
                            lt->pr + (size_t)BEAM_LINE_LANES * q,
                            lt->pi + (size_t)BEAM_LINE_LANES * q,
                            lt->ac + (size_t)q * lt->nb, lt->as + (size_t)q * lt->nb,
                            g0, T, c + ts, c + IM + ts, part + ts, part + IM + ts,
                            q == 0);
        else
            line_edge_body(lt->mag[q], lt->fc[LINE_ROT_FIXED * q + 2u],
                           lt->fs[LINE_ROT_FIXED * q + 2u],
                           lt->pr + (size_t)BEAM_LINE_LANES * q,
                           lt->pi + (size_t)BEAM_LINE_LANES * q,
                           lt->ac + (size_t)q * lt->nb, lt->as + (size_t)q * lt->nb,
                           g0, T, c + ts, c + IM + ts, part + ts, part + IM + ts,
                           q == 0);
    }
}

static void *line_alloc(size_t n_doubles) {
    /* Whole cache lines: a plane row is a multiple of sixteen orders. */
    size_t bytes = n_doubles * sizeof(double);
    bytes = (bytes + 63u) / 64u * 64u;
#ifdef _WIN32
    /* The Windows C runtime has no aligned_alloc; its aligned blocks must be
     * returned with _aligned_free, which line_release does. */
    return _aligned_malloc(bytes ? bytes : 64u, 64u);
#else
    return aligned_alloc(64u, bytes ? bytes : 64u);
#endif
}

/* Returns a block from line_alloc. */
static void line_release(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* One thread's scratch, `beam_line_scratch_doubles` long: a tile of one
 * target's accumulator, this thread's largest entries, and up to
 * LINE_EDGES edges' weights, moduli, sources and rotations. */
#define LINE_EDGES(A, R) ((uint64_t)(A) * (R) > BEAM_LINE_SINK_CHUNK \
                          ? (uint64_t)(A) * (R) : BEAM_LINE_SINK_CHUNK)
#define LINE_TILE_BLOCKS (BEAM_LINE_TILE / BEAM_LINE_BLOCK + 1u)

typedef struct {
    double   *acc;     /* [2 BEAM_LINE_TILE] re, then im        */
    double   *nmax;    /* [Wp]                                   */
    double   *w, *mag; /* [E]                                    */
    uint32_t *src;     /* [E]                                    */
    uint32_t *first;   /* [A + 1] a target's first edge          */
    double   *fang, *fsn, *fcs;  /* [3 E]                        */
    double   *aang, *asn, *acs;  /* [LINE_TILE_BLOCKS E]         */
    double   *pr, *pi;           /* [16 E] patterns              */
} LineScratch;

static void line_scratch_carve(double *base, uint32_t Wp, uint64_t E,
                               LineScratch *sc) {
    double *p = base;
    sc->acc = p;  p += 2u * BEAM_LINE_TILE;
    sc->nmax = p; p += Wp;
    sc->w = p;    p += E;
    sc->mag = p;  p += E;
    sc->fang = p; p += LINE_ROT_FIXED * E;
    sc->fsn = p;  p += LINE_ROT_FIXED * E;
    sc->fcs = p;  p += LINE_ROT_FIXED * E;
    sc->aang = p; p += LINE_TILE_BLOCKS * E;
    sc->asn = p;  p += LINE_TILE_BLOCKS * E;
    sc->acs = p;  p += LINE_TILE_BLOCKS * E;
    sc->pr = p;   p += BEAM_LINE_LANES * E;
    sc->pi = p;   p += BEAM_LINE_LANES * E;
    sc->src = (uint32_t *)p;
    sc->first = sc->src + E;
}

/* One layer's work for the targets tgt[b .. e), which share their sources:
 * each target's row of the stretch, scaled by `sv` into `next`, with this
 * thread's largest entries in sc->nmax. */
static void line_group(const BeamModel *m, const uint32_t *tgt, uint32_t b,
                       uint32_t e, const uint8_t *alive, const double *w,
                       double re, double qmax, double im0, double dim,
                       uint64_t k0, uint32_t Wp, const double *sv,
                       const double *cur, double *next, LineScratch *sc,
                       LineStoreFn store, int narrow) {
    const uint32_t A = m->n_alphabet, R = m->n_residues, SK = m->state_shift;
    const uint32_t rest = tgt[b] / A;
    uint32_t ne = 0;
    for (uint32_t k = b; k < e; ++k) {
        sc->first[k - b] = ne;
        const uint32_t x = tgt[k] % A;
        if (x >= R) continue;
        for (uint32_t a = 0; a < A; ++a) {
            const uint32_t src = a * SK + rest;
            if (!alive[src]) continue;
            const double wv = w[(size_t)src * R + x];
            if (wv == -INFINITY) continue;
            sc->w[ne] = wv;
            sc->mag[ne] = exp(re * wv - qmax);
            sc->src[ne++] = src;
        }
    }
    sc->first[e - b] = ne;
    if (ne) {
        line_fixed(sc->w, ne, dim, sc->fang, sc->fsn, sc->fcs);
        line_patterns(ne, sc->fcs, sc->fsn, sc->pr, sc->pi);
    }
    LineTile lt = {tgt + b, sc->first, e - b, sc->mag, sc->src, sc->fcs, sc->fsn,
                   sc->pr, sc->pi, sc->acs, sc->asn, 0u, narrow};
    for (uint32_t ts = 0; ts < Wp; ts += BEAM_LINE_TILE) {
        const uint32_t T = Wp - ts < BEAM_LINE_TILE ? Wp - ts : BEAM_LINE_TILE;
        const uint64_t g0 = k0 + ts;
        const uint64_t a0 = g0 - g0 % BEAM_LINE_BLOCK;
        const uint32_t nb = (uint32_t)((g0 + T - a0 + BEAM_LINE_BLOCK - 1u)
                                       / BEAM_LINE_BLOCK);
        if (ne) line_anchors(sc->w, ne, im0, dim, a0, nb, sc->aang, sc->asn, sc->acs);
        lt.nb = nb;
        line_tile(&lt, g0, ts, T, Wp, cur, next, sv, sc->acc, sc->nmax, store);
    }
}

/* The sink over end states sv2[b .. e): their tail edges' sum into `part`
 * (re then im, Wp each). Returns 0 if none of them has a tail. */
static int line_sink_chunk(const uint32_t *ends, uint32_t b, uint32_t e,
                           const double *tail, double re, double tmax,
                           double im0, double dim, uint64_t k0, uint32_t Wp,
                           const double *cur, double *part, LineScratch *sc,
                           int narrow) {
    uint32_t ne = 0;
    for (uint32_t k = b; k < e; ++k) {
        const double tv = tail[ends[k]];
        if (tv == -INFINITY) continue;
        sc->w[ne] = tv;
        sc->mag[ne] = exp(re * tv - tmax);
        sc->src[ne++] = ends[k];
    }
    if (!ne) return 0;
    line_fixed(sc->w, ne, dim, sc->fang, sc->fsn, sc->fcs);
    line_patterns(ne, sc->fcs, sc->fsn, sc->pr, sc->pi);
    const uint32_t one[2] = {0u, ne};
    LineTile lt = {NULL, one, 1u, sc->mag, sc->src, sc->fcs, sc->fsn,
                   sc->pr, sc->pi, sc->acs, sc->asn, 0u, narrow};
    for (uint32_t ts = 0; ts < Wp; ts += BEAM_LINE_TILE) {
        const uint32_t T = Wp - ts < BEAM_LINE_TILE ? Wp - ts : BEAM_LINE_TILE;
        const uint64_t g0 = k0 + ts;
        const uint64_t a0 = g0 - g0 % BEAM_LINE_BLOCK;
        const uint32_t nb = (uint32_t)((g0 + T - a0 + BEAM_LINE_BLOCK - 1u)
                                       / BEAM_LINE_BLOCK);
        line_anchors(sc->w, ne, im0, dim, a0, nb, sc->aang, sc->asn, sc->acs);
        lt.nb = nb;
        line_sink_tile(&lt, g0, ts, T, Wp, cur, part);
    }
    return 1;
}

struct BeamLineWork {
    uint32_t  Wp, nth;           /* the widest stretch; the threads     */
    uint64_t  per_thread;        /* doubles of one thread's scratch     */
    double   *w, *tail, *cur, *next, *pt, *scr;
    uint8_t  *alive, *dead, *used;
    uint32_t *gs;
    int      *ex;
};

void beam_line_work_free(BeamLineWork *wk) {
    if (!wk) return;
    free(wk->w); free(wk->tail); line_release(wk->cur); line_release(wk->next);
    line_release(wk->pt); line_release(wk->scr);
    free(wk->alive); free(wk->dead); free(wk->used);
    free(wk->gs); free(wk->ex);
    free(wk);
}

BeamLineWork *beam_line_work_new(const BeamModel *m, uint32_t Wp, uint32_t nth) {
    const uint32_t S = m->n_states, R = m->n_residues, A = m->n_alphabet;
    BeamLineWork *wk = (BeamLineWork *)calloc(1, sizeof *wk);
    if (!wk) return NULL;
    wk->Wp = (Wp + BEAM_LINE_LANES - 1u) / BEAM_LINE_LANES * BEAM_LINE_LANES;
    wk->nth = nth ? nth : 1u;
    wk->per_thread = beam_line_scratch_doubles(wk->Wp, A, R);
    const size_t RS = beam_line_row(wk->Wp);
    const uint32_t nchunks = (S + BEAM_LINE_SINK_CHUNK - 1u) / BEAM_LINE_SINK_CHUNK;
    wk->w     = (double *)malloc((size_t)S * R * sizeof(double));
    wk->tail  = (double *)malloc((size_t)S * sizeof(double));
    wk->cur   = (double *)line_alloc((size_t)S * RS);
    wk->next  = (double *)line_alloc((size_t)S * RS);
    wk->pt    = (double *)line_alloc((size_t)wk->Wp * 5u);
    wk->scr   = (double *)line_alloc((size_t)(wk->per_thread * wk->nth));
    wk->alive = (uint8_t *)malloc(S);
    wk->dead  = (uint8_t *)malloc(wk->Wp);
    wk->used  = (uint8_t *)malloc(nchunks ? nchunks : 1u);
    wk->gs    = (uint32_t *)malloc(((size_t)S + 1u) * sizeof(uint32_t));
    wk->ex    = (int *)malloc((size_t)wk->Wp * sizeof(int));
    if (!wk->w || !wk->tail || !wk->cur || !wk->next || !wk->pt || !wk->scr
        || !wk->alive || !wk->dead || !wk->used || !wk->gs || !wk->ex) {
        beam_line_work_free(wk);
        return NULL;
    }
    return wk;
}

BeamError jet_sweep_line(const BeamModel *m, BeamLineWork *wk, uint32_t len,
                         double re, double im0, double dim, uint64_t k0,
                         uint32_t nq, double *out_log_mod, double *out_phase,
                         size_t out_stride) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t W = nq;
    const uint32_t Wp = (W + BEAM_LINE_LANES - 1u) / BEAM_LINE_LANES * BEAM_LINE_LANES;
    const uint32_t nth = wk->nth;

    for (uint32_t j = 0; j < W; ++j) {
        out_log_mod[(size_t)j * out_stride] = -INFINITY;
        out_phase[(size_t)j * out_stride] = 0.0;
    }
    if (W == 0u) return BEAM_OK;
    if (k0 % BEAM_LINE_LANES != 0u || Wp > wk->Wp)
        return BEAM_ERANGE("%s: a stretch must start at a multiple of %u orders "
                           "and fit its workspace, not %llu + %u", __func__,
                           BEAM_LINE_LANES, (unsigned long long)k0, W);

    /* The workspace is reused from call to call: rows are read only after
     * they are written (a reached state's row in the layer before, the start
     * state's below), so nothing but the per-order scalars is reset. */
    const size_t RS = beam_line_row(Wp), IM = beam_line_im(Wp);
    const uint64_t per_thread = wk->per_thread;
    const LineStoreFn store = line_store_pick();
    const int narrow = line_narrow_pick();
    double   *w     = wk->w;
    double   *tail  = wk->tail;
    double   *cur   = wk->cur;
    double   *next  = wk->next;
    uint8_t  *alive = wk->alive;
    uint32_t *gs    = wk->gs;
    double   *pt    = wk->pt;
    int      *ex    = wk->ex;
    uint8_t  *dead  = wk->dead;
    double   *scr   = wk->scr;
    uint8_t  *used  = wk->used;
    memset(ex, 0, (size_t)Wp * sizeof(int));
    memset(dead, 0, Wp);
    LiveHandle h;
    if (beam_live_acquire(m, len, &h) != BEAM_OK)
        return BEAM_EALLOC("the complex line sweep's live frontier");
    const BeamLive *L = h.L;
    double *scale = pt;
    double *nmax  = pt + 1u * Wp;
    double *sv    = pt + 2u * Wp;      /* this layer's power-of-two scale */
    double *fre   = pt + 3u * Wp;
    double *fim   = pt + 4u * Wp;
    double *part  = NULL;
    uint32_t nalive = W;

    /* Only the start state is read before it is written. */
    memset(cur + (size_t)(S - 1) * RS, 0, RS * sizeof(double));
    for (uint32_t j = 0; j < Wp; ++j) scale[j] = 0.0;
    for (uint32_t j = 0; j < W; ++j) cur[(size_t)(S - 1) * RS + j] = 1.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_reached_mask(L, i, S, alive);
        /* The layer's largest weight comes with the weights; the smallest,
         * needed only left of the imaginary axis, takes a scan. */
        double hi = -INFINITY, lo = INFINITY;
        beam_layer_weights_live(m, i, len, w, &hi, alive);
        if (re < 0.0) {
            const uint32_t  nr = L->n_reached[i];
            const uint32_t *rv = L->reached[i];
            for (uint32_t k = 0; k < nr; ++k) {
                const double *row = w + (size_t)rv[k] * R;
                for (uint32_t x = 0; x < R; ++x)
                    if (row[x] != -INFINITY && row[x] < lo) lo = row[x];
            }
        }
        if (hi == -INFINITY) goto done;
        const double qmax = re >= 0.0 ? re * hi
                          : (lo == INFINITY ? -INFINITY : re * lo);
        if (qmax == -INFINITY) goto done;

        for (uint32_t j = 0; j < Wp; ++j) {
            sv[j] = (j < W && !dead[j]) ? ldexp(1.0, -ex[j]) : 0.0;
            nmax[j] = 0.0;
        }
        /* Groups: runs of targets that differ only in the newest residue.
         * The targets ascend, so a run ends where one reaches the next
         * multiple of A: one division a group rather than one a target. */
        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
        uint32_t ng = 0;
        uint64_t bound = 0u;
        for (uint32_t k = 0; k < nt; ++k)
            if (k == 0 || tgt[k] >= bound) {
                gs[ng++] = k;
                bound = ((uint64_t)tgt[k] / A + 1u) * A;
            }
        gs[ng] = nt;
#ifdef _OPENMP
        #pragma omp parallel num_threads((int)nth)
#endif
        {
#ifdef _OPENMP
            const uint32_t tid = (uint32_t)omp_get_thread_num();
#else
            const uint32_t tid = 0u;
#endif
            LineScratch sc;
            line_scratch_carve(scr + per_thread * tid, Wp, LINE_EDGES(A, R), &sc);
            for (uint32_t j = 0; j < Wp; ++j) sc.nmax[j] = 0.0;
#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 16)
#endif
            for (int64_t g = 0; g < (int64_t)ng; ++g)
                line_group(m, tgt, gs[g], gs[g + 1], alive, w, re, qmax, im0,
                           dim, k0, Wp, sv, cur, next, &sc, store, narrow);
            LINE_FENCE();          /* this thread's streamed rows, before the barrier */
#ifdef _OPENMP
            #pragma omp critical(beam_line_nmax)
#endif
            for (uint32_t j = 0; j < Wp; ++j)
                if (sc.nmax[j] > nmax[j]) nmax[j] = sc.nmax[j];
        }
        for (uint32_t j = 0; j < W; ++j) {
            if (dead[j]) continue;
            scale[j] += qmax + (double)ex[j] * BEAM_LN2;
            if (!(nmax[j] > 0.0)) { dead[j] = 1; --nalive; continue; }
            ex[j] = ilogb(nmax[j]);
        }
        if (nalive == 0u) goto done;
        double *tmp = cur; cur = next; next = tmp;
    }

    /* Sink: one more edge whose weight is the tail, with the shift taken over
     * the reached end states, again independent of every order. The sum is
     * taken over fixed chunks of end states, each by one thread, then over
     * the chunks in order: the same bits at any thread count. */
    beam_tail_weights(m, len, tail);
    {
        const uint32_t  ns = L->n_reached[len];
        const uint32_t *ends = L->reached[len];
        double tmax = -INFINITY;
        for (uint32_t k = 0; k < ns; ++k) {
            const double tv = tail[ends[k]];
            if (tv != -INFINITY && re * tv > tmax) tmax = re * tv;
        }
        if (tmax == -INFINITY) goto done;
        const uint32_t nc = (ns + BEAM_LINE_SINK_CHUNK - 1u) / BEAM_LINE_SINK_CHUNK;
        part = next;          /* the planes' other half is free by now */
#ifdef _OPENMP
        #pragma omp parallel num_threads((int)nth)
#endif
        {
#ifdef _OPENMP
            const uint32_t tid = (uint32_t)omp_get_thread_num();
#else
            const uint32_t tid = 0u;
#endif
            LineScratch sc;
            line_scratch_carve(scr + per_thread * tid, Wp, LINE_EDGES(A, R), &sc);
#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 1)
#endif
            for (int64_t c = 0; c < (int64_t)nc; ++c) {
                const uint32_t b = (uint32_t)c * BEAM_LINE_SINK_CHUNK;
                const uint32_t e = b + BEAM_LINE_SINK_CHUNK < ns ? b + BEAM_LINE_SINK_CHUNK : ns;
                used[c] = (uint8_t)line_sink_chunk(ends, b, e, tail, re, tmax, im0,
                                                   dim, k0, Wp, cur,
                                                   part + (size_t)c * RS, &sc,
                                                   narrow);
            }
        }
        int any = 0;
        for (uint32_t j = 0; j < W; ++j) { fre[j] = 0.0; fim[j] = 0.0; }
        for (uint32_t c = 0; c < nc; ++c) {
            if (!used[c]) continue;
            const double *p = part + (size_t)c * RS;
            if (!any) {
                for (uint32_t j = 0; j < W; ++j) { fre[j] = p[j]; fim[j] = p[IM + j]; }
                any = 1;
            } else {
                for (uint32_t j = 0; j < W; ++j) { fre[j] += p[j]; fim[j] += p[IM + j]; }
            }
        }
        if (!any) goto done;
        for (uint32_t j = 0; j < W; ++j) {
            if (dead[j]) continue;
            const double mod = hypot(fre[j], fim[j]);
            if (!(mod > 0.0)) continue;
            out_log_mod[(size_t)j * out_stride] = scale[j] + tmax + log(mod);
            out_phase[(size_t)j * out_stride] = atan2(fim[j], fre[j]);
        }
    }

done:
    beam_live_release(&h);
    return BEAM_OK;
}


/* ── Many real orders at once ────────────────────────────────── */

/**
 * `jet_sweep_fan` for the real orders qs[0 .. nq), rebuilt the way the line
 * sweep was: `ln Z_q(L)` per order into out_log_z[j * out_stride].
 *
 * WHAT CHANGED, AND WHY. The fan took `exp(q w - qmax)` from libm once per
 * edge per order, one length per thread: 0.7 s an order on the foundation,
 * where a complex order along a line had come down to 0.04 s. Profiled over
 * the figure steps, these real power sums had become 40% of the exact work
 * -- the series' ninety, the planner's grid, the lines' fingerprints.
 * Here every thread shares out each layer's targets, in groups with the same
 * sources and tiles of BEAM_REAL_TILE orders, and an edge's factors are taken
 * eight orders at a time by `vm_exp8` (beam/vmath.h). Each order's scale is
 * an exact power of two chosen a layer ahead, the next layer is written once
 * with non-temporal stores where the CPU has them, and nothing is cleared.
 *
 * WHAT IS KEPT. Each order's layer shift is the fan's: its largest weight
 * over the reached states that still carry ITS mass (its smallest, for
 * q < 0), because at q = 90 most states' mass has underflowed and a shift
 * taken over them all would flush the rest. An order dies where the fan's
 * did: no mass-carrying edge, a shift of -inf, or an empty layer.
 *
 * WHY THAT MATTERS: an order's value depends on the model and on q alone --
 * not on the other orders, the stretch, the thread count or the budget. Each
 * target's edges are summed in a fixed order by one thread, the extremes are
 * maxima and minima, and the sink's sum is taken over fixed chunks of end
 * states and then over the chunks in order. So `beam_model_log_mellin` and
 * `beam_model_log_mellin_many` agree to the bit, as their contract says.
 */
struct BeamRealWork {
    uint32_t  Wp, nth;
    uint64_t  per_thread;
    double   *w, *tail, *cur, *next, *pt, *scr;
    uint8_t  *alive, *dead, *used;
    uint32_t *gs;
    int      *ex;
};

void beam_real_work_free(BeamRealWork *wk) {
    if (!wk) return;
    free(wk->w); free(wk->tail); line_release(wk->cur); line_release(wk->next);
    line_release(wk->pt); line_release(wk->scr);
    free(wk->alive); free(wk->dead); free(wk->used);
    free(wk->gs); free(wk->ex);
    free(wk);
}

BeamRealWork *beam_real_work_new(const BeamModel *m, uint32_t Wp, uint32_t nth) {
    const uint32_t S = m->n_states, R = m->n_residues, A = m->n_alphabet;
    BeamRealWork *wk = (BeamRealWork *)calloc(1, sizeof *wk);
    if (!wk) return NULL;
    wk->Wp = (Wp + BEAM_REAL_LANES - 1u) / BEAM_REAL_LANES * BEAM_REAL_LANES;
    wk->nth = nth ? nth : 1u;
    wk->per_thread = beam_real_scratch_doubles(wk->Wp, A, R);
    const size_t RS = beam_real_row(wk->Wp);
    const uint32_t nchunks = (S + BEAM_LINE_SINK_CHUNK - 1u) / BEAM_LINE_SINK_CHUNK;
    wk->w     = (double *)malloc((size_t)S * R * sizeof(double));
    wk->tail  = (double *)malloc((size_t)S * sizeof(double));
    wk->cur   = (double *)line_alloc((size_t)S * RS);
    wk->next  = (double *)line_alloc((size_t)S * RS);
    wk->pt    = (double *)line_alloc((size_t)wk->Wp * 9u);
    wk->scr   = (double *)line_alloc((size_t)(wk->per_thread * wk->nth));
    wk->alive = (uint8_t *)malloc(S);
    wk->dead  = (uint8_t *)malloc(wk->Wp);
    wk->used  = (uint8_t *)malloc(nchunks ? nchunks : 1u);
    wk->gs    = (uint32_t *)malloc(((size_t)S + 1u) * sizeof(uint32_t));
    wk->ex    = (int *)malloc((size_t)wk->Wp * sizeof(int));
    if (!wk->w || !wk->tail || !wk->cur || !wk->next || !wk->pt || !wk->scr
        || !wk->alive || !wk->dead || !wk->used || !wk->gs || !wk->ex) {
        beam_real_work_free(wk);
        return NULL;
    }
    return wk;
}

typedef struct {
    double   *acc;     /* [BEAM_REAL_TILE]            */
    double   *nmax;    /* [Wp] this thread's largest   */
    double   *hi, *lo; /* [Wp] this thread's extremes  */
    double   *tmx;     /* [Wp] this thread's sink max  */
    double   *w;       /* [E] edge weights             */
    uint32_t *src;     /* [E]                          */
    uint32_t *first;   /* [A + 1]                      */
} RealScratch;

static void real_scratch_carve(double *base, uint32_t Wp, uint64_t E,
                               RealScratch *sc) {
    double *p = base;
    sc->acc = p;  p += BEAM_REAL_TILE;
    sc->nmax = p; p += Wp;
    sc->hi = p;   p += Wp;
    sc->lo = p;   p += Wp;
    sc->tmx = p;  p += Wp;
    sc->w = p;    p += E;
    sc->src = (uint32_t *)p;
    sc->first = sc->src + E;
}

typedef double real_v8d __attribute__((vector_size(64)));
typedef long long real_v8i __attribute__((vector_size(64)));
#define R8_LOAD(dst, p)  memcpy(&(dst), (p), sizeof(real_v8d))
#define R8_STORE(p, src) memcpy((p), &(src), sizeof(real_v8d))

/* a where the mask is set, else b; a macro, since a function returning a
 * vector by value changes the ABI between builds with and without AVX-512. */
#define REAL_SEL(msk, a, b) \
    ((real_v8d)(((msk) & (real_v8i)(a)) | (~(msk) & (real_v8i)(b))))

/* Each order's extremes over the reached states rv[b .. e) that carry its
 * mass: the rows' own max and (finite) min, offered to every order whose
 * mass there is nonzero -- as the fan found them, order by order. */
BEAM_EDGE_CLONES
static void real_extremes(const uint32_t *rv, uint32_t b, uint32_t e,
                          const double *w, uint32_t R, const double *cur,
                          size_t RS, uint32_t Wp, double *hi, double *lo) {
    for (uint32_t k = b; k < e; ++k) {
        const uint32_t s = rv[k];
        const double *row = w + (size_t)s * R;
        double rmax = -INFINITY, rmin = INFINITY;
        for (uint32_t x = 0; x < R; ++x) {
            if (row[x] > rmax) rmax = row[x];
            if (row[x] != -INFINITY && row[x] < rmin) rmin = row[x];
        }
        const double *c = cur + (size_t)s * RS;
        const real_v8d vmax = {rmax, rmax, rmax, rmax, rmax, rmax, rmax, rmax};
        const real_v8d vmin = {rmin, rmin, rmin, rmin, rmin, rmin, rmin, rmin};
        const real_v8d zero = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t j = 0; j < Wp; j += BEAM_REAL_LANES) {
            real_v8d cv, hv, lv;
            R8_LOAD(cv, c + j); R8_LOAD(hv, hi + j); R8_LOAD(lv, lo + j);
            const real_v8i has = cv != zero;
            hv = REAL_SEL(has & (vmax > hv), vmax, hv);
            lv = REAL_SEL(has & (vmin < lv), vmin, lv);
            R8_STORE(hi + j, hv); R8_STORE(lo + j, lv);
        }
    }
}

/* nx = acc * s and nmax = max(nmax, nx) over T orders; masses are never
 * negative, so the largest is the largest value. */
typedef void (*RealStoreFn)(uint32_t T, const double *acc, const double *s,
                            double *nx, double *nmax);

__attribute__((always_inline))
static inline void real_store_body(uint32_t T, const double *restrict acc,
                                   const double *restrict s,
                                   double *restrict nx, double *restrict nmax) {
    for (uint32_t j = 0; j < T; j += BEAM_REAL_LANES) {
        real_v8d a, sv, mv;
        R8_LOAD(a, acc + j); R8_LOAD(sv, s + j); R8_LOAD(mv, nmax + j);
        a *= sv;
        R8_STORE(nx + j, a);
        mv = REAL_SEL(a > mv, a, mv);
        R8_STORE(nmax + j, mv);
    }
}

static void real_store_plain(uint32_t T, const double *acc, const double *s,
                             double *nx, double *nmax) {
    real_store_body(T, acc, s, nx, nmax);
}

#if defined(__x86_64__) && defined(__GNUC__)
#define REAL_STORE_STREAM(NAME, TARGET, VT, VI, W, STREAM)                     \
__attribute__((target(TARGET)))                                                \
static void NAME(uint32_t T, const double *acc, const double *s, double *nx,   \
                 double *nmax) {                                               \
    typedef double VT __attribute__((vector_size(8 * (W))));                   \
    typedef long long VI __attribute__((vector_size(8 * (W))));                \
    for (uint32_t j = 0; j < T; j += (W)) {                                    \
        VT a, sv, mv;                                                          \
        memcpy(&a, acc + j, sizeof a); memcpy(&sv, s + j, sizeof sv);          \
        memcpy(&mv, nmax + j, sizeof mv);                                      \
        a *= sv;                                                               \
        STREAM(nx + j, a);                                                     \
        const VI g = a > mv;                                                   \
        mv = (VT)((g & (VI)a) | (~g & (VI)mv));                                \
        memcpy(nmax + j, &mv, sizeof mv);                                      \
    }                                                                          \
}
REAL_STORE_STREAM(real_store_nt512, "avx512f", rs_v8d, rs_v8i, 8u, LINE_STREAM512)
REAL_STORE_STREAM(real_store_nt256, "avx2", rs_v4d, rs_v4i, 4u, LINE_STREAM256)

static RealStoreFn real_store_pick(void) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return real_store_nt512;
    if (__builtin_cpu_supports("avx2")) return real_store_nt256;
    return real_store_plain;
}
#else
static RealStoreFn real_store_pick(void) { return real_store_plain; }
#endif

/* acc (= if first, else +=) exp(q w - qmax) c over T orders, eight at a
 * time. `q`, `qmax`, `c` and `acc` point at the tile's first order.
 *
 * THE ARGUMENT IS CLAMPED AT 0. For an edge whose source carries the order's
 * mass it is never positive -- qmax is q times that mass's largest weight
 * (smallest, for q < 0), and rounding is monotone -- so the clamp changes no
 * value. A source with no mass for the order can carry a heavier edge, and at
 * large q its factor would overflow to inf, and inf times its zero mass is
 * NaN; the old fan skipped such sources, this takes their factor as 1 times
 * 0. */
__attribute__((always_inline))
static inline void real_edge_body(double w, const double *restrict q,
                                  const double *restrict qmax, uint32_t T,
                                  const double *restrict c,
                                  double *restrict acc, int first) {
    const real_v8d zero = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t j = 0; j < T; j += BEAM_REAL_LANES) {
        real_v8d qv, mv, cv, f, a;
        R8_LOAD(qv, q + j); R8_LOAD(mv, qmax + j); R8_LOAD(cv, c + j);
        const real_v8d raw = qv * w - mv;
        const vm_v8d arg = REAL_SEL(raw > zero, zero, raw);
        vm_exp8(&arg, &f);
        if (first) {
            a = f * cv;
        } else {
            R8_LOAD(a, acc + j);
            a += f * cv;
        }
        R8_STORE(acc + j, a);
    }
}

/* One tile of one group: each target's edges into `acc`, then its row scaled
 * into `next`. */
BEAM_EDGE_CLONES
static void real_tile(const uint32_t *tgt, uint32_t ntg, const uint32_t *first,
                      const double *ew, const uint32_t *src, uint32_t ts,
                      uint32_t T, uint32_t Wp, const double *q,
                      const double *qmax, const double *cur, double *next,
                      const double *sv, double *acc, double *nmax,
                      RealStoreFn store) {
    const size_t RS = beam_real_row(Wp);
    for (uint32_t k = 0; k < ntg; ++k) {
        const uint32_t q0 = first[k], q1 = first[k + 1u];
        for (uint32_t e = q0; e < q1; ++e)
            real_edge_body(ew[e], q + ts, qmax + ts, T,
                           cur + (size_t)src[e] * RS + ts, acc, e == q0);
        if (q0 == q1) memset(acc, 0, BEAM_REAL_TILE * sizeof(double));
        store(T, acc, sv + ts, next + (size_t)tgt[k] * RS + ts, nmax + ts);
    }
}

/* The sink's tile: every end state's tail edge into `part`. */
BEAM_EDGE_CLONES
static void real_sink_tile(uint32_t ne, const double *ew, const uint32_t *src,
                           uint32_t ts, uint32_t T, uint32_t Wp,
                           const double *q, const double *tmax,
                           const double *cur, double *part) {
    const size_t RS = beam_real_row(Wp);
    for (uint32_t e = 0; e < ne; ++e)
        real_edge_body(ew[e], q + ts, tmax + ts, T,
                       cur + (size_t)src[e] * RS + ts, part + ts, e == 0);
}

static void real_group(const BeamModel *m, const uint32_t *tgt, uint32_t b,
                       uint32_t e, const uint8_t *alive, const double *w,
                       uint32_t Wp, const double *q, const double *qmax,
                       const double *sv, const double *cur, double *next,
                       RealScratch *sc, RealStoreFn store) {
    const uint32_t A = m->n_alphabet, R = m->n_residues, SK = m->state_shift;
    const uint32_t rest = tgt[b] / A;
    uint32_t ne = 0;
    for (uint32_t k = b; k < e; ++k) {
        sc->first[k - b] = ne;
        const uint32_t x = tgt[k] % A;
        if (x >= R) continue;
        for (uint32_t a = 0; a < A; ++a) {
            const uint32_t src = a * SK + rest;
            if (!alive[src]) continue;
            const double wv = w[(size_t)src * R + x];
            if (wv == -INFINITY) continue;
            sc->w[ne] = wv;
            sc->src[ne++] = src;
        }
    }
    sc->first[e - b] = ne;
    for (uint32_t ts = 0; ts < Wp; ts += BEAM_REAL_TILE) {
        const uint32_t T = Wp - ts < BEAM_REAL_TILE ? Wp - ts : BEAM_REAL_TILE;
        real_tile(tgt + b, e - b, sc->first, sc->w, sc->src, ts, T, Wp, q, qmax,
                  cur, next, sv, sc->acc, sc->nmax, store);
    }
}

BeamError jet_sweep_real(const BeamModel *m, BeamRealWork *wk, uint32_t len,
                         const double *qs, uint32_t nq, double *out_log_z,
                         size_t out_stride) {
    const uint32_t A = m->n_alphabet;
    const uint32_t R = m->n_residues;
    const uint32_t S = m->n_states;
    const uint32_t W = nq;
    const uint32_t Wp = (W + BEAM_REAL_LANES - 1u) / BEAM_REAL_LANES * BEAM_REAL_LANES;
    const uint32_t nth = wk->nth;

    for (uint32_t j = 0; j < W; ++j) out_log_z[(size_t)j * out_stride] = -INFINITY;
    if (W == 0u) return BEAM_OK;
    if (Wp > wk->Wp)
        return BEAM_ERANGE("%s: %u orders do not fit a workspace of %u", __func__,
                           W, wk->Wp);

    const size_t RS = beam_real_row(Wp);
    const uint64_t per_thread = wk->per_thread;
    const uint64_t E = LINE_EDGES(A, R);
    const RealStoreFn store = real_store_pick();
    double   *w = wk->w, *tail = wk->tail, *cur = wk->cur, *next = wk->next;
    double   *pt = wk->pt, *scr = wk->scr;
    uint8_t  *alive = wk->alive, *dead = wk->dead, *used = wk->used;
    uint32_t *gs = wk->gs;
    int      *ex = wk->ex;
    double *q     = pt;
    double *qmax  = pt + 1u * Wp;
    double *scale = pt + 2u * Wp;
    double *nmax  = pt + 3u * Wp;
    double *sv    = pt + 4u * Wp;
    double *hi    = pt + 5u * Wp;
    double *lo    = pt + 6u * Wp;
    double *tmax  = pt + 7u * Wp;
    double *fin   = pt + 8u * Wp;
    for (uint32_t j = 0; j < Wp; ++j) {
        q[j] = j < W ? qs[j] : 0.0;
        scale[j] = 0.0;
        ex[j] = 0;
        dead[j] = (uint8_t)(j >= W);
    }
    uint32_t nalive = W;
    LiveHandle h;
    if (beam_live_acquire(m, len, &h) != BEAM_OK)
        return BEAM_EALLOC("the real-order sweep's live frontier");
    const BeamLive *L = h.L;

    /* Only the start state is read before it is written. */
    memset(cur + (size_t)(S - 1) * RS, 0, RS * sizeof(double));
    for (uint32_t j = 0; j < W; ++j) cur[(size_t)(S - 1) * RS + j] = 1.0;

    for (uint32_t i = 0; i < len; ++i) {
        beam_reached_mask(L, i, S, alive);
        beam_layer_weights_live(m, i, len, w, NULL, alive);
        const uint32_t  nr = L->n_reached[i];
        const uint32_t *rv = L->reached[i];

        /* Each order's extremes over the states carrying its mass. */
        for (uint32_t j = 0; j < Wp; ++j) { hi[j] = -INFINITY; lo[j] = INFINITY; }
#ifdef _OPENMP
        #pragma omp parallel num_threads((int)nth)
#endif
        {
#ifdef _OPENMP
            const uint32_t tid = (uint32_t)omp_get_thread_num();
            const uint32_t nt_ = (uint32_t)omp_get_num_threads();
#else
            const uint32_t tid = 0u, nt_ = 1u;
#endif
            RealScratch sc;
            real_scratch_carve(scr + per_thread * tid, Wp, E, &sc);
            for (uint32_t j = 0; j < Wp; ++j) { sc.hi[j] = -INFINITY; sc.lo[j] = INFINITY; }
            const uint32_t b = (uint32_t)((uint64_t)nr * tid / nt_);
            const uint32_t e = (uint32_t)((uint64_t)nr * (tid + 1u) / nt_);
            real_extremes(rv, b, e, w, R, cur, RS, Wp, sc.hi, sc.lo);
#ifdef _OPENMP
            #pragma omp critical(beam_real_extremes)
#endif
            for (uint32_t j = 0; j < Wp; ++j) {
                if (sc.hi[j] > hi[j]) hi[j] = sc.hi[j];
                if (sc.lo[j] < lo[j]) lo[j] = sc.lo[j];
            }
        }
        for (uint32_t j = 0; j < W; ++j) {
            if (dead[j]) continue;
            double qm = -INFINITY;
            if (hi[j] != -INFINITY)
                qm = q[j] >= 0.0 ? q[j] * hi[j]
                   : (lo[j] == INFINITY ? -INFINITY : q[j] * lo[j]);
            if (qm == -INFINITY) { dead[j] = 1; q[j] = 0.0; --nalive; continue; }
            qmax[j] = qm;
        }
        if (nalive == 0u) goto done;
        for (uint32_t j = 0; j < Wp; ++j) {
            if (dead[j]) { qmax[j] = 0.0; sv[j] = 0.0; }
            else sv[j] = ldexp(1.0, -ex[j]);
            nmax[j] = 0.0;
        }

        /* Groups: runs of targets that differ only in the newest residue. */
        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
        uint32_t ng = 0;
        uint64_t bound = 0u;
        for (uint32_t k = 0; k < nt; ++k)
            if (k == 0 || tgt[k] >= bound) {
                gs[ng++] = k;
                bound = ((uint64_t)tgt[k] / A + 1u) * A;
            }
        gs[ng] = nt;
#ifdef _OPENMP
        #pragma omp parallel num_threads((int)nth)
#endif
        {
#ifdef _OPENMP
            const uint32_t tid = (uint32_t)omp_get_thread_num();
#else
            const uint32_t tid = 0u;
#endif
            RealScratch sc;
            real_scratch_carve(scr + per_thread * tid, Wp, E, &sc);
            for (uint32_t j = 0; j < Wp; ++j) sc.nmax[j] = 0.0;
#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 16)
#endif
            for (int64_t g = 0; g < (int64_t)ng; ++g)
                real_group(m, tgt, gs[g], gs[g + 1], alive, w, Wp, q, qmax, sv,
                           cur, next, &sc, store);
            LINE_FENCE();          /* this thread's streamed rows, before the barrier */
#ifdef _OPENMP
            #pragma omp critical(beam_real_nmax)
#endif
            for (uint32_t j = 0; j < Wp; ++j)
                if (sc.nmax[j] > nmax[j]) nmax[j] = sc.nmax[j];
        }
        for (uint32_t j = 0; j < W; ++j) {
            if (dead[j]) continue;
            scale[j] += qmax[j] + (double)ex[j] * BEAM_LN2;
            if (!(nmax[j] > 0.0)) { dead[j] = 1; q[j] = 0.0; --nalive; continue; }
            ex[j] = ilogb(nmax[j]);
        }
        if (nalive == 0u) goto done;
        double *tmp = cur; cur = next; next = tmp;
    }

    /* Sink: one more edge whose weight is the tail. Each order's shift is its
     * largest tilted tail over the end states carrying its mass, as the fan
     * took it; the sum is over fixed chunks of end states, each by one
     * thread, then over the chunks in order. */
    beam_tail_weights(m, len, tail);
    {
        const uint32_t  ns = L->n_reached[len];
        const uint32_t *ends = L->reached[len];
        for (uint32_t j = 0; j < Wp; ++j) tmax[j] = -INFINITY;
#ifdef _OPENMP
        #pragma omp parallel num_threads((int)nth)
#endif
        {
#ifdef _OPENMP
            const uint32_t tid = (uint32_t)omp_get_thread_num();
            const uint32_t nt_ = (uint32_t)omp_get_num_threads();
#else
            const uint32_t tid = 0u, nt_ = 1u;
#endif
            RealScratch sc;
            real_scratch_carve(scr + per_thread * tid, Wp, E, &sc);
            for (uint32_t j = 0; j < Wp; ++j) sc.tmx[j] = -INFINITY;
            const uint32_t b = (uint32_t)((uint64_t)ns * tid / nt_);
            const uint32_t e = (uint32_t)((uint64_t)ns * (tid + 1u) / nt_);
            for (uint32_t k = b; k < e; ++k) {
                const double tv = tail[ends[k]];
                if (tv == -INFINITY) continue;
                const double *c = cur + (size_t)ends[k] * RS;
                for (uint32_t j = 0; j < W; ++j)
                    if (c[j] > 0.0 && q[j] * tv > sc.tmx[j]) sc.tmx[j] = q[j] * tv;
            }
#ifdef _OPENMP
            #pragma omp critical(beam_real_tmax)
#endif
            for (uint32_t j = 0; j < Wp; ++j)
                if (sc.tmx[j] > tmax[j]) tmax[j] = sc.tmx[j];
        }
        for (uint32_t j = 0; j < Wp; ++j)
            if (dead[j] || tmax[j] == -INFINITY) {   /* no tilted mass reaches the sink */
                dead[j] = 1; q[j] = 0.0; tmax[j] = 0.0;
            }
        const uint32_t nc = (ns + BEAM_LINE_SINK_CHUNK - 1u) / BEAM_LINE_SINK_CHUNK;
        double *part = next;          /* the planes' other half is free by now */
#ifdef _OPENMP
        #pragma omp parallel num_threads((int)nth)
#endif
        {
#ifdef _OPENMP
            const uint32_t tid = (uint32_t)omp_get_thread_num();
#else
            const uint32_t tid = 0u;
#endif
            RealScratch sc;
            real_scratch_carve(scr + per_thread * tid, Wp, E, &sc);
#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 1)
#endif
            for (int64_t c = 0; c < (int64_t)nc; ++c) {
                const uint32_t b = (uint32_t)c * BEAM_LINE_SINK_CHUNK;
                const uint32_t e = b + BEAM_LINE_SINK_CHUNK < ns ? b + BEAM_LINE_SINK_CHUNK : ns;
                uint32_t ne = 0;
                for (uint32_t k = b; k < e; ++k) {
                    const double tv = tail[ends[k]];
                    if (tv == -INFINITY) continue;
                    sc.w[ne] = tv;
                    sc.src[ne++] = ends[k];
                }
                used[c] = (uint8_t)(ne > 0u);
                if (!ne) continue;
                double *pc = part + (size_t)c * RS;
                for (uint32_t ts = 0; ts < Wp; ts += BEAM_REAL_TILE) {
                    const uint32_t T = Wp - ts < BEAM_REAL_TILE ? Wp - ts : BEAM_REAL_TILE;
                    real_sink_tile(ne, sc.w, sc.src, ts, T, Wp, q, tmax, cur, pc);
                }
            }
        }
        int any = 0;
        for (uint32_t c = 0; c < nc; ++c) {
            if (!used[c]) continue;
            const double *p = part + (size_t)c * RS;
            if (!any) {
                for (uint32_t j = 0; j < W; ++j) fin[j] = p[j];
                any = 1;
            } else {
                for (uint32_t j = 0; j < W; ++j) fin[j] += p[j];
            }
        }
        if (!any) goto done;
        for (uint32_t j = 0; j < W; ++j) {
            if (dead[j] || !(fin[j] > 0.0)) continue;
            out_log_z[(size_t)j * out_stride] = scale[j] + tmax[j] + log(fin[j]);
        }
    }

done:
    beam_live_release(&h);
    return BEAM_OK;
}
