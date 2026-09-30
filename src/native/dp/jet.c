/**
 * @file jet.c
 * @brief The moment-jet forward sweep: one pass carrying r+1 moments.
 */
#include "beam/internal.h"


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
 *
 * WHAT THE LINE SHARES. Every order has the same real part, so the layer shift
 * and each edge's modulus `exp(re * w - qmax)` are one number for all of them,
 * taken once per edge. The shift comes from the REACHED states' weights,
 * which do not depend on any order, rather than from each order's mass as in
 * `jet_sweep_cfan`; on a reached state the two coincide.
 *
 * WHAT THE LINE RECURS. Order g's rotation `exp(i (im0 + g dim) w)` is order
 * g-1's times `exp(i dim w)`: one complex multiply where the reference takes a
 * cosine and a sine. It is taken DIRECTLY at every g that is a multiple of
 * BEAM_LINE_BLOCK, so the chain is at most that long and its rounding stays
 * at a few ulp; a stretch starting between anchors rebuilds its first rotation
 * from the anchor below with the same multiplies, in the same order.
 *
 * WHY THAT MATTERS: an order's value depends on the line and on g alone --
 * not on k0, nq, the thread count or the budget, all of which only decide
 * which stretch it is computed in. It agrees with `jet_sweep_cfan` to
 * rounding, not to the bit: the rotation is the same number reached by a
 * different sequence of operations.
 */
#define BEAM_LINE_BLOCK 32u

BeamError jet_sweep_line(const BeamModel *m, uint32_t len, double re,
                         double im0, double dim, uint64_t k0, uint32_t nq,
                         double *out_log_mod, double *out_phase,
                         size_t out_stride) {
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

    BEAM_BUDGET_OR_REFUSE("a complex tilted line sweep", len,
                          beam_jet_cfan_bytes(S, R, W));

    double  *w     = (double *)malloc((size_t)S * R * sizeof(double));
    double  *tail  = (double *)malloc((size_t)S * sizeof(double));
    double  *cur   = (double *)calloc((size_t)S * W * 2u, sizeof(double));
    double  *next  = (double *)calloc((size_t)S * W * 2u, sizeof(double));
    uint8_t *alive = (uint8_t *)malloc(S);
    double  *pt    = (double *)malloc((size_t)W * 4u * sizeof(double));
    uint8_t *dead  = (uint8_t *)calloc(W, 1);
    LiveHandle h;
    if (!w || !tail || !cur || !next || !alive || !pt || !dead
        || beam_live_acquire(m, len, &h) != BEAM_OK) {
        free(w); free(tail); free(cur); free(next); free(alive);
        free(pt); free(dead);
        return BEAM_EALLOC("the complex line-sweep buffers");
    }
    const BeamLive *L = h.L;
    double *scale = pt;
    double *nmax  = pt + 1u * W;
    double *fre   = pt + 2u * W;
    double *fim   = pt + 3u * W;
    const size_t W2 = (size_t)W * 2u;
    uint32_t nalive = W;

    for (uint32_t j = 0; j < W; ++j) {
        scale[j] = 0.0;
        cur[(size_t)(S - 1) * W2 + 2u * j] = 1.0;
    }

    /* The rotation of order k0 (the stretch's first) for weight v: direct at
     * the anchor at or below k0, then the recurrence up to k0 -- the same
     * operations a stretch starting at the anchor performs. */
#define LINE_FIRST_ROT(v, rr, ri) do {                                        \
        const uint64_t anc_ = k0 - k0 % BEAM_LINE_BLOCK;                       \
        const double a_ = (im0 + (double)anc_ * dim) * (v);                    \
        rr = cos(a_); ri = sin(a_);                                            \
        if (anc_ != k0) {                                                      \
            const double dr_ = cos(dim * (v)), di_ = sin(dim * (v));           \
            for (uint64_t g_ = anc_ + 1u; g_ <= k0; ++g_) {                    \
                const double t_ = rr * dr_ - ri * di_;                         \
                ri = rr * di_ + ri * dr_; rr = t_;                             \
            }                                                                  \
        }                                                                      \
    } while (0)

    for (uint32_t i = 0; i < len; ++i) {
        beam_reached_mask(L, i, S, alive);
        beam_layer_weights_live(m, i, len, w, NULL, alive);

        double hi = -INFINITY, lo = INFINITY;
        const uint32_t  nr = L->n_reached[i];
        const uint32_t *rv = L->reached[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(max:hi) \
                reduction(min:lo) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nr; ++k) {
            const double *row = w + (size_t)rv[k] * R;
            for (uint32_t x = 0; x < R; ++x) {
                if (row[x] > hi) hi = row[x];
                if (row[x] != -INFINITY && row[x] < lo) lo = row[x];
            }
        }
        if (hi == -INFINITY) goto done;
        const double qmax = re >= 0.0 ? re * hi
                          : (lo == INFINITY ? -INFINITY : re * lo);
        if (qmax == -INFINITY) goto done;

        const uint32_t  nt  = L->n_reached[i + 1];
        const uint32_t *tgt = L->reached[i + 1];
        for (uint32_t j = 0; j < W; ++j) nmax[j] = 0.0;
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
                const double mag = exp(re * wv - qmax);
                const double dr = cos(dim * wv), di = sin(dim * wv);
                double rr, ri;
                LINE_FIRST_ROT(wv, rr, ri);
                for (uint32_t j = 0; j < W; ++j) {
                    if (j) {
                        if ((k0 + j) % BEAM_LINE_BLOCK == 0u) {
                            const double ang = (im0 + (double)(k0 + j) * dim) * wv;
                            rr = cos(ang); ri = sin(ang);
                        } else {
                            const double tr = rr * dr - ri * di;
                            ri = rr * di + ri * dr; rr = tr;
                        }
                    }
                    if (dead[j]) continue;
                    const double cr = c[2u * j], ci = c[2u * j + 1u];
                    if (cr == 0.0 && ci == 0.0) continue;
                    const double er = mag * rr, ei = mag * ri;
                    nx[2u * j]      += er * cr - ei * ci;
                    nx[2u * j + 1u] += er * ci + ei * cr;
                }
            }
            for (uint32_t j = 0; j < W; ++j) {
                if (dead[j]) continue;
                const double v = fmax(fabs(nx[2u * j]), fabs(nx[2u * j + 1u]));
                if (v > nmax[j]) nmax[j] = v;
            }
        }
        for (uint32_t j = 0; j < W; ++j) {
            if (dead[j]) continue;
            scale[j] += qmax;
            if (!(nmax[j] > 0.0)) { dead[j] = 1; --nalive; }
        }
        if (nalive == 0u) goto done;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t k = 0; k < (int64_t)nt; ++k) {
            double *nx = next + (size_t)tgt[k] * W2;
            for (uint32_t j = 0; j < W; ++j) {
                if (dead[j]) continue;
                nx[2u * j]      = beam_rescale(nx[2u * j], nmax[j]);
                nx[2u * j + 1u] = beam_rescale(nx[2u * j + 1u], nmax[j]);
            }
        }
        for (uint32_t j = 0; j < W; ++j)
            if (!dead[j]) scale[j] += log(nmax[j]);

        for (uint32_t k = 0; k < nr; ++k)
            memset(cur + (size_t)rv[k] * W2, 0, W2 * sizeof(double));
        double *tmp = cur; cur = next; next = tmp;
    }

    /* Sink: one more edge whose weight is the tail, with the shift taken over
     * the reached end states, again independent of every order. */
    beam_tail_weights(m, len, tail);
    {
        const uint32_t  ns = L->n_reached[len];
        const uint32_t *sv = L->reached[len];
        double tmax = -INFINITY;
        for (uint32_t k = 0; k < ns; ++k) {
            const double tv = tail[sv[k]];
            if (tv != -INFINITY && re * tv > tmax) tmax = re * tv;
        }
        if (tmax == -INFINITY) goto done;
        for (uint32_t j = 0; j < W; ++j) { fre[j] = 0.0; fim[j] = 0.0; }
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t s = sv[k];
            const double tv = tail[s];
            if (tv == -INFINITY) continue;
            const double *c = cur + (size_t)s * W2;
            const double mag = exp(re * tv - tmax);
            const double dr = cos(dim * tv), di = sin(dim * tv);
            double rr, ri;
            LINE_FIRST_ROT(tv, rr, ri);
            for (uint32_t j = 0; j < W; ++j) {
                if (j) {
                    if ((k0 + j) % BEAM_LINE_BLOCK == 0u) {
                        const double ang = (im0 + (double)(k0 + j) * dim) * tv;
                        rr = cos(ang); ri = sin(ang);
                    } else {
                        const double tr = rr * dr - ri * di;
                        ri = rr * di + ri * dr; rr = tr;
                    }
                }
                if (dead[j]) continue;
                const double cr = c[2u * j], ci = c[2u * j + 1u];
                if (cr == 0.0 && ci == 0.0) continue;
                const double er = mag * rr, ei = mag * ri;
                fre[j] += er * cr - ei * ci;
                fim[j] += er * ci + ei * cr;
            }
        }
        for (uint32_t j = 0; j < W; ++j) {
            if (dead[j]) continue;
            const double mod = hypot(fre[j], fim[j]);
            if (!(mod > 0.0)) continue;
            out_log_mod[(size_t)j * out_stride] = scale[j] + tmax + log(mod);
            out_phase[(size_t)j * out_stride] = atan2(fim[j], fre[j]);
        }
    }
#undef LINE_FIRST_ROT

done:
    free(w); free(tail); free(cur); free(next); free(alive);
    free(pt); free(dead);
    beam_live_release(&h);
    return BEAM_OK;
}
