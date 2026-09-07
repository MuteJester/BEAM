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
