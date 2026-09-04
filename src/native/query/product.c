/**
 * @file product.c
 * @brief The trellis crossed with an automaton: sweeps over `(state, q)`.
 *
 * A constraint that relates positions to each other cannot be imposed by
 * deleting edges, because whether an edge is admissible depends on what was
 * read before it. The standard answer is to carry the automaton's state
 * alongside the trellis state and run the same recursion on the pair, which
 * is what this file does. The product has `n_states * |Q|` states per layer
 * and the sweep is otherwise the one in engine.c.
 *
 * There is no inverse transition table, and the design note that called for
 * one was wrong. The forward sweep must accumulate by TARGET so no two threads
 * write one accumulator, and inverting the automaton looked necessary for
 * that -- several `q` may reach one `q'` on the same residue. But the whole
 * automaton dimension of a target trellis state is written by the one thread
 * that owns that state, so the sweep can scatter forwards into
 * `next[t * Q ..]` and still have no two threads meet. Keeping the q-dimension
 * private to the target removes the inverse tables, their construction and
 * their memory.
 *
 * At `|Q| = 1` this file performs the same operations in the same order as
 * engine.c's mask sweep, and test_query.c requires the two to agree bit for
 * bit. That is a cross-check between two independent implementations of one
 * recursion, not a formality: it is what keeps the product path honest as it
 * grows.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/internal.h"
#include "beam/query.h"
#include "beam/query_internal.h"

/** The residues layer `i` may emit, as a set. NULL masks admit everything. */
static inline BeamResidueMask layer_set(const BeamResidueMask *mask,
                                        uint32_t i, uint32_t R) {
    return mask ? mask[i] : BEAM_RESIDUES_ALL(R);
}

/* ── Backward values over the product ────────────────────────── */

BeamError beam_query_betas_product(const BeamModel *m, uint32_t length,
                                   const BeamResidueMask *mask,
                                   const BeamAutomaton *aut,
                                   double *w, double *tail, double **beta) {
    const uint32_t A  = m->n_alphabet;
    const uint32_t R  = m->n_residues;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t Q  = aut->n_states;

    /*
     * The sink layer is where acceptance enters, and it is the only place it
     * does: a path that ends in a non-accepting automaton state contributes
     * no weight, and everything upstream follows from that one assignment.
     */
    beam_tail_weights(m, length, tail);
    for (uint32_t s = 0; s < S; ++s) {
        double *row = beta[length] + (size_t)s * Q;
        for (uint32_t q = 0; q < Q; ++q)
            row[q] = aut->accept[q] ? tail[s] : -INFINITY;
    }

    for (int32_t i = (int32_t)length - 1; i >= 0; --i) {
        beam_layer_weights(m, (uint32_t)i, length, w, NULL);
        const BeamResidueMask allowed = layer_set(mask, (uint32_t)i, R);
        const double *nb = beta[i + 1];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s64 = 0; s64 < (int64_t)S; ++s64) {
            const uint32_t s = (uint32_t)s64;
            const double *row = w + (size_t)s * R;
            const uint32_t base = (s % SK) * A;
            double *out = beta[i] + (size_t)s * Q;
            for (uint32_t q = 0; q < Q; ++q) {
                double mx = -INFINITY;
                for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u) {
                    const uint32_t x = (uint32_t)__builtin_ctz(mm);
                    if (row[x] == -INFINITY) continue;
                    const uint32_t qn =
                        beam_query_automaton_step(aut, (uint32_t)i, q, x);
                    const double nv = nb[(size_t)(base + x) * Q + qn];
                    if (nv == -INFINITY) continue;
                    const double v = row[x] + nv;
                    if (v > mx) mx = v;
                }
                if (mx == -INFINITY) { out[q] = -INFINITY; continue; }
                double acc = 0.0;
                for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u) {
                    const uint32_t x = (uint32_t)__builtin_ctz(mm);
                    if (row[x] == -INFINITY) continue;
                    const uint32_t qn =
                        beam_query_automaton_step(aut, (uint32_t)i, q, x);
                    const double nv = nb[(size_t)(base + x) * Q + qn];
                    if (nv == -INFINITY) continue;
                    acc += exp(row[x] + nv - mx);
                }
                out[q] = mx + log(acc);
            }
        }
    }
    return BEAM_OK;
}

/* ── The forward sweep over the product ──────────────────────── */

double beam_query_layer_max_product(const double *w, const double *cur,
                                    uint32_t n_states, uint32_t n_residues,
                                    uint32_t n_aut, BeamResidueMask allowed) {
    double mx = -INFINITY;
    for (uint32_t s = 0; s < n_states; ++s) {
        const double *cs = cur + (size_t)s * n_aut;
        bool reached = false;
        for (uint32_t q = 0; q < n_aut; ++q)
            if (cs[q] != 0.0) { reached = true; break; }
        if (!reached) continue;
        const double *row = w + (size_t)s * n_residues;
        for (BeamResidueMask mm = allowed; mm; mm &= mm - 1u)
            if (row[(uint32_t)__builtin_ctz(mm)] > mx)
                mx = row[(uint32_t)__builtin_ctz(mm)];
    }
    return mx;
}

bool beam_query_advance_product(const BeamModel *m, const BeamAutomaton *aut,
                                uint32_t layer, const double *w, double wmax,
                                const double *cur, double *next,
                                BeamResidueMask allowed, double *log_scale) {
    const uint32_t A  = m->n_alphabet;
    const uint32_t R  = m->n_residues;
    const uint32_t S  = m->n_states;
    const uint32_t SK = m->state_shift;
    const uint32_t Q  = aut->n_states;

    double nmax = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(max:nmax) \
            num_threads(beam_threads())
#endif
    for (int64_t t = 0; t < (int64_t)S; ++t) {
        double *dst = next + (size_t)t * Q;
        const uint32_t x = (uint32_t)t % A;
        /* The sentinel sits at index R and no mask has a bit there, so this
         * one test excludes both a forbidden residue and a non-residue. */
        if (!((allowed >> x) & 1u)) {
            for (uint32_t q = 0; q < Q; ++q) dst[q] = 0.0;
            continue;
        }
        for (uint32_t q = 0; q < Q; ++q) dst[q] = 0.0;

        const uint32_t rest = (uint32_t)t / A;
        for (uint32_t a = 0; a < A; ++a) {
            const uint32_t src = a * SK + rest;
            const double wv = w[(size_t)src * R + x];
            if (wv == -INFINITY) continue;
            const double e = exp(wv - wmax);
            const double *cs = cur + (size_t)src * Q;
            /*
             * Scattered forwards into this target's own slice. Every write
             * lands in `dst`, which belongs to the one thread holding `t`, so
             * the automaton's non-invertibility costs nothing here.
             */
            for (uint32_t q = 0; q < Q; ++q) {
                if (cs[q] == 0.0) continue;
                dst[beam_query_automaton_step(aut, layer, q, x)] += cs[q] * e;
            }
        }
        for (uint32_t q = 0; q < Q; ++q) if (dst[q] > nmax) nmax = dst[q];
    }
    *log_scale += wmax;
    if (!(nmax > 0.0)) return false;
    const size_t n_prod = (size_t)S * Q;
    for (size_t p = 0; p < n_prod; ++p) next[p] /= nmax;
    *log_scale += log(nmax);
    return true;
}
