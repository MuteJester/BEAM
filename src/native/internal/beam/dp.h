/**
 * @file dp.h
 * @brief Dynamic-programming entry points shared across files.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_DP_H
#define BEAM_INTERNAL_DP_H

#include "beam/model.h"
#include "beam/model_state.h"
#include "beam/table_access.h"
#include "beam/memory.h"

/* ── The per-layer rescale ───────────────────────────────────── */

/**
 * Crossings of the one invariant the linear sweeps depend on, since the last
 * reset. Zero at all times on every configuration measured; see rescale.c for
 * why that is checked rather than assumed, and for how little margin there is.
 *
 * A test hook, not a diagnostic for callers, and not thread-safe to read while
 * a sweep is running.
 */
unsigned long long beam_rescale_flushes(void);
void beam_rescale_flush_reset(void);
void beam_rescale_note_flush(void);

/**
 * Divide one accumulator by its layer's scale, noticing a value that vanishes.
 *
 * Bit-identical to `v / nmax`, which it has to be: these sweeps are pinned to
 * the last bit by `verification/check.py`, so a guard that computed the
 * quotient any other way -- multiplying by a reciprocal, say -- would move
 * every recorded number in the library and the guard would be indistinguishable
 * from a regression.
 *
 * Pass only accumulators that carry MASS. A jet's higher moments and the
 * expectation semiring's weighted sum can legitimately be zero or negative
 * without the state being dead, and flagging those would report a loss that
 * did not happen. Their companion mass vector is the one to guard.
 */
static inline double beam_rescale(double v, double nmax) {
    const double out = v / nmax;
    /* Predicted not-taken and never yet taken; the atomic is behind it. */
    if (out == 0.0 && v > 0.0) beam_rescale_note_flush();
    return out;
}

/* ── Cross-file helpers ──────────────────────────────────────── */

/**
 * Mixed log-weight of one residue in one direction, before the direction
 * weight is applied. `ctx_wide`, `ctx_mid` and `ctx_len` are pre-packed
 * context indices of order `order`, 3 and 2 respectively.
 */
double beam_mix(const BeamModel *m, uint32_t dir, uint32_t depth,
                    uint32_t ctx, uint32_t x);

/**
 * Edge weights of one DAG layer: `out[state * 20 + x]`, the full weight of
 * emitting `x` from `state` at position `i` of a length-`len` sequence.
 * Includes the forward term for the residue being emitted and the backward
 * term for the residue emitted `order` steps ago, which only now has its
 * complete right-hand context.
 *
 * `out` must hold `n_states * 20` doubles.
 *
 * `out_max`, if given, receives the largest weight in the layer. Every sweep
 * needs it to rescale, and computing it here costs nothing while the values
 * are still in register: computing it afterwards means a second pass over 31
 * MB per layer, which measured at a fifth of the sweep. A maximum is
 * order-independent, so the fused value is bit-identical to the separate one.
 * `-INFINITY` means the layer has no live edge at all.
 */
/** The layer's maximum edge weight, computed without building the layer. */
double beam_layer_max(const BeamModel *m, uint32_t i, uint32_t len);

void beam_layer_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            double *out, double *out_max);

/**
 * As above, but skipping states `alive` marks dead. When `out_max == NULL`,
 * this is safe only where the caller provably never reads a dead row. When
 * `out_max != NULL`, the maximum is over the MARKED states only -- the rows it
 * builds -- not over the whole layer, in either storage form. That is the
 * right rescale for a caller that consumes only those rows, and the only one
 * that does not defeat the point of marking: under the hierarchical prior
 * every state is live, so an all-state maximum would build the whole layer.
 * `beam_layer_max` is the all-state maximum when that is what is wanted.
 */
void beam_layer_weights_live(const BeamModel *m, uint32_t i, uint32_t len,
                                 double *out, double *out_max,
                                 const uint8_t *alive);

/**
 * Weights of the 20 edges leaving ONE state, identical to the corresponding
 * row of `beam_layer_weights`.
 *
 * For a consumer that visits states in an unpredictable order — the
 * best-first search in `algorithms/top_k.c` is the only one — building the
 * whole layer to
 * read one row of it is 3.9 million wasted edges per visit.
 */
void beam_state_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            uint32_t state, double *out);

/**
 * Sink weights: `out[state]`, settling the backward terms of the final
 * `min(order, len)` residues, which never get a full right-hand context
 * during the walk. `out` must hold `n_states` doubles.
 */
void beam_tail_weights(const BeamModel *m, uint32_t len, double *out);

/**
 * The largest `q * w` among the edges a tilted sweep will actually use.
 *
 * A tilted sweep rescales its layer by the maximum of the TILTED weights, and
 * for `q >= 0` that is `q * wmax`, since `q * w` rises with `w`. For `q < 0`
 * it rises as `w` FALLS, so the maximum sits at the smallest weight and
 * `q * wmax` is a minimum instead. Rescaling by it makes every
 * `exp(q * w - qmax)` an `exp` of a large positive number, and the layer
 * overflows to infinity.
 *
 * That is not a theoretical concern. Three sweeps did exactly this, and in the
 * open world of an earlier build, with backoff off, unobserved cells sat at
 * `ln(1e-300)`, so any `|q| > 1.03`
 * overflowed. `beam_model_log_mellin(-5)` returned `-inf` and
 * `beam_model_hill_number(-5)` returned `0.0`, both with BEAM_OK, where the
 * true values are finite; and at `q = -1.5` the answer came back finite and
 * WRONG, which is worse. `beam_betas_tilted` never had the defect because it
 * maximises the tilted quantity directly.
 *
 * Free when `q >= 0`, which is every quantity BEAM computes by default: the
 * caller has already found `wmax`, and this returns `q * wmax` without
 * looking at the layer. Only a negative tilt pays for the scan.
 *
 * `cur` restricts the scan to states the sweep has reached, whose edges are
 * the only ones that enter any sum. Returns `-INFINITY` when no reached state
 * has a live edge.
 */
static inline double beam_tilted_layer_max(const double *w, const double *cur,
                                           uint32_t n_states,
                                           uint32_t n_residues,
                                           double q, double wmax) {
    if (q >= 0.0) return q * wmax;
    double wmin = INFINITY;
    for (uint32_t s = 0; s < n_states; ++s) {
        if (cur && cur[s] == 0.0) continue;
        const double *row = w + (size_t)s * n_residues;
        for (uint32_t x = 0; x < n_residues; ++x)
            if (row[x] != -INFINITY && row[x] < wmin) wmin = row[x];
    }
    return wmin == INFINITY ? -INFINITY : q * wmin;
}

/** Compute `log_z[len]`. Returns BEAM_OK or an allocation error. */
/* `layer_logmass`, when non-NULL, receives `len` entries: the running log of
 * the trellis mass once residue i has been emitted, with the sink folded into
 * the last. Differencing it gives the per-layer factors log Z is a product of,
 * and the differences telescope to `*out` exactly. */
BeamError beam_compute_log_z(BeamModel *m, uint32_t len, double *out,
                             double *layer_logmass);

/**
 * Backward values of the trellis: `beta[i][state]`, the log of the total
 * weight of every completion from `state` at layer `i`. `beta` must be an
 * array of `len + 1` pointers, each to `n_states` doubles.
 */
/*
 * ALL THREE TAKE THE CALLER'S ACCOUNT, and that is the whole point of the
 * parameter. Each allocates a layer of edge weights -- `S * R` doubles, 31 MB
 * on an order-4 amino-acid model -- for the duration of the sweep, while the
 * caller is already holding a backward field it charged for separately. Two
 * checks against the same budget are not a budget: measured on an order-4
 * model, a 33.65 MB budget admitted a top-k search that held 46.87 MB, 39%
 * over, because the field passed one check and the layer passed the other.
 *
 * There is no null-account form. An optional account is what `positional.c`
 * had first, where the reader tested for null and the writer did not, and the
 * one caller that passed nothing dereferenced it.
 */
BeamError beam_betas(const BeamModel *m, uint32_t len, double **beta,
                     BeamMemAccount *acct);

/**
 * Backward values of the `q`-TILTED trellis, i.e. with every edge weight
 * scaled by `q` before exponentiating. `q = 1` reproduces `beam_betas`.
 *
 * Needed wherever a posterior under a tilted law is required, such as
 * attributing surprisal to depths at `q != 1`.
 */
BeamError beam_betas_tilted(const BeamModel *m, uint32_t len, double q,
                            double **beta, BeamMemAccount *acct);

/**
 * `best[i][s]`: the weight of the BEST completion from state `s` at layer `i`,
 * sink term included. Max-plus where the two above are log-sum-exp.
 *
 * Which one guides a top-k search is the difference between a second and a
 * minute. `acc + beta` bounds the best completion from above but not tightly
 * -- beta sums over every completion, so on a 194,481-state trellis it
 * overshoots by many nats and the frontier degenerates towards breadth-first,
 * expanding millions of prefixes that can never win. `acc + best` is the exact
 * weight of the best completion, so the pop order is the true order and every
 * expansion is one the answer needed.
 *
 * Exactness is unchanged and so is the reported probability: at the final
 * layer there is nothing left to choose, so `best[len]` IS the sink weight,
 * exactly as `beta[len]` is.
 *
 * It differs from its two siblings in three ways and each is deliberate: the
 * semiring, the absence of a tilt, and that it visits only the states the
 * forward frontier reached. That last one means `best[i][s]` is `-INFINITY`
 * for every unreached state, where `beta[i][s]` may be finite -- so the
 * relation between them is `best <= beta` and not equality outside the seed.
 *
 * `best` must hold `len + 1` rows of `n_states` doubles, which the caller
 * owns and budgets; this budgets only its own layer of edge weights.
 */
BeamError beam_best_completions(const BeamModel *m, uint32_t len,
                                double **best, BeamMemAccount *acct);

/**
 * Lengths the model actually generates.
 *
 * `finalize` zeroes the prior of any length whose trellis has no surviving
 * path, so a finite prior implies a finite partition function and this single
 * test is sufficient. Before that renormalisation the two could disagree, and
 * a length with prior mass but no support would poison every mean it entered.
 */
static inline bool beam_length_live(const BeamModel *m, uint32_t len) {
    return len >= 1 && len <= m->max_len && isfinite(m->log_pi[len]);
}


/* ── Moment jet ──────────────────────────────────────────────── */

/** Highest moment order the jet sweep carries. */
#define BEAM_MAX_MOMENT 16u

/**
 * One forward pass carrying `n_moments + 1` moments of the q-tilted law.
 *
 * The arithmetic here is a separate numerical path from `tilted_log_z` in
 * math/diversity.c even though both compute tilted quantities. They are not
 * to be consolidated on the grounds that they look alike: each has its own
 * regression value.
 */
/** Binomial coefficients up to `n`, for the moment recurrence. */
void beam_binomials(double c[][BEAM_MAX_MOMENT + 1], uint32_t n);

BeamError jet_sweep(const BeamModel *m, uint32_t len, double q,
                          uint32_t order, double *out_moments,
                          double *out_log_z);

/**
 * Peak bytes one `beam_compute_log_z` sweep holds, from the same expression
 * it allocates from.
 *
 * Here rather than beside the allocation because there are now two callers
 * that must agree: the sweep, which refuses a length it cannot fit, and
 * `beam_model_finalize`, which divides the budget across the lengths it runs
 * concurrently. Two copies of this arithmetic would drift, and the direction
 * they would drift in is a driver that admits more lengths than the sweep
 * will accept.
 *
 * The layer of edge weights dominates: `S * R` doubles is 31 MB on an order-4
 * amino-acid model and grows with the alphabet's ceiling order, while the
 * state vectors and the frontier index arrays are linear in `S`.
 */
static inline uint64_t beam_logz_sweep_bytes(uint32_t n_states,
                                             uint32_t n_residues,
                                             uint32_t state_shift) {
    return (uint64_t)n_states * n_residues * sizeof(double)
         + 3ull * n_states * sizeof(double)
         + 3ull * n_states * sizeof(uint32_t)
         + (uint64_t)n_states * sizeof(uint8_t)       /* needed-row mask */
         + 2ull * state_shift * sizeof(uint32_t)
         + ((uint64_t)state_shift + 1u) * sizeof(uint32_t);
}

/**
 * Peak bytes one TILTED forward sweep holds: a layer of edge weights,
 * `n_vec` state vectors, and one byte per state for the alive mask.
 *
 * `n_vec` is 4 for the tilted partition function and 6 for the min-plus /
 * max-plus range, which is the only thing that differs between the two
 * callers. The layer of edge weights dominates both.
 *
 * Here rather than beside either allocation because there are four callers in
 * two files -- each sweep's own guard and each driver's concurrency bound --
 * and they must agree. `math/diversity.c` and `math/extrema.c` each carried a
 * byte-identical private copy of this, each with a comment saying the figure
 * came from one expression "so what is refused and what is scheduled agree by
 * construction". Two copies is what that sentence rules out, and budget.h says
 * so directly: "A budget check computed from its own copy of the formula is
 * not a check."
 */
static inline uint64_t beam_tilted_sweep_bytes(uint32_t n_states,
                                               uint32_t n_residues,
                                               uint32_t n_vec) {
    return (uint64_t)n_states * n_residues * sizeof(double)
         + (uint64_t)n_vec * n_states * sizeof(double)
         + n_states;
}

/**
 * Peak bytes one `beam_model_path_count` sweep holds: the residue-mask bits
 * and three state vectors. No layer of edge weights, because path counting
 * never asks what an edge weighs -- which is why this is so much smaller than
 * every other sweep in the library.
 */
static inline uint64_t beam_path_count_bytes(uint32_t n_states) {
    return (uint64_t)n_states * sizeof(uint32_t)
         + 3ull * n_states * sizeof(double);
}

/**
 * Peak bytes one `jet_sweep` holds, from the same expression it allocates
 * from.
 *
 * Here rather than beside the allocation because four callers -- the Mellin
 * transform, the tilted cumulants and both moment entry points -- run these
 * sweeps concurrently over lengths and must divide the memory budget by one
 * sweep's size. A caller computing that from its own copy of the formula is
 * how a guard and the thing it guards drift apart.
 *
 * `n_moments` is `order + 1`, the width of the jet the sweep carries.
 */
static inline uint64_t beam_jet_sweep_bytes(uint32_t n_states,
                                            uint32_t n_residues,
                                            uint32_t n_moments) {
    const uint64_t S = n_states;
    return S * (uint64_t)n_residues * sizeof(double)     /* one layer   */
         + S * sizeof(double)                            /* the sink    */
         + 2ull * n_moments * S * sizeof(double)         /* cur, next   */
         + S;                                            /* alive       */
}

/**
 * `jet_sweep` at `order = 0` for `nq` tilts in ONE traversal of the
 * length-`len` trellis: `out_log_z[j * out_stride]` is bit-identical to the
 * `out_log_z` of `jet_sweep(m, len, qs[j], 0u, ...)`. The layer of edge
 * weights and the frontier are shared; every tilt keeps its own rescaling.
 * See the definition in dp/jet.c for why that is enough to keep the bits.
 */
BeamError jet_sweep_fan(const BeamModel *m, uint32_t len, const double *qs,
                        uint32_t nq, double *out_log_z, size_t out_stride);

/**
 * Peak bytes one `jet_sweep_fan` over `n_tilts` tilts holds, from the same
 * expressions it allocates from.
 *
 * `beam_jet_sweep_bytes` with one mass plane per TILT where that has one per
 * moment, plus seven scalars per tilt and the list of tilts still alive. The
 * mass planes are what grow: at 31 tilts on an order-4 amino-acid model they
 * are 97 MB of a 128 MB sweep, against 3 MB for one tilt.
 *
 * Two callers must agree on it -- the sweep's own guard and the driver in
 * math/mellin.c, which sizes both the fusion width and the concurrency from
 * it -- so it lives here and not beside either. OpenMP's private copies of the
 * per-tilt reductions are not counted: `n_tilts` doubles per thread, a few
 * hundred bytes against a sweep measured in megabytes.
 */
static inline uint64_t beam_jet_fan_bytes(uint32_t n_states,
                                          uint32_t n_residues,
                                          uint32_t n_tilts) {
    const uint64_t S = n_states;
    const uint64_t T = n_tilts;
    return S * (uint64_t)n_residues * sizeof(double)     /* one layer   */
         + S * sizeof(double)                            /* the sink    */
         + 2ull * T * S * sizeof(double)                 /* cur, next   */
         + S                                             /* alive       */
         + T * (7u * sizeof(double) + sizeof(uint32_t)); /* per tilt    */
}

/**
 * `jet_sweep_fan` at complex tilts `q_re[j] + i q_im[j]`: per length, the
 * log-modulus and the phase of `sum over paths of exp(q * raw)`. See the
 * definition in dp/jet.c for the rescaling and what cancellation costs.
 */
BeamError jet_sweep_cfan(const BeamModel *m, uint32_t len, const double *q_re,
                         const double *q_im, uint32_t nq, double *out_log_mod,
                         double *out_phase, size_t out_stride);

typedef struct BeamLineWork BeamLineWork;

/**
 * `jet_sweep_cfan` for `nq` orders `re + i (im0 + g dim)`, `g = k0 ..`, on one
 * vertical line: the modulus factor shared by the line, the rotations by
 * sixteen independent recurrences anchored directly every 128 orders, all
 * threads sharing out each layer's targets, in the workspace `wk`. `k0`
 * must be a multiple of BEAM_LINE_LANES. An order's bits depend on the line
 * and on g only. See dp/jet.c.
 */
BeamError jet_sweep_line(const BeamModel *m, BeamLineWork *wk, uint32_t len,
                         double re, double im0, double dim, uint64_t k0,
                         uint32_t nq, double *out_log_mod, double *out_phase,
                         size_t out_stride);

/**
 * A line sweep's planes and scratch for stretches up to `Wp` orders with
 * `n_threads` threads, `beam_jet_line_bytes` of them, made once per line and
 * reused by every (length, stretch): fresh planes each time had the system
 * clear 12.7 GB per length of a 2,048-order foundation line.
 */
BeamLineWork *beam_line_work_new(const BeamModel *m, uint32_t Wp,
                                 uint32_t n_threads);
void beam_line_work_free(BeamLineWork *wk);

/**
 * s[k] = sin x[k], c[k] = cos x[k], four at a time: within two ulp of 1 for
 * |x| up to about 8e5 (libm beyond), the same bits on every CPU and whatever
 * the call's size. See math/vtrig.c.
 */
void beam_sincos_many(const double *x, double *s, double *c, size_t n);

/** Orders a line sweep advances at once; a stretch starts at a multiple. */
#define BEAM_LINE_LANES 16u
/** Orders between a line's directly taken rotations (its anchors). */
#define BEAM_LINE_BLOCK 128u
/** Orders a thread walks a group of targets through at a time. */
#define BEAM_LINE_TILE 256u
/** End states summed together before the sink's partial sums are added. */
#define BEAM_LINE_SINK_CHUNK 256u

/**
 * How `beam_model_log_mellin_line` splits `n_orders` orders: stretches of
 * `width` orders (the last may be shorter), `n_stretches` of them per length,
 * each swept by all `n_threads` threads in turn. A `width` of 0 means not even
 * one chain of orders fits the budget. See math/mellin.c.
 */
typedef struct {
    uint32_t width;
    uint32_t n_stretches;
} BeamLinePlan;

BeamLinePlan beam_line_plan(uint32_t n_states, uint32_t n_residues,
                            uint32_t n_alphabet, uint32_t n_orders,
                            uint32_t n_threads, uint64_t budget);

/**
 * A state's row in a line sweep's planes: the real parts of its `Wp` orders,
 * one cache line, the imaginary parts, one more line. Unpadded, a 2,048-order
 * row was 32 KB and its halves 16 KB apart, so a group's source rows and
 * both halves of each fell in the same cache sets: that width ran at 47.7 ms
 * an order on the foundation against 37.4 at 1,536.
 */
static inline size_t beam_line_im(uint32_t Wp) { return (size_t)Wp + 8u; }
static inline size_t beam_line_row(uint32_t Wp) { return 2u * (size_t)Wp + 16u; }

/** Doubles of one thread's scratch in a line sweep of `Wp` padded orders. */
static inline uint64_t beam_line_scratch_doubles(uint32_t Wp, uint32_t A,
                                                 uint32_t R) {
    const uint64_t AR = (uint64_t)A * R;
    const uint64_t E = AR > BEAM_LINE_SINK_CHUNK ? AR : BEAM_LINE_SINK_CHUNK;
    const uint64_t blocks = BEAM_LINE_TILE / BEAM_LINE_BLOCK + 1u;
    const uint64_t d = 2u * BEAM_LINE_TILE + Wp + 2u * E + 9u * E + 3u * blocks * E
                     + 2u * BEAM_LINE_LANES * E;
    const uint64_t u = E + A + 1u;                    /* uint32 slots */
    return (d + (u + 1u) / 2u + 7u) / 8u * 8u;        /* whole lines  */
}

/**
 * Peak bytes one `jet_sweep_line` over `Wp` padded orders holds with
 * `n_threads` threads: the layer's weights and the sink's, two planes of
 * complex accumulators, the group index, the per-order scalars, and each
 * thread's scratch.
 */
static inline uint64_t beam_jet_line_bytes(uint32_t n_states, uint32_t n_residues,
                                           uint32_t n_alphabet, uint32_t Wp,
                                           uint32_t n_threads) {
    const uint64_t S = n_states;
    return S * (uint64_t)n_residues * sizeof(double)     /* one layer   */
         + S * sizeof(double)                            /* the sink    */
         + 2ull * S * beam_line_row(Wp) * sizeof(double) /* cur, next   */
         + S + (S + 1u) * sizeof(uint32_t)               /* alive, groups */
         + (uint64_t)Wp * (5u * sizeof(double) + sizeof(int) + 1u)
         + (S + BEAM_LINE_SINK_CHUNK - 1u) / BEAM_LINE_SINK_CHUNK
         + (uint64_t)n_threads * sizeof(double)
           * beam_line_scratch_doubles(Wp, n_alphabet, n_residues);
}

/**
 * Peak bytes one `jet_sweep_cfan` over `n_tilts` tilts holds: `beam_jet_fan_bytes`
 * with each mass plane twice as wide (re, im) and one more scalar per tilt.
 */
static inline uint64_t beam_jet_cfan_bytes(uint32_t n_states,
                                           uint32_t n_residues,
                                           uint32_t n_tilts) {
    const uint64_t S = n_states;
    const uint64_t T = n_tilts;
    return S * (uint64_t)n_residues * sizeof(double)     /* one layer   */
         + S * sizeof(double)                            /* the sink    */
         + 4ull * T * S * sizeof(double)                 /* cur, next   */
         + S                                             /* alive       */
         + T * (8u * sizeof(double) + sizeof(uint32_t)); /* per tilt    */
}

#endif /* BEAM_INTERNAL_DP_H */
