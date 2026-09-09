/**
 * @file rescale.c
 * @brief The per-layer rescale, and the invariant it is not allowed to break.
 *
 * Every linear forward sweep in this library divides its accumulator by the
 * layer's largest entry and carries `log(nmax)` alongside, which is what keeps
 * the mass in range over a trellis whose paths span hundreds of nats. The
 * division has one failure mode, and it is silent: a state whose accumulator is
 * far enough below the layer maximum divides to exactly 0.0, and every sweep
 * here treats 0.0 as "no mass at this state" and stops propagating it. The
 * answer then comes back finite, plausible, and short.
 *
 * That is not hypothetical. It is the same arithmetic that made
 * `beam_query_marginals` return rows summing to 5.9e-264 instead of 1, and the
 * states that vanished were not marginal ones -- they included the pinned
 * positions the query was about. Both constrained sweeps were moved to log
 * space because of it; the unconstrained ones here were not, on the grounds
 * that a state 300 orders of magnitude below its layer maximum contributes
 * nothing to an unconstrained total.
 *
 * That reasoning is sound and it is also the whole risk: it holds only while
 * nothing that matters gets that far down. Measured across amino-acid and
 * nucleotide models at orders 2 to 6 and lengths to 90, no state has ever been
 * flushed -- but in the OPEN world the smallest surviving value is 4.94e-324,
 * which is the smallest positive double there is. The margin is one unit in the
 * last place. A change to the smoothing, the length range or the tilt could
 * cross it, and nothing in the published answers would look different.
 *
 * So the division goes through one function that counts the crossings, and a
 * test asserts the count is zero over a matrix of configurations including the
 * ones with no margin left. The counter is not a diagnostic for callers; it
 * exists because this invariant is INVISIBLE in the answers, which is the same
 * reason `beam_live_selfcheck` exists.
 */
#include "beam/internal.h"


/*
 * Process-global rather than per-model or per-sweep, deliberately: a sweep does
 * not own the invariant, the arithmetic does, and every sweep in the library
 * shares one answer to "has anything been lost". It is not thread-safe to READ
 * while sweeps are running, which does not matter for its only caller -- a test
 * that resets, runs one thing, and reads.
 */
static unsigned long long g_flushes = 0;

void beam_rescale_flush_reset(void) { g_flushes = 0; }

unsigned long long beam_rescale_flushes(void) { return g_flushes; }

void beam_rescale_note_flush(void) {
#ifdef _OPENMP
    #pragma omp atomic
#endif
    g_flushes++;
}
