/**
 * @file edge_weights.c
 * @brief Edge weights over the trellis: per state, per layer, and the tail.
 */
#include "beam/internal.h"
#include "beam/mixing.h"

#ifdef _OPENMP
#include <omp.h>
#endif



/* ── Specialising this file on the alphabet ──────────────────── */

typedef enum { ALPHA_AA = 0, ALPHA_NT = 1, ALPHA_GEN = 2 } AlphaKind;

static inline AlphaKind alpha_kind(const BeamModel *m) {
    if (m->n_alphabet == BEAM_AA_RADIX) return ALPHA_AA;
    if (m->n_alphabet == BEAM_NT_RADIX) return ALPHA_NT;
    return ALPHA_GEN;
}

#define BEAM_KCAT2(a, b) a##b
#define BEAM_KCAT(a, b)  BEAM_KCAT2(a, b)
#define BEAM_KFN(name)   BEAM_KCAT(name, BEAM_K_SUF)

#define BEAM_K_SUF _aa
#define BEAM_K_A   BEAM_AA_RADIX
#define BEAM_K_R   BEAM_AA_RESIDUES
#include "edge_weights_kernels.h"
#undef BEAM_K_SUF
#undef BEAM_K_A
#undef BEAM_K_R

#define BEAM_K_SUF _nt
#define BEAM_K_A   BEAM_NT_RADIX
#define BEAM_K_R   BEAM_NT_RESIDUES
#include "edge_weights_kernels.h"
#undef BEAM_K_SUF
#undef BEAM_K_A
#undef BEAM_K_R

#define BEAM_K_SUF _gen
#define BEAM_K_A   m->n_alphabet
#define BEAM_K_R   m->n_residues
#include "edge_weights_kernels.h"
#undef BEAM_K_SUF
#undef BEAM_K_A
#undef BEAM_K_R

/* ── Public entry points ─────────────────────────────────────── */

#define BEAM_DISPATCH(m, fn, ...)                                             \
    do {                                                                      \
        switch (alpha_kind(m)) {                                              \
        case ALPHA_AA: fn##_aa(__VA_ARGS__);  break;                          \
        case ALPHA_NT: fn##_nt(__VA_ARGS__);  break;                          \
        default:       fn##_gen(__VA_ARGS__); break;                          \
        }                                                                     \
    } while (0)

void beam_state_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            uint32_t state, double *out) {
    BEAM_DISPATCH(m, state_weights, m, i, len, state, out);
}

void beam_layer_weights(const BeamModel *m, uint32_t i, uint32_t len,
                            double *out, double *out_max) {
    BEAM_DISPATCH(m, layer_weights, m, i, len, out, out_max);
}

void beam_layer_weights_live(const BeamModel *m, uint32_t i, uint32_t len,
                                 double *out, double *out_max,
                                 const uint8_t *alive) {
    BEAM_DISPATCH(m, layer_weights_live, m, i, len, out, out_max, alive);
}

void beam_tail_weights(const BeamModel *m, uint32_t len, double *out) {
    BEAM_DISPATCH(m, tail_weights, m, len, out);
}

double beam_layer_max(const BeamModel *m, uint32_t i, uint32_t len) {
    switch (alpha_kind(m)) {
    case ALPHA_AA: return layer_max_aa(m, i, len);
    case ALPHA_NT: return layer_max_nt(m, i, len);
    default:       return layer_max_gen(m, i, len);
    }
}
