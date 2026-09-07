/**
 * @file common.h
 * @brief Shared types, macros, and platform abstractions for BEAM.
 */
#ifndef BEAM_COMMON_H
#define BEAM_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

/* ── Result / error codes and the error context ─────────────── */

/*
 * Moved to beam/errors.h, and included here so every existing translation
 * unit still compiles unchanged. The enum, BEAM_FAIL and the thread-local
 * context are all still public and still spelled the same; what changed is
 * that they now live in a file named after them, with the two calling
 * protocols and the append-only rule written down beside them.
 */
#include "beam/errors.h"

/* ── Graph variant ──────────────────────────────────────────── */

typedef enum {
    BEAM_VARIANT_AAP       = 0,  /* Amino Acid Positional           */
    BEAM_VARIANT_NDP       = 1,  /* Nucleotide Double Positional    */
    BEAM_VARIANT_NAIVE     = 2,  /* Naive LZ76 (no position)        */
    BEAM_VARIANT_NAIVE_POS = 3,  /* Naive positional: "{AA}_{pos}"  */
    BEAM_VARIANT_FLAT_FB   = 4,  /* Flattened FlashBack: "{a}{b}_{k}" */
} BeamVariant;

/* ── Compiler hints ─────────────────────────────────────────── */

#if defined(__GNUC__) || defined(__clang__)
  #define BEAM_LIKELY(x)   __builtin_expect(!!(x), 1)
  #define BEAM_UNLIKELY(x) __builtin_expect(!!(x), 0)
  #define BEAM_INLINE       static inline __attribute__((always_inline))
#else
  #define BEAM_LIKELY(x)   (x)
  #define BEAM_UNLIKELY(x) (x)
  #define BEAM_INLINE       static inline
#endif


/* ── Numerical constants ────────────────────────────────────── */

#define BEAM_EPS       1e-300
#define BEAM_LOG_EPS  (-690.7755278982137)  /* log(1e-300) */
#define BEAM_NAN       ((double)NAN)

/* ── Alphabet ──────────────────────────────────────────────── */

#define BEAM_AA_COUNT  20

/** Sentinel characters for sequence boundaries. */
#define BEAM_START_SENTINEL '@'
#define BEAM_END_SENTINEL   '$'

/**
 * Map a character to a bit index for the walk dictionary bitmask.
 *
 * Bits 0-19:  Standard amino acids (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)
 * Bit 20:     Start sentinel '@'
 * Bit 21:     End sentinel '$'
 * Bit 31:     Non-standard (ignored in bitmask checks)
 */
uint8_t beam_aa_to_bit(char c);  /* implemented in src/native/core/common.c */

/**
 * Compute the character bitmask for a string of length `len`.
 * Each amino acid sets its corresponding bit in the uint32.
 */
BEAM_INLINE uint32_t beam_aa_bitmask(const char *s, uint32_t len) {
    uint32_t mask = 0;
    for (uint32_t i = 0; i < len; i++)
        mask |= (1u << beam_aa_to_bit(s[i]));
    return mask;
}

#endif /* BEAM_COMMON_H */
