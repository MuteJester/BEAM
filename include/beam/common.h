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

#endif /* BEAM_COMMON_H */
