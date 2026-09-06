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

/* ── Logging ────────────────────────────────────────────────── */

/**
 * Log severity levels (libgit2-style severity ladder).
 *
 * Set the max level via beam_log_set(). Messages above the threshold
 * are silently dropped (zero overhead if level < threshold).
 */
typedef enum {
    BEAM_LOG_NONE  = 0,   /* logging disabled                          */
    BEAM_LOG_ERROR = 1,   /* non-fatal errors worth reporting          */
    BEAM_LOG_WARN  = 2,   /* recoverable issues (skipped seqs, slow)   */
    BEAM_LOG_INFO  = 3,   /* progress and timing                       */
    BEAM_LOG_DEBUG = 4,   /* algorithm decisions, internal metrics      */
    BEAM_LOG_TRACE = 5,   /* per-item detail (very verbose)            */
} BeamLogLevel;

/**
 * Log callback signature.
 *
 * @param level   Severity of the message.
 * @param msg     Null-terminated message string.
 * @param data    User-provided context pointer (from beam_log_set).
 */
typedef void (*beam_log_fn)(BeamLogLevel level, const char *msg, void *data);

/**
 * Set the global log callback and maximum verbosity level.
 *
 * Only messages with level <= max_level are delivered to the callback.
 * Pass callback=NULL or max_level=BEAM_LOG_NONE to disable logging.
 *
 * @param max_level  Maximum level to emit (BEAM_LOG_NONE disables all).
 * @param callback   Function to receive messages. NULL = disable.
 * @param data       User context passed to every callback invocation.
 *
 * Example:
 *   void my_logger(BeamLogLevel lvl, const char *msg, void *data) {
 *       fprintf(stderr, "[BEAM %d] %s\n", lvl, msg);
 *   }
 *   beam_log_set(BEAM_LOG_INFO, my_logger, NULL);
 */
void beam_log_set(BeamLogLevel max_level, beam_log_fn callback, void *data);

/**
 * Get the current log level (for fast short-circuit checks).
 */
BeamLogLevel beam_log_level(void);

/* Internal: emit a log message (do not call directly — use macros). */
void beam_log_emit(BeamLogLevel level, const char *fmt, ...);

/**
 * Log macros — the primary interface for library code.
 *
 * These check the level BEFORE formatting, so disabled levels have
 * zero cost (no vsnprintf call). Define BEAM_NO_LOG at compile time
 * to eliminate all logging code entirely.
 *
 * Usage:
 *   BEAM_INFO("graph built: %u nodes, %u edges (%.1f ms)", nn, ne, ms);
 *   BEAM_WARN("sequence '%s' skipped: length 0", seq);
 *   BEAM_DEBUG("string pool: %u slots, load=%.2f", cap, load);
 */
#ifdef BEAM_NO_LOG
  #define BEAM_LOG(lvl, ...) ((void)0)
#else
  #define BEAM_LOG(lvl, ...) \
      do { if ((lvl) <= beam_log_level()) beam_log_emit((lvl), __VA_ARGS__); } while(0)
#endif

#define BEAM_ERROR(...) BEAM_LOG(BEAM_LOG_ERROR, __VA_ARGS__)
#define BEAM_WARN(...)  BEAM_LOG(BEAM_LOG_WARN,  __VA_ARGS__)
#define BEAM_INFO(...)  BEAM_LOG(BEAM_LOG_INFO,  __VA_ARGS__)
#define BEAM_DEBUG(...) BEAM_LOG(BEAM_LOG_DEBUG, __VA_ARGS__)
#define BEAM_TRACE(...) BEAM_LOG(BEAM_LOG_TRACE, __VA_ARGS__)

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
