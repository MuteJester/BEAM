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

/* ── Result / error codes ───────────────────────────────────── */

typedef enum {
    BEAM_OK = 0,

    /* Memory */
    BEAM_ERR_ALLOC,                /* malloc/calloc/realloc failed   */

    /* Input validation */
    BEAM_ERR_NULL_ARG,             /* required pointer is NULL       */
    BEAM_ERR_EMPTY_INPUT,          /* empty sequence list or array   */
    BEAM_ERR_INVALID_SEQUENCE,     /* bad characters or length       */
    BEAM_ERR_INVALID_VARIANT,      /* unknown BeamVariant value       */
    BEAM_ERR_LENGTH_MISMATCH,      /* array lengths don't match      */
    BEAM_ERR_PARAM_OUT_OF_RANGE,   /* parameter outside valid range  */

    /* Graph state */
    BEAM_ERR_NOT_BUILT,            /* graph not yet built/finalized  */
    BEAM_ERR_NO_LIVE_PATHS,        /* no live initial states exist   */
    BEAM_ERR_HAS_CYCLES,           /* graph is not a DAG             */

    /* Gene data */
    BEAM_ERR_NO_GENE_DATA,         /* gene op on graph without genes */
    BEAM_ERR_GENE_NOT_FOUND,       /* gene name not in string pool   */

    /* Graph operations */
    BEAM_ERR_VARIANT_MISMATCH,     /* set op on different variants   */
    BEAM_ERR_MISSING_EDGE,         /* walk requires a missing edge   */

    /* IO */
    BEAM_ERR_IO_OPEN,              /* cannot open file               */
    BEAM_ERR_IO_READ,              /* read error or unexpected EOF   */
    BEAM_ERR_IO_WRITE,             /* write error                    */
    BEAM_ERR_IO_CORRUPT,           /* CRC mismatch or bad structure  */
    BEAM_ERR_IO_VERSION,           /* unsupported format version     */

    /* Numerical */
    BEAM_ERR_CONVERGENCE,          /* iterative method did not converge */

    /* Internal (library bugs — should never reach the user) */
    BEAM_ERR_OVERFLOW,             /* fixed-size buffer exceeded     */
    BEAM_ERR_INTERNAL,             /* invariant violation (report!)  */

    /* Backward compatibility aliases (map old codes to new) */
    BEAM_ERR_INVALID_ARG   = BEAM_ERR_NULL_ARG,
    BEAM_ERR_NOT_FINALIZED = BEAM_ERR_NOT_BUILT,
    BEAM_ERR_EMPTY_DATA    = BEAM_ERR_EMPTY_INPUT,
    BEAM_ERR_IO            = BEAM_ERR_IO_OPEN,
} BeamError;

/* ── Thread-local error context ────────────────────────────── */

/**
 * Get the human-readable error message for the last error.
 * Returns "" if no error has been set. Thread-safe (thread-local storage).
 */
const char *beam_error_message(void);

/**
 * Get the error code of the last error.
 */
BeamError beam_last_error(void);

/**
 * Set the error context. Called internally before returning an error code.
 * Supports printf-style formatting.
 *
 * Usage:
 *   beam_set_error(BEAM_ERR_NULL_ARG, "sequences must not be NULL");
 *   return BEAM_ERR_NULL_ARG;
 *
 * Or use the convenience macro:
 *   return BEAM_FAIL(BEAM_ERR_NULL_ARG, "sequences must not be NULL");
 */
void beam_set_error(BeamError code, const char *fmt, ...);

/**
 * Clear the error state.
 */
void beam_clear_error(void);

/**
 * Convenience macro: set error and return the code in one line.
 */
#define BEAM_FAIL(code, ...) (beam_set_error((code), __VA_ARGS__), (code))

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
uint8_t beam_aa_to_bit(char c);  /* implemented in lib/core/common.c */

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
