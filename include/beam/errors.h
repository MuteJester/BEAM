/**
 * @file errors.h
 * @brief How BEAM reports failure: the codes, the message context, and the
 *        two calling protocols.
 *
 * Split out of `common.h`, which is the header a new contributor reads first
 * and which had grown into a grab-bag. Everything here was already public;
 * nothing moved between translation units, and `common.h` includes this file
 * so every existing include still compiles unchanged.
 *
 * ── The two protocols ──────────────────────────────────────────────────────
 *
 * MOST entry points return `BeamError` and set a message alongside it. Check
 * the code; read `beam_error_message()` for the detail.
 *
 * A FEW return a `double` and cannot carry a code, because their answer
 * occupies the whole return type. These are:
 *
 *     beam_model_path_count          beam_model_path_count_total
 *     beam_model_accumulator_max     beam_model_log_pseq
 *     beam_model_log_partition       beam_model_log_length_prior
 *
 * Their contract: CLEAR the error on entry, SET it on failure, and return a
 * sentinel (0.0, or -INFINITY for the log-domain ones). A caller distinguishes
 * failure from a real answer by `beam_last_error() != BEAM_OK`, never by the
 * value -- 0.0 is a legitimate path count and -inf is the legitimate log
 * probability of a sequence the model cannot generate.
 *
 * That distinction is the whole reason this section exists. Four of these six
 * did not follow the protocol: `path_count_total` returned 0.0 on an
 * unfinalized model without setting anything, so the value read as "this model
 * generates nothing"; the two length accessors did the same with -inf. Every
 * one of those was a failure a caller could not tell from an answer.
 *
 * `beam_model_log_pseq` is the remaining exception and is deliberately left
 * alone for now: it returns -inf for input it cannot parse as well as for a
 * sequence of probability zero, which `beam_model_explain` reports as
 * BEAM_ERR_INVALID_SEQUENCE from the identical guard. Reconciling the two is a
 * contract change on the hottest function in the library and is pinned by
 * `tests/python/model/test_scoring.py`, so it is a decision rather than a
 * cleanup.
 *
 * ── Adding a code ──────────────────────────────────────────────────────────
 *
 * APPEND, never insert. Every member below is pinned to the value the
 * compiler gave it before this file existed, because the values were
 * positional and nothing recorded them: inserting a code in the natural place
 * renumbered twenty others, and `verification/check.py` pins the error
 * FUNCTION symbols but not the enum values, so it would not have noticed.
 *
 * The compatibility aliases share values with the codes they alias, so a
 * `switch` cannot carry a `case` for both halves of a pair -- that is a
 * duplicate-case error whose cause is not obvious from the message.
 */
#ifndef BEAM_ERRORS_H
#define BEAM_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result / error codes ───────────────────────────────────── */

typedef enum {
    BEAM_OK = 0,

    /* Memory */
    BEAM_ERR_ALLOC              = 1,   /* malloc/calloc/realloc failed      */

    /* Input validation */
    BEAM_ERR_NULL_ARG           = 2,   /* required pointer is NULL          */
    BEAM_ERR_EMPTY_INPUT        = 3,   /* empty sequence list or array      */
    BEAM_ERR_INVALID_SEQUENCE   = 4,   /* bad characters or length          */
    BEAM_ERR_INVALID_VARIANT    = 5,   /* unknown BeamVariant value         */
    BEAM_ERR_LENGTH_MISMATCH    = 6,   /* array lengths don't match         */
    BEAM_ERR_PARAM_OUT_OF_RANGE = 7,   /* parameter outside valid range     */

    /* Graph state */
    BEAM_ERR_NOT_BUILT          = 8,   /* graph not yet built/finalized     */
    BEAM_ERR_NO_LIVE_PATHS      = 9,   /* no live initial states exist      */
    BEAM_ERR_HAS_CYCLES         = 10,  /* graph is not a DAG                */

    /* Gene data */
    BEAM_ERR_NO_GENE_DATA       = 11,  /* gene op on graph without genes    */
    BEAM_ERR_GENE_NOT_FOUND     = 12,  /* gene name not in string pool      */

    /* Graph operations */
    BEAM_ERR_VARIANT_MISMATCH   = 13,  /* set op on different variants      */
    BEAM_ERR_MISSING_EDGE       = 14,  /* walk requires a missing edge      */

    /* IO */
    BEAM_ERR_IO_OPEN            = 15,  /* cannot open file                  */
    BEAM_ERR_IO_READ            = 16,  /* read error or unexpected EOF      */
    BEAM_ERR_IO_WRITE           = 17,  /* write error                       */
    BEAM_ERR_IO_CORRUPT         = 18,  /* CRC mismatch or bad structure     */
    BEAM_ERR_IO_VERSION         = 19,  /* unsupported format version        */

    /* Numerical */
    BEAM_ERR_CONVERGENCE        = 20,  /* iterative method did not converge */

    /* Internal (library bugs — should never reach the user) */
    BEAM_ERR_OVERFLOW           = 21,  /* fixed-size buffer exceeded        */
    BEAM_ERR_INTERNAL           = 22,  /* invariant violation (report!)     */

    /* Append new codes HERE, above the aliases, taking the next free value. */

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
 *
 * THREAD-LOCAL is load-bearing, not an implementation note. An error raised
 * inside an OpenMP worker is invisible to the thread that started the region,
 * so a parallel driver must either refuse before the region opens or read
 * `beam_last_error()` back inside each iteration. Both patterns are in use;
 * see `beam_budget_concurrency` and the per-length drivers in math/.
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
 *
 * A code without a message reaches Python as a bare "invalid argument", and
 * worse: the message buffer is only ever overwritten, never cleared by a bare
 * return, so the caller is served whatever unrelated failure came before.
 * Always pass a message.
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

#ifdef __cplusplus
}
#endif

#endif /* BEAM_ERRORS_H */
