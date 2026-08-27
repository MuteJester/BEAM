/**
 * @file errors.c
 * @brief Thread-local error code and message, set by every failing entry point.
 */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "beam/common.h"

/* ── Thread-local error context ────────────────────────────── */

#define BEAM_ERR_MSG_SIZE 512

/* Thread-local storage: C11 _Thread_local, or MSVC __declspec(thread) */
#ifdef _MSC_VER
  #define TLS __declspec(thread)
#else
  #define TLS _Thread_local
#endif

static TLS BeamError  tl_error_code = BEAM_OK;
static TLS char      tl_error_msg[BEAM_ERR_MSG_SIZE] = "";

const char *beam_error_message(void) {
    return tl_error_msg;
}

BeamError beam_last_error(void) {
    return tl_error_code;
}

void beam_set_error(BeamError code, const char *fmt, ...) {
    tl_error_code = code;
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(tl_error_msg, BEAM_ERR_MSG_SIZE, fmt, ap);
        va_end(ap);
    } else {
        tl_error_msg[0] = '\0';
    }
}

void beam_clear_error(void) {
    tl_error_code = BEAM_OK;
    tl_error_msg[0] = '\0';
}
