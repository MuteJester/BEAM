/**
 * @file logging.c
 * @brief Optional log sink. Silent unless a caller installs a callback.
 */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "beam/common.h"

/* ── Logging ────────────────────────────────────────────────── */

static BeamLogLevel  g_log_level = BEAM_LOG_NONE;
static beam_log_fn   g_log_cb    = NULL;
static void         *g_log_data  = NULL;

void beam_log_set(BeamLogLevel max_level, beam_log_fn callback, void *data) {
    g_log_level = callback ? max_level : BEAM_LOG_NONE;
    g_log_cb    = callback;
    g_log_data  = data;
}

BeamLogLevel beam_log_level(void) {
    return g_log_level;
}

void beam_log_emit(BeamLogLevel level, const char *fmt, ...) {
    if (!g_log_cb || level > g_log_level) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log_cb(level, buf, g_log_data);
}
